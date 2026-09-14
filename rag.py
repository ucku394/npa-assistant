# ============================================================
# RAG.PY
# Система поиска нормативных правовых актов
# Республика Беларусь
# ============================================================

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types
from supabase import Client

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    RAG_MATCH_THRESHOLD,
    RAG_MATCH_COUNT,
    RAG_FINAL_COUNT,
)

from prompts import RAG_RELEVANCE_PROMPT


logger = logging.getLogger(__name__)


# ============================================================
# Gemini
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ============================================================

def _create_embedding(user_query: str) -> list[float]:
    """
    Создаёт embedding пользовательского вопроса.
    Выполняется синхронно — вызывается через asyncio.to_thread().
    """

    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=user_query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768,
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini не вернул embedding для запроса."
        )

    return response.embeddings[0].values


def _search_chunks(
    supabase: Client,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    """
    Ищет нормативные фрагменты через существующую
    функцию Supabase match_npa_chunks.
    """

    response = supabase.rpc(
        "match_npa_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": RAG_MATCH_THRESHOLD,
            "match_count": RAG_MATCH_COUNT,
        }
    ).execute()

    if not response.data:
        return []

    return response.data


def _format_chunk(chunk: dict[str, Any]) -> str:
    """
    Приводит chunk из Supabase к единому текстовому виду.
    """

    doc_name = chunk.get("doc_name") or "НПА"

    point_num = (
        chunk.get("point_num")
        or chunk.get("article")
        or chunk.get("section")
        or "-"
    )

    content = chunk.get("content") or ""

    return (
        f"Документ: {doc_name}\n"
        f"Пункт/статья: {point_num}\n"
        f"Текст:\n{content}"
    )


# ============================================================
# RERANKING
# ============================================================

def _score_chunk(
    user_query: str,
    chunk: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """
    Оценивает релевантность одного нормативного фрагмента.

    Возвращает:
        (score, chunk)
    """

    chunk_text = _format_chunk(chunk)

    prompt = RAG_RELEVANCE_PROMPT.format(
        user_query=user_query,
        chunk=chunk_text,
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=10,
            ),
        )

        text = (response.text or "").strip()

        # Защита от неожиданного ответа модели
        score = int(text)

        if score not in (0, 1, 2, 3):
            score = 0

    except Exception as exc:
        logger.warning(
            "Ошибка reranking chunk: %s",
            exc
        )

        # При ошибке не выбрасываем chunk полностью.
        # Оставляем исходную semantic similarity,
        # если она присутствует.
        score = 1

    return score, chunk


async def _rerank_chunks(
    user_query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Проверяет релевантность найденных chunks.

    Сначала получаем 10 кандидатов,
    затем оставляем наиболее релевантные.
    """

    if not chunks:
        return []

    tasks = [
        asyncio.to_thread(
            _score_chunk,
            user_query,
            chunk,
        )
        for chunk in chunks
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    scored_chunks = []

    for result in results:

        if isinstance(result, Exception):
            logger.warning(
                "Ошибка обработки reranking result: %s",
                result,
            )
            continue

        score, chunk = result

        chunk_copy = dict(chunk)
        chunk_copy["_rerank_score"] = score

        scored_chunks.append(chunk_copy)

    # Сначала более высокий rerank score.
    # Затем исходная similarity, если она есть.
    scored_chunks.sort(
        key=lambda item: (
            item.get("_rerank_score", 0),
            item.get("similarity", 0),
            item.get("score", 0),
        ),
        reverse=True,
    )

    return scored_chunks[:RAG_FINAL_COUNT]


# ============================================================
# ПОЛУЧЕНИЕ КОНТЕКСТА
# ============================================================

async def retrieve_context(
    user_query: str,
    supabase: Client,
) -> dict[str, Any]:
    """
    Главная функция RAG.

    Возвращает:
        {
            "chunks": [...],
            "retrieved_text": "...",
            "found": True/False,
            "candidate_count": 10,
            "final_count": 5
        }
    """

    if not user_query or not user_query.strip():
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    logger.info(
        "RAG: поиск по запросу: %s",
        user_query,
    )

    # --------------------------------------------------------
    # Шаг 1. Embedding
    # --------------------------------------------------------

    query_vector = await asyncio.to_thread(
        _create_embedding,
        user_query,
    )

    logger.info(
        "RAG: embedding создан."
    )

    # --------------------------------------------------------
    # Шаг 2. Semantic search
    # --------------------------------------------------------

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
    )

    logger.info(
        "RAG: найдено кандидатов: %s",
        len(candidate_chunks),
    )

    if not candidate_chunks:
        return {
            "chunks": [],
            "retrieved_text": (
                "Релевантные нормативные акты "
                "в доступной базе не найдены."
            ),
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    # --------------------------------------------------------
    # Шаг 3. Reranking
    # --------------------------------------------------------

    final_chunks = await _rerank_chunks(
        user_query,
        candidate_chunks,
    )

    logger.info(
        "RAG: после reranking: %s",
        len(final_chunks),
    )

    # --------------------------------------------------------
    # Шаг 4. Формирование контекста
    # --------------------------------------------------------

    context_parts = []

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        chunk_text = _format_chunk(chunk)

        rerank_score = chunk.get(
            "_rerank_score",
            0,
        )

        similarity = chunk.get(
            "similarity",
            chunk.get("score", None),
        )

        metadata = (
            f"\nРелевантность: {rerank_score}/3"
        )

        if similarity is not None:
            metadata += (
                f"\nSemantic similarity: {similarity}"
            )

        context_parts.append(
            f"===== ИСТОЧНИК {index} =====\n"
            f"{chunk_text}"
            f"{metadata}"
        )

    retrieved_text = "\n\n".join(
        context_parts
    )

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": len(candidate_chunks),
        "final_count": len(final_chunks),
    }


# ============================================================
# ИСТОЧНИКИ
# ============================================================

def get_source_names(
    chunks: list[dict[str, Any]]
) -> list[str]:
    """
    Возвращает уникальные названия НПА.
    """

    sources = []

    for chunk in chunks:

        name = chunk.get(
            "doc_name",
            "НПА",
        )

        if name and name not in sources:
            sources.append(name)

    return sources