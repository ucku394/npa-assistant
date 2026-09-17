import asyncio
import logging
import os
from typing import Any, Optional

from supabase import Client

from config import (
    SUPABASE_MATCH_THRESHOLD,
    SUPABASE_MATCH_COUNT,
)

from embedding import get_query_embedding

logger = logging.getLogger(__name__)

# Сколько документов передавать дальше в AI
RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))


def _safe_float(value: Any) -> Optional[float]:
    """
    Безопасно преобразует значение similarity/score в float.
    """
    try:
        if value is None or value == "":
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def _semantic_score(chunk: dict[str, Any]) -> float:
    """
    Получает semantic similarity из результата Supabase.
    """

    value = _safe_float(
        chunk.get("similarity", chunk.get("score"))
    )

    return value if value is not None else 0.0


def _format_chunk(chunk: dict[str, Any]) -> str:
    """
    Приводит один найденный фрагмент НПА
    к удобному для AI формату.
    """

    doc_name = str(
        chunk.get("doc_name") or "НПА"
    ).strip()

    point_num = (
        chunk.get("point_num")
        or chunk.get("article")
        or chunk.get("section")
        or ""
    )

    point_num = str(point_num).strip()

    content = str(
        chunk.get("content") or ""
    ).strip()

    result = f"Документ: {doc_name}\n"

    if point_num:
        result += f"Пункт/статья: {point_num}\n"

    result += f"Текст:\n{content}"

    return result


def _search_chunks(
    supabase: Client,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    """
    Выполняет векторный поиск в Supabase.
    """

    response = supabase.rpc(
        "match_npa_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": SUPABASE_MATCH_THRESHOLD,
            "match_count": SUPABASE_MATCH_COUNT,
        },
    ).execute()

    return response.data or []


def _sort_by_semantic_similarity(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Сортирует найденные фрагменты по semantic similarity.

    Чем выше similarity, тем ближе НПА
    к заданному пользователем вопросу.
    """

    result = []

    for index, chunk in enumerate(chunks):

        item = dict(chunk)

        item["_semantic_score"] = _semantic_score(chunk)

        # Сохраняем исходную позицию.
        # Она используется как дополнительный стабильный критерий.
        item["_original_index"] = index

        result.append(item)

    result.sort(
        key=lambda item: (
            item["_semantic_score"],
            -item["_original_index"],
        ),
        reverse=True,
    )

    return result


def _build_retrieved_text(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Формирует контекст НПА для передачи в AI.
    """

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_text = _format_chunk(chunk)

        semantic = chunk.get(
            "_semantic_score"
        )

        metadata = []

        if semantic is not None:
            metadata.append(
                f"semantic similarity: {semantic:.4f}"
            )

        metadata_text = ""

        if metadata:
            metadata_text = (
                "\n"
                + "\n".join(metadata)
            )

        blocks.append(
            f"===== ИСТОЧНИК {index} =====\n"
            f"{chunk_text}"
            f"{metadata_text}"
        )

    return "\n\n".join(blocks)


async def retrieve_context(
    user_query: str,
    supabase: Client,
) -> dict[str, Any]:
    """
    Основная функция RAG.

    Алгоритм:

    1. Получаем embedding вопроса через E5.
    2. Ищем похожие фрагменты НПА в Supabase.
    3. Сортируем их по semantic similarity.
    4. Берём TOP-N.
    5. Формируем контекст для AI.

    Gemini здесь НЕ используется.
    """

    user_query = (
        user_query or ""
    ).strip()

    if not user_query:

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    logger.info(
        "RAG | query=%s",
        user_query,
    )

    # =========================================================
    # ШАГ 1. EMBEDDING ВОПРОСА
    # =========================================================

    query_vector = await asyncio.to_thread(
        get_query_embedding,
        user_query,
    )

    logger.info(
        "RAG | embedding dimension=%s",
        len(query_vector),
    )

    if len(query_vector) != 384:

        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{len(query_vector)}; expected 384."
        )

    # =========================================================
    # ШАГ 2. ПОИСК В SUPABASE
    # =========================================================

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
    )

    logger.info(
        "RAG | candidates=%s",
        len(candidate_chunks),
    )

    if not candidate_chunks:

        logger.info(
            "RAG | no relevant chunks found"
        )

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    # =========================================================
    # ШАГ 3. СОРТИРОВКА ПО SEMANTIC SIMILARITY
    # =========================================================

    ranked_chunks = _sort_by_semantic_similarity(
        candidate_chunks
    )

    # =========================================================
    # ШАГ 4. TOP-N
    # =========================================================

    final_chunks = ranked_chunks[
        :RAG_FINAL_COUNT
    ]

    logger.info(
        "RAG | final chunks=%s",
        len(final_chunks),
    )

    # Логируем результаты для диагностики
    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        doc_name = str(
            chunk.get("doc_name") or "НПА"
        ).strip()

        point_num = (
            chunk.get("point_num")
            or chunk.get("article")
            or chunk.get("section")
            or ""
        )

        similarity = chunk.get(
            "_semantic_score",
            0.0,
        )

        logger.info(
            "RAG | TOP %s | similarity=%.4f | "
            "document=%s | point=%s",
            index,
            similarity,
            doc_name,
            point_num,
        )

    # =========================================================
    # ШАГ 5. ФОРМИРУЕМ КОНТЕКСТ
    # =========================================================

    retrieved_text = _build_retrieved_text(
        final_chunks
    )

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": len(candidate_chunks),
        "final_count": len(final_chunks),
    }


def get_source_references(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает список источников
    в формате:

    Документ — пункт/статья
    """

    result = []

    seen = set()

    for chunk in chunks:

        doc_name = str(
            chunk.get("doc_name") or "НПА"
        ).strip()

        point_num = (
            chunk.get("point_num")
            or chunk.get("article")
            or chunk.get("section")
            or ""
        )

        point_num = str(
            point_num
        ).strip()

        key = (
            doc_name,
            point_num,
        )

        if key in seen:
            continue

        seen.add(key)

        if point_num:

            result.append(
                f"{doc_name} — {point_num}"
            )

        else:

            result.append(
                doc_name
            )

    return result


def get_source_names(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает уникальные названия НПА.
    """

    result = []

    seen = set()

    for chunk in chunks:

        name = str(
            chunk.get("doc_name") or "НПА"
        ).strip()

        if name and name not in seen:

            seen.add(name)

            result.append(name)

    return result
