import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    RAG_MATCH_THRESHOLD,
    RAG_MATCH_COUNT,
    RAG_FINAL_COUNT,
)

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# EMBEDDING
# ============================================================

async def create_query_embedding(
    user_query: str,
) -> list[float]:
    """
    Создаёт embedding пользовательского вопроса.
    """

    def generate_embedding():

        return gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,

            contents=user_query,

            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )

    response = await asyncio.to_thread(
        generate_embedding
    )

    if not response or not response.embeddings:
        raise RuntimeError(
            "Gemini не вернул embedding."
        )

    return response.embeddings[0].values


# ============================================================
# SUPABASE SEARCH
# ============================================================

async def search_supabase(
    supabase,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    """
    Семантический поиск в Supabase.
    """

    def execute_search():

        return supabase.rpc(
            "match_npa_chunks",
            {
                "query_embedding": query_vector,

                "match_threshold":
                    RAG_MATCH_THRESHOLD,

                "match_count":
                    RAG_MATCH_COUNT,
            },
        ).execute()

    response = await asyncio.to_thread(
        execute_search
    )

    if not response:
        return []

    return response.data or []


# ============================================================
# FORMAT CHUNKS
# ============================================================

def format_chunk_for_reranking(
    index: int,
    chunk: dict[str, Any],
) -> str:
    """
    Формирует компактное представление
    одного нормативного фрагмента.
    """

    doc_name = chunk.get(
        "doc_name",
        "НПА",
    )

    point_num = chunk.get(
        "point_num",
        "-",
    )

    content = chunk.get(
        "content",
        "",
    )

    return (
        f"[CHUNK {index}]\n"
        f"Документ: {doc_name}\n"
        f"Пункт/статья: {point_num}\n"
        f"Текст: {content}"
    )


# ============================================================
# RERANK PROMPT
# ============================================================

RERANK_SYSTEM_PROMPT = """
Ты выполняешь reranking нормативных фрагментов
для экспертной системы по охране труда,
промышленной и пожарной безопасности
в Республике Беларусь.

Твоя задача — определить, насколько каждый
найденный фрагмент действительно помогает
ответить на вопрос пользователя.

Оцени каждый CHUNK по шкале:

0 — нерелевантен;
1 — косвенно связан;
2 — существенно связан;
3 — непосредственно отвечает на вопрос.

ВАЖНО:

- оценивай только предоставленный текст;
- не придумывай отсутствующие нормы;
- не используй законодательство РФ;
- не добавляй собственные юридические выводы;
- если фрагмент похож по словам, но не отвечает
  на вопрос по существу — ставь 0 или 1.

Ответ ОБЯЗАТЕЛЬНО должен быть валидным JSON.

Формат:

{
  "scores": [
    {
      "chunk": 0,
      "score": 3,
      "reason": "Краткое объяснение"
    }
  ]
}
"""


# ============================================================
# RERANK
# ============================================================

async def rerank_chunks(
    user_query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Оценивает все найденные чанки одним запросом Gemini.
    """

    if not chunks:
        return []

    formatted_chunks = []

    for index, chunk in enumerate(chunks):

        formatted_chunks.append(
            format_chunk_for_reranking(
                index,
                chunk,
            )
        )

    chunks_text = "\n\n".join(
        formatted_chunks
    )

    prompt = f"""
{RERANK_SYSTEM_PROMPT}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:

{user_query}

НАЙДЕННЫЕ ФРАГМЕНТЫ:

{chunks_text}
"""

    def call_gemini():

        return gemini_client.models.generate_content(
            model=CHAT_MODEL,

            contents=prompt,

            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "max_output_tokens": 2000,
            },
        )

    try:

        response = await asyncio.to_thread(
            call_gemini
        )

        if not response or not response.text:
            logger.warning(
                "Reranking: Gemini не вернул результат."
            )

            return chunks

        data = json.loads(
            response.text
        )

        scores = data.get(
            "scores",
            [],
        )

        score_map = {}

        for item in scores:

            try:

                index = int(
                    item.get("chunk")
                )

                score = int(
                    item.get("score", 0)
                )

                score = max(
                    0,
                    min(3, score)
                )

                score_map[index] = score

            except Exception:
                continue

        # ----------------------------------------------------
        # Добавляем оценки к чанкам
        # ----------------------------------------------------

        enriched = []

        for index, chunk in enumerate(chunks):

            chunk_copy = dict(chunk)

            chunk_copy["_rerank_score"] = (
                score_map.get(index, 0)
            )

            enriched.append(
                chunk_copy
            )

        # ----------------------------------------------------
        # Сначала релевантность Gemini,
        # затем исходное similarity
        # ----------------------------------------------------

        enriched.sort(
            key=lambda item: (
                item.get(
                    "_rerank_score",
                    0,
                ),

                float(
                    item.get(
                        "similarity",
                        0,
                    ) or 0
                ),
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Отбрасываем полностью нерелевантные
        # ----------------------------------------------------

        relevant = [
            chunk
            for chunk in enriched
            if chunk.get(
                "_rerank_score",
                0,
            ) >= 1
        ]

        if not relevant:
            return []

        return relevant[
            :RAG_FINAL_COUNT
        ]

    except Exception as error:

        logger.warning(
            "Ошибка reranking: %s",
            error,
            exc_info=True,
        )

        # Безопасный fallback:
        # возвращаем исходные результаты Supabase,
        # а не выдумываем оценки.
        return chunks[
            :RAG_FINAL_COUNT
        ]


# ============================================================
# FORMAT FINAL CONTEXT
# ============================================================

def format_context(
    chunks: list[dict[str, Any]],
) -> str:
    """
    Формирует нормативный контекст
    для основной модели.
    """

    if not chunks:
        return (
            "Релевантные нормативные "
            "фрагменты не найдены."
        )

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        doc_name = chunk.get(
            "doc_name",
            "НПА",
        )

        point_num = chunk.get(
            "point_num",
            "-",
        )

        content = chunk.get(
            "content",
            "",
        )

        similarity = chunk.get(
            "similarity"
        )

        block = (
            f"[ИСТОЧНИК {index}]\n"
            f"Документ: {doc_name}\n"
            f"Пункт/статья: {point_num}\n"
        )

        if similarity is not None:

            block += (
                f"Релевантность поиска: "
                f"{similarity}\n"
            )

        block += (
            f"Текст НПА:\n"
            f"{content}"
        )

        blocks.append(
            block
        )

    return "\n\n====================\n\n".join(
        blocks
    )


# ============================================================
# MAIN RAG
# ============================================================

async def retrieve_context(
    user_query: str,
    supabase,
) -> dict[str, Any]:
    """
    Полный RAG pipeline:

    1. Embedding
    2. Supabase search
    3. Gemini reranking
    4. Формирование финального контекста
    """

    logger.info(
        "RAG: поиск по запросу: %s",
        user_query[:300],
    )

    # --------------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------------

    query_vector = (
        await create_query_embedding(
            user_query
        )
    )

    # --------------------------------------------------------
    # 2. Semantic search
    # --------------------------------------------------------

    candidates = (
        await search_supabase(
            supabase,
            query_vector,
        )
    )

    logger.info(
        "RAG: найдено кандидатов: %s",
        len(candidates),
    )

    if not candidates:

        return {
            "found": False,
            "chunks": [],
            "retrieved_text": "",
            "candidate_count": 0,
            "final_count": 0,
        }

    # --------------------------------------------------------
    # 3. Reranking
    # --------------------------------------------------------

    final_chunks = (
        await rerank_chunks(
            user_query,
            candidates,
        )
    )

    logger.info(
        "RAG: после reranking: %s",
        len(final_chunks),
    )

    # --------------------------------------------------------
    # 4. Context
    # --------------------------------------------------------

    retrieved_text = (
        format_context(
            final_chunks
        )
    )

    return {
        "found": bool(final_chunks),

        "chunks": final_chunks,

        "retrieved_text":
            retrieved_text,

        "candidate_count":
            len(candidates),

        "final_count":
            len(final_chunks),
    }


# ============================================================
# SOURCES
# ============================================================

def get_source_names(
    chunks: list[dict[str, Any]],
) -> list[str]:
    """
    Возвращает уникальные названия НПА.
    """

    sources = []

    for chunk in chunks:

        name = chunk.get(
            "doc_name"
        )

        if (
            name
            and name not in sources
        ):

            sources.append(name)

    return sources
