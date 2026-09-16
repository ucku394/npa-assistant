# ============================================================
# RAG.PY
# Поиск нормативных правовых актов
# Республика Беларусь
# ============================================================

import asyncio
import logging
import os
import re
from typing import Any, Optional

from google import genai
from supabase import Client

from config import (
    GEMINI_API_KEY,
    SUPABASE_MATCH_THRESHOLD,
    SUPABASE_MATCH_COUNT,
)

from embedding import get_query_embedding
from prompts import RAG_RELEVANCE_PROMPT


logger = logging.getLogger(__name__)

# Можно переопределить через переменные окружения, не меняя config.py.
RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() in {
    "1", "true", "yes", "on"
}
RAG_MIN_RERANK_SCORE = int(os.getenv("RAG_MIN_RERANK_SCORE", "1"))
RAG_RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "gemini-3.6-flash")

_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _semantic_score(chunk: dict[str, Any]) -> float:
    value = _safe_float(
        chunk.get("similarity", chunk.get("score"))
    )
    return value if value is not None else 0.0


def _format_chunk(chunk: dict[str, Any]) -> str:
    doc_name = str(chunk.get("doc_name") or "НПА").strip()

    point_num = (
        chunk.get("point_num")
        or chunk.get("article")
        or chunk.get("section")
        or ""
    )
    point_num = str(point_num).strip()

    content = str(chunk.get("content") or "").strip()

    result = f"Документ: {doc_name}\n"
    if point_num:
        result += f"Пункт/статья: {point_num}\n"
    result += f"Текст:\n{content}"

    return result


def _extract_rerank_score(text: Any) -> Optional[int]:
    """
    Gemini иногда возвращает не только '3', а '3.', 'Оценка: 3',
    пустую строку или другой служебный текст.

    Извлекаем только отдельную цифру 0..3.
    """
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None

    match = re.search(r"(?<!\d)([0-3])(?!\d)", cleaned)
    if not match:
        return None

    score = int(match.group(1))
    return score if score in (0, 1, 2, 3) else None


def _score_chunk(
    user_query: str,
    chunk: dict[str, Any],
) -> tuple[Optional[int], dict[str, Any]]:
    if _gemini is None:
        return None, chunk

    chunk_text = _format_chunk(chunk)

    prompt = RAG_RELEVANCE_PROMPT.format(
        user_query=user_query,
        chunk=chunk_text,
    )

    try:
        response = _gemini.models.generate_content(
            model=RAG_RERANK_MODEL,
            contents=prompt,
            config={
                "temperature": 0,
                "max_output_tokens": 8,
            },
        )

        raw_text = getattr(response, "text", "") or ""
        score = _extract_rerank_score(raw_text)

        if score is None:
            logger.warning(
                "RAG reranking returned invalid score | raw=%r | fallback=semantic",
                raw_text,
            )

        return score, chunk

    except Exception as exc:
        logger.warning(
            "RAG reranking failed | error=%s | fallback=semantic",
            exc,
        )
        return None, chunk


async def _rerank_chunks(
    user_query: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    if not RAG_RERANK_ENABLED or _gemini is None:
        logger.info("RAG | reranking disabled/unavailable; semantic ranking only.")
        result = [dict(chunk) for chunk in chunks]
        result.sort(key=_semantic_score, reverse=True)
        return result[:RAG_FINAL_COUNT]

    tasks = [
        asyncio.to_thread(_score_chunk, user_query, chunk)
        for chunk in chunks
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored = []

    for original_index, result in enumerate(results):
        chunk = chunks[original_index]

        if isinstance(result, Exception):
            logger.warning(
                "RAG reranking result failed | error=%s | fallback=semantic",
                result,
            )
            rerank_score = None
        else:
            rerank_score, _ = result

        item = dict(chunk)
        item["_rerank_score"] = rerank_score
        item["_semantic_score"] = _semantic_score(chunk)

        # Для сортировки:
        # - успешный rerank выше;
        # - semantic similarity используется как стабильный tie-breaker;
        # - при ошибке rerank не ставим искусственный score=1.
        effective_rerank = (
            rerank_score if rerank_score is not None else -1
        )
        item["_effective_rank"] = (
            effective_rerank,
            item["_semantic_score"],
            -original_index,
        )

        scored.append(item)

    scored.sort(
        key=lambda item: item["_effective_rank"],
        reverse=True,
    )

    # Если модель отдала только нулевые результаты, всё равно оставляем
    # наиболее близкие semantic chunks, чтобы не потерять контекст.
    positive = [
        item for item in scored
        if item.get("_rerank_score") is not None
        and item["_rerank_score"] >= RAG_MIN_RERANK_SCORE
    ]

    fallback = [
        item for item in scored
        if item.get("_rerank_score") is None
    ]

    if positive:
        selected = positive[:RAG_FINAL_COUNT]

        # Если положительных результатов мало, добавляем лучшие semantic
        # fallback-кандидаты, но только если они действительно близки.
        if len(selected) < RAG_FINAL_COUNT:
            for item in fallback:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= RAG_FINAL_COUNT:
                    break
    else:
        selected = sorted(
            scored,
            key=lambda item: item["_semantic_score"],
            reverse=True,
        )[:RAG_FINAL_COUNT]

    return selected


def _search_chunks(
    supabase: Client,
    query_vector: list[float],
) -> list[dict[str, Any]]:
    response = supabase.rpc(
        "match_npa_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": SUPABASE_MATCH_THRESHOLD,
            "match_count": SUPABASE_MATCH_COUNT,
        },
    ).execute()

    return response.data or []


def _build_retrieved_text(chunks: list[dict[str, Any]]) -> str:
    blocks = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_text = _format_chunk(chunk)

        rerank_score = chunk.get("_rerank_score")
        semantic = chunk.get("_semantic_score")

        metadata = []

        if rerank_score is not None:
            metadata.append(f"RAG-релевантность: {rerank_score}/3")

        if semantic:
            metadata.append(
                f"semantic similarity: {semantic:.4f}"
            )

        metadata_text = ""
        if metadata:
            metadata_text = "\n" + "\n".join(metadata)

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
    Основной RAG pipeline:

    1. Local E5 query embedding.
    2. Supabase vector search.
    3. Gemini reranking.
    4. Final context.
    """

    user_query = (user_query or "").strip()

    if not user_query:
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    logger.info("RAG | query=%s", user_query)

    # Local E5, а не Gemini Embedding API.
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
            f"Unexpected embedding dimension: {len(query_vector)}; expected 384."
        )

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
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    final_chunks = await _rerank_chunks(
        user_query,
        candidate_chunks,
    )

    logger.info(
        "RAG | final chunks=%s",
        len(final_chunks),
    )

    retrieved_text = _build_retrieved_text(final_chunks)

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
    Возвращает уникальные ссылки только по финальным chunks,
    которые реально были переданы модели.
    """

    result = []
    seen = set()

    for chunk in chunks:
        doc_name = str(chunk.get("doc_name") or "НПА").strip()

        point_num = (
            chunk.get("point_num")
            or chunk.get("article")
            or chunk.get("section")
            or ""
        )
        point_num = str(point_num).strip()

        key = (doc_name, point_num)

        if key in seen:
            continue

        seen.add(key)

        if point_num:
            result.append(f"{doc_name} — {point_num}")
        else:
            result.append(doc_name)

    return result


def get_source_names(
    chunks: list[dict[str, Any]],
) -> list[str]:
    result = []
    seen = set()

    for chunk in chunks:
        name = str(chunk.get("doc_name") or "НПА").strip()

        if name and name not in seen:
            seen.add(name)
            result.append(name)

    return result
