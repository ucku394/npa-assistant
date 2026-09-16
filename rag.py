# ============================================================
# RAG.PY — поиск нормативных фрагментов
# Republic of Belarus
# ============================================================

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types
from supabase import Client

from config import (
    CHAT_MODEL,
    GEMINI_API_KEY,
    SUPABASE_MATCH_COUNT,
    SUPABASE_MATCH_THRESHOLD,
)
from embedding import get_query_embedding
from prompts import RAG_RELEVANCE_PROMPT

logger = logging.getLogger(__name__)

_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def _search_chunks(supabase: Client, query_vector: list[float]) -> list[dict[str, Any]]:
    response = supabase.rpc(
        "match_npa_chunks",
        {
            "query_embedding": query_vector,
            "match_threshold": SUPABASE_MATCH_THRESHOLD,
            "match_count": SUPABASE_MATCH_COUNT,
        },
    ).execute()
    return response.data or []


def _format_chunk(chunk: dict[str, Any]) -> str:
    doc_name = str(chunk.get("doc_name") or "НПА").strip()
    point_num = str(
        chunk.get("point_num")
        or chunk.get("article")
        or chunk.get("section")
        or "-"
    ).strip()
    content = str(chunk.get("content") or "").strip()
    return f"Документ: {doc_name}\nПункт/статья: {point_num}\nТекст:\n{content}"


def _semantic_score(chunk: dict[str, Any]) -> float:
    for key in ("similarity", "score"):
        value = chunk.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _score_chunk(user_query: str, chunk: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    # If Gemini is unavailable, semantic similarity remains the fallback signal.
    if _gemini is None:
        return -1, chunk

    prompt = RAG_RELEVANCE_PROMPT.format(
        user_query=user_query,
        chunk=_format_chunk(chunk),
    )

    try:
        response = _gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=5,
            ),
        )
        text = (response.text or "").strip()
        score = int(text)
        if score not in (0, 1, 2, 3):
            score = 0
        return score, chunk
    except Exception as exc:
        logger.warning("RAG reranking failed: %s", exc)
        return -1, chunk


async def _rerank_chunks(user_query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []

    tasks = [
        asyncio.to_thread(_score_chunk, user_query, chunk)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("RAG reranking result failed: %s", result)
            continue
        score, chunk = result
        item = dict(chunk)
        item["_rerank_score"] = score
        scored.append(item)

    # When reranking is unavailable (-1), semantic similarity decides the order.
    scored.sort(
        key=lambda item: (
            item.get("_rerank_score", -1),
            _semantic_score(item),
        ),
        reverse=True,
    )

    # Do not feed completely unrelated chunks to the answer model when
    # the reranker explicitly rejected them.
    useful = [x for x in scored if x.get("_rerank_score", -1) != 0]
    if not useful:
        useful = scored[:1]

    return useful[: max(1, min(6, SUPABASE_MATCH_COUNT))]


async def retrieve_context(user_query: str, supabase: Client) -> dict[str, Any]:
    if not user_query or not user_query.strip():
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    logger.info("RAG | search: %s", user_query)

    query_vector = await asyncio.to_thread(get_query_embedding, user_query)
    logger.info("RAG | embedding dimension: %s", len(query_vector))

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
    )
    logger.info("RAG | candidates: %s", len(candidate_chunks))

    if not candidate_chunks:
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
        }

    final_chunks = await _rerank_chunks(user_query, candidate_chunks)
    logger.info("RAG | final chunks: %s", len(final_chunks))

    context_parts = []
    for index, chunk in enumerate(final_chunks, 1):
        rerank = chunk.get("_rerank_score", -1)
        similarity = _semantic_score(chunk)
        metadata = f"\nРелевантность: {rerank}/3" if rerank >= 0 else ""
        if similarity:
            metadata += f"\nSemantic similarity: {similarity:.4f}"

        context_parts.append(
            f"===== ИСТОЧНИК {index} =====\n"
            f"{_format_chunk(chunk)}"
            f"{metadata}"
        )

    retrieved_text = "\n\n".join(context_parts)

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": len(candidate_chunks),
        "final_count": len(final_chunks),
    }


def get_source_names(chunks: list[dict[str, Any]]) -> list[str]:
    result = []
    for chunk in chunks:
        name = str(chunk.get("doc_name") or "НПА").strip()
        if name and name not in result:
            result.append(name)
    return result
