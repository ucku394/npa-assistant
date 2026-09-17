import asyncio
import logging
import os
import re
from typing import Any, Dict, List

from embeddings import get_query_embedding

logger = logging.getLogger(__name__)

RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_identifier(value: Any) -> str:
    """
    Делает безопасную часть SOURCE_ID.
    """
    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_./-]+", "", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def _get_document_name(chunk: Dict[str, Any]) -> str:
    """
    Пытаемся определить название НПА из разных возможных полей БД.
    """
    for key in (
        "document",
        "document_name",
        "doc_name",
        "title",
        "npa_name",
        "source",
    ):
        value = chunk.get(key)

        if value:
            return str(value).strip()

    return "Неизвестный НПА"


def _get_point_number(chunk: Dict[str, Any]) -> str:
    """
    Получаем пункт/статью/раздел.
    """
    for key in (
        "point",
        "point_number",
        "article",
        "article_number",
        "paragraph",
        "section",
    ):
        value = chunk.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def _extract_npa_number(document_name: str) -> str:
    """
    Пытаемся получить номер НПА из его названия.

    Например:
    'Постановление № 175 от 28.11.2008'
    -> '175'
    """
    if not document_name:
        return ""

    patterns = [
        r"№\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
        r"N\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, document_name, flags=re.IGNORECASE)

        if match:
            return match.group(1)

    return ""


def build_source_id(chunk: Dict[str, Any], index: int = 0) -> str:
    """
    Создаёт стабильный идентификатор источника.

    Приоритет:
    1. source_id из БД
    2. npa_number + point
    3. название документа + point
    4. fallback по индексу
    """

    # Если SOURCE_ID уже есть в БД — используем его.
    existing_source_id = (
        chunk.get("source_id")
        or chunk.get("_source_id")
    )

    if existing_source_id:
        return _normalize_identifier(existing_source_id)

    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)

    npa_number = _extract_npa_number(document_name)

    if npa_number:
        base = f"NPA_{npa_number}"
    else:
        base = f"NPA_{_normalize_identifier(document_name)[:80]}"

    if point:
        normalized_point = _normalize_identifier(point)
        return f"{base}_P{normalized_point}"

    if index:
        return f"{base}_CHUNK{index}"

    return base


# ============================================================
# ФОРМАТИРОВАНИЕ
# ============================================================

def _format_chunk(chunk: Dict[str, Any], index: int) -> str:
    """
    Форматирует один RAG-фрагмент для передачи AI.
    """

    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    )

    similarity = _safe_float(
        chunk.get("similarity")
        or chunk.get("score")
        or chunk.get("distance"),
        0.0,
    )

    source_id = build_source_id(chunk, index)

    chunk["_source_id"] = source_id

    lines = [
        f"SOURCE_ID: {source_id}",
        f"DOCUMENT: {document_name}",
    ]

    if point:
        lines.append(f"POINT_OR_ARTICLE: {point}")

    lines.append(f"TEXT: {str(text).strip()}")

    # Similarity нужен для внутреннего контроля,
    # но модель не должна использовать его как юридическое доказательство.
    lines.append(f"SEARCH_SIMILARITY: {similarity:.4f}")

    return "\n".join(lines)


def _build_retrieved_text(chunks: List[Dict[str, Any]]) -> str:
    """
    Формирует весь RAG-контекст.
    """

    blocks = []

    for index, chunk in enumerate(chunks, start=1):
        source_id = build_source_id(chunk, index)

        chunk["_source_id"] = source_id

        block = [
            f"===== RAG SOURCE {index} =====",
            _format_chunk(chunk, index),
            f"===== END RAG SOURCE {index} =====",
        ]

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


# ============================================================
# SUPABASE
# ============================================================

def _search_chunks(
    supabase,
    query_vector: List[float],
) -> List[Dict[str, Any]]:

    response = (
        supabase
        .rpc(
            "match_npa_chunks",
            {
                "query_embedding": query_vector,
                "match_count": max(RAG_FINAL_COUNT * 3, 15),
            },
        )
        .execute()
    )

    return response.data or []


def _semantic_score(chunk: Dict[str, Any]) -> float:
    """
    Унифицированное получение similarity.
    """

    if "similarity" in chunk:
        return _safe_float(chunk["similarity"])

    if "score" in chunk:
        return _safe_float(chunk["score"])

    # Если RPC возвращает distance,
    # меньшая distance = большая близость.
    if "distance" in chunk:
        distance = _safe_float(chunk["distance"])
        return 1.0 - distance

    return 0.0


def _sort_by_semantic_similarity(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    return sorted(
        chunks,
        key=_semantic_score,
        reverse=True,
    )


# ============================================================
# ОСНОВНОЙ RAG
# ============================================================

async def retrieve_context(
    user_query: str,
    supabase,
) -> Dict[str, Any]:

    logger.info(
        "RAG | query=%s",
        user_query,
    )

    # --------------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------------

    query_vector = await asyncio.to_thread(
        get_query_embedding,
        user_query,
    )

    if not query_vector:
        logger.warning("RAG | embedding is empty")

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
        }

    if len(query_vector) != 384:
        raise ValueError(
            f"Unexpected embedding dimension: {len(query_vector)}. "
            f"Expected 384."
        )

    # --------------------------------------------------------
    # 2. Search Supabase
    # --------------------------------------------------------

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
    )

    candidate_count = len(candidate_chunks)

    logger.info(
        "RAG | candidates=%s",
        candidate_count,
    )

    if not candidate_chunks:
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
        }

    # --------------------------------------------------------
    # 3. Semantic sorting
    # --------------------------------------------------------

    ranked_chunks = _sort_by_semantic_similarity(
        candidate_chunks
    )

    # --------------------------------------------------------
    # 4. TOP-N
    # --------------------------------------------------------

    final_chunks = ranked_chunks[:RAG_FINAL_COUNT]

    # --------------------------------------------------------
    # 5. SOURCE_ID
    # --------------------------------------------------------

    source_references = []

    for index, chunk in enumerate(final_chunks, start=1):

        source_id = build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = source_id

        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = (
                f"{document_name} — пункт/статья {point}"
            )
        else:
            reference = document_name

        source_references.append(
            {
                "source_id": source_id,
                "reference": reference,
            }
        )

        logger.info(
            "RAG | TOP %s | source=%s | similarity=%.4f",
            index,
            source_id,
            _semantic_score(chunk),
        )

    # --------------------------------------------------------
    # 6. Context
    # --------------------------------------------------------

    retrieved_text = _build_retrieved_text(
        final_chunks
    )

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": candidate_count,
        "final_count": len(final_chunks),
        "source_references": source_references,
    }


# ============================================================
# ИСТОЧНИКИ
# ============================================================

def get_source_references(
    chunks: List[Dict[str, Any]],
) -> List[str]:

    references = []
    seen = set()

    for index, chunk in enumerate(chunks, start=1):

        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = (
                f"{document_name} — пункт/статья {point}"
            )
        else:
            reference = document_name

        if reference not in seen:
            seen.add(reference)
            references.append(reference)

    return references


def get_source_names(
    chunks: List[Dict[str, Any]],
) -> List[str]:

    names = []
    seen = set()

    for chunk in chunks:

        document_name = _get_document_name(chunk)

        if document_name not in seen:
            seen.add(document_name)
            names.append(document_name)

    return names
