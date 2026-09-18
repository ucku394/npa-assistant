import asyncio
import hashlib
import logging
import os
import re
from typing import Any, Dict, List

from embedding import get_query_embedding

logger = logging.getLogger(__name__)

RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))

# Для юридического RAG лучше дать модели больше кандидатов,
# а затем выбрать наиболее релевантные.
RAG_CANDIDATE_COUNT = max(
    RAG_FINAL_COUNT * 6,
    30,
)


# ============================================================
# ОПРЕДЕЛЕНИЕ ОБЛАСТИ ЗАПРОСА
# ============================================================

def detect_legal_domain(user_query: str) -> str:
    """
    Определяет основную правовую область запроса.

    Приоритет:
    1. промышленная безопасность
    2. пожарная безопасность
    3. электробезопасность
    4. охрана труда
    5. санитарные требования
    6. general
    """

    query = str(user_query or "").strip().lower()

    if not query:
        return "general"

    # --------------------------------------------------------
    # ПРОМЫШЛЕННАЯ БЕЗОПАСНОСТЬ
    # --------------------------------------------------------

    industrial_patterns = [
        r"\bпромышленн\w*\s+безопасност\w*",
        r"\bопасн\w*\s+производственн\w*\s+объект\w*",
        r"\bпотенциально\s+опасн\w*\s+объект\w*",
        r"\bпоо\b",
        r"\bопо\b",
        r"\bавари\w*\s+на\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bтехническ\w*\s+расследован\w*\s+авари\w*",
        r"\bрегистрац\w*\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bэкспертиз\w*\s+промышленн\w*\s+безопасност\w*",
        r"\bэксплуатац\w*\s+опасн\w*\s+производственн\w*\s+объект\w*",
        r"\bлиценз\w*\s+промышленн\w*\s+безопасност\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in industrial_patterns
    ):
        return "industrial_safety"

    # --------------------------------------------------------
    # ПОЖАРНАЯ БЕЗОПАСНОСТЬ
    # --------------------------------------------------------

    fire_patterns = [
        r"\bпожарн\w*\s+безопасност\w*",
        r"\bпротивопожарн\w*",
        r"\bпожар\w*",
        r"\bогнетушител\w*",
        r"\bэвакуац\w*",
        r"\bпожарн\w*\s+сигнализац\w*",
        r"\bсистем\w*\s+оповещен\w*\s+о\s+пожар\w*",
        r"\bвзрывопожароопасн\w*",
        r"\bпожароопасн\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in fire_patterns
    ):
        return "fire_safety"

    # --------------------------------------------------------
    # ЭЛЕКТРОБЕЗОПАСНОСТЬ
    # --------------------------------------------------------

    electrical_patterns = [
        r"\bэлектробезопасност\w*",
        r"\bгрупп\w*\s+по\s+электробезопасност\w*",
        r"\bэлектроустановк\w*",
        r"\bэлектроустановк\w*\s+до\s+\d+\s*кВ",
        r"\bэлектроустановк\w*\s+свыше\s+\d+\s*кВ",
        r"\bэлектротехническ\w*\s+персонал\w*",
        r"\bэлектротехнологическ\w*\s+персонал\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in electrical_patterns
    ):
        return "electrical_safety"

    # --------------------------------------------------------
    # ОХРАНА ТРУДА
    # --------------------------------------------------------

    occupational_patterns = [
        r"\bохран\w*\s+труд\w*",
        r"\bохране\s+труда\b",
        r"\bинструктаж\w*",
        r"\bстажировк\w*",
        r"\bсредств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bсиз\b",
        r"\bпровер\w*\s+знани\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bобучен\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bинструкци\w*\s+по\s+охран\w*\s+труд\w*",
        r"\bнесчастн\w*\s+случа\w*",
        r"\bпрофессиональн\w*\s+заболеван\w*",
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедосмотр\w*",
        r"\bработ\w*\s+на\s+высот\w*",
        r"\bпогрузочно-разгрузочн\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in occupational_patterns
    ):
        return "occupational_safety"

    # --------------------------------------------------------
    # САНИТАРНЫЕ ТРЕБОВАНИЯ
    # --------------------------------------------------------

    sanitary_patterns = [
        r"\bсанитар\w*",
        r"\bсанитарно-эпидемиологическ\w*",
        r"\bгигиен\w*",
        r"\bмикроклимат\w*",
        r"\bсанитарн\w*\s+норм\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in sanitary_patterns
    ):
        return "sanitary"

    return "general"


# ============================================================
# ОПРЕДЕЛЕНИЕ ТЕМЫ
# ============================================================

def detect_topic(user_query: str) -> str:
    """
    Определяет специализированную тему запроса.
    """

    query = str(user_query or "").strip().lower()

    if not query:
        return "general"

    # --------------------------------------------------------
    # НЕСЧАСТНЫЕ СЛУЧАИ
    # --------------------------------------------------------

    accident_patterns = [
        r"\bнесчастн\w*\s+случа\w*",
        r"\bрасследован\w*\s+несчастн\w*",
        r"\bучет\w*\s+несчастн\w*",
        r"\bпотерпевш\w*",
        r"\bтравм\w*\s+на\s+производств\w*",
        r"\bтравм\w*\s+работник\w*",
        r"\bпроисшеств\w*\s+на\s+производств\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in accident_patterns
    ):
        return "accident_investigation"

    # --------------------------------------------------------
    # АТТЕСТАЦИЯ РАБОЧИХ МЕСТ
    # --------------------------------------------------------

    workplace_attestation_patterns = [
        r"\bаттестаци\w*\s+рабоч\w*\s+мест\w*",
        r"\bоценк\w*\s+услов\w*\s+труд\w*",
        r"\bуслов\w*\s+труд\w*\s+при\s+аттестаци\w*",
        r"\bаттестаци\w*\s+по\s+услов\w*\s+труд\w*",
    ]

    if any(
        re.search(pattern, query, flags=re.IGNORECASE)
        for pattern in workplace_attestation_patterns
    ):
        return "workplace_attestation"

    return "general"


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _normalize_identifier(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    text = re.sub(
        r"[^A-Za-zА-Яа-яЁё0-9_./-]+",
        "",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def _get_document_name(
    chunk: Dict[str, Any],
) -> str:

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


def _get_point_number(
    chunk: Dict[str, Any],
) -> str:

    # В БД основное поле — point_num.
    for key in (
        "point_num",
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

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    )

    text = str(text).strip()

    if not text:
        return ""

    patterns = [
        r"\bпункт(?:а|ом)?\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bп\.\s*([0-9]+(?:\.[0-9]+)*)\b",
        r"\bстатья\s+([0-9]+(?:\.[0-9]+)*)\b",
        r"\bст\.\s*([0-9]+(?:\.[0-9]+)*)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return ""


def _extract_npa_number(
    document_name: str,
) -> str:

    if not document_name:
        return ""

    patterns = [
        r"№\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
        r"N\s*([0-9]+(?:[-/][A-Za-zА-Яа-я0-9]+)*)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            document_name,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return ""


def build_source_id(
    chunk: Dict[str, Any],
    index: int = 0,
) -> str:
    """
    Создаёт стабильный SOURCE_ID.

    Приоритет:

    1. source_id из БД
    2. NPA + point/article
    3. NPA + hash текста
    """

    existing_source_id = (
        chunk.get("source_id")
        or chunk.get("_source_id")
    )

    if existing_source_id:

        return _normalize_identifier(
            existing_source_id
        )

    document_name = _get_document_name(
        chunk
    )

    point = _get_point_number(
        chunk
    )

    npa_number = _extract_npa_number(
        document_name
    )

    if npa_number:

        base = f"NPA_{npa_number}"

    else:

        normalized_document = (
            _normalize_identifier(
                document_name
            )
        )

        base = (
            f"NPA_{normalized_document[:80]}"
            if normalized_document
            else "NPA_UNKNOWN"
        )

    if point:

        normalized_point = (
            _normalize_identifier(
                point
            )
        )

        return (
            f"{base}_P{normalized_point}"
        )

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    )

    stable_string = (
        f"{document_name}|{str(text).strip()}"
    )

    digest = hashlib.sha1(
        stable_string.encode("utf-8")
    ).hexdigest()[:12]

    return f"{base}_H{digest}"


# ============================================================
# ФОРМАТИРОВАНИЕ RAG
# ============================================================

def _format_chunk(
    chunk: Dict[str, Any],
    index: int,
) -> str:

    document_name = _get_document_name(
        chunk
    )

    point = _get_point_number(
        chunk
    )

    text = (
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("chunk_text")
        or ""
    )

    source_id = build_source_id(
        chunk,
        index,
    )

    chunk["_source_id"] = source_id

    legal_domain = (
        chunk.get("legal_domain")
        or ""
    )

    topic = (
        chunk.get("topic")
        or ""
    )

    lines = [
        f"SOURCE_ID: {source_id}",
        f"DOCUMENT: {document_name}",
    ]

    if legal_domain:

        lines.append(
            f"LEGAL_DOMAIN: {legal_domain}"
        )

    if topic:

        lines.append(
            f"TOPIC: {topic}"
        )

    if point:

        lines.append(
            f"POINT_OR_ARTICLE: {point}"
        )

    lines.append(
        "TEXT:"
    )

    lines.append(
        str(text).strip()
    )

    return "\n".join(lines)


def _build_retrieved_text(
    chunks: List[Dict[str, Any]],
) -> str:

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        source_id = build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = source_id

        block = [
            f"===== RAG SOURCE {index} =====",
            _format_chunk(
                chunk,
                index,
            ),
            f"===== END RAG SOURCE {index} =====",
        ]

        blocks.append(
            "\n".join(block)
        )

    return "\n\n".join(blocks)


# ============================================================
# SUPABASE
# ============================================================

def _search_chunks(
    supabase,
    query_vector: List[float],
    legal_domain: str,
    topic: str,
) -> List[Dict[str, Any]]:

    response = (
        supabase
        .rpc(
            "match_npa_chunks_v3",
            {
                "match_count": RAG_CANDIDATE_COUNT,
                "match_threshold": 0.0,
                "query_embedding": query_vector,
                "legal_domain_filter": legal_domain,
                "topic_filter": topic,
            },
        )
        .execute()
    )

    return response.data or []


# ============================================================
# SEMANTIC SORT
# ============================================================

def _semantic_score(
    chunk: Dict[str, Any],
) -> float:

    if "similarity" in chunk:

        return _safe_float(
            chunk["similarity"]
        )

    if "score" in chunk:

        return _safe_float(
            chunk["score"]
        )

    if "distance" in chunk:

        distance = _safe_float(
            chunk["distance"]
        )

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
# DIVERSIFICATION
# ============================================================

def _get_document_key(
    chunk: Dict[str, Any],
) -> str:

    return (
        _get_document_name(chunk)
        .strip()
        .lower()
    )


def _diversify_chunks(
    chunks: List[Dict[str, Any]],
    limit: int,
    max_per_document: int = 4,
) -> List[Dict[str, Any]]:
    """
    Мягкая диверсификация.

    В юридическом RAG несколько последовательных пунктов
    одного НПА могут быть необходимы для полного ответа.

    Поэтому один документ может занять до 4 позиций TOP-N.
    """

    selected: List[Dict[str, Any]] = []

    per_document: Dict[str, int] = {}

    for chunk in chunks:

        document_key = _get_document_key(
            chunk
        )

        count = per_document.get(
            document_key,
            0,
        )

        if count >= max_per_document:
            continue

        selected.append(chunk)

        per_document[document_key] = (
            count + 1
        )

        if len(selected) >= limit:
            break

    # Если строгая диверсификация не заполнила TOP-N,
    # добираем остальные по исходному semantic ranking.
    if len(selected) < limit:

        selected_ids = {
            id(chunk)
            for chunk in selected
        }

        for chunk in chunks:

            if id(chunk) in selected_ids:
                continue

            selected.append(chunk)

            if len(selected) >= limit:
                break

    return selected


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
    # 0. Domain + topic
    # --------------------------------------------------------

    legal_domain = detect_legal_domain(
        user_query
    )

    topic = detect_topic(
        user_query
    )

    logger.info(
        "RAG | legal_domain=%s | topic=%s",
        legal_domain,
        topic,
    )

    # --------------------------------------------------------
    # 1. Embedding
    # --------------------------------------------------------

    query_vector = await asyncio.to_thread(
        get_query_embedding,
        user_query,
    )

    if not query_vector:

        logger.warning(
            "RAG | embedding is empty"
        )

        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
            "legal_domain": legal_domain,
            "topic": topic,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    if len(query_vector) != 384:

        raise ValueError(
            "Unexpected embedding dimension: "
            f"{len(query_vector)}. "
            "Expected 384."
        )

    # --------------------------------------------------------
    # 2. Supabase
    # --------------------------------------------------------

    candidate_chunks = await asyncio.to_thread(
        _search_chunks,
        supabase,
        query_vector,
        legal_domain,
        topic,
    )

    candidate_count = len(
        candidate_chunks
    )

    logger.info(
        "RAG | domain=%s | candidates=%s",
        legal_domain,
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
            "legal_domain": legal_domain,
            "topic": topic,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    # --------------------------------------------------------
    # 3. Statistics
    # --------------------------------------------------------

    domain_specific_count = sum(
        1
        for chunk in candidate_chunks
        if chunk.get("legal_domain")
        == legal_domain
    )

    topic_specific_count = sum(
        1
        for chunk in candidate_chunks
        if (
            chunk.get("topic")
            or "general"
        ) == topic
    )

    logger.info(
        "RAG | domain=%s | specialized=%s | "
        "topic=%s | topic_specific=%s",
        legal_domain,
        domain_specific_count,
        topic,
        topic_specific_count,
    )

    # --------------------------------------------------------
    # 4. Semantic ranking
    # --------------------------------------------------------

    ranked_chunks = (
        _sort_by_semantic_similarity(
            candidate_chunks
        )
    )

    # --------------------------------------------------------
    # 5. Final TOP-N
    # --------------------------------------------------------

    final_chunks = _diversify_chunks(
        ranked_chunks,
        RAG_FINAL_COUNT,
        max_per_document=4,
    )

    # --------------------------------------------------------
    # 6. SOURCE IDs + references
    # --------------------------------------------------------

    source_references = []

    for index, chunk in enumerate(
        final_chunks,
        start=1,
    ):

        source_id = build_source_id(
            chunk,
            index,
        )

        chunk["_source_id"] = source_id

        document_name = (
            _get_document_name(chunk)
        )

        point = _get_point_number(
            chunk
        )

        if point:

            reference = (
                f"{document_name} — "
                f"пункт/статья {point}"
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
            "RAG | TOP %s | domain=%s | "
            "topic=%s | source=%s | similarity=%.4f",
            index,
            legal_domain,
            topic,
            source_id,
            _semantic_score(chunk),
        )

    # --------------------------------------------------------
    # 7. Build context
    # --------------------------------------------------------

    retrieved_text = (
        _build_retrieved_text(
            final_chunks
        )
    )

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": candidate_count,
        "final_count": len(final_chunks),
        "source_references": source_references,
        "legal_domain": legal_domain,
        "topic": topic,
        "domain_specific_count": domain_specific_count,
        "topic_specific_count": topic_specific_count,
    }


# ============================================================
# ИСТОЧНИКИ
# ============================================================

def get_source_references(
    chunks: List[Dict[str, Any]],
) -> List[str]:

    references = []
    seen = set()

    for chunk in chunks:

        document_name = (
            _get_document_name(chunk)
        )

        point = _get_point_number(
            chunk
        )

        if point:

            reference = (
                f"{document_name} — "
                f"пункт/статья {point}"
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

        document_name = (
            _get_document_name(chunk)
        )

        if document_name not in seen:

            seen.add(document_name)
            names.append(document_name)

    return names
