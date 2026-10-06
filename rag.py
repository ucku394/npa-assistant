import asyncio
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional

from embedding import get_query_embeddings
from rag_query_profile import build_universal_query_profile
from rag_query_generator import build_universal_search_queries, build_search_queries
from rag_query_modes import _attestation_query_mode, _accident_query_mode
from rag_query_classifier import (
    detect_legal_domain,
    detect_topic,
    detect_special_category,
    _minor_special_issue,
    detect_query_intents,
    detect_primary_intent,
    _is_labor_code_query,
    is_cross_reference_query,
    _is_target_briefing_query,
    _is_responsible_briefing_query,
    detect_scope_target,
)

logger = logging.getLogger(__name__)


# ============================================================
# НАСТРОЙКИ
# ============================================================

RAG_FINAL_COUNT = int(os.getenv("RAG_FINAL_COUNT", "5"))

RAG_CANDIDATE_COUNT = max(
    RAG_FINAL_COUNT * 10,
    50,
)

RAG_MIN_CANDIDATES = 10
TARGETED_SEARCH_LIMIT = 50


# ============================================================
# ОПРЕДЕЛЕНИЕ ОБЛАСТИ ЗАПРОСА
# ============================================================

def _extract_query_terms(user_query: str) -> List[str]:
    query = str(user_query or "").strip().lower()
    terms: List[str] = []

    keyword_patterns = [
        r"\bсиз\b",
        r"\bсредств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bспецодежд\w*",
        r"\bспецобув\w*",
        r"\bкостюм\w*",
        r"\bперчат\w*",
        r"\bне\s+выдан\w*",
        r"\bневыдач\w*",
        r"\bповрежден\w*",
        r"\bповрежд\w*",
        r"\bнеисправн\w*",
        r"\bне\s+обеспечен\w*",
        r"\bбез\s+сиз\b",
        r"\bбез\s+средств\w*\s+индивидуальн\w*\s+защит\w*",
        r"\bотказ\w*",
        r"\bне\s+приступ\w*",
        r"\bприостанов\w*",
        r"\bработник\w*",
        r"\bдейств\w*",
        r"\bперв\w*\s+шаг\w*",
        r"\bправ\w*",
        r"\bобеспечен\w*",
        r"\bвыдач\w*",
        r"\bисправн\w*",
        r"\bзащитн\w+\s+свойств\w*",
        r"\bмедицинск\w*\s+осмотр\w*",
        r"\bмедицинск\w*",
        r"\bмедосмотр\w*",
        r"\bпредварительн\w*",
        r"\bпериодическ\w*",
        r"\bвнеочередн\w*",
        r"\bобязательн\w*",
        r"\bработающ\w*",
        r"\bза\s+чей\s+счет\b",
        r"\bза\s+чей\s+сч[её]т\b",
        r"\bза\s+сч[её]т\b",
        r"\bсчет\b",
        r"\bоплат\w*",
        r"\bфинанс\w*",
        r"\bрасход\w*",
        r"\bзатрат\w*",
        r"\bсредств\w*",
        r"\bиспользован\w*",
        r"\bбывш\w*\s+в\s+употреблен\w*",
        r"\bпериод\w*\s+использован\w*",
        r"\bсрок\w*\s+носк\w*",
        r"\bаттестаци\w*",
        r"\bрабоч\w*\s+мест\w*",
        r"\bуслов\w*\s+труд\w*",
        r"\bоснован\w*",
        r"\bпроведен\w*",
        r"\bпериодичност\w*",
        r"\bрезультат\w*",
        r"\bкомисс\w*",
        r"\bвредн\w*",
        r"\bопасн\w*",
        r"\bнесчастн\w*\s+случа\w*",
        r"\bрасследован\w*",
        r"\bучет\w*",
        r"\bакт\w*\s+н[-–—]?\s*1\b",
        r"\bформа\w*\s+н[-–—]?\s*1\b",
        r"\bн[-–—]?\s*1\b",
        r"\bпострадавш\w*",
        r"\bпотерпевш\w*",
        r"\bродственник\w*",
        r"\bвруч\w*",
        r"\bутвержденн?\w*",
        r"\bокончан\w*\s+расследован\w*",
        r"\bрабоч\w*\s+дн\w*",
        r"\bсрок\w*",
        r"\bстажиров\w*",
        r"\bпродолжительност\w*\s+стажиров\w*",
        r"\bне\s+менее\s+двух\b",
        r"\bрабоч\w*\s+смен\w*",
        r"\bповышенн\w*\s+опасност\w*",
        r"\bсамостоятельн\w*\s+работ\w*",
        r"\bдопуск\w*",
        r"\bпровер\w*\s+знан\w*",
        r"\b№\s*175\b",
        r"\bпромышленн\w*\s+безопасност\w*",
        r"\bопасн\w*\s+производственн\w*\s+объект\w*",
        r"\bопо\b",
        r"\bпоо\b",
        r"\bпроизводственн\w*\s+контрол\w*",
        r"\bтехническ\w*\s+устройств\w*",
        r"\bавари\w*",
        r"\bинцидент\w*",
        r"\bработ\w*\s+на\s+высот\w*",
        r"\bработающ\w*\s+1\s+групп\w*",
        r"\bпервая\s+групп\w*",
        r"\bпостановлен\w*\s+№\s*11\b",
        r"\bпожарн\w*\s+безопасност\w*",
        r"\bпожар\w*",
        r"\bогнетушител\w*",
        r"\bэвакуац\w*",
    ]

    for pattern in keyword_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        for match in matches:
            value = str(match).strip().lower()
            if value and value not in terms:
                terms.append(value)

    return terms


def _keyword_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
) -> float:
    if not query_terms:
        return 0.0

    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    point = str(chunk.get("point_num") or "").lower()

    text = f"{document_name} {content} {point}"
    matched = sum(1 for term in query_terms if term in text)
    return matched / len(query_terms)


# ============================================================
# СПЕЦИАЛЬНЫЙ SCORE ДЛЯ ЮРИДИЧЕСКОЙ ТЕМЫ
# ============================================================

def _topic_relevance_score(
    chunk: Dict[str, Any],
    topic: str,
) -> float:
    if not topic or topic == "general":
        return 0.0

    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    db_topic = str(chunk.get("topic") or "").lower()

    score = 0.0

    if topic == "portable_ladder":
        for phrase, weight in (
            ("переносная лестница", 0.85),
            ("приставная лестница", 0.80),
            ("лестница-стремянка", 0.80),
            ("стремянка", 0.75),
            ("испытание лестницы", 0.70),
            ("осмотр лестницы", 0.85),
            ("не реже одного раза в шесть месяцев", 0.95),
        ):
            if phrase in text:
                score += weight

    if topic == "occupational_briefing":
        if db_topic == "occupational_briefing":
            score += 1.0

        target_instruction_markers = [
            "целевой инструктаж",
            "разовых работ",
            "не связанных с прямыми обязанностями",
            "прямыми обязанностями",
            "наряд-допуск",
            "наряду-допуску",
        ]
        target_matches = sum(1 for marker in target_instruction_markers if marker in content)
        if target_matches:
            score += min(target_matches * 0.45, 1.80)

        briefing_markers = [
            "вводный инструктаж",
            "проводит инструктаж",
            "специалист по охране труда",
            "уполномоченное должностное лицо",
            "руководитель организации",
            "руководитель структурного подразделения",
        ]
        matches = sum(1 for marker in briefing_markers if marker in content or marker in document_name)
        if matches:
            score += min(matches * 0.30, 1.20)

        if "175" in document_name:
            score += 0.80

    elif topic == "ppe_nonprovision":
        if db_topic == "ppe_nonprovision":
            score += 1.0

        is_npa_209 = bool(
            re.search(r"№\s*209\b", document_name, flags=re.IGNORECASE)
            or re.search(r"\b209\b", document_name, flags=re.IGNORECASE)
        )
        if is_npa_209:
            score += 1.20

        if "сиз" in document_name or (
            "средств" in document_name
            and "индивидуальн" in document_name
            and "защит" in document_name
        ):
            score += 0.70

        if "сиз" in content:
            score += 0.35
        if "средств" in content and "индивидуальн" in content and "защит" in content:
            score += 0.35

        problem_markers = [
            "не выдан",
            "невыдач",
            "поврежден",
            "поврежд",
            "неисправн",
            "не обеспечен",
            "отказ",
            "не приступ",
            "приостанов",
        ]
        problem_matches = sum(1 for marker in problem_markers if marker in content)
        if problem_matches:
            score += min(problem_matches * 0.25, 0.75)

    elif topic == "medical_examinations":
        if db_topic == "medical_examinations":
            score += 1.0

        if (
            re.search(r"№\s*74\b", document_name, flags=re.IGNORECASE)
            or re.search(r"\b74\b", document_name, flags=re.IGNORECASE)
        ):
            score += 1.20

        if "медицинск" in document_name and "осмотр" in document_name:
            score += 0.80

        if "медицинск" in content:
            score += 0.30
        if "осмотр" in content:
            score += 0.30

        financing_markers = ["за счет", "за счёт", "оплата", "расход", "средств", "финанс", "затрат"]
        if any(marker in content for marker in financing_markers):
            score += 0.60

    elif topic == "workplace_attestation":
        if db_topic == "workplace_attestation":
            score += 1.0
        if "аттестаци" in document_name:
            score += 0.80
        if "аттестаци" in content:
            score += 0.35
        if "рабоч" in content and "мест" in content:
            score += 0.15

        attestation_mode = _attestation_query_mode(
            str(chunk.get("_user_query_for_scoring") or "")
        )
        if attestation_mode == "periodicity":
            if "один раз в пять лет" in content:
                score += 1.20
            if "срок действия результатов аттестации" in content:
                score += 0.90
            if "очередн" in content and "аттестаци" in content:
                score += 0.35
            if re.search(r"\b19\b", str(chunk.get("point_num") or "")):
                score += 0.90

        if "253" in document_name and (
            "аттестаци" in document_name or "аттестаци" in content or "рабоч" in content
        ):
            score += 1.50

    elif topic == "portable_ladder":
        if db_topic == "portable_ladder":
            score += 1.50
        if re.search(r"№\s*11\b", document_name, flags=re.IGNORECASE):
            score += 1.60
        if "работ на высоте" in document_name:
            score += 0.90
        point = _normalize_point_identifier(_get_point_number(chunk))
        if point == "53":
            score += 1.20
        if point == "54":
            score += 1.60
        if "осмотр" in content:
            score += 0.60
        if "исправн" in content:
            score += 0.35
        if "испытан" in content:
            score += 0.15

    elif topic == "height_work_training":
        if db_topic == "height_work_training":
            score += 1.50
        if re.search(r"№\s*11\b", document_name, flags=re.IGNORECASE):
            score += 1.40
        if "работ" in document_name and "высот" in document_name:
            score += 0.90
        if "работающ" in content and "1 группы" in content:
            score += 1.00
        if "первая группа" in content or "1 группы" in content:
            score += 0.70
        if "обучен" in content:
            score += 0.35
        if "первая помощь" in content:
            score += 0.50
        if re.search(r"\b50\b", str(chunk.get("point_num") or ""), flags=re.IGNORECASE):
            score += 1.20

    elif topic == "accident_investigation":
        if db_topic == "accident_investigation":
            score += 1.0
        if "несчаст" in document_name:
            score += 0.80
        if "расследован" in document_name:
            score += 0.45
        if "расследован" in content:
            score += 0.35
        if "несчаст" in content:
            score += 0.25

        if re.search(r"\bн[-–—]?\s*1\b", content, re.IGNORECASE):
            score += 0.90
        if "акт" in content and re.search(r"\bн[-–—]?\s*1\b", content, re.IGNORECASE):
            score += 0.55

        accident_markers = [
            "пострадавш",
            "потерпевш",
            "родственник",
            "вруч",
            "утвержден",
            "утверждён",
            "рабочих дней",
            "рабочие дни",
            "окончани",
        ]
        marker_matches = sum(1 for marker in accident_markers if marker in content)
        if marker_matches:
            score += min(marker_matches * 0.15, 0.75)

    return min(max(score, 0.0), 2.5)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_./-]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_point_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"[.,;:]+$", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_.-]+", "", text)
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"[.,;:]+$", "", text)
    return text.strip("_")


def _normalize_source_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"[.,;:]+$", "", text)
    return _normalize_identifier(text)


def _get_document_name(chunk: Dict[str, Any]) -> str:
    for key in (
        "doc_name",
        "document",
        "document_name",
        "doc",
        "title",
        "npa_name",
        "source",
    ):
        value = chunk.get(key)
        if value:
            return str(value).strip()
    return "Неизвестный НПА"


def _get_point_number(chunk: Dict[str, Any]) -> str:
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

    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""
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
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return ""


def _extract_npa_number(document_name: str) -> str:
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


def build_source_id(
    chunk: Dict[str, Any],
    index: int = 0,
) -> str:
    existing_source_id = chunk.get("source_id") or chunk.get("_source_id")
    if existing_source_id:
        return _normalize_source_id(existing_source_id)

    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)
    npa_number = _extract_npa_number(document_name)

    if npa_number:
        base = f"NPA_{npa_number}"
    else:
        normalized_document = _normalize_identifier(document_name)
        base = (
            f"NPA_{normalized_document[:80]}"
            if normalized_document
            else "NPA_UNKNOWN"
        )

    if point:
        normalized_point = _normalize_point_identifier(point)
        if normalized_point:
            return f"{base}_P{normalized_point}"

    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""
    stable_string = f"{document_name}|{str(text).strip()}"
    digest = hashlib.sha1(stable_string.encode("utf-8")).hexdigest()[:12]

    return f"{base}_H{digest}"


# ============================================================
# ФОРМИРОВАНИЕ КОНТЕКСТА
# ============================================================

def _format_chunk(
    chunk: Dict[str, Any],
    index: int,
) -> str:
    document_name = _get_document_name(chunk)
    point = _get_point_number(chunk)
    text = chunk.get("content") or chunk.get("text") or chunk.get("chunk_text") or ""

    source_id = build_source_id(chunk, index)
    chunk["_source_id"] = source_id

    legal_domain = chunk.get("legal_domain") or ""
    topic = chunk.get("topic")
    semantic = _semantic_score(chunk)
    combined = chunk.get("_combined_score")

    lines = [
        f"SOURCE_ID: {source_id}",
        f"DOCUMENT: {document_name}",
    ]

    if legal_domain:
        lines.append(f"LEGAL_DOMAIN: {legal_domain}")
    if topic:
        lines.append(f"TOPIC: {topic}")
    if point:
        lines.append(f"POINT_OR_ARTICLE: {point}")

    lines.append(f"SEMANTIC_SCORE: {semantic:.4f}")
    if combined is not None:
        lines.append(f"RAG_SCORE: {float(combined):.4f}")

    lines.append("TEXT:")
    lines.append(str(text).strip())

    return "\n".join(lines)


def _build_retrieved_text(
    chunks: List[Dict[str, Any]],
) -> str:
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
# SUPABASE — VECTOR SEARCH
# ============================================================

def _search_chunks(
    supabase,
    query_vector: List[float],
    search_query: str,
    legal_domain: Optional[str],
    topic_filter: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        response = (
            supabase.rpc(
                "hybrid_search_npa_chunks",
                {
                    "query_embedding": query_vector,
                    "search_query": search_query,
                    "match_count": RAG_CANDIDATE_COUNT,
                    "domain_filter": legal_domain,
                    "topic_filter": topic_filter,
                    "semantic_limit": max(RAG_CANDIDATE_COUNT, 40),
                    "lexical_limit": max(RAG_CANDIDATE_COUNT, 40),
                    "rrf_k": 60,
                },
            ).execute()
        )

        data = response.data or []
        for chunk in data:
            chunk["similarity"] = _safe_float(
                chunk.get("semantic_score"),
                _safe_float(chunk.get("similarity")),
            )
            chunk["_hybrid_rrf_score"] = _safe_float(chunk.get("rrf_score"))
            chunk["_hybrid_final_score"] = _safe_float(chunk.get("final_score"))

        return data

    except Exception as exc:
        logger.warning(
            "RAG | hybrid search failed; fallback to semantic_search_npa_chunks | error=%s",
            exc,
        )

        response = (
            supabase.rpc(
                "semantic_search_npa_chunks",
                {
                    "query_embedding": query_vector,
                    "match_count": RAG_CANDIDATE_COUNT,
                    "domain_filter": legal_domain,
                    "topic_filter": topic_filter,
                },
            ).execute()
        )

        data = response.data or []
        for chunk in data:
            chunk["similarity"] = _safe_float(chunk.get("similarity"))
            chunk["_hybrid_rrf_score"] = 0.0
            chunk["_hybrid_final_score"] = 0.0

        return data


# ============================================================
# TARGETED SEARCH (СГРУППИРОВАННЫЙ В ОДИН ЗАПРОС)
# ============================================================
def _targeted_portable_ladder_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    """Адресный поиск действующих требований к лестницам по Правилам № 11."""
    try:
        response = (
            supabase.table("npa_chunks")
            .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%№ 11%")
            .or_(
                "point_num.ilike.%53%,point_num.ilike.%54%,"
                "content.ilike.%лестниц%,content.ilike.%стремянк%"
            )
            .limit(30)
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        for chunk in results:
            point = _get_point_number(chunk).strip().rstrip(".")
            if point in {"53", "54"}:
                chunk["_portable_ladder_targeted"] = True
        results.sort(
            key=lambda chunk: (
                1 if _get_point_number(chunk).strip().rstrip(".") == "54" else 0,
                1 if _get_point_number(chunk).strip().rstrip(".") == "53" else 0,
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )
        logger.info(
            "RAG | portable ladder targeted | points=%s",
            [f"{_get_document_name(x)}#{_get_point_number(x)}" for x in results],
        )
        return results
    except Exception as exc:
        logger.warning("RAG | portable ladder targeted search failed: %s", exc)
        return []

def _targeted_height_work_training_search(supabase) -> List[Dict[str, Any]]:
    """Нормативно полный набор именно для алгоритма обучения 1 группы.

    Не используем общий поиск по всему НПА № 11: он часто вытесняет
    нужные пункты 48-51 нерелевантным п. 17 о наряде-допуске.
    """
    results: List[Dict[str, Any]] = []
    try:
        height = (
            supabase.table("npa_chunks")
            .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%№ 11%")
            .limit(120)
            .execute()
        )
        for chunk in (height.data or []):
            point = _get_point_number(chunk).strip().rstrip(".")
            if point in {"48", "49", "50", "51"}:
                results.append(chunk)
    except Exception as exc:
        logger.warning("RAG | height rules targeted search failed: %s", exc)

    try:
        training = (
            supabase.table("npa_chunks")
            .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%175%")
            .limit(160)
            .execute()
        )
        for chunk in (training.data or []):
            point = _get_point_number(chunk).strip().rstrip(".")
            if point in {"36", "37", "38", "39", "40", "44", "45"}:
                results.append(chunk)
    except Exception as exc:
        logger.warning("RAG | instruction 175 targeted search failed: %s", exc)

    results = _deduplicate_chunks(results)
    for chunk in results:
        chunk["_height_training_targeted"] = True
    return results

def _execute_combined_targeted_search(
    supabase,
    filters_list: List[str],
    domain: str = "occupational_safety",
    limit: int = TARGETED_SEARCH_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Выполняет один сгруппированный запрос через .or_()
    вместо десятка последовательных вызовов.
    """
    if not filters_list:
        return []

    combined_query = ",".join(filters_list)

    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,"
                "doc_type,"
                "point_num,"
                "content,"
                "legal_domain,"
                "topic,"
                "source_url"
            )
            .eq("legal_domain", domain)
            .or_(combined_query)
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as exc:
        logger.warning("RAG | combined targeted search failed: %s", exc)
        return []


def _targeted_attestation_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """
    Узкий lexical-search для аттестации рабочих мест.

    Для вопроса о периодичности специально ищем норму п. 19 Положения
    о порядке проведения аттестации рабочих мест по условиям труда.
    """
    mode = _attestation_query_mode(user_query)

    if mode == "periodicity":
        # Для периодичности нельзя использовать общий OR-запрос с
        # doc_name.ilike.%253%: он совпадает со всеми chunks НПА №253
        # и может вытеснить п. 19 из лимита. Сначала ограничиваемся №253,
        # затем ищем внутри него признаки нормы о пятилетнем сроке.
        try:
            response = (
                supabase.table("npa_chunks")
                .select(
                    "doc_name,"
                    "doc_type,"
                    "point_num,"
                    "content,"
                    "legal_domain,"
                    "topic,"
                    "source_url"
                )
                .eq("legal_domain", "occupational_safety")
                .ilike("doc_name", "%253%")
                .or_(
                    "content.ilike.%срок действия результатов аттестации составляет пять лет%,"
                    "content.ilike.%один раз в пять лет%,"
                    "content.ilike.%приказ об утверждении очередной аттестации%,"
                    "point_num.ilike.%19%"
                )
                .limit(TARGETED_SEARCH_LIMIT)
                .execute()
            )
            return _deduplicate_chunks(response.data or [])
        except Exception as exc:
            logger.warning(
                "RAG | attestation periodicity targeted search failed: %s",
                exc,
            )
            return []
    elif mode == "extraordinary":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%внеочередная аттестация%",
            "content.ilike.%переаттестация%",
            "content.ilike.%в течение шести месяцев%",
            "point_num.ilike.%17%",
        ]
    elif mode == "results":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%результаты аттестации%",
            "content.ilike.%дополнительный отпуск%",
            "content.ilike.%профессиональное пенсионное страхование%",
        ]
    elif mode == "commission":
        queries = [
            "doc_name.ilike.%253%",
            "content.ilike.%комиссия по проведению аттестации%",
            "content.ilike.%комиссия%",
        ]
    else:
        queries = [
            "doc_name.ilike.%253%",
            "doc_name.ilike.%аттестаци%",
            "content.ilike.%аттестаци%",
        ]

    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_medical_exam_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%74%",
        "doc_name.ilike.%медицинск%",
        "doc_name.ilike.%осмотр%",
        "content.ilike.%медицинск%",
        "content.ilike.%медосмотр%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_ppe_nonprovision_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%209%",
        "doc_name.ilike.%СИЗ%",
        "doc_name.ilike.%средств%индивидуальн%защит%",
        "content.ilike.%не выдан%",
        "content.ilike.%невыдач%",
        "content.ilike.%поврежден%",
        "content.ilike.%поврежд%",
        "content.ilike.%неисправн%",
        "content.ilike.%не обеспечен%",
        "content.ilike.%отказ%",
        "content.ilike.%не приступ%",
        "content.ilike.%приостанов%",
        "content.ilike.%работник%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_training_suspension_search(supabase) -> List[Dict[str, Any]]:
    """Точечный поиск ст. 49 ТК РБ для непройденного ОТ-обучения."""
    try:
        response = (
            supabase.table("npa_chunks")
            .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
            .eq("legal_domain", "occupational_safety")
            .eq("doc_name", "Трудовой кодекс Республики Беларусь 2026")
            .or_("point_num.eq.Статья 49,point_num.eq.49")
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        for chunk in results:
            chunk["_training_suspension_targeted"] = True
        return results
    except Exception as exc:
        logger.warning("RAG | training-suspension targeted search failed: %s", exc)
        return []


def _targeted_osh_knowledge_frequency_search(
    supabase,
    subject: str = "working_persons",
) -> List[Dict[str, Any]]:
    """Точечный поиск норм № 175 о периодичности проверки знаний с учетом категории работающих."""
    try:
        if subject == "managers_specialists":
            filters = [
                "point_num.eq.42",
                "point_num.eq.42.",
                "point_num.eq.43",
                "point_num.eq.43.",
                "content.ilike.%руководители и специалисты%периодическую проверку знаний%",
                "content.ilike.%не реже одного раза в три года%",
            ]
        elif subject == "workers":
            filters = [
                "point_num.eq.51",
                "point_num.eq.51.",
                "content.ilike.%рабочие%периодическую проверку знаний%",
                "content.ilike.%не реже одного раза в 12 месяцев%",
            ]
        else:
            filters = [
                "point_num.eq.42",
                "point_num.eq.42.",
                "point_num.eq.51",
                "point_num.eq.51.",
                "point_num.eq.53",
                "point_num.eq.53.",
                "content.ilike.%периодическую проверку знаний%",
            ]

        response = (
            supabase.table("npa_chunks")
            .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%175%")
            .or_(",".join(filters))
            .limit(TARGETED_SEARCH_LIMIT)
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        for chunk in results:
            chunk["_osh_knowledge_frequency_targeted"] = True
            chunk["_osh_knowledge_frequency_subject"] = subject
        return results
    except Exception as exc:
        logger.warning("RAG | OHS knowledge frequency targeted search failed: %s", exc)
        return []


def _targeted_occupational_training_search(supabase) -> List[Dict[str, Any]]:
    queries = [
        "doc_name.ilike.%175%",
        "doc_name.ilike.%Инструкци%",
        "content.ilike.%стажиров%",
        "content.ilike.%продолжительн%стажиров%",
        "content.ilike.%срок%стажиров%",
        "content.ilike.%не менее двух%",
        "content.ilike.%рабочих дней%",
        "content.ilike.%рабочих смен%",
        "content.ilike.%повышенной опасностью%",
        "content.ilike.%допуск к самостоятельной работе%",
        "content.ilike.%самостоятельной работе%",
        "content.ilike.%проверка знаний%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_ppe_refusal_search(
    supabase,
) -> List[Dict[str, Any]]:
    """
    Точечный поиск ст. 11 Закона № 356-З:
    право работника отказаться от порученной работы при
    непредоставлении СИЗ, непосредственно обеспечивающих безопасность труда.
    """
    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .eq("doc_name", "Закон об охране труда от 23 июня 2008 г. № 356-З")
            .or_("point_num.eq.Статья 11,point_num.eq.11")
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        if results:
            for chunk in results:
                chunk["_ppe_refusal_targeted"] = True
            return results

        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%356-З%")
            .or_(
                "content.ilike.%отказ от выполнения порученной работы%,"
                "content.ilike.%непредоставлении ему средств индивидуальной защиты%,"
                "content.ilike.%непосредственно обеспечивающих безопасность труда%"
            )
            .limit(20)
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        for chunk in results:
            chunk["_ppe_refusal_targeted"] = True
        return results
    except Exception as exc:
        logger.warning(
            "RAG | PPE refusal targeted search failed: %s",
            exc,
        )
        return []


def _targeted_work_break_briefing_search(
    supabase,
) -> List[Dict[str, Any]]:
    """
    Точечный lexical-search для п. 27 Инструкции № 175:
    внеплановый инструктаж при перерыве в работе по профессии
    (в должности) более шести месяцев.
    """
    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%175%")
            .or_("point_num.eq.27,point_num.eq.27.")
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        if results:
            for chunk in results:
                chunk["_work_break_targeted"] = True
            return results

        # Резервный поиск, если point_num в БД хранится не как точное "27".
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%175%")
            .or_(
                "content.ilike.%перерывах в работе%,"
                "content.ilike.%более шести месяцев%,"
                "content.ilike.%внеплановый инструктаж%"
            )
            .limit(20)
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])
        for chunk in results:
            chunk["_work_break_targeted"] = True
        return results
    except Exception as exc:
        logger.warning(
            "RAG | work-break briefing targeted search failed: %s",
            exc,
        )
        return []


def _targeted_occupational_briefing_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """Точечный сбор нормативного каркаса для процедурных вопросов об инструктажах."""
    target_mode = _is_target_briefing_query(user_query)
    normalized_query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    procedure_mode = bool(
        re.search(
            r"\b(алгоритм|порядок|пошагов|что делать|как провести|как организовать|"
            r"последовательност|процедур)\b",
            normalized_query,
        )
    )

    try:
        # Ключевой блок действующей Инструкции № 175. Забираем пункты
        # адресно, чтобы semantic search не вытеснил этапы процедуры.
        point_values = []
        for point in range(16, 36):
            point_values.extend([str(point), f"{point}."])

        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", "%175%")
            .in_("point_num", point_values)
            .limit(100)
            .execute()
        )
        results = _deduplicate_chunks(response.data or [])

        # Резерв, если point_num в загрузке хранится нестандартно.
        if not results:
            results = _execute_combined_targeted_search(
                supabase,
                [
                    "doc_name.ilike.%175%",
                    "content.ilike.%вводный инструктаж%",
                    "content.ilike.%первичный инструктаж%",
                    "content.ilike.%повторный инструктаж%",
                    "content.ilike.%внеплановый инструктаж%",
                    "content.ilike.%целевой инструктаж%",
                    "content.ilike.%проверка знаний%",
                    "content.ilike.%регистрац%",
                ],
                limit=80,
            )

        priority_points = {
            "16": 1000, "17": 980, "18": 960, "20": 940,
            "22": 1000, "26": 900, "27": 880, "28": 860,
            "29": 840, "30": 820, "31": 980, "35": 960,
        }

        def briefing_priority(chunk: Dict[str, Any]) -> tuple:
            point = _normalize_point_identifier(_get_point_number(chunk))
            content = str(chunk.get("content") or "").lower()
            score = priority_points.get(point, 500)
            for marker, bonus in (
                ("вводный инструктаж", 80),
                ("первичный инструктаж", 80),
                ("повторный инструктаж", 70),
                ("внеплановый инструктаж", 70),
                ("целевой инструктаж", 70),
                ("проверка знаний", 60),
                ("регистрац", 50),
            ):
                if marker in content:
                    score += bonus
            return score, point

        results.sort(key=briefing_priority, reverse=True)

        for chunk in results:
            chunk["_occupational_briefing_targeted"] = True
            chunk["_procedure_targeted"] = procedure_mode

        # Только для запроса именно о целевом инструктаже добавляем
        # специальные нормы о разовых работах/наряде-допуске.
        if target_mode:
            extra = (
                supabase.table("npa_chunks")
                .select(
                    "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
                )
                .eq("legal_domain", "occupational_safety")
                .ilike("doc_name", "%175%")
                .or_(
                    "content.ilike.%целевой инструктаж%,"
                    "content.ilike.%разовых работ%,"
                    "content.ilike.%ликвидации последствий аварий%,"
                    "content.ilike.%наряд-допуск%"
                )
                .limit(30)
                .execute()
            )
            extra_results = _deduplicate_chunks(extra.data or [])
            for chunk in extra_results:
                chunk["_occupational_briefing_targeted"] = True
                chunk["_procedure_targeted"] = procedure_mode
            results = _deduplicate_chunks(results + extra_results)

        logger.info(
            "RAG | occupational briefing targeted | procedure=%s | points=%s",
            procedure_mode,
            [
                f"{_get_document_name(chunk)}#{_get_point_number(chunk)}"
                for chunk in results[:30]
            ],
        )
        return results
    except Exception as exc:
        logger.warning(
            "RAG | occupational briefing targeted search failed: %s",
            exc,
        )
        return []


def _targeted_accident_worker_not_report_search(supabase) -> List[Dict[str, Any]]:
    """Точечный lexical-search для ситуации, когда работник не сообщил о НС."""
    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,"
                "doc_type,"
                "point_num,"
                "content,"
                "legal_domain,"
                "topic,"
                "source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .or_("doc_name.ilike.%30%,doc_name.ilike.%81 144%")
            .or_(
                "content.ilike.%не сообщил%,"
                "content.ilike.%не сообщила%,"
                "content.ilike.%сообщить о несчастном случае%,"
                "content.ilike.%сообщить о происшедшем несчастном случае%,"
                "content.ilike.%сообщить непосредственному руководителю%,"
                "content.ilike.%сообщить руководителю%,"
                "content.ilike.%немедленно сообщить%,"
                "content.ilike.%обязан сообщить%"
            )
            .limit(TARGETED_SEARCH_LIMIT)
            .execute()
        )
        return _deduplicate_chunks(response.data or [])
    except Exception as exc:
        logger.warning(
            "RAG | accident worker-not-report targeted search failed: %s",
            exc,
        )
        return []


def _targeted_lifting_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    """Точечный lexical-search для норм ручного подъема/перемещения грузов."""
    try:
        target_doc = (
            "Об утверждении Межотраслевых правил по охране труда "
            "при проведении погрузочно-разгрузочных работ от 26 января 2018 г. № 12"
        )

        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", "occupational_safety")
            .ilike("doc_name", f"%{target_doc}%")
            .or_(
                "point_num.ilike.%86%,"
                "content.ilike.%50 кг%,"
                "content.ilike.%разовый подъем%,"
                "content.ilike.%разового подъема%,"
                "content.ilike.%погрузочно-разгрузоч%"
            )
            .limit(TARGETED_SEARCH_LIMIT)
            .execute()
        )

        results = _deduplicate_chunks(response.data or [])
        results.sort(
            key=lambda chunk: (
                1 if re.search(r"(?<!\d)86\.?\b", _get_point_number(chunk)) else 0,
                1 if "50 кг" in str(chunk.get("content") or "").lower() else 0,
                1 if "разовый подъем" in str(chunk.get("content") or "").lower() else 0,
            ),
            reverse=True,
        )
        return results
    except Exception as exc:
        logger.warning(
            "RAG | lifting targeted search failed: %s",
            exc,
        )
        return []


def _targeted_accident_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    mode = _accident_query_mode(user_query)

    if mode == "worker_did_not_report":
        return _targeted_accident_worker_not_report_search(supabase)

    queries = [
        "doc_name.ilike.%Правила%",
        "doc_name.ilike.%30%",
        "content.ilike.%группов%",
        "content.ilike.%одновременно с двумя%",
        "content.ilike.%два и более%",
        "content.ilike.%специальному расследованию%",
        "content.ilike.%немедленно сообщает%",
        "content.ilike.%сообщает%",
        "content.ilike.%уведом%",
        "content.ilike.%прокуратур%",
        "content.ilike.%государственной инспекции труда%",
        "doc_name.ilike.%несчаст%",
        "doc_name.ilike.%расслед%",
        "content.ilike.%Н-1%",
        "content.ilike.%н-1%",
        "content.ilike.%форма Н-1%",
        "content.ilike.%форма н-1%",
        "content.ilike.%акт Н-1%",
        "content.ilike.%акт н-1%",
        "content.ilike.%пострадавш%",
        "content.ilike.%потерпевш%",
        "content.ilike.%родственник%",
        "content.ilike.%вруч%",
        "content.ilike.%утвержден%",
        "content.ilike.%утверждён%",
        "content.ilike.%рабочих дней%",
        "content.ilike.%окончани%расследован%",
        "content.ilike.%расследован%несчаст%",
    ]
    results = _execute_combined_targeted_search(supabase, queries)
    return _deduplicate_chunks(results)


def _targeted_minor_search(supabase, user_query: str = "") -> List[Dict[str, Any]]:
    """
    Прямой lexical search для специальных гарантий несовершеннолетних.
    Нужен как страховка: vector search не должен быть единственным способом
    найти статью 276, если запрос явно содержит возраст работника.
    """
    issue = _minor_special_issue(user_query)
    if not issue:
        return []

    base_queries = [
        "doc_name.ilike.%Трудовой кодекс%",
        "content.ilike.%моложе восемнадцати лет%",
        "content.ilike.%несовершеннолетн%",
    ]

    article_queries = {
        "holiday_weekend": [
            "point_num.ilike.%276%",
            "content.ilike.%статья 276%",
            "content.ilike.%государственные праздники%",
            "content.ilike.%праздничные дни%",
            "content.ilike.%выходные дни%",
        ],
        "night_overtime": [
            "point_num.ilike.%276%",
            "content.ilike.%статья 276%",
            "content.ilike.%ночным%",
            "content.ilike.%сверхурочным%",
        ],
        "prohibited_work": [
            "point_num.ilike.%274%",
            "content.ilike.%статья 274%",
            "content.ilike.%тяжелых работах%",
            "content.ilike.%вредными%",
        ],
        "medical": [
            "point_num.ilike.%275%",
            "content.ilike.%статья 275%",
            "content.ilike.%медицинских осмотров%",
        ],
        "leave": [
            "point_num.ilike.%277%",
            "content.ilike.%статья 277%",
            "content.ilike.%трудовые отпуска%",
        ],
        "work_time": [
            "point_num.ilike.%278%",
            "point_num.ilike.%279%",
            "content.ilike.%сокращенной продолжительности%",
        ],
    }

    results = _execute_combined_targeted_search(
        supabase,
        base_queries + article_queries.get(issue, []),
    )
    return _deduplicate_chunks(results)


def _targeted_law_scope_search(
    supabase,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    """Точечный lexical-поиск для вопросов о сфере действия НПА."""
    target = detect_scope_target(user_query) or {}
    document_key = str(target.get("document_key") or "").lower()

    if document_key == "356-з":
        try:
            response = (
                supabase.table("npa_chunks")
                .select(
                    "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
                )
                .eq("legal_domain", "occupational_safety")
                .or_(
                    "doc_name.ilike.%356-З%,"
                    "doc_name.ilike.%356 З%,"
                    "content.ilike.%сфера действия настоящего Закона%,"
                    "content.ilike.%применяется в отношении всех работодателей%,"
                    "point_num.ilike.%3%"
                )
                .limit(TARGETED_SEARCH_LIMIT)
                .execute()
            )
            results = _deduplicate_chunks(response.data or [])
            for chunk in results:
                chunk["_law_scope_targeted"] = True
                chunk["_law_scope_target_document"] = "356-з"
            return results
        except Exception as exc:
            logger.warning("RAG | law scope targeted search failed: %s", exc)
            return []

    if document_key == "трудовой кодекс":
        results = _execute_combined_targeted_search(
            supabase,
            [
                "doc_name.ilike.%Трудовой кодекс%",
                "content.ilike.%сфера действия%",
                "content.ilike.%трудовые отношения%",
            ],
        )
        for chunk in results:
            chunk["_law_scope_targeted"] = True
            chunk["_law_scope_target_document"] = "трудовой кодекс"
        return _deduplicate_chunks(results)

    # Общие Правила по охране труда — постановление Минтруда № 53, пункт 2.
    normalized_query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())
    rules_match = re.search(r"\bправила\s+по\s+охране\s+труда\b", normalized_query)
    if rules_match and not re.match(r"\s+при\b", normalized_query[rules_match.end():]):
        try:
            response = (
                supabase.table("npa_chunks")
                .select("doc_name,doc_type,point_num,content,legal_domain,topic,source_url")
                .eq("legal_domain", "occupational_safety")
                .or_(
                    "doc_name.ilike.%Правила по охране труда № 53%,"
                    "doc_name.ilike.%Правила по охране труда%,"
                    "content.ilike.%распространяются на работодателей%,"
                    "content.ilike.%организационно-правовых форм и форм собственности%"
                )
                .limit(TARGETED_SEARCH_LIMIT)
                .execute()
            )
            results = _deduplicate_chunks(response.data or [])
            for chunk in results:
                chunk["_law_scope_targeted"] = True
                chunk["_law_scope_target_document"] = "правила по охране труда 53"
            return results
        except Exception as exc:
            logger.warning("RAG | Rules No.53 scope targeted search failed: %s", exc)
            return []

    return []


def _deduplicate_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()

    for chunk in chunks:
        document = _get_document_name(chunk)
        point = _get_point_number(chunk)
        content = str(chunk.get("content") or "").strip()

        key = (
            document.lower(),
            point.lower(),
            content[:300].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(chunk)

    return unique


async def _get_targeted_chunks(
    supabase,
    topic: str,
    user_query: str = "",
) -> List[Dict[str, Any]]:
    if detect_special_category(user_query) == "minor":
        minor_results = await asyncio.to_thread(
            _targeted_minor_search,
            supabase,
            user_query,
        )
        if minor_results:
            return minor_results

    if query_profile := build_universal_query_profile(user_query):
        if query_profile.get("event") == "law_scope":
            scope_results = await asyncio.to_thread(
                _targeted_law_scope_search,
                supabase,
                user_query,
            )
            if scope_results:
                return scope_results
        if query_profile.get("event") == "lifting_and_moving_loads":
            lifting_results = await asyncio.to_thread(
                _targeted_lifting_search,
                supabase,
                user_query,
            )
            if lifting_results:
                for chunk in lifting_results:
                    chunk["_lifting_constraint_targeted"] = True
                return lifting_results

        frequency_qualifiers = query_profile.get("qualifiers") or []
        if (
            "osh_knowledge_check_frequency_managers_specialists" in frequency_qualifiers
            or "osh_knowledge_check_frequency_workers" in frequency_qualifiers
            or "osh_knowledge_check_frequency" in frequency_qualifiers
        ):
            subject = query_profile.get("subject") or "working_persons"
            frequency_results = await asyncio.to_thread(
                _targeted_osh_knowledge_frequency_search,
                supabase,
                subject,
            )
            if frequency_results:
                return frequency_results

        if "suspension_for_unpassed_osh_training" in (query_profile.get("qualifiers") or []):
            suspension_results = await asyncio.to_thread(
                _targeted_training_suspension_search,
                supabase,
            )
            if suspension_results:
                return suspension_results

        if (
            "refusal_due_to_no_ppe" in (query_profile.get("qualifiers") or [])
            or "ppe_nonprovision_action" in (query_profile.get("qualifiers") or [])
        ):
            refusal_results = await asyncio.to_thread(
                _targeted_ppe_refusal_search,
                supabase,
            )
            if refusal_results:
                for chunk in refusal_results:
                    chunk["_ppe_nonprovision_action_targeted"] = (
                        "ppe_nonprovision_action"
                        in (query_profile.get("qualifiers") or [])
                    )
                return refusal_results

        if (
            query_profile.get("event") == "occupational_briefing"
            and "work_break_over_six_months" in (query_profile.get("qualifiers") or [])
        ):
            work_break_results = await asyncio.to_thread(
                _targeted_work_break_briefing_search,
                supabase,
            )
            if work_break_results:
                return work_break_results

    if topic == "portable_ladder":
        return await asyncio.to_thread(_targeted_portable_ladder_search, supabase, user_query)
    if topic == "height_work_training":
        return await asyncio.to_thread(_targeted_height_work_training_search, supabase)
    if topic == "ppe_nonprovision":
        return await asyncio.to_thread(_targeted_ppe_nonprovision_search, supabase)
    if topic == "medical_examinations":
        return await asyncio.to_thread(_targeted_medical_exam_search, supabase)
    if topic == "workplace_attestation":
        return await asyncio.to_thread(
            _targeted_attestation_search,
            supabase,
            user_query,
        )
    if topic == "occupational_briefing":
        return await asyncio.to_thread(_targeted_occupational_briefing_search, supabase, user_query)
    if topic == "accident_investigation":
        return await asyncio.to_thread(
            _targeted_accident_search,
            supabase,
            user_query,
        )
    if topic == "occupational_training":
        return await asyncio.to_thread(_targeted_occupational_training_search, supabase)
    return []


# ============================================================
# SEMANTIC SCORE
# ============================================================

def _semantic_score(chunk: Dict[str, Any]) -> float:
    if "similarity" in chunk:
        return _safe_float(chunk["similarity"])
    if "score" in chunk:
        return _safe_float(chunk["score"])
    if "distance" in chunk:
        return 1.0 - _safe_float(chunk["distance"])
    return 0.0


# ============================================================
# DIVERSIFICATION
# ============================================================

def _get_document_key(chunk: Dict[str, Any]) -> str:
    return _get_document_name(chunk).strip().lower()


# ============================================================
# ПРОВЕРКА НУЖНОГО НПА
# ============================================================

def _is_occupational_training_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_175 = bool(
        re.search(r"№\s*175\b", document_name, flags=re.IGNORECASE)
        or ("инструкци" in document_name and re.search(r"\b175\b", document_name, flags=re.IGNORECASE))
    )

    has_internship = "стажиров" in content
    has_duration = "продолжительн" in content or "срок" in content
    has_minimum_two = "не менее двух" in content
    has_working_period = "рабочих дней" in content or "рабочих смен" in content

    return is_npa_175 or (has_internship and (has_minimum_two or has_working_period or has_duration))


def _is_accident_investigation_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    has_n1 = bool(re.search(r"\bн[-–—]?\s*1\b", document_name + " " + content, re.IGNORECASE))
    has_accident = "несчаст" in document_name or "несчаст" in content
    has_investigation = "расследован" in document_name or "расследован" in content
    has_victim = "пострадавш" in content or "потерпевш" in content or "родственник" in content

    return has_n1 or (has_accident and has_investigation) or (has_accident and has_victim)


def _is_attestation_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()
    return "253" in document_name and ("аттестаци" in document_name or "аттестаци" in content)


def _is_medical_exam_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_74 = bool(
        re.search(r"№\s*74\b", document_name, flags=re.IGNORECASE)
        or re.search(r"\b74\b", document_name, flags=re.IGNORECASE)
    )
    is_medical_document = "медицинск" in document_name and "осмотр" in document_name
    return is_npa_74 or is_medical_document


def _is_ppe_nonprovision_document(chunk: Dict[str, Any]) -> bool:
    document_name = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or "").lower()

    is_npa_209 = bool(
        re.search(r"№\s*209\b", document_name, flags=re.IGNORECASE)
        or re.search(r"\b209\b", document_name, flags=re.IGNORECASE)
    )
    is_ppe_document = "сиз" in document_name or (
        "средств" in document_name and "индивидуальн" in document_name and "защит" in document_name
    )
    has_problem_context = any(
        marker in content
        for marker in (
            "не выдан",
            "невыдач",
            "поврежден",
            "поврежд",
            "неисправн",
            "не обеспечен",
            "отказ",
            "не приступ",
            "приостанов",
        )
    )
    return is_npa_209 or (is_ppe_document and has_problem_context)


# ============================================================
# МНОГОЗАПРОСНЫЙ ЮРИДИЧЕСКИЙ ПОИСК
# ============================================================

def _merge_search_results(
    groups: List[List[Dict[str, Any]]],
    query_roles: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for group_index, group in enumerate(groups):
        role = query_roles[group_index] if group_index < len(query_roles) else "semantic"

        for chunk in group:
            document = _get_document_name(chunk)
            point = _get_point_number(chunk)
            content = str(chunk.get("content") or "").strip()
            key = (
                document.lower(),
                point.lower(),
                content[:300].lower(),
            )
            score = _semantic_score(chunk)

            if key not in merged:
                item = dict(chunk)
                item["_best_similarity"] = score
                item["_search_hits"] = 1
                item["_query_roles"] = [role]
                merged[key] = item
                continue

            item = merged[key]
            if score > _safe_float(item.get("_best_similarity")):
                item["_best_similarity"] = score
                for field in ("similarity", "score", "distance"):
                    if field in chunk:
                        item[field] = chunk[field]
                        break

            item["_search_hits"] = int(item.get("_search_hits", 0)) + 1

            if chunk.get("_accident_worker_not_report_targeted"):
                item["_accident_worker_not_report_targeted"] = True
            if chunk.get("_height_training_targeted"):
                item["_height_training_targeted"] = True
            if chunk.get("_occupational_briefing_targeted"):
                item["_occupational_briefing_targeted"] = True
            if chunk.get("_procedure_targeted"):
                item["_procedure_targeted"] = True
            roles = item.setdefault("_query_roles", [])
            if role not in roles:
                roles.append(role)

    return list(merged.values())


def _intent_relevance_score(
    chunk: Dict[str, Any],
    intents: List[str],
) -> float:
    if not intents:
        return 0.0

    text = " ".join([
        str(chunk.get("doc_name") or chunk.get("document") or ""),
        str(chunk.get("content") or chunk.get("text") or ""),
    ]).lower()

    markers = {
        "refusal": ["отказ", "не приступ", "приостанов", "не выполнять"],
        "danger": ["угроз", "опасност", "жизни", "здоров", "риск"],
        "employee_right": ["имеет право", "право работника", "вправе"],
        "employer_duty": ["обязан", "обязанность нанимателя", "наниматель обязан", "работодатель обязан"],
        "procedure": ["порядок", "действия работника", "немедленно сообщ", "уведом"],
        "responsible_person": [
            "проводит вводный инструктаж",
            "проводит инструктаж",
            "специалист по охране труда",
            "уполномоченное должностное лицо",
            "руководитель организации",
        ],
        "liability": ["ответственност", "штраф", "взыскан", "наказан"],
    }

    score = 0.0
    for intent in intents:
        matches = sum(1 for marker in markers.get(intent, []) if marker in text)
        score += min(matches * 0.18, 0.35)

    return min(score, 1.0)


def _primary_intent_relevance_score(
    chunk: Dict[str, Any],
    primary_intent: Optional[str],
    topic: str,
) -> float:
    if not primary_intent:
        return 0.0

    role = _legal_chunk_role(
        chunk,
        topic,
        [primary_intent],
        primary_intent=primary_intent,
    )

    if role == primary_intent:
        return 1.0

    return 0.0


def _exact_match_score(
    chunk: Dict[str, Any],
    user_query: str,
    topic: str,
) -> float:
    """Strong lexical/exact-match signal for legal RAG."""
    document_name = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = re.sub(r"\s+", " ", f"{document_name} {content}").strip()
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    if not text or not query:
        return 0.0

    score = 0.90 if len(query) >= 18 and query in text else 0.0

    phrase_weights = {
        "ppe_nonprovision": (
            ("средства индивидуальной защиты", 0.40),
            ("не выданы", 0.40),
            ("не обеспечен средствами индивидуальной защиты", 0.40),
            ("не приступать к работе", 0.35),
            ("приостановить работу", 0.35),
            ("отказаться от выполнения работы", 0.30),
        ),
        "medical_examinations": (
            ("медицинский осмотр", 0.45),
            ("обязательный медицинский осмотр", 0.40),
            ("предварительный медицинский осмотр", 0.35),
            ("периодический медицинский осмотр", 0.35),
        ),
        "workplace_attestation": (
            ("аттестация рабочих мест", 0.50),
            ("условия труда", 0.30),
            ("вредные условия труда", 0.35),
            ("полный рабочий день", 0.25),
            ("периодичность аттестации", 0.70),
            ("один раз в пять лет", 0.90),
            ("срок действия результатов аттестации", 0.75),
            ("очередной аттестации", 0.55),
        ),
        "accident_investigation": (
            ("акт формы н-1", 0.45),
            ("форма н-1", 0.45),
            ("несчастный случай на производстве", 0.45),
            ("окончания расследования", 0.30),
            ("пострадавшему", 0.25),
        ),
        "occupational_training": (
            ("стажировка", 0.40),
            ("самостоятельной работе", 0.35),
            ("не менее двух рабочих дней", 0.45),
            ("повышенной опасностью", 0.30),
            ("проверки знаний", 0.25),
        ),
    }

    if topic == "occupational_briefing":
        mode = (
            "target"
            if _is_target_briefing_query(user_query)
            else (
                "responsible"
                if _is_responsible_briefing_query(user_query)
                else "generic"
            )
        )

        target_phrases = (
            ("целевой инструктаж", 0.75),
            ("разовых работ", 0.60),
            ("не связанных с прямыми обязанностями", 0.90),
            ("не связаны с прямыми обязанностями", 0.90),
            ("наряд-допуск", 0.55),
            ("наряду-допуску", 0.65),
            ("наряд допуск", 0.45),
        )
        responsible_phrases = (
            ("вводный инструктаж", 0.70),
            ("проводит инструктаж", 0.60),
            ("специалист по охране труда", 0.60),
            ("уполномоченное должностное лицо", 0.60),
            ("руководитель организации", 0.25),
            ("руководитель структурного подразделения", 0.25),
        )

        if mode == "target":
            for phrase, weight in target_phrases:
                if phrase in text:
                    score += weight
            responsible_matches = sum(
                1 for phrase, _ in responsible_phrases if phrase in text
            )
            score -= min(responsible_matches * 0.12, 0.36)
        elif mode == "responsible":
            for phrase, weight in responsible_phrases:
                if phrase in text:
                    score += weight
            target_matches = sum(
                1 for phrase, _ in target_phrases if phrase in text
            )
            score -= min(target_matches * 0.10, 0.30)
        else:
            for phrase, weight in target_phrases + responsible_phrases:
                if phrase in text:
                    score += weight * 0.55
    else:
        for phrase, weight in phrase_weights.get(topic, ()):
            if phrase in text:
                score += weight

    if topic in ("occupational_briefing", "occupational_training") and "175" in document_name:
        score += 0.08

    return max(0.0, min(score, 1.00))


def _minor_special_relevance_score(
    chunk: Dict[str, Any],
    special_category: Optional[str],
    issue: Optional[str],
) -> float:
    """Точечный score для специальных гарантий несовершеннолетних."""
    if special_category != "minor":
        return 0.0

    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"
    score = 0.0

    if issue == "holiday_weekend":
        if re.search(r"(?<!\d)276(?!\d)", point) or "статья 276" in text:
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.28
        if "государственные праздники" in text or "праздничные дни" in text:
            score += 0.22
        if "выходные дни" in text:
            score += 0.12
    elif issue == "night_overtime":
        if re.search(r"(?<!\d)276(?!\d)", point) or "статья 276" in text:
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.25
        if "ночным" in text or "сверхурочным" in text:
            score += 0.20
    elif issue == "prohibited_work":
        if re.search(r"(?<!\d)274(?!\d)", point) or "статья 274" in text:
            score += 0.50
        if "лиц моложе восемнадцати лет" in text:
            score += 0.25
        if "тяжелых работах" in text or "вредными" in text or "опасными условиями" in text:
            score += 0.20
    elif issue == "medical":
        if re.search(r"(?<!\d)275(?!\d)", point) or "статья 275" in text:
            score += 0.50
        if "лиц моложе восемнадцати лет" in text:
            score += 0.25
        if "медицинских осмотров" in text:
            score += 0.20
    elif issue == "leave":
        if re.search(r"(?<!\d)277(?!\d)", point) or "статья 277" in text:
            score += 0.50
        if "работникам моложе восемнадцати лет" in text:
            score += 0.30
        if "трудовые отпуска" in text:
            score += 0.20
    elif issue == "work_time":
        if re.search(r"(?<!\d)278(?!\d)|(?<!\d)279(?!\d)", point):
            score += 0.50
        if "работников моложе восемнадцати лет" in text:
            score += 0.20
        if "сокращенной продолжительности" in text:
            score += 0.20

    if "моложе восемнадцати лет" in text or "несовершеннолетн" in text:
        score += 0.10

    return min(score, 1.00)



def _constraint_scope_relevance_score(
    chunk: Dict[str, Any],
    constraint: Dict[str, Any],
) -> float:
    """Оценивает совпадение области действия количественной нормы."""
    if not constraint or constraint.get("type") != "maximum":
        return 0.0

    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"
    scope = constraint.get("scope")
    score = 0.0

    if scope == "manual_loading_unloading":
        if any(marker in text for marker in (
            "погрузочно-разгрузоч",
            "погрузочно разгрузоч",
            "погрузочных работ",
            "разгрузочных работ",
        )):
            score += 0.45
        if re.search(r"(?<!\d)86\.?\b", point) or "пункт 86" in text:
            score += 0.30
    elif scope == "manual_handling":
        if any(marker in text for marker in (
            "вручную",
            "ручное перемещение",
            "ручной перенос",
            "перемещение тяжестей",
            "подъем тяжестей",
        )):
            score += 0.25

    if constraint.get("unit") == "kg":
        if any(marker in text for marker in (
            "50 кг",
            "50 килограмм",
            "не более 50 кг",
            "не более 50 килограмм",
        )):
            score += 0.20

    return min(score, 1.0)


def _ppe_refusal_relevance_score(
    chunk: Dict[str, Any],
) -> float:
    """
    Узкий score для вопроса о праве работника отказаться от работы
    при непредоставлении СИЗ. Приоритет — ст. 11 Закона № 356-З.
    """
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"

    score = 0.0

    if chunk.get("_ppe_refusal_targeted"):
        score += 0.80
    if "356-з" in document or ("закон" in document and "охране труда" in document):
        score += 0.35
    if re.search(r"статья\s*11|ст\.\s*11|(?<!\d)11(?!\d)", point):
        score += 1.00
    if "отказ от выполнения порученной работы" in text:
        score += 0.70
    if "непредоставлении ему средств индивидуальной защиты" in text:
        score += 0.90
    if "непосредственно обеспечивающих безопасность труда" in text:
        score += 0.70
    if "имеет право" in text and "работник" in text:
        score += 0.35
    if "незамедлительно письменно сообщить работодателю" in text:
        score += 0.25

    return min(score, 4.00)


def _osh_knowledge_frequency_relevance_score(
    chunk: Dict[str, Any],
    subject: str = "working_persons",
) -> float:
    """Узкий score для пп. 42/43/51/53 Инструкции № 175 с учетом категории."""
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"
    score = 0.0

    if chunk.get("_osh_knowledge_frequency_targeted"):
        score += 0.80
    if "175" in document:
        score += 0.35
    if "периодическ" in text and "провер" in text and "знан" in text:
        score += 0.55

    if subject == "managers_specialists":
        if re.search(r"(?<!\d)42\.?\b", point):
            score += 1.40
        if re.search(r"(?<!\d)43\.?\b", point):
            score += 1.20
        if "руководители и специалисты" in text:
            score += 0.60
        if "не реже одного раза в три года" in text:
            score += 1.20
        if "не позднее месяца со дня назначения" in text:
            score += 0.30
        if "рабочие" in text and "руководители и специалисты" not in text:
            score -= 0.80
    elif subject == "workers":
        if re.search(r"(?<!\d)51\.?\b", point):
            score += 1.40
        if "рабочие" in text:
            score += 0.35
        if "не реже одного раза в 12 месяцев" in text or "не реже одного раза в год" in text:
            score += 1.20
        if "повышенной опасностью" in text:
            score += 0.30
        if "опасных производственных объектах" in text or "потенциально опасных объектах" in text:
            score += 0.25
        if "руководители и специалисты" in text and "рабочие" not in text:
            score -= 0.80
    else:
        if re.search(r"(?<!\d)42\.?\b", point):
            score += 0.90
        if re.search(r"(?<!\d)51\.?\b", point):
            score += 0.90
        if re.search(r"(?<!\d)53\.?\b", point):
            score += 0.70
        if "не реже одного раза в три года" in text or "не реже одного раза в 12 месяцев" in text:
            score += 0.80

    return min(max(score, 0.0), 5.00)


def _work_break_briefing_relevance_score(
    chunk: Dict[str, Any],
) -> float:
    """Точечный score для п. 27 Инструкции № 175."""
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"

    score = 0.0

    if chunk.get("_work_break_targeted"):
        score += 0.60
    if "175" in document:
        score += 0.20
    if re.search(r"(?<!\d)27(?!\d)", point):
        score += 0.80
    if "внеплановый инструктаж" in text:
        score += 0.45
    if "перерывах в работе" in text or "перерыв в работе" in text:
        score += 0.40
    if "более шести месяцев" in text or "более 6 месяцев" in text:
        score += 0.50
    if "по профессии" in text or "в должности" in text:
        score += 0.20

    return min(score, 2.50)


def _universal_query_relevance_score(
    chunk: Dict[str, Any],
    profile: Dict[str, Any],
) -> float:
    """Универсальный reranker: насколько chunk отвечает именно вопросу."""
    document = _get_document_name(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    point = _get_point_number(chunk).lower()
    text = f"{document} {point} {content}"
    score = 0.0

    qtype = profile.get("question_type") or "general"
    event = profile.get("event")
    state = profile.get("action_state")
    qualifiers = set(profile.get("qualifiers") or [])
    phrases = profile.get("legal_phrases") or []

    def hits(markers):
        return sum(1 for marker in markers if marker in text)

    score += min(sum(1 for phrase in phrases if phrase.lower() in text) * 0.16, 0.48)

    if event == "law_scope":
        target_document = str(profile.get("target_document") or "").lower()
        target_article = str(profile.get("target_article") or "").lower()

        if target_document == "356-з":
            doc_hit = "356-з" in document or "356 з" in document
            article_hit = bool(target_article and point.rstrip(".") == target_article.rstrip("."))
            scope_phrase_hits = hits((
                "сфера действия настоящего закона",
                "применяется в отношении всех работодателей",
                "работающих граждан республики беларусь",
                "иностранных граждан и лиц без гражданства",
            ))
            score += 0.55 if doc_hit else -0.45
            if article_hit:
                score += 0.95
            if scope_phrase_hits:
                score += min(scope_phrase_hits * 0.22, 0.66)
            if chunk.get("_law_scope_targeted"):
                score += 0.80

        elif target_document == "правила по охране труда 53":
            doc_hit = (
                "правила по охране труда № 53" in document
                or "правила по охране труда 53" in document
            )
            article_hit = bool(target_article and point.rstrip(".") == target_article.rstrip("."))
            scope_phrase_hits = hits((
                "распространяются на работодателей",
                "независимо от их организационно-правовых форм и форм собственности",
                "различные виды экономической деятельности",
                "требования по охране труда",
            ))
            score += 0.75 if doc_hit else -0.45
            if article_hit:
                score += 1.00
            if scope_phrase_hits:
                score += min(scope_phrase_hits * 0.20, 0.80)
            if chunk.get("_law_scope_targeted"):
                score += 0.90

        elif target_document == "трудовой кодекс":
            if "трудовой кодекс" in document or "трудовои кодекс" in document:
                score += 0.65
            else:
                score -= 0.25
            score += min(hits(("сфера действия", "трудовые отношения")) * 0.18, 0.36)
            if chunk.get("_law_scope_targeted"):
                score += 0.50

    if event == "work_accident":
        score += min(hits(("несчастный случай", "несчастном случае")) * 0.10, 0.20)
        score += min(hits(("сообщить", "сообщает", "сообщают", "уведомить")) * 0.07, 0.21)
        if profile.get("recipient") == "immediate_supervisor":
            score += min(hits(("непосредственному руководителю", "должностному лицу страхователя", "руководителю")) * 0.10, 0.30)
        if state == "not_done":
            score += min(hits(("не сообщил", "не сообщила", "несвоевременно")) * 0.14, 0.28)

    if event == "lifting_and_moving_loads":
        score += min(hits(("подъем", "подъема", "перемещение", "перемещения", "тяжест")) * 0.07, 0.28)
        score += min(hits(("вручную", "ручн")) * 0.10, 0.20)
        score += min(hits(("предельно допустим", "предельные нормы", "нормы подъема")) * 0.18, 0.36)
        constraint = profile.get("constraint") or {}
        constraint_subject = constraint.get("subject")
        if constraint.get("type") == "maximum":
            scope_score = _constraint_scope_relevance_score(chunk, constraint)
            chunk["_constraint_scope_score"] = scope_score
            score += scope_score * 0.45

        if "men" in qualifiers or constraint_subject == "adult_male":
            score += min(hits(("мужчин", "мужчина", "мужского пола", "работающим мужчиной")) * 0.16, 0.32)
            wrong_subject_hits = hits(("женщин", "женщина", "лиц моложе восемнадцати лет", "несовершеннолетн"))
            score -= min(wrong_subject_hits * 0.18, 0.36)
            if hits(("50 кг", "50 килограмм", "не более 50 кг", "не более 50 килограмм")):
                score += 0.42
            if hits(("погрузочно-разгрузочн", "погрузочно разгрузочн")):
                score += 0.22
            if hits(("разовый подъем", "разового подъема", "разовом подъеме")):
                score += 0.18
            if hits(("пункт 86", "п. 86", "26.01.2018", "постановления 12")):
                score += 0.20
        elif "women" in qualifiers or constraint_subject == "adult_female":
            score += min(hits(("женщин", "женщина", "женского пола")) * 0.16, 0.32)
            wrong_subject_hits = hits(("мужчин", "мужчина", "лиц моложе восемнадцати лет", "несовершеннолетн"))
            score -= min(wrong_subject_hits * 0.18, 0.36)
        if qtype == "limit":
            score += min(hits(("кг", "килограмм", "масса", "вес")) * 0.08, 0.16)

    if event == "workplace_attestation":
        score += min(hits(("аттестаци", "рабочих мест", "условия труда")) * 0.08, 0.24)
        if qtype == "frequency":
            score += min(hits(("периодичност", "срок действия результатов", "пять лет", "очередн")) * 0.15, 0.45)

    if event == "occupational_briefing":
        score += min(hits(("инструктаж", "охране труда")) * 0.08, 0.16)
        if qtype == "kind":
            score += min(hits(("виды инструктажей", "вводный инструктаж", "первичный инструктаж", "повторный инструктаж", "внеплановый инструктаж", "целевой инструктаж")) * 0.13, 0.39)
        elif qtype == "who":
            score += min(hits(("проводит", "специалист по охране труда", "уполномоченное должностное лицо", "руководитель")) * 0.12, 0.36)

    if event == "ppe":
        score += min(hits(("средств индивидуальной защиты", "сиз", "обеспеч", "выдач")) * 0.08, 0.24)
        if state == "not_done":
            score += min(hits(("не выдан", "не обеспечен", "неисправн", "поврежден")) * 0.14, 0.28)

    if event == "medical_exam":
        score += min(hits(("медицинск", "осмотр", "обязательн", "предварительн", "периодическ")) * 0.08, 0.32)

    if event == "occupational_training":
        score += min(hits(("стажиров", "самостоятельной работе", "рабочих дней", "рабочих смен")) * 0.10, 0.35)

    focus_markers = {
        "what_to_do": ("порядок", "действия", "обязан", "должен", "немедленно", "следует"),
        "who": ("проводит", "обязан", "должностному лицу", "ответствен", "назнач"),
        "frequency": ("периодичност", "раз в", "срок действия", "пять лет"),
        "limit": ("предельн", "норм", "максимальн", "кг", "килограмм", "допустим"),
        "whether": ("вправе", "имеет право", "может", "допускается", "разрешается"),
        "responsibility": ("ответствен", "взыскан", "штраф", "нарушен"),
        "term": ("срок", "в течение", "не позднее", "дней", "месяц", "лет"),
        "document": ("положение", "правила", "инструкция", "кодекс", "постановлен", "приказ"),
        "kind": ("виды инструктаж", "вводный", "первичный", "повторный", "внеплановый", "целевой"),
    }
    if qtype in focus_markers:
        score += min(hits(focus_markers[qtype]) * 0.08, 0.24)

    if state == "not_done":
        negative_hits = hits(("не сообщил", "не выдан", "не обеспечен", "не прошел", "не проведен", "не допущен", "не приступ"))
        if negative_hits:
            score += min(negative_hits * 0.10, 0.20)

    return max(0.0, min(score, 1.0))


def _legal_authority_relevance_score(
    chunk: Dict[str, Any],
    user_query: str,
    query_profile: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Legal-authority reranker.

    Semantic similarity is useful for discovery, but for legal questions
    an exact normative target (document/article/point) must outrank a
    merely semantically similar document such as a general code provision.
    """
    profile = query_profile or {}
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    query = re.sub(r"\s+", " ", str(user_query or "").strip().lower())

    score = 0.0

    target_document = str(profile.get("target_document") or "").strip().lower()
    target_article = str(profile.get("target_article") or "").strip().lower()

    if target_document:
        normalized_doc = re.sub(r"[^a-zа-яё0-9]+", " ", document)
        normalized_target = re.sub(r"[^a-zа-яё0-9]+", " ", target_document)
        if normalized_target and normalized_target in normalized_doc:
            score += 0.75

    if target_article:
        normalized_point = point.rstrip(".").strip()
        normalized_article = target_article.rstrip(".").strip()
        if normalized_point == normalized_article:
            score += 1.15

    # Explicit NPA number in the user query is a strong legal signal.
    npa_numbers = re.findall(r"№\s*([0-9]+(?:[-/]?[а-яa-z0-9]+)?)", query, flags=re.IGNORECASE)
    if npa_numbers:
        for number in npa_numbers:
            if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", document, flags=re.IGNORECASE):
                score += 0.55
                break

    # Explicit article/point marker in the query.
    article_match = re.search(
        r"\b(?:статья|ст\.?|пункт|п\.?)\s*([0-9]+(?:\.[0-9]+)*)",
        query,
        flags=re.IGNORECASE,
    )
    if article_match:
        requested_point = article_match.group(1).rstrip(".")
        if point.rstrip(".") == requested_point:
            score += 0.85

    # When the user explicitly asks about the Labour Code, unrelated NPA
    # documents should not win solely because their wording is semantically close.
    if "трудовой кодекс" in query or "трудовом кодексе" in query:
        if "трудовой кодекс" in document or "трудовои кодекс" in document:
            score += 0.35
        else:
            score -= 0.35

    # Same legal domain is a weak positive signal; domain-specific topic
    # and existing targeted markers remain handled by the other rerankers.
    if chunk.get("legal_domain") == "occupational_safety":
        score += 0.05

    # Targeted lexical retrieval is stronger evidence than raw cosine similarity.
    if any(
        chunk.get(flag)
        for flag in (
            "_law_scope_targeted",
            "_ppe_refusal_targeted",
            "_work_break_targeted",
            "_training_suspension_targeted",
            "_osh_knowledge_frequency_targeted",
        )
    ):
        score += 0.30

    return score


def _legal_relevance_score(
    chunk: Dict[str, Any],
    query_terms: List[str],
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
    user_query: str = "",
    special_category: Optional[str] = None,
    special_issue: Optional[str] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> float:
    semantic = _safe_float(chunk.get("_best_similarity", _semantic_score(chunk)))
    hybrid_score = _safe_float(chunk.get("_hybrid_final_score"))
    keyword = _keyword_score(chunk, query_terms)
    exact_score = _exact_match_score(chunk, user_query, topic)

    briefing_mode_bonus = 0.0
    if topic == "occupational_briefing":
        mode = (
            "target"
            if _is_target_briefing_query(user_query)
            else (
                "responsible"
                if _is_responsible_briefing_query(user_query)
                else "generic"
            )
        )
        content_lower = str(chunk.get("content") or "").lower()
        document_lower = _get_document_name(chunk).lower()

        target_matches = sum(
            marker in content_lower
            for marker in (
                "целевой инструктаж",
                "разовых работ",
                "не связанных с прямыми обязанностями",
                "прямыми обязанностями",
                "наряд-допуск",
                "наряду-допуску",
            )
        )
        responsible_matches = sum(
            marker in content_lower
            for marker in (
                "вводный инструктаж",
                "проводит инструктаж",
                "специалист по охране труда",
                "уполномоченное должностное лицо",
            )
        )

        if mode == "target":
            briefing_mode_bonus += min(target_matches * 0.08, 0.24)
            briefing_mode_bonus -= min(responsible_matches * 0.04, 0.12)
            if "175" in document_lower:
                briefing_mode_bonus += 0.04
        elif mode == "responsible":
            briefing_mode_bonus += min(responsible_matches * 0.08, 0.24)
            briefing_mode_bonus -= min(target_matches * 0.04, 0.12)

    chunk["_briefing_mode_bonus"] = briefing_mode_bonus
    topic_score = min(_topic_relevance_score(chunk, topic), 1.0)
    intent_score = _intent_relevance_score(chunk, intents)
    primary_score = _primary_intent_relevance_score(chunk, primary_intent, topic)
    hits = int(chunk.get("_search_hits", 1))
    repeated_bonus = min(max(hits - 1, 0) * 0.025, 0.10)
    topic_weight = 0.06 if cross_reference else 0.08
    keyword_weight = 0.10 if cross_reference else 0.12

    labor_code_bonus = 0.0
    if labor_code_query:
        document_name = _get_document_name(chunk).lower()
        if (
            "трудовой кодекс" in document_name
            or "трудовои кодекс" in document_name
            or "трудовой_кодекс" in document_name
        ):
            labor_code_bonus = 0.10

    special_category_bonus = _minor_special_relevance_score(
        chunk,
        special_category,
        special_issue,
    )
    chunk["_special_category_bonus"] = special_category_bonus

    universal_score = _universal_query_relevance_score(
        chunk,
        query_profile or build_universal_query_profile(user_query),
    )
    chunk["_universal_score"] = universal_score
    if query_profile and query_profile.get("event") == "law_scope":
        chunk["_scope_score"] = universal_score

    authority_score = _legal_authority_relevance_score(
        chunk,
        user_query,
        query_profile or build_universal_query_profile(user_query),
    )
    chunk["_legal_authority_score"] = authority_score

    return (
        semantic * 0.27
        + hybrid_score * 0.12
        + exact_score * 0.16
        + universal_score * (0.34 if query_profile and query_profile.get("event") == "law_scope" else 0.22)
        + keyword * keyword_weight
        + topic_score * topic_weight
        + intent_score * 0.05
        + primary_score * 0.10
        + authority_score * 0.13
        + briefing_mode_bonus
        + labor_code_bonus
        + special_category_bonus
        + repeated_bonus
    )


def _legal_chunk_role(
    chunk: Dict[str, Any],
    topic: str,
    intents: List[str],
    primary_intent: Optional[str] = None,
) -> str:
    document = str(chunk.get("doc_name") or chunk.get("document") or "").lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {content}"

    if topic == "ppe_nonprovision" and (
        "209" in document
        or "средств индивидуальной защиты" in document
        or "сиз" in document
    ) and any(
        marker in content
        for marker in (
            "не выдан",
            "невыдач",
            "поврежден",
            "неисправн",
            "отказ",
        )
    ):
        return "ppe_specific"

    employer_markers = (
        "обязанность нанимателя",
        "обязанности нанимателя",
        "наниматель обязан",
        "наниматель обеспечивает",
        "наниматель должен",
        "работодатель обязан",
        "обязан обеспечить",
    )

    employee_right_markers = (
        "имеет право",
        "право работника",
        "работник вправе",
        "вправе",
    )

    procedure_markers = (
        "действия работника",
        "немедленно сообщ",
        "уведом",
        "порядок действий",
        "не приступать",
        "приостановить работу",
    )

    danger_markers = (
        "угроз",
        "опасност",
        "жизни и здоров",
        "жизни или здоров",
    )

    liability_markers = (
        "ответственност",
        "штраф",
        "взыскан",
        "дисциплинарн",
    )

    responsible_person_markers = (
        "проводит вводный инструктаж",
        "проводит инструктаж",
        "специалист по охране труда",
        "уполномоченное должностное лицо",
        "руководитель организации",
        "руководитель структурного подразделения",
    )

    if primary_intent == "responsible_person":
        ordered_checks = [
            ("responsible_person", responsible_person_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("employee_right", employee_right_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent == "employer_duty":
        ordered_checks = [
            ("employer_duty", employer_markers),
            ("employee_right", employee_right_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent == "employee_right":
        ordered_checks = [
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]
    elif primary_intent in ("refusal", "danger", "procedure"):
        ordered_checks = [
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("liability", liability_markers),
        ]
    else:
        ordered_checks = [
            ("employee_right", employee_right_markers),
            ("employer_duty", employer_markers),
            ("procedure", procedure_markers),
            ("danger", danger_markers),
            ("liability", liability_markers),
        ]

    for role, markers in ordered_checks:
        if any(marker in text for marker in markers):
            return role

    return "general"



def _accident_worker_not_report_relevance_score(
    chunk: Dict[str, Any],
) -> float:
    """
    Узкий score для вопроса: работник/потерпевший не сообщил о НС
    непосредственному руководителю/нанимателю.

    Наличие слова «сообщить» само по себе недостаточно: нормы о сообщении
    в прокуратуру, ГИТ, страховщику, оформлении Н-1 и вручении акта относятся
    к другим этапам процедуры расследования.
    """
    document = _get_document_name(chunk).lower()
    point = _get_point_number(chunk).lower()
    content = str(chunk.get("content") or chunk.get("text") or "").lower()
    text = f"{document} {point} {content}"

    score = 0.0

    direct_phrases = (
        ("не сообщил", 0.45),
        ("не сообщила", 0.45),
        ("не сообщив", 0.40),
        ("сообщить о несчастном случае", 0.45),
        ("сообщить о происшедшем несчастном случае", 0.50),
        ("сообщить непосредственному руководителю", 0.60),
        ("непосредственному руководителю", 0.45),
        ("сообщить руководителю", 0.40),
        ("немедленно сообщить", 0.35),
        ("обязан сообщить", 0.35),
    )

    for phrase, weight in direct_phrases:
        if phrase in text:
            score += weight

    if "30" in document:
        score += 0.12
    if "81 144" in document or "81-144" in document:
        score += 0.10

    unrelated_markers = (
        "группов",
        "акт н-1",
        "форма н-1",
        "вручение потерпевшему",
        "вручение родственникам",
        "после окончания расследования",
        "прокуратур",
        "государственной инспекции труда",
        "страховщик",
    )
    unrelated_matches = sum(marker in text for marker in unrelated_markers)
    score -= min(unrelated_matches * 0.12, 0.48)

    return max(0.0, min(score, 1.50))


def _select_legal_diverse_chunks(
    ranked_chunks: List[Dict[str, Any]],
    limit: int,
    topic: str,
    intents: List[str],
    cross_reference: bool,
    primary_intent: Optional[str] = None,
    labor_code_query: bool = False,
    special_category: Optional[str] = None,
    special_issue: Optional[str] = None,
    accident_mode: Optional[str] = None,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not ranked_chunks or limit <= 0:
        return []

    selected: List[Dict[str, Any]] = []
    selected_keys = set()
    roles_seen = set()
    documents_seen: Dict[str, int] = {}
    points_seen = set()

    def _key(chunk: Dict[str, Any]):
        return (
            _get_document_key(chunk),
            _get_point_number(chunk).lower(),
            str(chunk.get("content") or "")[:200].lower(),
        )

    def _point_key(chunk: Dict[str, Any]):
        point = _get_point_number(chunk).strip().lower()
        if not point:
            return None
        return (
            _get_document_key(chunk),
            point,
        )

    def _add(
        chunk: Dict[str, Any],
        max_per_document: int = 4,
        max_per_point: int = 1,
    ) -> bool:
        key = _key(chunk)
        document_key = _get_document_key(chunk)
        point_key = _point_key(chunk)

        if key in selected_keys:
            return False

        # Один и тот же пункт/статья не должен занимать несколько мест
        # финального юридического контекста. Разные пункты одного документа
        # по-прежнему могут попадать в ответ.
        if point_key is not None and point_key in points_seen:
            return False

        if documents_seen.get(document_key, 0) >= max_per_document:
            return False

        selected.append(chunk)
        selected_keys.add(key)
        documents_seen[document_key] = documents_seen.get(document_key, 0) + 1
        if point_key is not None:
            points_seen.add(point_key)
        return True

    if topic == "portable_ladder":
        # Для лестниц действующая норма № 11 п. 54 должна быть
        # нормативным ядром. Не позволяем общему score вытеснить её.
        ladder_pool = [chunk for chunk in ranked_chunks if chunk.get("_portable_ladder_targeted")]
        ladder_pool.sort(
            key=lambda chunk: (
                1 if _normalize_point_identifier(_get_point_number(chunk)) == "54" else 0,
                1 if _normalize_point_identifier(_get_point_number(chunk)) == "53" else 0,
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )
        for required_point in ("54", "53"):
            for chunk in ladder_pool:
                if _normalize_point_identifier(_get_point_number(chunk)) == required_point:
                    _add(chunk, max_per_document=4, max_per_point=1)
                    break
        for chunk in ladder_pool:
            if len(selected) >= min(limit, 3):
                break
            _add(chunk, max_per_document=4, max_per_point=1)
        if selected:
            logger.info(
                "RAG | portable ladder final | sources=%s",
                [f"{_get_document_name(x)}#{_get_point_number(x)}" for x in selected],
            )
            return selected


    if topic == "height_work_training":
        # Алгоритм обучения должен быть нормативно полным, а не состоять
        # из пяти наиболее похожих фрагментов. Сначала фиксируем ключевые
        # пункты Правил № 11 (48-51), затем добираем процедурные нормы № 175.
        height_pool = [chunk for chunk in ranked_chunks if chunk.get("_height_training_targeted")]
        priority_11 = {"48": 1000, "49": 990, "50": 980, "51": 970}
        priority_175 = {"36": 900, "37": 890, "38": 880, "39": 870, "40": 860, "44": 850, "45": 840}

        def _height_priority(chunk):
            doc = _get_document_name(chunk).lower()
            point = _get_point_number(chunk).strip().rstrip(".")
            if re.search(r"№\s*11\b", doc) or "работ на высоте" in doc:
                base = priority_11.get(point, 40)
            elif re.search(r"№\s*175\b", doc) or "инструкци" in doc and "175" in doc:
                base = priority_175.get(point, 20)
            else:
                base = 10
            return (base, _safe_float(chunk.get("_combined_score")))

        # Для этого сценария пп. 48-51 Правил № 11 являются обязательным
        # нормативным ядром. Сначала добавляем их детерминированно по одному,
        # чтобы общий score/лимит документа не мог вытеснить, например, п. 51.
        logger.info(
            "RAG | height training targeted points=%s",
            [
                f"{_get_document_name(chunk)}#{_get_point_number(chunk)}"
                for chunk in height_pool
            ],
        )

        height_pool.sort(key=_height_priority, reverse=True)

        for required_point in ("48", "49", "50", "51"):
            for chunk in height_pool:
                doc = _get_document_name(chunk).lower()
                point = _get_point_number(chunk).strip().rstrip(".")
                if required_point != point:
                    continue
                if not (
                    re.search(r"№\s*11\b", doc)
                    or "работ на высоте" in doc
                ):
                    continue
                _add(chunk, max_per_document=5, max_per_point=1)
                break

        # После обязательного ядра добираем процедурные нормы Инструкции №175.
        for chunk in height_pool:
            if _add(chunk, max_per_document=5, max_per_point=1):
                if len(selected) >= min(limit, 9):
                    return selected

        if selected:
            return selected

    if accident_mode == "worker_did_not_report":
        # Для этого intent финальный отбор идёт из узкого targeted-пула.
        # Общий vector/hybrid-поиск не должен вытеснять нужные нормы
        # правилами про Н-1, групповые НС, прокуратуру или вручение акта.
        targeted_pool = [
            chunk for chunk in ranked_chunks
            if chunk.get("_accident_worker_not_report_targeted")
        ]

        targeted_pool.sort(
            key=lambda chunk: (
                _accident_worker_not_report_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in targeted_pool:
            relevance = _accident_worker_not_report_relevance_score(chunk)
            if relevance < 0.45:
                continue
            if _add(chunk, max_per_document=4, max_per_point=1):
                if len(selected) >= limit:
                    return selected

        # Добор разрешён только из основных НПА №30 и №81-144
        # и только при наличии прямой связи с сообщением о НС.
        fallback_pool = [
            chunk for chunk in ranked_chunks
            if _accident_worker_not_report_relevance_score(chunk) >= 0.35
            and (
                "30" in _get_document_name(chunk).lower()
                or "81 144" in _get_document_name(chunk).lower()
                or "81-144" in _get_document_name(chunk).lower()
            )
        ]
        fallback_pool.sort(
            key=lambda chunk: (
                _accident_worker_not_report_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in fallback_pool:
            _add(chunk, max_per_document=4, max_per_point=1)
            if len(selected) >= limit:
                return selected

    if special_category == "minor":
        # Для конкретного ограничения несовершеннолетнего сначала выбираем
        # именно норму, регулирующую этот вопрос. Для holiday_weekend это
        # должна быть ст. 276, а не случайные статьи главы 20.
        if special_issue:
            issue_threshold = 0.70

            for chunk in ranked_chunks:
                special_score = _minor_special_relevance_score(
                    chunk,
                    special_category,
                    special_issue,
                )
                if special_score < issue_threshold:
                    continue

                # max_per_point=1 не позволяет двум chunks одной статьи
                # занимать два из пяти мест финального контекста.
                if _add(
                    chunk,
                    max_per_document=3,
                    max_per_point=1,
                ):
                    break

            # Если основной нормы недостаточно для заполнения контекста,
            # добираем только действительно релевантные нормы той же
            # специальной категории. Слабые совпадения (например, общие
            # статьи ТК о труде) сюда не должны попадать на раннем этапе.
            category_threshold = 0.50

            for chunk in ranked_chunks:
                if len(selected) >= min(3, limit):
                    break

                if _minor_special_relevance_score(
                    chunk,
                    special_category,
                    special_issue,
                ) < category_threshold:
                    continue

                _add(
                    chunk,
                    max_per_document=3,
                    max_per_point=1,
                )

    # Для явного вопроса о сфере действия сначала выбираем норму
    # внутри указанного НПА. Это защищает от вытеснения ст. 3
    # семантически похожими правилами по другим темам.
    if (query_profile or {}).get("event") == "law_scope":
        target_document = str((query_profile or {}).get("target_document") or "").lower()
        scope_pool = []

        for chunk in ranked_chunks:
            text_lower = (
                f"{_get_document_name(chunk)} "
                f"{_get_point_number(chunk)} "
                f"{str(chunk.get('content') or '')}"
            ).lower()

            if target_document == "356-з":
                if ("356-з" in text_lower or "356 з" in text_lower) and (
                    _get_point_number(chunk).rstrip(".") == "3"
                    or "сфера действия настоящего закона" in text_lower
                    or "применяется в отношении всех работодателей" in text_lower
                ):
                    scope_pool.append(chunk)
            elif target_document == "правила по охране труда 53":
                if (
                    ("правила по охране труда № 53" in text_lower
                     or "правила по охране труда 53" in text_lower)
                    and (
                        _get_point_number(chunk).rstrip(".") == "2"
                        or "распространяются на работодателей" in text_lower
                        or "организационно-правовых форм и форм собственности" in text_lower
                    )
                ):
                    scope_pool.append(chunk)
            elif target_document == "трудовой кодекс":
                if "трудовой кодекс" in text_lower:
                    scope_pool.append(chunk)
            else:
                if "сфера действия" in text_lower or "на кого распространяется" in text_lower:
                    scope_pool.append(chunk)

        scope_pool.sort(
            key=lambda chunk: (
                _safe_float(chunk.get("_scope_score")),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in scope_pool:
            if _add(chunk, max_per_document=4, max_per_point=1):
                if len(selected) >= min(3, limit):
                    return selected

    # Универсальный приоритет для количественных ограничений.
    # Сначала отбираем норму, соответствующую субъекту и операции запроса,
    # чтобы общие нормы ТК/КоАП не вытесняли прямое числовое ограничение.
    constraint = (query_profile or {}).get("constraint") or {}
    if constraint.get("type") == "maximum" and constraint.get("unit") == "kg":
        constraint_candidates = []
        for chunk in ranked_chunks:
            cscore = _universal_query_relevance_score(chunk, query_profile or {})

            if chunk.get("_lifting_constraint_targeted"):
                cscore += 0.80
            text_lower = (
                f"{_get_document_name(chunk)} "
                f"{_get_point_number(chunk)} "
                f"{str(chunk.get('content') or '')}"
            ).lower()

            if constraint.get("subject") == "adult_male":
                if any(x in text_lower for x in ("50 кг", "50 килограмм", "не более 50 кг", "не более 50 килограмм")):
                    cscore += 0.50
                if re.search(r"\b12\b", text_lower) and (
                    "погрузочно-разгрузоч" in text_lower
                    or "погрузочно разгрузоч" in text_lower
                ):
                    cscore += 0.35
                if re.search(r"\b86\b", text_lower) and (
                    "50 кг" in text_lower
                    or "разовый подъем" in text_lower
                    or "разового подъема" in text_lower
                ):
                    cscore += 0.35
                if "погрузочно-разгрузоч" in text_lower or "погрузочно разгрузоч" in text_lower:
                    cscore += 0.25
                if any(x in text_lower for x in ("пункт 86", "п. 86", "26.01.2018", "постановления 12")):
                    cscore += 0.25
                # Для вопроса именно о мужчинах общая норма о ручном
                # перемещении тяжестей без числового ограничения слабее,
                # чем прямой норматив о разовом подъеме.
                if any(x in text_lower for x in ("разовый подъем", "разового подъема", "разовом подъеме")):
                    cscore += 0.18
                if any(x in text_lower for x in ("женщин", "женщина", "несовершеннолетн", "моложе восемнадцати лет")) and not any(x in text_lower for x in ("мужчин", "мужчина", "работающим мужчиной")):
                    cscore -= 0.45
            elif constraint.get("subject") == "adult_female":
                if any(x in text_lower for x in ("женщин", "женщина", "женского пола")):
                    cscore += 0.20
                if any(x in text_lower for x in ("мужчин", "мужчина", "несовершеннолетн", "моложе восемнадцати лет")):
                    cscore -= 0.35

            constraint_candidates.append((cscore, _safe_float(chunk.get("_combined_score")), chunk))

        constraint_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        constraint_selected = False
        for cscore, _, chunk in constraint_candidates:
            if cscore < 0.35:
                continue
            if _add(chunk, max_per_document=4, max_per_point=1):
                constraint_selected = True
                if len(selected) >= limit:
                    return selected

        # Если прямой числовой норматив найден, общий pool не должен его вытеснить.
        if constraint_selected:
            for cscore, _, candidate in constraint_candidates:
                if len(selected) >= limit:
                    break
                if cscore < 0.20:
                    continue
                _add(candidate, max_per_document=4, max_per_point=1)

    ppe_refusal_qualifier = "refusal_due_to_no_ppe" in (
        (query_profile or {}).get("qualifiers") or []
    )
    ppe_nonprovision_action_qualifier = "ppe_nonprovision_action" in (
        (query_profile or {}).get("qualifiers") or []
    )

    if ppe_refusal_qualifier or ppe_nonprovision_action_qualifier:
        ppe_refusal_pool = [
            chunk for chunk in ranked_chunks
            if _ppe_refusal_relevance_score(chunk) >= 1.20
        ]
        ppe_refusal_pool.sort(
            key=lambda chunk: (
                _ppe_refusal_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in ppe_refusal_pool:
            if _add(chunk, max_per_document=3, max_per_point=1):
                if len(selected) >= limit:
                    return selected

        if selected:
            # После прямой нормы можно добрать только связанные с ней
            # материалы, не вытесняя саму ст. 11.
            for chunk in ranked_chunks:
                if len(selected) >= limit:
                    break
                if _ppe_refusal_relevance_score(chunk) >= 0.45:
                    _add(chunk, max_per_document=3, max_per_point=1)

    # Процедурные вопросы по инструктажам: сначала сохраняем
    # нормативный каркас из точечного поиска № 175.
    if topic == "occupational_briefing":
        briefing_pool = [
            chunk for chunk in ranked_chunks
            if chunk.get("_occupational_briefing_targeted")
        ]
        priority = {"16": 1000, "22": 1000, "31": 980, "35": 960}
        briefing_pool.sort(
            key=lambda chunk: (
                priority.get(
                    _normalize_point_identifier(_get_point_number(chunk)),
                    500,
                ),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )
        for required_point in ("16", "22", "31", "35"):
            for chunk in briefing_pool:
                if _normalize_point_identifier(_get_point_number(chunk)) == required_point:
                    _add(chunk, max_per_document=4, max_per_point=1)
                    break
        for chunk in briefing_pool:
            if len(selected) >= min(limit, 5):
                break
            _add(chunk, max_per_document=4, max_per_point=1)

    # Для вопросов о периодичности проверки знаний приоритет имеет
    # соответствующая категория пп. 42/43 или 51 Инструкции № 175.
    frequency_qualifiers = (query_profile or {}).get("qualifiers") or []
    frequency_subject = (query_profile or {}).get("subject") or "working_persons"
    knowledge_frequency_qualifier = any(
        q in frequency_qualifiers
        for q in (
            "osh_knowledge_check_frequency_managers_specialists",
            "osh_knowledge_check_frequency_workers",
            "osh_knowledge_check_frequency",
        )
    )

    if knowledge_frequency_qualifier:
        frequency_pool = [
            chunk for chunk in ranked_chunks
            if _osh_knowledge_frequency_relevance_score(
                chunk,
                frequency_subject,
            ) >= 1.20
        ]
        frequency_pool.sort(
            key=lambda chunk: (
                _osh_knowledge_frequency_relevance_score(
                    chunk,
                    frequency_subject,
                ),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )
        for chunk in frequency_pool:
            if _add(chunk, max_per_document=3, max_per_point=1):
                if len(selected) >= limit:
                    return selected
        if selected:
            for chunk in ranked_chunks:
                if len(selected) >= limit:
                    break
                if _osh_knowledge_frequency_relevance_score(
                    chunk,
                    frequency_subject,
                ) >= 0.70:
                    _add(chunk, max_per_document=3, max_per_point=1)

    # Для запроса о перерыве более шести месяцев п. 27 Инструкции № 175
    # является прямой нормой. Он должен попасть в контекст раньше общих
    # пунктов № 25/26/30, даже если vector/hybrid search ранжировал их выше.
    work_break_qualifier = "work_break_over_six_months" in (
        (query_profile or {}).get("qualifiers") or []
    )

    if work_break_qualifier:
        work_break_pool = [
            chunk for chunk in ranked_chunks
            if _work_break_briefing_relevance_score(chunk) >= 0.80
        ]
        work_break_pool.sort(
            key=lambda chunk: (
                _work_break_briefing_relevance_score(chunk),
                _safe_float(chunk.get("_combined_score")),
            ),
            reverse=True,
        )

        for chunk in work_break_pool:
            if _add(chunk, max_per_document=3, max_per_point=1):
                if len(selected) >= limit:
                    return selected

        # Если точная норма найдена, общий отбор может только добирать
        # контекст, но не заменять п. 27.
        if selected:
            for chunk in ranked_chunks:
                if len(selected) >= limit:
                    break
                if _work_break_briefing_relevance_score(chunk) >= 0.35:
                    _add(chunk, max_per_document=3, max_per_point=1)

    if primary_intent:
        for chunk in ranked_chunks:
            role = _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            )

            if role != primary_intent:
                continue

            if labor_code_query and primary_intent == "employer_duty":
                document_name = _get_document_name(chunk).lower()
                is_labor_code = (
                    "трудовой кодекс" in document_name
                    or "трудовои кодекс" in document_name
                    or "трудовой_кодекс" in document_name
                )
                if not is_labor_code:
                    continue

            _add(chunk, max_per_document=3)

            if len(selected) >= min(2, limit):
                break

    if labor_code_query and len(selected) < limit:
        for chunk in ranked_chunks:
            document_name = _get_document_name(chunk).lower()
            is_labor_code = (
                "трудовой кодекс" in document_name
                or "трудовои кодекс" in document_name
                or "трудовой_кодекс" in document_name
            )

            if not is_labor_code:
                continue

            if _add(chunk, max_per_document=3):
                if len(selected) >= min(limit, 3):
                    break

    if cross_reference:
        max_per_document = 2
        for chunk in ranked_chunks:
            role = _legal_chunk_role(
                chunk,
                topic,
                intents,
                primary_intent=primary_intent,
            )

            if role in roles_seen:
                continue

            if _add(chunk, max_per_document=max_per_document):
                roles_seen.add(role)

            if len(selected) >= limit:
                return selected

    # Для расследования несчастных случаев ограничиваем один НПА тремя
    # фрагментами. Это предотвращает ситуацию, когда 4 из 5 мест финального
    # контекста занимает один документ (например, НПА №30), хотя в кандидатах
    # есть релевантные нормы других НПА. Для остальных тем сохраняем прежнюю
    # политику отбора.
    if topic == "accident_investigation" and not primary_intent and not cross_reference:
        max_per_document = 3
    else:
        max_per_document = 3 if primary_intent else (2 if cross_reference else 4)

    for chunk in ranked_chunks:
        _add(chunk, max_per_document=max_per_document)
        if len(selected) >= limit:
            return selected

    for chunk in ranked_chunks:
        key = _key(chunk)
        if key in selected_keys:
            continue
        selected.append(chunk)
        selected_keys.add(key)
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
    logger.info("RAG | query=%s", user_query)

    legal_domain = detect_legal_domain(user_query)
    topic = detect_topic(user_query)
    query_terms = _extract_query_terms(user_query)
    intents = detect_query_intents(user_query)
    cross_reference = is_cross_reference_query(intents)
    primary_intent = detect_primary_intent(intents, user_query)
    labor_code_query = _is_labor_code_query(user_query)
    special_category = detect_special_category(user_query)
    special_issue = _minor_special_issue(user_query) if special_category == "minor" else None
    query_profile = build_universal_query_profile(user_query)

    search_queries = build_search_queries(
        user_query,
        topic,
        legal_domain,
        intents,
    )

    logger.info(
        "RAG | query_input | %r",
        user_query,
    )
    logger.info(
        "RAG | classify | domain=%s | topic=%s | intents=%s | primary=%s | cross_reference=%s | labor_code=%s | special_category=%s | special_issue=%s | question_type=%s | subject=%s | event=%s | action=%s | state=%s | target_document=%s | target_article=%s",
        legal_domain,
        topic,
        intents,
        primary_intent,
        cross_reference,
        labor_code_query,
        special_category,
        special_issue,
        query_profile.get("question_type"),
        query_profile.get("subject"),
        query_profile.get("event"),
        query_profile.get("action"),
        query_profile.get("action_state"),
        query_profile.get("target_document"),
        query_profile.get("target_article"),
    )
    constraint = query_profile.get("constraint") or {}
    logger.info(
        "RAG | constraint | type=%s | value=%s | unit=%s | subject=%s | action=%s | scope=%s | scope_signals=%s",
        constraint.get("type"),
        constraint.get("value"),
        constraint.get("unit"),
        constraint.get("subject"),
        constraint.get("action"),
        constraint.get("scope"),
        constraint.get("scope_signals"),
    )
    logger.info(
        "RAG | search_queries | count=%s | queries=%s",
        len(search_queries),
        search_queries,
    )

    query_vectors = await asyncio.to_thread(
        get_query_embeddings,
        search_queries,
    )

    valid_pairs = [
        (search_query, vector)
        for search_query, vector in zip(search_queries, query_vectors)
        if vector
    ]

    valid_queries = [search_query for search_query, _ in valid_pairs]
    query_vectors = [vector for _, vector in valid_pairs]

    for vector in query_vectors:
        if len(vector) != 384:
            raise ValueError(f"Unexpected embedding dimension: {len(vector)}. Expected 384.")

    if not query_vectors:
        logger.warning("RAG | all embeddings are empty")
        return {
            "chunks": [],
            "retrieved_text": "",
            "found": False,
            "candidate_count": 0,
            "final_count": 0,
            "source_references": [],
            "legal_domain": legal_domain,
            "topic": topic,
            "intents": intents,
            "cross_reference": cross_reference,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    async def _run_search(
        vector: List[float],
        search_query: str,
        remove_domain_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            _search_chunks,
            supabase,
            vector,
            search_query,
            None if remove_domain_filter else legal_domain,
            None,
        )

    # Keep concurrent Supabase searches bounded. This lowers peak RAM while
    # preserving the same queries, search limits, scoring and final selection.
    clean_groups: List[List[Dict[str, Any]]] = []
    search_concurrency = max(1, int(os.getenv("RAG_SEARCH_CONCURRENCY", "2")))

    for batch_start in range(0, len(valid_queries), search_concurrency):
        batch_indices = range(
            batch_start,
            min(batch_start + search_concurrency, len(valid_queries)),
        )
        batch_results = await asyncio.gather(
            *[
                _run_search(
                    query_vectors[index],
                    valid_queries[index],
                    remove_domain_filter=(cross_reference and index == 0),
                )
                for index in batch_indices
            ],
            return_exceptions=True,
        )

        for index, result in zip(batch_indices, batch_results):
            if isinstance(result, Exception):
                logger.warning(
                    "RAG | search failed | query=%s | error=%s",
                    valid_queries[index],
                    result,
                )
                clean_groups.append([])
                continue

            clean_groups.append(result or [])

    query_roles = [
        "main" if index == 0 else "expanded"
        for index in range(len(clean_groups))
    ]

    candidate_chunks = _merge_search_results(
        clean_groups,
        query_roles,
    )

    targeted_chunks = await _get_targeted_chunks(
        supabase,
        topic,
        user_query,
    )

    if targeted_chunks:
        accident_worker_not_report_mode = (
            topic == "accident_investigation"
            and _accident_query_mode(user_query) == "worker_did_not_report"
        )

        if accident_worker_not_report_mode:
            for chunk in targeted_chunks:
                chunk["_accident_worker_not_report_targeted"] = True

        targeted_merged = _merge_search_results([targeted_chunks], ["targeted"])
        candidate_chunks = _merge_search_results(
            [candidate_chunks, targeted_merged],
            ["merged", "targeted"],
        )

    candidate_count = len(candidate_chunks)

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
            "intents": intents,
            "cross_reference": cross_reference,
            "domain_specific_count": 0,
            "topic_specific_count": 0,
        }

    domain_specific_count = sum(
        1 for chunk in candidate_chunks if chunk.get("legal_domain") == legal_domain
    )
    topic_specific_count = sum(
        1 for chunk in candidate_chunks if chunk.get("topic") == topic
    )

    for chunk in candidate_chunks:
        # Передаём исходный вопрос в topic-score, чтобы различать
        # подтипы внутри workplace_attestation.
        chunk["_user_query_for_scoring"] = user_query

        chunk["_combined_score"] = _legal_relevance_score(
            chunk,
            query_terms,
            topic,
            intents,
            cross_reference,
            primary_intent=primary_intent,
            labor_code_query=labor_code_query,
            user_query=user_query,
            special_category=special_category,
            special_issue=special_issue,
            query_profile=query_profile,
        )

    ranked_chunks = sorted(
        candidate_chunks,
        key=lambda chunk: chunk.get("_combined_score", 0.0),
        reverse=True,
    )

    for rank, chunk in enumerate(ranked_chunks[:10], start=1):
        logger.info(
            "RAG | candidate | rank=%s | score=%.4f | sim=%.4f | exact=%.4f | universal=%.4f | authority=%.4f | scope=%.4f | topic=%.4f | intent=%.4f | primary=%.4f | briefing=%.4f | special=%.4f | doc=%s | point=%s",
            rank,
            _safe_float(chunk.get("_combined_score")),
            _safe_float(chunk.get("_best_similarity", _semantic_score(chunk))),
            _safe_float(_exact_match_score(chunk, user_query, topic)),
            _safe_float(chunk.get("_universal_score")),
            _safe_float(chunk.get("_legal_authority_score")),
            _safe_float(chunk.get("_constraint_scope_score")),
            _safe_float(_topic_relevance_score(chunk, topic)),
            _safe_float(_intent_relevance_score(chunk, intents)),
            _safe_float(_primary_intent_relevance_score(chunk, primary_intent, topic)),
            _safe_float(chunk.get("_briefing_mode_bonus")),
            _safe_float(chunk.get("_special_category_bonus")),
            _get_document_name(chunk),
            _get_point_number(chunk),
        )

    final_limit = (
        max(RAG_FINAL_COUNT, 9)
        if topic == "height_work_training"
        else max(RAG_FINAL_COUNT, 7)
        if cross_reference
        else RAG_FINAL_COUNT
    )

    accident_mode = (
        _accident_query_mode(user_query)
        if topic == "accident_investigation"
        else None
    )

    final_chunks = _select_legal_diverse_chunks(
        ranked_chunks,
        final_limit,
        topic,
        intents,
        cross_reference,
        primary_intent=primary_intent,
        labor_code_query=labor_code_query,
        special_category=special_category,
        special_issue=special_issue,
        accident_mode=accident_mode,
        query_profile=query_profile,
    )

    logger.info(
        "RAG | final | count=%s | accident_mode=%s | special_category=%s | special_issue=%s | sources=%s",
        len(final_chunks),
        accident_mode,
        special_category,
        special_issue,
        [
            f"{_get_document_name(chunk)}#{_get_point_number(chunk)}"
            for chunk in final_chunks
        ],
    )

    source_references = []

    for index, chunk in enumerate(final_chunks, start=1):
        source_id = build_source_id(chunk, index)
        chunk["_source_id"] = source_id

        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = f"{document_name} — пункт/статья {point}"
        else:
            reference = document_name

        source_references.append(
            {
                "source_id": source_id,
                "reference": reference,
            }
        )

    retrieved_text = _build_retrieved_text(final_chunks)

    return {
        "chunks": final_chunks,
        "retrieved_text": retrieved_text,
        "found": bool(final_chunks),
        "candidate_count": candidate_count,
        "final_count": len(final_chunks),
        "source_references": source_references,
        "legal_domain": legal_domain,
        "topic": topic,
        "intents": intents,
        "query_profile": query_profile,
        "cross_reference": cross_reference,
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
        document_name = _get_document_name(chunk)
        point = _get_point_number(chunk)

        if point:
            reference = f"{document_name} — пункт/статья {point}"
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