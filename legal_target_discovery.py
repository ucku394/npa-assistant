"""Universal legal target discovery for the Belarus NPA RAG.

The module never invents article numbers or NPA names. It discovers targets only
from chunks that actually exist in the indexed npa_chunks table.
"""

from typing import Any, Dict, List
import logging
import re

logger = logging.getLogger(__name__)


def _text(chunk: Dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(key) or "")
        for key in ("doc_name", "point_num", "content", "topic", "legal_domain")
    ).lower()


def _point(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("point_num") or "").strip().rstrip(".")


def _score(chunk: Dict[str, Any], plan: Dict[str, Any]) -> float:
    text = _text(chunk)
    score = 0.0

    for phrase in plan.get("legal_phrases") or []:
        phrase = str(phrase).strip().lower()
        if phrase and phrase in text:
            score += 2.0

    for concept in plan.get("legal_concepts") or []:
        concept = str(concept).strip().lower()
        if concept and concept in text:
            score += 0.8

    domain = str(plan.get("domain") or "").strip().lower()
    if domain and str(chunk.get("legal_domain") or "").lower() == domain:
        score += 1.0

    target_doc = str(plan.get("target_document") or "").strip().lower()
    if target_doc and target_doc in text:
        score += 1.5

    target_point = str(plan.get("target_point") or "").strip().lower()
    if target_point and _point(chunk).lower() == target_point:
        score += 2.0

    # Нормативная формулировка обычно информативнее одиночного термина.
    for marker in (
        "обязан",
        "имеет право",
        "не допускается",
        "отстран",
        "не прошедш",
        "порядок",
        "проводится",
        "допускается",
    ):
        if marker in text:
            score += 0.15

    return score


def _build_or_filters(plan: Dict[str, Any]) -> str:
    values: List[str] = []
    phrases = list(plan.get("legal_phrases") or [])
    concepts = list(plan.get("legal_concepts") or [])
    queries = list(plan.get("search_queries") or [])

    # Keep the DB request bounded. Prefer normative phrases and concepts.
    terms = []
    for value in phrases + concepts + queries:
        value = str(value or "").strip()
        if value and value not in terms:
            terms.append(value)
        if len(terms) >= 8:
            break

    for value in terms:
        safe = value.replace("%", "").replace(",", " ")
        if safe:
            values.append(f"content.ilike.%{safe}%")
            values.append(f"doc_name.ilike.%{safe}%")

    return ",".join(values)


def discover_legal_targets(
    supabase,
    plan: Dict[str, Any],
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """Discover real NPA/article candidates from indexed chunks."""
    if not plan:
        return []

    filters = _build_or_filters(plan)
    if not filters:
        return []

    domain = str(plan.get("domain") or "").strip()
    # The planner is hard-coded to BY. Do not let it widen jurisdiction.
    if not domain:
        domain = "occupational_safety"

    try:
        response = (
            supabase.table("npa_chunks")
            .select(
                "doc_name,doc_type,point_num,content,legal_domain,topic,source_url"
            )
            .eq("legal_domain", domain)
            .or_(filters)
            .limit(max(10, min(int(limit), 80)))
            .execute()
        )
        rows = response.data or []
    except Exception as exc:
        logger.warning("LEGAL TARGET DISCOVERY | search failed: %s", exc)
        return []

    for row in rows:
        row["_legal_target_score"] = _score(row, plan)
        row["_legal_target_discovered"] = True

    rows.sort(
        key=lambda row: (
            float(row.get("_legal_target_score") or 0.0),
            len(str(row.get("content") or "")),
        ),
        reverse=True,
    )

    # Keep several points from the same NPA but avoid flooding the candidate set
    # with identical chunks.
    result: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("doc_name") or "").strip().lower(),
            _point(row).lower(),
            str(row.get("content") or "").strip()[:180].lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= 20:
            break

    logger.info(
        "LEGAL TARGET DISCOVERY | plan_intent=%s | action=%s | candidates=%s",
        plan.get("intent"),
        plan.get("action"),
        [
            f"{row.get('doc_name')}#{_point(row)}"
            for row in result[:10]
        ],
    )
    return result
