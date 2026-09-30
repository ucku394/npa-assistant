"""Photo findings -> Belarus NPA verification."""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from ai_router import generate_vision_legal_json
from prompts import VISUAL_LEGAL_VERIFICATION_PROMPT
from rag import build_source_id, retrieve_context

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Legal verification did not return JSON.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Legal verification must return an object.")
    return value


def _validated_basis(chunks: List[Dict[str, Any]], data: Dict[str, Any]):
    by_id = {}
    for index, chunk in enumerate(chunks, 1):
        source_id = chunk.get("_source_id") or build_source_id(chunk, index)
        if source_id:
            chunk["_source_id"] = source_id
            by_id[str(source_id)] = chunk

    result = []
    for item in data.get("legal_basis") or []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        chunk = by_id.get(source_id)
        if not chunk:
            continue
        result.append({
            "source_id": source_id,
            "document": str(
                item.get("document")
                or chunk.get("doc_name")
                or chunk.get("document")
                or "НПА"
            ).strip(),
            "point": str(
                item.get("point")
                or chunk.get("point_num")
                or ""
            ).strip(),
        })
    return result


async def verify_finding(
    finding: Dict[str, Any],
    supabase,
) -> Dict[str, Any]:
    raw_query = str(finding.get("description") or "").strip()
    if not raw_query:
        return {
            "status": "not_confirmed",
            "violation": "",
            "legal_basis": [],
            "verification_needed": ["Нет описания визуального наблюдения."],
            "risk_level": "medium",
        }

    # Do not force every visual finding into fire safety: PPE/tool/electrical
    # findings should retrieve occupational-safety requirements.
    finding_text = (raw_query + " " + str(finding.get("visual_evidence") or "")).lower()
    if any(marker in finding_text for marker in (
        "сварочн", "welding", "щиток", "helmet", "очки", "glasses",
        "перчат", "glove", "каск", "сиз", "ppe", "кабел", "cable",
        "электроинструмент", "grinder", "болгар", "ушм", "дрел", "drill",
        "пил", "tool", "инструмент", "рукоят", "кожух", "guard",
    )):
        domain_hint = "охрана труда"
    elif any(marker in finding_text for marker in (
        "огнетуш", "extinguisher", "искр", "sparks", "горюч", "combust",
        "пожар", "fire", "средства пожаротушения",
    )):
        domain_hint = "пожарная безопасность"
    else:
        domain_hint = "охрана труда и пожарная безопасность"
    query = f"нормативные требования Республики Беларусь, {domain_hint}: {raw_query}"

    rag = await retrieve_context(query, supabase)
    chunks = rag.get("chunks") or []
    context = rag.get("retrieved_text") or ""

    if not chunks or not context:
        return {
            "status": "potential",
            "violation": raw_query,
            "evidence": str(finding.get("visual_evidence") or ""),
            "legal_basis": [],
            "corrective_action": "",
            "verification_needed": [
                "Не найден достаточный нормативный контекст; требуется проверка специалистом."
            ],
            "risk_level": finding.get("risk_level", "medium"),
        }

    prompt = VISUAL_LEGAL_VERIFICATION_PROMPT.replace("{finding}", json.dumps(finding, ensure_ascii=False)).replace("{retrieved_text}", context)
    try:
        answer = await asyncio.to_thread(generate_vision_legal_json, prompt)
        data = _parse_json(answer)
    except Exception as exc:
        logger.warning(
            "INSPECTION | legal verification unavailable; preserving visual finding | error=%s",
            exc,
        )
        return {
            "status": "potential",
            "violation": raw_query,
            "evidence": str(finding.get("visual_evidence") or ""),
            "legal_basis": [],
            "corrective_action": "",
            "verification_needed": [
                "Проверить факт непосредственно на месте.",
                "Сопоставить факт с применимым НПА Республики Беларусь.",
            ],
            "risk_level": finding.get("risk_level", "medium"),
            "rag": {
                "candidate_count": rag.get("candidate_count", 0),
                "final_count": rag.get("final_count", 0),
                "legal_domain": rag.get("legal_domain"),
                "topic": rag.get("topic"),
            },
        }

    basis = _validated_basis(chunks, data)
    status = data.get("status")
    if status not in {"confirmed", "potential", "not_confirmed"}:
        status = "potential"
    if status == "confirmed" and not basis:
        status = "potential"

    data["status"] = status
    data["legal_basis"] = basis
    data["violation"] = str(data.get("violation") or raw_query).strip()
    data["evidence"] = str(
        data.get("evidence")
        or finding.get("visual_evidence")
        or ""
    ).strip()
    data["corrective_action"] = str(
        data.get("corrective_action") or ""
    ).strip()
    data["verification_needed"] = [
        str(x).strip()
        for x in (data.get("verification_needed") or [])
        if str(x).strip()
    ]
    if data.get("risk_level") not in {"low", "medium", "high", "critical"}:
        data["risk_level"] = finding.get("risk_level", "medium")
    data["rag"] = {
        "candidate_count": rag.get("candidate_count", 0),
        "final_count": rag.get("final_count", 0),
        "legal_domain": rag.get("legal_domain"),
        "topic": rag.get("topic"),
    }
    return data


async def verify_findings(
    findings: List[Dict[str, Any]],
    supabase,
) -> List[Dict[str, Any]]:
    results = []
    # Последовательно: бесплатные модели имеют более жёсткие rate limits.
    for finding in findings[:5]:
        try:
            results.append(await verify_finding(finding, supabase))
        except Exception as exc:
            logger.exception("INSPECTION | verification failed: %s", exc)
            results.append({
                "status": "potential",
                "violation": finding.get("description", ""),
                "evidence": finding.get("visual_evidence", ""),
                "legal_basis": [],
                "corrective_action": "",
                "verification_needed": [
                    "Автоматическая юридическая проверка не завершилась; "
                    "требуется проверка специалистом."
                ],
                "risk_level": finding.get("risk_level", "medium"),
            })
    return results
