"""Legal relevance and evidence gating for Belarus OHS/industrial-safety RAG.

This module deliberately separates semantic similarity from legal relevance.
A chunk can be semantically similar while belonging to the wrong legal subject
(e.g. "repeat knowledge testing" vs. "repeated administrative offence").
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


_TOPIC_POLICIES: Dict[str, Dict[str, List[str]]] = {
    "knowledge_testing": {
        "required_any": [
            "проверка знаний",
            "проверки знаний",
            "проверке знаний",
            "проверку знаний",
        ],
        "required_context_any": [
            "охране труда",
            "требований охраны труда",
            "требования охраны труда",
            "безопасности труда",
        ],
        "forbidden": [
            "административное правонарушение",
            "административного правонарушения",
            "повторность правонарушения",
            "повторное правонарушение",
            "административная ответственность",
            "состав административного правонарушения",
        ],
    },
    "occupational_briefing": {
        "required_any": ["инструктаж", "инструктажей", "инструктажа"],
        "required_context_any": [
            "охране труда",
            "безопасности труда",
        ],
        "forbidden": [
            "административное правонарушение",
            "медицинский осмотр",
        ],
    },
    "medical_examinations": {
        "required_any": ["медицинский осмотр", "медосмотр", "медицинские осмотры"],
        "required_context_any": [
            "работник",
            "работающих",
            "наниматель",
            "охране труда",
        ],
        "forbidden": [
            "медицинская помощь",
            "медицинская тайна",
            "оказание медицинской помощи",
        ],
    },
    "workplace_attestation": {
        "required_any": [
            "аттестация рабочих мест",
            "аттестации рабочих мест",
            "условия труда",
        ],
        "required_context_any": [
            "рабочих мест",
            "условий труда",
            "аттестации",
        ],
        "forbidden": [
            "аттестация персонала",
            "аттестация работников",
            "квалификационная аттестация",
        ],
    },
    "ppe_nonprovision": {
        "required_any": [
            "средств индивидуальной защиты",
            "средства индивидуальной защиты",
            "сиз",
        ],
        "required_context_any": [
            "работник",
            "наниматель",
            "охране труда",
            "выдач",
        ],
        "forbidden": [
            "информационная безопасность",
            "информационной безопасности",
            "защита информации",
        ],
    },
    "accident_investigation": {
        "required_any": [
            "несчастный случай",
            "несчастного случая",
            "расследование несчастного случая",
        ],
        "required_context_any": [
            "производстве",
            "расследован",
            "пострадавш",
            "акт н-1",
        ],
        "forbidden": [
            "административное правонарушение",
            "уголовное дело",
        ],
    },
}


def _text(chunk: Dict[str, Any]) -> str:
    values = [
        chunk.get("doc_name"),
        chunk.get("document"),
        chunk.get("document_name"),
        chunk.get("title"),
        chunk.get("content"),
        chunk.get("text"),
        chunk.get("point_num"),
    ]
    return " ".join(str(v) for v in values if v).lower()


def _query_text(query_profile: Optional[Dict[str, Any]]) -> str:
    if not query_profile:
        return ""
    values: List[str] = []
    for key in ("legal_phrases", "qualifiers"):
        value = query_profile.get(key) or []
        if isinstance(value, str):
            values.append(value)
        else:
            values.extend(str(item) for item in value)
    return " ".join(values).lower()


def _count_hits(text: str, markers: Iterable[str]) -> int:
    return sum(1 for marker in markers if marker.lower() in text)


def legal_relevance_adjustment(
    chunk: Dict[str, Any],
    topic: str,
    query_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return additive score and diagnostics for legal-domain relevance.

    The adjustment is intentionally conservative: it strongly penalizes an
    explicit wrong legal subject, while requiring only a modest topical match.
    """

    policy = _TOPIC_POLICIES.get(topic)
    if not policy:
        return {
            "score": 0.0,
            "required_hits": 0,
            "context_hits": 0,
            "forbidden_hits": 0,
            "legal_match": 0.0,
            "hard_negative": False,
        }

    text = _text(chunk)
    required_hits = _count_hits(text, policy["required_any"])
    context_hits = _count_hits(text, policy["required_context_any"])
    forbidden_hits = _count_hits(text, policy["forbidden"])

    legal_match = 0.0
    if required_hits:
        legal_match += 0.55
    if context_hits:
        legal_match += min(context_hits * 0.12, 0.30)

    # Explicit wrong-domain legal terminology is a hard negative. The penalty
    # is large enough to defeat superficial semantic similarity.
    hard_negative = forbidden_hits > 0
    penalty = min(forbidden_hits * 0.85, 1.70)

    # If a topic is known but its defining legal vocabulary is absent, apply a
    # smaller penalty rather than deleting the chunk. This preserves recall.
    weak_topic_penalty = 0.18 if required_hits == 0 else 0.0

    score = legal_match - penalty - weak_topic_penalty

    return {
        "score": score,
        "required_hits": required_hits,
        "context_hits": context_hits,
        "forbidden_hits": forbidden_hits,
        "legal_match": min(legal_match, 1.0),
        "hard_negative": hard_negative,
    }


def evidence_gate(
    ranked_chunks: List[Dict[str, Any]],
    topic: str,
    query_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide whether the ranked context contains usable legal evidence.

    This is not an answer-quality judge. It only checks whether retrieval found
    at least one chunk that belongs to the requested legal subject.
    """

    if not ranked_chunks:
        return {
            "sufficient": False,
            "reason": "no_candidates",
            "best_score": 0.0,
            "legal_match": 0.0,
        }

    policy = _TOPIC_POLICIES.get(topic)
    if not policy:
        best = ranked_chunks[0]
        return {
            "sufficient": float(best.get("_combined_score", 0.0)) >= 0.55,
            "reason": "generic_threshold",
            "best_score": float(best.get("_combined_score", 0.0)),
            "legal_match": float(best.get("_legal_match", 0.0)),
        }

    for chunk in ranked_chunks[:10]:
        if chunk.get("_legal_hard_negative"):
            continue
        if int(chunk.get("_legal_required_hits", 0)) <= 0:
            continue

        # Topic-specific evidence is accepted when the defining legal concept
        # is present and the candidate has not been classified as wrong-domain.
        if float(chunk.get("_legal_match", 0.0)) >= 0.55:
            return {
                "sufficient": True,
                "reason": "topic_evidence",
                "best_score": float(chunk.get("_combined_score", 0.0)),
                "legal_match": float(chunk.get("_legal_match", 0.0)),
            }

    return {
        "sufficient": False,
        "reason": "no_topic_evidence",
        "best_score": float(ranked_chunks[0].get("_combined_score", 0.0)),
        "legal_match": float(ranked_chunks[0].get("_legal_match", 0.0)),
    }


def build_second_pass_queries(
    original_query: str,
    topic: str,
    query_profile: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate deterministic lexical/legal queries for a second retrieval pass."""

    q = str(original_query or "").strip()
    if not q:
        return []

    phrases = list((query_profile or {}).get("legal_phrases") or [])
    queries: List[str] = [q]

    for phrase in phrases[:4]:
        phrase = str(phrase).strip()
        if phrase and phrase.lower() not in q.lower():
            queries.append(phrase)

    topic_queries = {
        "knowledge_testing": [
            "проверка знаний требований охраны труда порядок проведения",
            "периодичность проверки знаний требований охраны труда",
            "повторная проверка знаний требований охраны труда сроки порядок",
        ],
        "occupational_briefing": [
            "инструктаж по охране труда порядок проведения",
            "виды инструктажей по охране труда",
        ],
        "medical_examinations": [
            "обязательные медицинские осмотры работников порядок периодичность",
        ],
        "workplace_attestation": [
            "аттестация рабочих мест по условиям труда порядок сроки",
        ],
        "ppe_nonprovision": [
            "обеспечение работников средствами индивидуальной защиты порядок",
        ],
        "accident_investigation": [
            "расследование несчастного случая на производстве порядок",
        ],
    }

    queries.extend(topic_queries.get(topic, []))
    return list(dict.fromkeys(queries))


def legal_policy(topic: str) -> Dict[str, List[str]]:
    """Expose a copy for tests/evaluation without allowing mutation."""
    policy = _TOPIC_POLICIES.get(topic, {})
    return {key: list(value) for key, value in policy.items()}
