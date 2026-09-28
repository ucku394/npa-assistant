"""Query Planner 2.0 for the Belarus OHS/PB legal RAG.

The planner is deliberately thin: existing classifiers/profiles/query generators remain
the source of truth for individual signals. This module composes those signals into
one explicit retrieval plan so rag.py does not have to know how a query was built.

Pipeline:
    question -> understanding -> query plan -> hybrid retrieval -> rerank -> evidence gate
"""

from __future__ import annotations

from typing import Any, Dict, List

from rag_query_classifier import (
    detect_legal_domain,
    detect_topic,
    detect_special_category,
    _minor_special_issue,
    detect_query_intents,
    detect_primary_intent,
    _is_labor_code_query,
    is_cross_reference_query,
)
from rag_query_profile import build_universal_query_profile
from rag_query_generator import build_search_queries
from rag_engine.legal_relevance import legal_policy
from rag_engine.topic_planner import build_topic_hierarchy, hierarchy_preferred_terms


def _unique(values: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def _build_negative_concepts(
    topic: str,
    profile: Dict[str, Any],
) -> List[str]:
    policy = legal_policy(topic)
    negatives = list(policy.get("forbidden") or [])

    qualifiers = set(profile.get("qualifiers") or [])
    if topic == "knowledge_testing":
        # Semantic similarity around "повторная" can pull КоАП material about
        # repeated offences. Make this exclusion explicit in the plan.
        negatives.extend([
            "административное правонарушение",
            "административная ответственность",
            "повторное правонарушение",
        ])

    if topic == "accident_investigation" and "special_investigation" in qualifiers:
        negatives.extend([
            "обычное расследование несчастного случая",
            "общий порядок расследования",
            "акт н-1 без специального расследования",
            "административное правонарушение",
        ])

    if topic == "occupational_briefing":
        if profile.get("question_type") == "kind":
            negatives.extend([
                "кто проводит вводный инструктаж",
                "специалист по охране труда проводит",
            ])
        elif profile.get("question_type") == "who":
            negatives.extend([
                "виды инструктажей",
                "целевой инструктаж разовые работы",
            ])

    if "refusal_due_to_no_ppe" in qualifiers:
        negatives.extend([
            "выдача сиз сама по себе",
            "срок носки сиз без вопроса о праве отказаться",
        ])

    return _unique(negatives)


def _build_source_constraints(
    domain: str,
    topic: str,
    profile: Dict[str, Any],
    labor_code_query: bool,
) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {
        "legal_domain": domain,
        "topic": topic,
        "preferred_documents": [],
        "preferred_terms": [],
        "forbidden_terms": list(legal_policy(topic).get("forbidden") or []),
    }

    if labor_code_query:
        constraints["preferred_documents"].append("Трудовой кодекс Республики Беларусь")

    preferred_documents = {
        "knowledge_testing": [
            "проверка знаний",
            "обучение по охране труда",
            "охрана труда",
        ],
        "occupational_briefing": [
            "Инструкция № 175",
            "инструктаж по охране труда",
        ],
        "ppe_nonprovision": [
            "СИЗ",
            "№ 209",
            "средства индивидуальной защиты",
        ],
        "medical_examinations": [
            "медицинские осмотры",
            "№ 74",
        ],
        "workplace_attestation": [
            "№ 253",
            "аттестация рабочих мест",
        ],
        "accident_investigation": [
            "расследование несчастных случаев",
            "№ 30",
        ],
    }
    constraints["preferred_documents"].extend(
        preferred_documents.get(topic, [])
    )

    constraints["preferred_terms"] = list(profile.get("legal_phrases") or [])
    constraints["preferred_documents"] = _unique(
        constraints["preferred_documents"]
    )
    constraints["preferred_terms"] = _unique(constraints["preferred_terms"])
    constraints["forbidden_terms"] = _unique(constraints["forbidden_terms"])
    return constraints


def _assign_query_role(
    query: str,
    index: int,
    original: str,
    profile: Dict[str, Any],
) -> str:
    if index == 0 and query.strip().lower() == original.strip().lower():
        return "exact"

    legal_phrases = {
        str(value).strip().lower()
        for value in (profile.get("legal_phrases") or [])
        if str(value).strip()
    }
    if query.strip().lower() in legal_phrases:
        return "legal"

    query_lower = query.lower()
    if any(
        marker in query_lower
        for marker in (
            "трудовой кодекс",
            "статья ",
            "пункт ",
            "постановления ",
            "инструкция №",
            "приказ ",
        )
    ):
        return "document"

    return "semantic"



def _query_type_prefixes(question_type: str, preferred_term: str) -> List[str]:
    """Build deterministic, intent-preserving lexical queries."""
    if not preferred_term:
        return []
    prefixes = {
        "kind": [
            f"виды {preferred_term}",
            f"какой вид {preferred_term}",
            f"какой {preferred_term}",
        ],
        "who": [
            f"кто проводит {preferred_term}",
            f"кто отвечает за {preferred_term}",
        ],
        "frequency": [
            f"периодичность {preferred_term}",
            f"как часто проводится {preferred_term}",
            f"сроки проведения {preferred_term}",
        ],
        "what_to_do": [
            f"порядок действий при {preferred_term}",
            f"что делать при {preferred_term}",
        ],
        "whether": [
            f"имеет ли право {preferred_term}",
            f"допускается ли {preferred_term}",
        ],
        "limit": [
            f"предельные нормы {preferred_term}",
            f"нормы {preferred_term}",
        ],
    }
    return prefixes.get(question_type, [preferred_term])


def _refine_search_queries(original: str, generated: List[str], hierarchy: Dict[str, Any], profile: Dict[str, Any], negatives: List[str]) -> List[str]:
    """Prioritize exact/hierarchy/legal queries without changing question intent."""
    preferred = hierarchy_preferred_terms(hierarchy)
    preferred_term = preferred[0] if preferred else ""
    qtype = str(profile.get("question_type") or "general")
    result = [original] if original else []
    result.extend(_query_type_prefixes(qtype, preferred_term))
    for phrase in profile.get("legal_phrases") or []:
        phrase = str(phrase).strip()
        if phrase and phrase.lower() != preferred_term.lower():
            result.append(phrase)
    result.extend(generated)
    negative_lower = [str(x).strip().lower() for x in negatives if str(x).strip()]
    cleaned = []
    seen = set()
    for index, query in enumerate(result):
        query = str(query or "").strip()
        key = query.lower()
        if not query or key in seen or (
            index != 0 and any(n in key for n in negative_lower)
        ):
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned[:8]


def _build_query_slots(original, generated, hierarchy, profile, source_constraints):
    preferred = hierarchy_preferred_terms(hierarchy)
    preferred_term = preferred[0] if preferred else ""
    qtype = str(profile.get("question_type") or "general")
    legal = [str(v).strip() for v in (profile.get("legal_phrases") or []) if str(v).strip()]

    # object = предмет регулирования, а не готовая нормативная фраза.
    # Для проверки знаний электротехнического персонала это именно
    # «электротехнический персонал», тогда как «проверка знаний ...»
    # относится к legal/intent слотам.
    object_term = preferred_term
    object_markers = (
        "проверка знаний ",
        "периодичность ",
        "инструктаж ",
        "аттестация ",
    )
    for marker in object_markers:
        if object_term.lower().startswith(marker):
            object_term = object_term[len(marker):].strip()
            break
    docs = [str(v).strip() for v in (source_constraints.get("preferred_documents") or []) if str(v).strip()]
    slots = []
    if original: slots.append({"role":"exact","query":original})
    if object_term: slots.append({"role":"object","query":object_term})
    if preferred_term:
        if qtype == "who" and preferred_term.startswith("проверка "):
            intent = "кто проводит проверку " + preferred_term[len("проверка "):]
        else:
            intent = (_query_type_prefixes(qtype, preferred_term) or [preferred_term])[0]
        slots.append({"role":"intent","query":intent})
    if legal: slots.append({"role":"legal","query":legal[0]})
    if docs: slots.append({"role":"document","query":" ".join(_unique(docs[:2]))})
    parts=[str(profile.get("event") or "").strip(),str(profile.get("action_state") or "").strip()]
    parts=[p for p in parts if p]
    if parts and preferred_term: slots.append({"role":"condition","query":" ".join(_unique(parts+[preferred_term]))})
    synonym = " ".join(preferred[:2]) if len(preferred)>1 else (legal[1] if len(legal)>1 else "")
    if synonym and synonym.lower()!=preferred_term.lower(): slots.append({"role":"synonym","query":synonym})
    for q in generated:
        q=str(q or "").strip()
        if q: slots.append({"role":"recovery","query":q}); break
    return slots


def _select_query_slots(original, generated, hierarchy, profile, negatives, source_constraints):
    candidates=_build_query_slots(original,generated,hierarchy,profile,source_constraints)+[{"role":"generated","query":str(q or "").strip()} for q in generated]
    negatives=[str(v).strip().lower() for v in negatives if str(v).strip()]
    queries=[]; roles=[]; seen=set()
    for item in candidates:
        q=str(item.get("query") or "").strip(); role=str(item.get("role") or "generated"); key=q.lower()
        if not q or key in seen: continue
        if role!="exact" and any(n in key for n in negatives): continue
        seen.add(key); queries.append(q); roles.append(role)
        if len(queries)>=8: break
    return queries,roles

def build_query_plan(user_query: str) -> Dict[str, Any]:
    """Build a deterministic retrieval plan from one user question."""

    original = str(user_query or "").strip()
    domain = detect_legal_domain(original)
    topic = detect_topic(original)
    intents = detect_query_intents(original)
    primary_intent = detect_primary_intent(intents, original)
    cross_reference = is_cross_reference_query(intents)
    labor_code_query = _is_labor_code_query(original)
    special_category = detect_special_category(original)
    special_issue = (
        _minor_special_issue(original)
        if special_category == "minor"
        else None
    )
    profile = build_universal_query_profile(original)
    hierarchy = build_topic_hierarchy(original, topic)

    if topic == "general" and hierarchy.get("topic") != "general":
        topic = hierarchy["topic"]
        domain = hierarchy["domain"]

    generated_search_queries = build_search_queries(
        original,
        topic,
        domain,
        intents,
    )

    negative_concepts = _build_negative_concepts(topic, profile)
    source_constraints = _build_source_constraints(
        domain,
        topic,
        profile,
        labor_code_query,
    )
    search_queries, query_roles = _select_query_slots(
        original, generated_search_queries, hierarchy, profile, negative_concepts, source_constraints
    )

    source_constraints["preferred_terms"] = _unique(
        (source_constraints.get("preferred_terms") or [])
        + hierarchy_preferred_terms(hierarchy)
    )

    return {
        "original": original,
        "domain": domain,
        "topic": topic,
        "subtopic": hierarchy.get("subtopic"),
        "legal_object": hierarchy.get("legal_object"),
        "topic_confidence": hierarchy.get("confidence"),
        "topic_signals": hierarchy.get("signals") or [],
        "topic_hierarchy": hierarchy,
        "intents": intents,
        "primary_intent": primary_intent,
        "question_type": profile.get("question_type"),
        "subject": profile.get("subject"),
        "event": profile.get("event"),
        "action": profile.get("action"),
        "action_state": profile.get("action_state"),
        "profile": profile,
        "search_queries": search_queries,
        "query_roles": query_roles,
        "query_slots": [{"role": r, "query": q} for r, q in zip(query_roles, search_queries)],
        "negative_concepts": negative_concepts,
        "source_constraints": source_constraints,
        "special_category": special_category,
        "special_issue": special_issue,
        "cross_reference": cross_reference,
        "labor_code_query": labor_code_query,
    }


def plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return only safe, compact diagnostics for production logs."""

    return {
        "domain": plan.get("domain"),
        "topic": plan.get("topic"),
        "subtopic": plan.get("subtopic"),
        "legal_object": plan.get("legal_object"),
        "topic_confidence": plan.get("topic_confidence"),
        "question_type": plan.get("question_type"),
        "event": plan.get("event"),
        "action": plan.get("action"),
        "action_state": plan.get("action_state"),
        "primary_intent": plan.get("primary_intent"),
        "query_count": len(plan.get("search_queries") or []),
        "query_roles": plan.get("query_roles") or [],
        "query_slots": plan.get("query_slots") or [],
        "negative_count": len(plan.get("negative_concepts") or []),
        "preferred_documents": (
            plan.get("source_constraints", {}).get("preferred_documents") or []
        ),
    }
