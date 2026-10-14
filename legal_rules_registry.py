"""Declarative registry for legal-source gates and topic-specific RAG priorities.

Keep legal rule metadata here; orchestration and retrieval implementations stay in
ChatService and rag_engine.rag. Adding a rule should normally require registry data
and a regression test, not another copy of the gate algorithm.
"""

from typing import Any, Callable, Dict, List, Tuple


MANDATORY_SOURCE_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "article_49",
        "qualifiers": ("suspension_for_osh_violation",),
        "document_contains": "трудовой кодекс",
        "points": ("статья 49", "49"),
        "label": "ст. 49 ТК РБ",
    },
    {
        "id": "article_11",
        "qualifiers": ("refusal_due_to_no_ppe", "ppe_nonprovision_action"),
        "document_contains": "356-з",
        "points": ("статья 11", "11"),
        "label": "ст. 11 Закона № 356-З",
    },
    {
        "id": "instruction_list_compiler",
        "qualifiers": ("instruction_list_compiler",),
        "document_contains": "№ 176",
        "points": ("11", "пункт 11"),
        "label": "п. 11 постановления № 176 (перечень инструкций)",
    },
    {
        "id": "instruction_osh_service_role",
        "qualifiers": ("instruction_osh_service_role",),
        "document_contains": "№ 176",
        "points": ("11", "пункт 11"),
        "label": "п. 11 постановления № 176 (роль службы ОТ)",
    },
    {
        "id": "instruction_developer",
        "qualifiers": ("instruction_developer",),
        "document_contains": "№ 176",
        "points": ("14", "пункт 14"),
        "label": "п. 14 постановления № 176",
    },
)


# Topic-specific selection metadata. The implementation in rag.py is shared;
# only the source flag, point priority and selection limits vary by topic.
RAG_SELECTION_RULES: Dict[str, Dict[str, Any]] = {
    "lathe_work": {
        "target_flag": "_lathe_safety_targeted",
        "priority_points": ("84", "85", "40"),
        "max_items": 5,
        "max_per_document": 5,
        "log_label": "lathe safety",
    },
    "portable_ladder": {
        "target_flag": "_portable_ladder_targeted",
        "priority_points": ("54", "53"),
        "max_items": 3,
        "max_per_document": 4,
        "log_label": "portable ladder",
    },
}


def evaluate_mandatory_source_rules(
    query_profile: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    answer: str,
    extract_source_ids: Callable[[str], List[str]],
    build_source_id: Callable[[Dict[str, Any], int], str],
) -> Dict[str, Any]:
    """Evaluate all active mandatory-source rules against retrieved chunks and answer.

    The returned failure keys are stable diagnostic identifiers. A rule fails closed
    if its source was not retrieved or was retrieved but not cited in the answer.
    """
    qualifiers = set((query_profile or {}).get("qualifiers") or [])
    cited_ids = set(extract_source_ids(str(answer or "")))
    failures: Dict[str, bool] = {}
    details: Dict[str, Any] = {}
    active_rules: List[str] = []

    for rule in MANDATORY_SOURCE_RULES:
        if not qualifiers.intersection(rule["qualifiers"]):
            continue

        rule_id = rule["id"]
        active_rules.append(rule_id)
        expected_points = {
            str(point).strip().lower().rstrip(".")
            for point in rule["points"]
        }
        source_ids: List[str] = []

        for index, chunk in enumerate(chunks or [], start=1):
            document = str(chunk.get("doc_name") or "").strip().lower()
            point = str(chunk.get("point_num") or "").strip().lower().rstrip(".")
            if rule["document_contains"] in document and point in expected_points:
                source_id = chunk.get("_source_id") or build_source_id(chunk, index)
                if source_id:
                    source_ids.append(str(source_id))

        missing = not source_ids
        not_cited = bool(source_ids) and not bool(cited_ids.intersection(source_ids))
        failures[f"{rule_id}_missing"] = missing
        failures[f"{rule_id}_not_cited"] = not_cited
        details[f"{rule_id}_source_ids"] = source_ids
        details[f"{rule_id}_missing"] = missing
        details[f"{rule_id}_not_cited"] = not_cited

    return {
        "required": bool(active_rules),
        "active_rules": active_rules,
        "failures": failures,
        "details": details,
        "passed": not any(failures.values()),
    }
