"""Offline evaluation framework for the Belarus OHS/PB RAG."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_query_classifier import detect_topic, detect_legal_domain
from rag_query_generator import build_search_queries
from rag_query_profile import build_universal_query_profile
from rag_engine.query_planner import build_query_plan

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "evaluation_cases.json"


def load_cases() -> list[dict[str, Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases) or 1
    topic_ok = domain_ok = legal_ok = 0
    hierarchy_labeled = hierarchy_ok = subtopic_ok = legal_object_ok = 0
    knowledge_leak = briefing_leak = 0
    failures = []
    by_topic = defaultdict(
        lambda: {
            "total": 0,
            "topic_ok": 0,
            "domain_ok": 0,
            "legal_ok": 0,
            "hierarchy_labeled": 0,
            "hierarchy_ok": 0,
            "subtopic_ok": 0,
            "legal_object_ok": 0,
        }
    )

    for case in cases:
        q = case["question"]
        plan = build_query_plan(q)
        topic = plan["topic"]
        domain = plan["domain"]
        profile = plan["profile"]
        queries = plan["search_queries"]
        hierarchy = plan.get("topic_hierarchy") or {}
        predicted_subtopic = plan.get("subtopic") or hierarchy.get("subtopic")
        predicted_legal_object = plan.get("legal_object") or hierarchy.get("legal_object")
        query_text = "\n".join(queries).lower()

        acceptable = set(case.get("acceptable_topics") or [case["topic"]])
        topic_pass = topic in acceptable
        domain_pass = domain == case.get("expected_domain", "occupational_safety")
        required = case.get("required_terms", [])
        legal_pass = not required or any(term.lower() in query_text for term in required)

        hierarchy_expected = (
            "expected_subtopic" in case or "expected_legal_object" in case
        )
        subtopic_pass = (
            "expected_subtopic" not in case
            or predicted_subtopic == case["expected_subtopic"]
        )
        legal_object_pass = (
            "expected_legal_object" not in case
            or predicted_legal_object == case["expected_legal_object"]
        )
        hierarchy_pass = subtopic_pass and legal_object_pass

        bucket = by_topic[case["topic"]]
        bucket["total"] += 1
        bucket["topic_ok"] += int(topic_pass)
        bucket["domain_ok"] += int(domain_pass)
        bucket["legal_ok"] += int(legal_pass)
        bucket["hierarchy_labeled"] += int(hierarchy_expected)
        bucket["hierarchy_ok"] += int(hierarchy_expected and hierarchy_pass)
        bucket["subtopic_ok"] += int(hierarchy_expected and subtopic_pass)
        bucket["legal_object_ok"] += int(hierarchy_expected and legal_object_pass)

        topic_ok += int(topic_pass)
        domain_ok += int(domain_pass)
        legal_ok += int(legal_pass)
        hierarchy_labeled += int(hierarchy_expected)
        hierarchy_ok += int(hierarchy_expected and hierarchy_pass)
        subtopic_ok += int(hierarchy_expected and subtopic_pass)
        legal_object_ok += int(hierarchy_expected and legal_object_pass)

        if case["topic"] == "knowledge_testing" and "вводный инструктаж" in query_text:
            knowledge_leak += 1
        if case["topic"] == "occupational_briefing" and "проверка знаний" in query_text:
            briefing_leak += 1

        if not (topic_pass and domain_pass and legal_pass and hierarchy_pass):
            failures.append(
                {
                    "id": case["id"],
                    "question": q,
                    "expected_topic": case["topic"],
                    "predicted_topic": topic,
                    "expected_domain": case.get("expected_domain"),
                    "predicted_domain": domain,
                    "expected_subtopic": case.get("expected_subtopic"),
                    "predicted_subtopic": predicted_subtopic,
                    "expected_legal_object": case.get("expected_legal_object"),
                    "predicted_legal_object": predicted_legal_object,
                    "question_type": profile.get("question_type"),
                    "query_roles": plan.get("query_roles"),
                    "negative_concepts": plan.get("negative_concepts"),
                    "queries": queries,
                }
            )

    for bucket in by_topic.values():
        bucket["topic_accuracy"] = round(bucket["topic_ok"] / bucket["total"], 4)
        bucket["domain_accuracy"] = round(bucket["domain_ok"] / bucket["total"], 4)
        bucket["legal_query_coverage"] = round(bucket["legal_ok"] / bucket["total"], 4)
        if bucket["hierarchy_labeled"]:
            bucket["hierarchy_accuracy"] = round(
                bucket["hierarchy_ok"] / bucket["hierarchy_labeled"], 4
            )
            bucket["subtopic_accuracy"] = round(
                bucket["subtopic_ok"] / bucket["hierarchy_labeled"], 4
            )
            bucket["legal_object_accuracy"] = round(
                bucket["legal_object_ok"] / bucket["hierarchy_labeled"], 4
            )
        else:
            bucket["hierarchy_accuracy"] = None
            bucket["subtopic_accuracy"] = None
            bucket["legal_object_accuracy"] = None

    return {
        "total": len(cases),
        "topic_accuracy": round(topic_ok / total, 4),
        "domain_accuracy": round(domain_ok / total, 4),
        "legal_query_coverage": round(legal_ok / total, 4),
        "hierarchy_labeled_cases": hierarchy_labeled,
        "hierarchy_coverage": round(hierarchy_labeled / total, 4),
        "hierarchy_accuracy": (
            round(hierarchy_ok / hierarchy_labeled, 4)
            if hierarchy_labeled
            else None
        ),
        "subtopic_accuracy": (
            round(subtopic_ok / hierarchy_labeled, 4)
            if hierarchy_labeled
            else None
        ),
        "legal_object_accuracy": (
            round(legal_object_ok / hierarchy_labeled, 4)
            if hierarchy_labeled
            else None
        ),
        "by_topic": dict(sorted(by_topic.items())),
        "knowledge_testing_briefing_leak_rate": round(knowledge_leak / total, 4),
        "briefing_knowledge_testing_leak_rate": round(briefing_leak / total, 4),
        "failure_count": len(failures),
        "failures": failures,
    }


if __name__ == "__main__":
    result = evaluate_cases(load_cases())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if (
        result["knowledge_testing_briefing_leak_rate"]
        or result["briefing_knowledge_testing_leak_rate"]
    ):
        raise SystemExit(2)
