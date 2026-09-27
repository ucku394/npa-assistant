"""Offline evaluation framework for the Belarus OHS/PB RAG."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from rag_query_classifier import detect_topic, detect_legal_domain
from rag_query_generator import build_search_queries
from rag_query_profile import build_universal_query_profile

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "evaluation_cases.json"

def load_cases() -> list[dict[str, Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))

def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases) or 1
    topic_ok = domain_ok = legal_ok = 0
    knowledge_leak = briefing_leak = 0
    failures = []
    by_topic = defaultdict(lambda: {"total": 0, "topic_ok": 0, "domain_ok": 0, "legal_ok": 0})
    for case in cases:
        q = case["question"]
        topic = detect_topic(q)\n        domain = detect_legal_domain(q)
        profile = build_universal_query_profile(q)
        queries = build_search_queries(q, topic, "occupational_safety", [])
        query_text = "\n".join(queries).lower()
        acceptable = set(case.get("acceptable_topics") or [case["topic"]])
        topic_pass = topic in acceptable\n        domain_pass = domain == case.get("expected_domain", "occupational_safety")
        required = case.get("required_terms", [])
        legal_pass = not required or any(term.lower() in query_text for term in required)
        bucket = by_topic[case["topic"]]
        bucket["total"] += 1
        bucket["topic_ok"] += int(topic_pass)\n        bucket["domain_ok"] += int(domain_pass)
        bucket["legal_ok"] += int(legal_pass)
        topic_ok += int(topic_pass)\n        domain_ok += int(domain_pass)
        legal_ok += int(legal_pass)
        if case["topic"] == "knowledge_testing" and "вводный инструктаж" in query_text:
            knowledge_leak += 1
        if case["topic"] == "occupational_briefing" and "проверка знаний" in query_text:
            briefing_leak += 1
        if not (topic_pass and domain_pass and legal_pass):
            failures.append({
                "id": case["id"], "question": q,
                "expected_topic": case["topic"], "predicted_topic": topic,\n                "expected_domain": case.get("expected_domain"), "predicted_domain": domain,
                "question_type": profile.get("question_type"), "queries": queries,
            })
    for bucket in by_topic.values():
        bucket["topic_accuracy"] = round(bucket["topic_ok"] / bucket["total"], 4)\n        bucket["domain_accuracy"] = round(bucket["domain_ok"] / bucket["total"], 4)
        bucket["legal_query_coverage"] = round(bucket["legal_ok"] / bucket["total"], 4)
    return {
        "total": len(cases),
        "topic_accuracy": round(topic_ok / total, 4),\n        "domain_accuracy": round(domain_ok / total, 4),
        "legal_query_coverage": round(legal_ok / total, 4),
        "by_topic": dict(sorted(by_topic.items())),
        "knowledge_testing_briefing_leak_rate": round(knowledge_leak / total, 4),
        "briefing_knowledge_testing_leak_rate": round(briefing_leak / total, 4),
        "failure_count": len(failures),
        "failures": failures,
    }

if __name__ == "__main__":
    result = evaluate_cases(load_cases())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["knowledge_testing_briefing_leak_rate"] or result["briefing_knowledge_testing_leak_rate"]:
        raise SystemExit(2)