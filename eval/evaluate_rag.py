"""Offline evaluation framework for the Belarus OHS/PB RAG."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from rag_query_classifier import detect_topic
from rag_query_generator import build_search_queries
from rag_query_profile import build_universal_query_profile
ROOT=Path(__file__).resolve().parent
DATASET=ROOT/"evaluation_cases.json"
def load_cases()->list[dict[str,Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))
def evaluate_cases(cases:list[dict[str,Any]])->dict[str,Any]:
    total=len(cases) or 1; topic_ok=legal_ok=0; knowledge_leak=briefing_leak=0; failures=[]
    for case in cases:
        q=case["question"]; topic=detect_topic(q); profile=build_universal_query_profile(q)
        queries=build_search_queries(q,topic,"occupational_safety",[]); text="\n".join(queries).lower()
        topic_pass=topic in set(case.get("acceptable_topics") or [case["topic"]])
        required=case.get("required_terms",[]); legal_pass=not required or any(x.lower() in text for x in required)
        if case["topic"]=="knowledge_testing" and "вводный инструктаж" in text: knowledge_leak+=1
        if case["topic"]=="occupational_briefing" and "проверка знаний" in text: briefing_leak+=1
        topic_ok+=int(topic_pass); legal_ok+=int(legal_pass)
        if not (topic_pass and legal_pass): failures.append({"id":case["id"],"question":q,"expected_topic":case["topic"],"predicted_topic":topic,"question_type":profile.get("question_type"),"queries":queries})
    return {"total":len(cases),"topic_accuracy":round(topic_ok/total,4),"legal_query_coverage":round(legal_ok/total,4),"knowledge_testing_briefing_leak_rate":round(knowledge_leak/total,4),"briefing_knowledge_testing_leak_rate":round(briefing_leak/total,4),"failures":failures}
if __name__=="__main__":
    result=evaluate_cases(load_cases());print(json.dumps(result,ensure_ascii=False,indent=2))
    if result["knowledge_testing_briefing_leak_rate"] or result["briefing_knowledge_testing_leak_rate"]: raise SystemExit(2)
