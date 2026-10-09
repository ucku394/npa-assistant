"""RAG quality evaluation harness.

Usage:
  python evaluation/rag_eval.py evaluation/questions.json

The dataset is intentionally extensible to 200+ cases. Each case can define
expected_topic, expected_documents and expected_points. The evaluator reports:
Recall@K (topic/document/point), grounding rate, and average evidence count.
"""
import asyncio
import json
import sys
from pathlib import Path

from core.chat_service import ChatService


async def run(path: str) -> None:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    service = ChatService()
    rows = []

    for case in cases:
        result = await service.process_text(case["question"])
        rag = result.get("rag") or {}
        chunks = result.get("evidence_map") or []
        docs = {str(x.get("document", "")).lower() for x in chunks}
        points = {str(x.get("point", "")).strip() for x in chunks if x.get("point")}
        expected_docs = {str(x).lower() for x in case.get("expected_documents", [])}
        expected_points = {str(x).strip() for x in case.get("expected_points", [])}

        topic_hit = not case.get("expected_topic") or rag.get("topic") == case["expected_topic"]
        doc_hit = not expected_docs or any(any(exp in doc for exp in expected_docs) for doc in docs)
        point_hit = not expected_points or bool(points & expected_points)

        # Missing verification metadata must never be treated as a successful
        # grounding check. A case is grounded only when the request succeeded
        # and both Evidence Gate stages explicitly returned passed=True.
        grounding = rag.get("grounding")
        claim_evidence = rag.get("claim_evidence")
        grounding_known = isinstance(grounding, dict) and "passed" in grounding
        claim_evidence_known = isinstance(claim_evidence, dict) and "passed" in claim_evidence
        grounded = bool(
            result.get("success") is True
            and grounding_known
            and grounding.get("passed") is True
            and claim_evidence_known
            and claim_evidence.get("passed") is True
        )

        rows.append({
            "id": case["id"],
            "topic_hit": topic_hit,
            "document_hit": doc_hit,
            "point_hit": point_hit,
            "grounded": grounded,
            "grounding_known": grounding_known,
            "claim_evidence_known": claim_evidence_known,
            "error": result.get("error"),
            "request_id": result.get("request_id"),
            "candidate_count": rag.get("candidate_count", 0),
            "final_count": rag.get("final_count", 0),
            "evidence_count": len(chunks),
        })

    n = max(len(rows), 1)
    print(json.dumps({
        "cases": len(rows),
        "topic_recall": round(sum(x["topic_hit"] for x in rows) / n, 3),
        "document_recall": round(sum(x["document_hit"] for x in rows) / n, 3),
        "point_recall": round(sum(x["point_hit"] for x in rows) / n, 3),
        "grounding_rate": round(sum(x["grounded"] for x in rows) / n, 3),
        "grounding_metadata_coverage": round(
            sum(x["grounding_known"] and x["claim_evidence_known"] for x in rows) / n, 3
        ),
        "avg_candidates": round(sum(x["candidate_count"] for x in rows) / n, 2),
        "avg_final": round(sum(x["final_count"] for x in rows) / n, 2),
        "avg_evidence": round(sum(x["evidence_count"] for x in rows) / n, 2),
        "failed_cases": [
            {
                "id": x["id"],
                "topic_hit": x["topic_hit"],
                "document_hit": x["document_hit"],
                "point_hit": x["point_hit"],
                "grounded": x["grounded"],
                "error": x["error"],
                "request_id": x["request_id"],
            }
            for x in rows
            if not x["grounded"] or not x["topic_hit"] or not x["document_hit"] or not x["point_hit"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluation/rag_eval.py evaluation/questions.json")
    asyncio.run(run(sys.argv[1]))
