"""Run a cross-topic retrieval benchmark against the live Supabase corpus.

Usage:
    python evaluate_rag_retrieval.py
    python evaluate_rag_retrieval.py --dataset retrieval_eval_cases.json --output retrieval_eval_results.json

Required environment:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY

This script measures retrieval, not answer correctness. It does not modify the database.
"""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from rag import retrieve_context

ROOT = Path(__file__).resolve().parent


def expected_doc_match(doc_name, expected_values):
    value = str(doc_name or "").casefold()
    return any(str(expected).casefold() in value for expected in expected_values)


def chunk_doc_name(chunk):
    return str(
        chunk.get("doc_name")
        or chunk.get("document_name")
        or chunk.get("source_document")
        or ""
    )


async def evaluate_case(case, supabase):
    result = await retrieve_context(case["query"], supabase, request_id=f"eval-{case['id']}")
    expected = case["expected_document_contains"]

    search_docs = []
    for diagnostic in result.get("search_diagnostics", []):
        for group in ("hybrid_top", "semantic_top", "lexical_top", "top_results"):
            for item in diagnostic.get(group, []) or []:
                doc = item.get("document") or item.get("doc_name") or ""
                if doc:
                    search_docs.append(doc)

    final_docs = [chunk_doc_name(chunk) for chunk in result.get("chunks", [])]
    search_hit = any(expected_doc_match(doc, expected) for doc in search_docs)
    final_hit = any(expected_doc_match(doc, expected) for doc in final_docs)

    return {
        "id": case["id"],
        "query": case["query"],
        "expected_document_contains": expected,
        "expected_domain": case.get("domain"),
        "expected_topic": case.get("topic"),
        "actual_domain": result.get("legal_domain"),
        "actual_topic": result.get("topic"),
        "candidate_count": result.get("candidate_count", 0),
        "final_count": result.get("final_count", len(result.get("chunks", []))),
        "expected_document_in_search_results": search_hit,
        "expected_document_in_final_context": final_hit,
        "failure_reason": result.get("failure_reason"),
        "search_modes": sorted({
            str(item.get("mode") or "unknown")
            for item in result.get("search_diagnostics", [])
        }),
        "final_sources": [
            {
                "document": chunk_doc_name(chunk),
                "point": chunk.get("point_num") or chunk.get("point_path"),
            }
            for chunk in result.get("chunks", [])
        ],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "retrieval_eval_cases.json"))
    parser.add_argument("--output", default=str(ROOT / "retrieval_eval_results.json"))
    args = parser.parse_args()

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment or .env file.")

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    supabase = create_client(url, key)
    results = []
    for case in dataset["cases"]:
        try:
            results.append(await evaluate_case(case, supabase))
        except Exception as exc:
            results.append({
                "id": case.get("id"),
                "query": case.get("query"),
                "error": f"{type(exc).__name__}: {exc}",
                "expected_document_in_search_results": False,
                "expected_document_in_final_context": False,
            })

    total = len(results)
    searched = sum(bool(row.get("expected_document_in_search_results")) for row in results)
    final = sum(bool(row.get("expected_document_in_final_context")) for row in results)
    domain_correct = sum(
        row.get("actual_domain") == row.get("expected_domain")
        for row in results if row.get("expected_domain")
    )
    topic_checked = [row for row in results if row.get("expected_topic")]
    topic_correct = sum(row.get("actual_topic") == row.get("expected_topic") for row in topic_checked)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.get("version"),
        "jurisdiction": dataset.get("jurisdiction"),
        "metrics": {
            "cases": total,
            "expected_document_search_hit_rate": round(searched / total, 4) if total else 0,
            "expected_document_final_context_hit_rate": round(final / total, 4) if total else 0,
            "domain_accuracy": round(domain_correct / total, 4) if total else 0,
            "topic_accuracy_on_labeled_cases": round(topic_correct / len(topic_checked), 4) if topic_checked else None,
        },
        "cases": results,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
