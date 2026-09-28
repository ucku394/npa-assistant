from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import retrieve_context
from .dataset import CASES
from .metrics import mean, precision_at_k, recall_at_k, reciprocal_rank
from .schema import (
    EvalCase,
    chunk_document_matches,
    chunk_is_forbidden,
    chunk_matches,
)


def load_jsonl(path: str) -> list[EvalCase]:
    cases = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"{path}:{line_no}: question is required")
        cases.append(EvalCase(
            id=raw.get("id") or f"prod-{line_no:04d}",
            question=question,
            expected_documents=raw.get("expected_documents", []),
            expected_points=raw.get("expected_points", []),
            forbidden_documents=raw.get("forbidden_documents", []),
            forbidden_phrases=raw.get("forbidden_phrases", []),
            tags=raw.get("tags", []),
        ))
    return cases


async def run_case(case: EvalCase, supabase):
    result = await retrieve_context(case.question, supabase)
    chunks = result.get("chunks") or []
    hit_ranks = [rank for rank, chunk in enumerate(chunks, 1) if chunk_matches(case, chunk)]
    document_hit_ranks = [
        rank for rank, chunk in enumerate(chunks, 1) if chunk_document_matches(case, chunk)
    ]
    forbidden_ranks = [rank for rank, chunk in enumerate(chunks, 1) if chunk_is_forbidden(case, chunk)]
    return {
        "id": case.id,
        "question": case.question,
        "tags": case.tags,
        "candidate_count": result.get("candidate_count", 0),
        "final_count": result.get("final_count", 0),
        "hit_ranks": hit_ranks,
        "document_hit_ranks": document_hit_ranks,
        "forbidden_ranks": forbidden_ranks,
        "recall@3": recall_at_k(hit_ranks, 3),
        "recall@5": recall_at_k(hit_ranks, 5),
        "document@3": recall_at_k(document_hit_ranks, 3),
        "document@5": recall_at_k(document_hit_ranks, 5),
        "exact_point@3": recall_at_k(hit_ranks, 3),
        "exact_point@5": recall_at_k(hit_ranks, 5),
        "mrr": reciprocal_rank(hit_ranks),
        "precision@5": precision_at_k(hit_ranks, min(5, len(chunks) or 1)),
        "forbidden_rate": 1.0 if forbidden_ranks else 0.0,
        "empty": 1.0 if not chunks else 0.0,
    }


def summarize(results):
    summary = {
        "cases": len(results),
        "recall@3": mean(r["recall@3"] for r in results),
        "recall@5": mean(r["recall@5"] for r in results),
        "document@3": mean(r["document@3"] for r in results),
        "document@5": mean(r["document@5"] for r in results),
        "exact_point@3": mean(r["exact_point@3"] for r in results),
        "exact_point@5": mean(r["exact_point@5"] for r in results),
        "mrr": mean(r["mrr"] for r in results),
        "precision@5": mean(r["precision@5"] for r in results),
        "forbidden_rate": mean(r["forbidden_rate"] for r in results),
        "empty_rate": mean(r["empty"] for r in results),
    }
    by_tag = defaultdict(list)
    for result in results:
        for tag in result["tags"] or ["untagged"]:
            by_tag[tag].append(result)
    summary["by_tag"] = {
        tag: {
            "cases": len(items),
            "recall@5": mean(x["recall@5"] for x in items),
            "document@5": mean(x["document@5"] for x in items),
            "exact_point@5": mean(x["exact_point@5"] for x in items),
            "mrr": mean(x["mrr"] for x in items),
            "forbidden_rate": mean(x["forbidden_rate"] for x in items),
            "empty_rate": mean(x["empty"] for x in items),
        }
        for tag, items in sorted(by_tag.items())
    }
    return summary


def check_regression(current, baseline, max_regression):
    failures = []
    for key in (
        "recall@3", "recall@5", "document@3", "document@5",
        "exact_point@3", "exact_point@5", "mrr", "precision@5",
    ):
        if key in baseline and current.get(key, 0) < baseline[key] - max_regression:
            failures.append(f"{key}: {baseline[key]:.4f} -> {current.get(key, 0):.4f}")
    for key in ("forbidden_rate", "empty_rate"):
        if key in baseline and current.get(key, 0) > baseline[key] + max_regression:
            failures.append(f"{key}: {baseline[key]:.4f} -> {current.get(key, 0):.4f}")
    return failures


async def main():
    parser = argparse.ArgumentParser(description="Evaluate NPA RAG retrieval.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--dataset")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--baseline")
    parser.add_argument("--max-regression", type=float, default=0.02)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset) if args.dataset else CASES
    if args.tag:
        cases = [c for c in cases if any(t in args.tag for t in c.tags)]
    if args.limit > 0:
        cases = cases[:args.limit]

    from supabase import create_client
    from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    results = []
    for index, case in enumerate(cases, 1):
        item = await run_case(case, supabase)
        results.append(item)
        print(
            f"[{index:03d}/{len(cases):03d}] {case.id} "
            f"R@5={item['recall@5']:.0%} "
            f"P@5={item['exact_point@5']:.0%} "
            f"D@5={item['document@5']:.0%} "
            f"MRR={item['mrr']:.3f} "
            f"forbidden={item['forbidden_rate']:.0%}"
        )

    summary = summarize(results)
    print("\\n=== RAG EVALUATION ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    payload = {"summary": summary, "results": results}
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.baseline and Path(args.baseline).exists():
        baseline_payload = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = baseline_payload.get("summary") or {}
        failures = check_regression(summary, baseline, args.max_regression)
        if failures:
            print("\\nREGRESSION GATE: FAILED")
            for failure in failures:
                print(" -", failure)
            raise SystemExit(2)
        print("\\nREGRESSION GATE: PASSED")


if __name__ == "__main__":
    asyncio.run(main())
