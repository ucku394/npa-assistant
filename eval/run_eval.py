from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import retrieve_context
from .dataset import CASES
from .metrics import mean, precision_at_k, recall_at_k, reciprocal_rank
from .schema import chunk_is_forbidden, chunk_matches

async def run_case(case, supabase):
    result = await retrieve_context(case.question, supabase)
    chunks = result.get("chunks") or []
    hit_ranks = [rank for rank, chunk in enumerate(chunks, 1) if chunk_matches(case, chunk)]
    forbidden_ranks = [rank for rank, chunk in enumerate(chunks, 1) if chunk_is_forbidden(case, chunk)]
    return {
        "id": case.id,
        "question": case.question,
        "tags": case.tags,
        "candidate_count": result.get("candidate_count", 0),
        "final_count": result.get("final_count", 0),
        "hit_ranks": hit_ranks,
        "forbidden_ranks": forbidden_ranks,
        "recall@3": recall_at_k(hit_ranks, 3),
        "recall@5": recall_at_k(hit_ranks, 5),
        "mrr": reciprocal_rank(hit_ranks),
        "precision@5": precision_at_k(hit_ranks, min(5, len(chunks) or 1)),
        "forbidden_rate": 1.0 if forbidden_ranks else 0.0,
    }

async def main():
    parser = argparse.ArgumentParser(description="Evaluate NPA RAG retrieval.")
    parser.add_argument("--limit", type=int, default=len(CASES))
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    from supabase import create_client
    from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    cases = [c for c in CASES if not args.tag or any(t in args.tag for t in c.tags)]
    cases = cases[:max(0, args.limit)]

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required')
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    results = []
    for index, case in enumerate(cases, 1):
        item = await run_case(case, supabase)
        results.append(item)
        print(f"[{index:03d}/{len(cases):03d}] {case.id} R@5={item['recall@5']:.0%} MRR={item['mrr']:.3f} forbidden={item['forbidden_rate']:.0%}")

    summary = {
        "cases": len(results),
        "recall@3": mean(r["recall@3"] for r in results),
        "recall@5": mean(r["recall@5"] for r in results),
        "mrr": mean(r["mrr"] for r in results),
        "precision@5": mean(r["precision@5"] for r in results),
        "forbidden_rate": mean(r["forbidden_rate"] for r in results),
        "empty_rate": mean(1.0 if r["final_count"] == 0 else 0.0 for r in results),
    }
    print("\n=== RAG EVALUATION ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

if __name__ == "__main__":
    asyncio.run(main())
