"""Online retrieval evaluation against the live Supabase corpus.

Run with:
    SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... python -m eval.evaluate_retrieval

The script intentionally evaluates the RAG retrieval layer only. It does not call
Gemini/OpenRouter, so failures are attributable to retrieval rather than generation.
Gold labels are document-level (gold_source_ids); chunk-level labels can be added
later in gold_chunk_ids after manual verification.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from supabase import create_client

from rag import build_source_id, retrieve_context
from rag_engine.legal_relevance import legal_policy

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "evaluation_cases.json"


def load_cases() -> list[dict[str, Any]]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def source_matches(source_id: str, gold_ids: list[str]) -> bool:
    source_id = str(source_id or "").strip()
    for gold in gold_ids:
        gold = str(gold or "").strip()
        if source_id == gold or source_id.startswith(gold + "_"):
            return True
    return False


def allowed_domains(expected_domain: str) -> set[str]:
    if expected_domain == "occupational_safety":
        # The Labor Code and some cross-cutting NPA are currently stored as
        # general-domain documents and are valid OHS evidence.
        return {"occupational_safety", "general"}
    return {expected_domain}


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    chunks = list(result.get("chunks") or [])
    gold = list(case.get("gold_source_ids") or [])
    gold_chunks = set(str(value) for value in (case.get("gold_chunk_ids") or []))

    source_ids = [
        build_source_id(chunk, index)
        for index, chunk in enumerate(chunks, start=1)
    ]

    slot_hits = defaultdict(bool)
    slot_hit_ranks = {}
    for rank, (chunk, source_id) in enumerate(zip(chunks, source_ids), start=1):
        if not source_matches(source_id, gold):
            continue
        for role in (chunk.get("_query_roles") or []):
            role = str(role or "").strip() or "unknown"
            slot_hits[role] = True
            slot_hit_ranks.setdefault(role, rank)

    ranks = [
        index
        for index, source_id in enumerate(source_ids, start=1)
        if source_matches(source_id, gold)
    ]

    expected_domain = case.get("expected_domain", "occupational_safety")
    allowed = allowed_domains(expected_domain)
    wrong_domain = sum(
        1
        for chunk in chunks[:10]
        if str(chunk.get("legal_domain") or "").strip()
        and str(chunk.get("legal_domain")).strip() not in allowed
    )

    policy = legal_policy(case["topic"])
    forbidden = [
        str(value).lower()
        for value in (policy.get("forbidden") or [])
    ]
    forbidden_hits = 0
    for chunk in chunks[:10]:
        text = " ".join(
            str(chunk.get(key) or "")
            for key in ("doc_name", "content", "search_text")
        ).lower()
        if any(marker and marker in text for marker in forbidden):
            forbidden_hits += 1

    evidence = result.get("evidence") or {}

    chunk_ids = [str(chunk.get("id") or "").strip() for chunk in chunks]
    chunk_ranks = [
        index
        for index, chunk_id in enumerate(chunk_ids, start=1)
        if chunk_id and chunk_id in gold_chunks
    ]
    first_chunk_rank = chunk_ranks[0] if chunk_ranks else None

    first_rank = ranks[0] if ranks else None
    return {
        "id": case["id"],
        "topic": case["topic"],
        "question": case["question"],
        "gold_source_ids": gold,
        "retrieved_source_ids": source_ids[:10],
        "hit_at_5": any(rank <= 5 for rank in ranks),
        "hit_at_10": any(rank <= 10 for rank in ranks),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "chunk_recall_at_5": (
            any(rank <= 5 for rank in chunk_ranks)
            if gold_chunks else None
        ),
        "chunk_recall_at_10": (
            any(rank <= 10 for rank in chunk_ranks)
            if gold_chunks else None
        ),
        "chunk_mrr": (
            (1.0 / first_chunk_rank)
            if first_chunk_rank else 0.0
        ) if gold_chunks else None,
        "precision_at_5": (
            sum(source_matches(source_ids[i], gold) for i in range(min(5, len(source_ids))))
            / min(5, len(source_ids))
            if source_ids else 0.0
        ),
        "wrong_domain_at_10": wrong_domain > 0,
        "wrong_domain_count_at_10": wrong_domain,
        "forbidden_source_at_10": forbidden_hits > 0,
        "forbidden_source_count_at_10": forbidden_hits,
        "evidence_sufficient": bool(evidence.get("sufficient")),
        "evidence_reason": evidence.get("reason"),
        "query_slot_hits": dict(slot_hits),
        "query_slot_first_rank": slot_hit_ranks,
    }


async def run() -> dict[str, Any]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required."
        )

    supabase = create_client(url, key)
    cases = load_cases()
    limit = int(os.getenv("RAG_EVAL_LIMIT", str(len(cases))))
    cases = cases[:limit]
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        try:
            result = await retrieve_context(case["question"], supabase)
            rows.append(evaluate_case(case, result))
            if index % 10 == 0:
                print(f"evaluated {index}/{len(cases)}")
        except Exception as exc:
            rows.append({
                "id": case["id"],
                "topic": case["topic"],
                "question": case["question"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    valid = [row for row in rows if "error" not in row]
    total = len(valid) or 1
    chunk_labeled = [row for row in valid if row.get("chunk_mrr") is not None]
    chunk_total = len(chunk_labeled) or 1

    def mean(key: str) -> float:
        return round(sum(float(row[key]) for row in valid) / total, 4)

    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_topic[row["topic"]].append(row)

    slot_stats = {}
    all_roles = sorted({role for row in valid for role in (row.get("query_slot_hits") or {})})
    for role in all_roles:
        hits = sum(bool((row.get("query_slot_hits") or {}).get(role)) for row in valid)
        ranks = [float((row.get("query_slot_first_rank") or {})[role]) for row in valid if role in (row.get("query_slot_first_rank") or {})]
        slot_stats[role] = {
            "gold_hit_rate": round(hits / total, 4),
            "mean_first_rank": round(sum(ranks) / len(ranks), 4) if ranks else None,
            "cases_with_rank": len(ranks),
        }

    report = {
        "total_cases": len(cases),
        "successful_cases": len(valid),
        "failed_cases": len(cases) - len(valid),
        "recall_at_5": mean("hit_at_5"),
        "recall_at_10": mean("hit_at_10"),
        "mrr": mean("mrr"),
        "precision_at_5": mean("precision_at_5"),
        "chunk_labeled_cases": len(chunk_labeled),
        "chunk_recall_at_5": (
            round(sum(float(row["chunk_recall_at_5"]) for row in chunk_labeled) / chunk_total, 4)
            if chunk_labeled else None
        ),
        "chunk_recall_at_10": (
            round(sum(float(row["chunk_recall_at_10"]) for row in chunk_labeled) / chunk_total, 4)
            if chunk_labeled else None
        ),
        "chunk_mrr": (
            round(sum(float(row["chunk_mrr"]) for row in chunk_labeled) / chunk_total, 4)
            if chunk_labeled else None
        ),
        "wrong_domain_rate_at_10": mean("wrong_domain_at_10"),
        "forbidden_source_rate_at_10": mean("forbidden_source_at_10"),
        "evidence_sufficiency_rate": mean("evidence_sufficient"),
        "query_slot_effectiveness": slot_stats,
        "by_topic": {
            topic: {
                "cases": len(items),
                "recall_at_5": round(
                    sum(float(x["hit_at_5"]) for x in items) / len(items), 4
                ),
                "recall_at_10": round(
                    sum(float(x["hit_at_10"]) for x in items) / len(items), 4
                ),
                "mrr": round(
                    sum(float(x["mrr"]) for x in items) / len(items), 4
                ),
                "precision_at_5": round(
                    sum(float(x["precision_at_5"]) for x in items) / len(items), 4
                ),
                "wrong_domain_rate_at_10": round(
                    sum(float(x["wrong_domain_at_10"]) for x in items) / len(items), 4
                ),
                "forbidden_source_rate_at_10": round(
                    sum(float(x["forbidden_source_at_10"]) for x in items) / len(items), 4
                ),
                "evidence_sufficiency_rate": round(
                    sum(float(x["evidence_sufficient"]) for x in items) / len(items), 4
                ),
            }
            for topic, items in sorted(by_topic.items())
        },
        "failures": [
            row for row in rows
            if "error" in row
            or not row.get("hit_at_10")
            or row.get("wrong_domain_at_10")
            or row.get("forbidden_source_at_10")
        ],
    }
    return report


if __name__ == "__main__":
    report = asyncio.run(run())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output_path = os.getenv("RAG_EVAL_OUTPUT")
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
        print(f"Evaluation report written to {output_path}")
    print(rendered)
