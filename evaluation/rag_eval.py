"""RAG evaluation harness with per-case diagnostics and dataset validation.

Usage:
  python evaluation/rag_eval.py evaluation/questions.json

This evaluates retrieval/grounding signals, not legal correctness by itself.
Cases without verified official references are explicitly reported.
"""
import asyncio
import json
import re
import sys
from datetime import date
from urllib.parse import urlparse
from collections import defaultdict
from pathlib import Path

from core.chat_service import ChatService


def validate_dataset(cases):
    errors = []
    warnings = []
    seen = set()
    if not isinstance(cases, list):
        raise ValueError("Dataset root must be a JSON array.")
    for index, case in enumerate(cases):
        label = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label}: expected an object")
            continue
        case_id = str(case.get("id") or "").strip()
        question = str(case.get("question") or "").strip()
        if not case_id:
            errors.append(f"{label}: missing id")
        elif case_id in seen:
            errors.append(f"{label}: duplicate id {case_id}")
        seen.add(case_id)
        if not question:
            errors.append(f"{label}: missing question")
        if not case.get("expected_topic"):
            warnings.append(f"{case_id or label}: no expected_topic")
        if not case.get("expected_documents") and not case.get("expected_points"):
            warnings.append(f"{case_id or label}: no document/point retrieval target")
        if case.get("reference_verified") is True:
            reference_url = str(case.get("reference_url") or "").strip()
            if not reference_url:
                errors.append(f"{case_id or label}: reference_verified=true but reference_url is missing")
            else:
                parsed = urlparse(reference_url)
                host = (parsed.hostname or "").lower().rstrip(".")
                official_domains = ("pravo.by", "etalonline.by", "ncpi.gov.by")
                official_host = any(host == domain or host.endswith("." + domain) for domain in official_domains)
                if parsed.scheme != "https" or not official_host:
                    errors.append(
                        f"{case_id or label}: verified reference must use HTTPS on an approved Belarus legal-information domain"
                    )
            verified_on = str(case.get("verified_on") or "").strip()
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", verified_on):
                    raise ValueError("expected ISO date")
                date.fromisoformat(verified_on)
            except ValueError:
                errors.append(f"{case_id or label}: reference_verified=true requires verified_on in YYYY-MM-DD format")
            if case.get("jurisdiction") != "BY":
                errors.append(f"{case_id or label}: verified reference must explicitly set jurisdiction='BY'")
    return errors, warnings


def _normalize_point(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:пункт(?:а|е|ом)?|п\.|point)\s*", "", text)
    text = re.sub(r"[\s.]+$", "", text)
    return text


def _safe_rate(rows, key):
    return round(sum(bool(row.get(key)) for row in rows) / max(len(rows), 1), 3)


def summarize(rows):
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    category_metrics = {}
    for category, items in sorted(by_category.items()):
        category_metrics[category] = {
            "cases": len(items),
            "topic_accuracy": _safe_rate(items, "topic_hit"),
            "document_recall": _safe_rate(items, "document_hit"),
            "point_recall": _safe_rate(items, "point_hit"),
            "grounding_rate": _safe_rate(items, "grounded"),
            "grounding_metadata_coverage": _safe_rate(items, "grounding_known"),
        }
    return {
        "cases": len(rows),
        "topic_accuracy": _safe_rate(rows, "topic_hit"),
        "document_recall": _safe_rate(rows, "document_hit"),
        "point_recall": _safe_rate(rows, "point_hit"),
        "grounding_rate": _safe_rate(rows, "grounded"),
        "grounding_metadata_coverage": _safe_rate(rows, "grounding_known"),
        "official_reference_coverage": _safe_rate(rows, "reference_verified"),
        "avg_candidates": round(sum(x["candidate_count"] for x in rows) / max(len(rows), 1), 2),
        "avg_final": round(sum(x["final_count"] for x in rows) / max(len(rows), 1), 2),
        "avg_evidence": round(sum(x["evidence_count"] for x in rows) / max(len(rows), 1), 2),
        "by_category": category_metrics,
        "failed_cases": [
            {
                "id": x["id"],
                "category": x["category"],
                "question": x["question"],
                "topic_hit": x["topic_hit"],
                "document_hit": x["document_hit"],
                "point_hit": x["point_hit"],
                "source_id_hit": x["source_id_hit"],
                "grounded": x["grounded"],
                "grounding_reason": x["grounding_reason"],
                "failure_reason": x["failure_reason"],
                "error": x["error"],
                "request_id": x["request_id"],
                "expected_documents": x["expected_documents"],
                "retrieved_documents": x["retrieved_documents"],
                "expected_points": x["expected_points"],
                "retrieved_points": x["retrieved_points"],
            }
            for x in rows
            if (
                not x["grounded"] or not x["topic_hit"] or not x["document_hit"]
                or not x["point_hit"] or not x["source_id_hit"] or x["error"]
            )
        ],
    }


async def run(path: str) -> None:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    errors, warnings = validate_dataset(cases)
    if errors:
        print(json.dumps({"dataset_valid": False, "validation_errors": errors, "validation_warnings": warnings},
                         ensure_ascii=False, indent=2))
        raise SystemExit(2)

    service = ChatService()
    rows = []
    for case in cases:
        case_id = str(case["id"])
        question = str(case["question"])
        try:
            result = await service.process_text(question)
            result = result if isinstance(result, dict) else {}
        except Exception as exc:
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}

        rag = result.get("rag") or {}
        chunks = result.get("evidence_map") or []
        if not isinstance(chunks, list):
            chunks = []
        docs = {str(x.get("document", "")).strip().lower() for x in chunks if isinstance(x, dict)}
        points = {_normalize_point(x.get("point")) for x in chunks if isinstance(x, dict) and x.get("point")}
        source_ids = {str(x.get("source_id", "")).strip() for x in chunks if isinstance(x, dict) and x.get("source_id")}
        expected_docs = [str(x).strip() for x in case.get("expected_documents", [])]
        expected_points = [str(x).strip() for x in case.get("expected_points", [])]
        expected_source_ids = {str(x).strip() for x in case.get("expected_source_ids", [])}

        topic_hit = not case.get("expected_topic") or rag.get("topic") == case["expected_topic"]
        doc_hit = not expected_docs or any(
            expected.lower() in document
            for expected in expected_docs for document in docs
        )
        point_hit = not expected_points or bool({_normalize_point(x) for x in expected_points} & points)
        source_id_hit = not expected_source_ids or bool(expected_source_ids & source_ids)

        grounding = rag.get("grounding")
        claim_evidence = rag.get("claim_evidence")
        grounding_known = (
            isinstance(grounding, dict) and grounding.get("passed") is not None
            and isinstance(claim_evidence, dict) and claim_evidence.get("passed") is not None
        )
        grounded = bool(
            result.get("success") is True
            and isinstance(grounding, dict) and grounding.get("passed") is True
            and isinstance(claim_evidence, dict) and claim_evidence.get("passed") is True
        )
        search_diagnostics = rag.get("search_diagnostics") or []
        rows.append({
            "id": case_id,
            "question": question,
            "category": str(case.get("category") or case.get("expected_topic") or "uncategorized"),
            "topic_hit": topic_hit,
            "document_hit": doc_hit,
            "point_hit": point_hit,
            "source_id_hit": source_id_hit,
            "grounded": grounded,
            "grounding_known": grounding_known,
            "reference_verified": case.get("reference_verified") is True,
            "grounding_reason": (claim_evidence or {}).get("reason") or (grounding or {}).get("reason"),
            "failure_reason": rag.get("failure_reason") or result.get("error"),
            "error": result.get("error"),
            "request_id": result.get("request_id") or rag.get("request_id"),
            "candidate_count": rag.get("candidate_count", 0) or 0,
            "final_count": rag.get("final_count", len(chunks)) or 0,
            "evidence_count": len(chunks),
            "expected_documents": expected_docs,
            "retrieved_documents": sorted(docs),
            "expected_points": expected_points,
            "retrieved_points": sorted(points),
            "search_diagnostics": search_diagnostics[:5] if isinstance(search_diagnostics, list) else [],
        })

    report = summarize(rows)
    report["dataset_valid"] = True
    report["validation_warnings"] = warnings
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python evaluation/rag_eval.py evaluation/questions.json")
    asyncio.run(run(sys.argv[1]))
