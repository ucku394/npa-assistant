"""Tests for evaluation dataset schema and reporting."""
import os

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

from evaluation.rag_eval import summarize, validate_dataset


def test_dataset_rejects_duplicate_ids_and_missing_questions():
    errors, _ = validate_dataset([
        {"id": "RAG-001", "question": "Question?", "expected_topic": "general"},
        {"id": "RAG-001", "question": "", "expected_topic": "general"},
    ])
    assert any("duplicate id RAG-001" in error for error in errors)
    assert any("missing question" in error for error in errors)


def test_verified_reference_requires_official_url():
    errors, _ = validate_dataset([
        {
            "id": "RAG-001",
            "question": "Question?",
            "expected_topic": "general",
            "reference_verified": True,
            "jurisdiction": "BY",
            "verified_on": "2026-10-09",
        }
    ])
    assert any("reference_url is missing" in error for error in errors)


def test_verified_reference_rejects_unapproved_domain_and_missing_date():
    errors, _ = validate_dataset([
        {
            "id": "RAG-002",
            "question": "Question?",
            "expected_topic": "general",
            "reference_verified": True,
            "reference_url": "https://example.com/law",
            "jurisdiction": "BY",
        }
    ])
    assert any("approved Belarus legal-information domain" in error for error in errors)
    assert any("requires verified_on" in error for error in errors)


def test_verified_reference_accepts_official_belarus_domain():
    errors, _ = validate_dataset([
        {
            "id": "RAG-003",
            "question": "Question?",
            "expected_topic": "general",
            "reference_verified": True,
            "reference_url": "https://pravo.by/document/example",
            "verified_on": "2026-10-09",
            "jurisdiction": "BY",
        }
    ])
    assert errors == []


def test_dataset_warns_when_no_retrieval_target_exists():
    errors, warnings = validate_dataset([
        {"id": "RAG-001", "question": "Question?", "expected_topic": "general"}
    ])
    assert errors == []
    assert any("no document/point retrieval target" in warning for warning in warnings)


def test_summary_reports_per_category_and_grounding():
    rows = [
        {
            "category": "ppe",
            "topic_hit": True,
            "document_hit": True,
            "point_hit": True,
            "grounded": True,
            "grounding_known": True,
            "reference_verified": True,
            "candidate_count": 4,
            "final_count": 3,
            "evidence_count": 2,
            "id": "RAG-001",
            "question": "Question?",
            "source_id_hit": True,
            "grounding_reason": None,
            "failure_reason": None,
            "error": None,
            "request_id": "abc123",
            "expected_documents": ["356-З"],
            "retrieved_documents": ["закон № 356-З"],
            "expected_points": [],
            "retrieved_points": ["3"],
        }
    ]
    report = summarize(rows)
    assert report["cases"] == 1
    assert report["by_category"]["ppe"]["grounding_rate"] == 1.0
    assert report["official_reference_coverage"] == 1.0
    assert report["failed_cases"] == []



def test_point_normalization_accepts_prefix_and_trailing_period():
    from evaluation.rag_eval import _normalize_point

    assert _normalize_point("пункт 54.") == "54"
    assert _normalize_point("54") == "54"
    assert _normalize_point("2.3.") == "2.3"
