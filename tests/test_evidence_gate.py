"""Regression tests for the fail-closed Evidence Gate checks."""
import os

# ChatService creates a Supabase client at module import. These dummy values
# allow unit tests to exercise pure validation methods without network calls.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

from core.chat_service import ChatService


def test_grounding_rejects_unknown_source_id():
    result = ChatService._grounding_check(
        "Требование установлено [SOURCE:NPA_UNKNOWN_1].",
        ["NPA_KNOWN_1"],
    )
    assert result["passed"] is False
    assert result["unknown_source_ids"] == ["NPA_UNKNOWN_1"]


def test_grounding_accepts_known_source_id():
    result = ChatService._grounding_check(
        "Требование установлено [SOURCE:NPA_KNOWN_1].",
        ["NPA_KNOWN_1"],
    )
    assert result["passed"] is True
    assert result["unknown_source_ids"] == []


def test_claim_evidence_rejects_missing_cited_evidence():
    result = ChatService._claim_evidence_check(
        answer="Требование установлено [SOURCE:NPA_MISSING_1].",
        evidence_map=[],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "no_cited_evidence"


def test_claim_evidence_rejects_reference_not_present_in_evidence():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 85 требуется выполнить действие [SOURCE:NPA_RULE_84].",
        evidence_map=[
            {
                "source_id": "NPA_RULE_84",
                "document": "Правила по охране труда",
                "point": "84.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_reference_not_found_in_cited_evidence"
    assert result["unsupported_references"] == ["пункт 85"]


def test_claim_evidence_accepts_reference_matching_source_point():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 84 требуется выполнить действие [SOURCE:NPA_RULE_84].",
        evidence_map=[
            {
                "source_id": "NPA_RULE_84",
                "document": "Правила по охране труда",
                "point": "84.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is True


def test_claim_evidence_does_not_confuse_article_and_paragraph_numbers():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 49 требуется выполнить действие [SOURCE:NPA_ARTICLE_49].",
        evidence_map=[
            {
                "source_id": "NPA_ARTICLE_49",
                "document": "Трудовой кодекс",
                "point": "статья 49",
                "excerpt": "Работодатель вправе действовать в предусмотренном порядке.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_reference_not_found_in_cited_evidence"
