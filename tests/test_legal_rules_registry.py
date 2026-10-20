"""Tests for the declarative legal rules registry."""
from legal_rules_registry import MANDATORY_SOURCE_RULES, RAG_SELECTION_RULES, evaluate_mandatory_source_rules


def _source_id(chunk, index):
    return chunk.get("_source_id") or f"NPA_TEST_{index}"


def _evaluate(profile, chunks, answer):
    return evaluate_mandatory_source_rules(
        query_profile=profile,
        chunks=chunks,
        answer=answer,
        extract_source_ids=lambda text: [token for token in text.split() if token.startswith("NPA_")],
        build_source_id=_source_id,
    )


def test_registry_has_expected_rules():
    ids = {rule["id"] for rule in MANDATORY_SOURCE_RULES}
    assert {"article_49", "article_11", "instruction_list_compiler", "instruction_osh_service_role", "instruction_developer"} <= ids


def test_required_source_cited_passes():
    result = _evaluate(
        {"qualifiers": ["suspension_for_osh_violation"]},
        [{"doc_name": "Трудовой кодекс Республики Беларусь 2026", "point_num": "Статья 49.", "_source_id": "NPA_TK_ARTICLE_49"}],
        "Основание NPA_TK_ARTICLE_49",
    )
    assert result["passed"] is True
    assert result["failures"] == {"article_49_missing": False, "article_49_not_cited": False}


def test_missing_required_source_fails():
    result = _evaluate({"qualifiers": ["suspension_for_osh_violation"]}, [], "Ответ NPA_OTHER")
    assert result["passed"] is False
    assert result["failures"]["article_49_missing"] is True


def test_uncited_required_source_fails():
    result = _evaluate(
        {"qualifiers": ["instruction_developer"]},
        [{"doc_name": "Постановление № 176", "point_num": "Пункт 14.", "_source_id": "NPA_RESOLUTION_176_P14"}],
        "Ответ NPA_OTHER",
    )
    assert result["passed"] is False
    assert result["failures"]["instruction_developer_not_cited"] is True


def test_inactive_rules_do_not_block_general_queries():
    result = _evaluate({"qualifiers": []}, [], "Общий ответ")
    assert result["passed"] is True
    assert result["active_rules"] == []
    assert result["failures"] == {}


def test_rag_topic_priorities_are_registry_data():
    assert RAG_SELECTION_RULES["lathe_work"]["priority_points"] == ("84", "85", "40")
    assert RAG_SELECTION_RULES["portable_ladder"]["priority_points"] == ("54", "53")


# Evidence Gate regression cases: deterministic checks, no LLM/Supabase calls.
from core.chat_service import ChatService


def _claim_check(answer, evidence, question_type="general", question=""):
    return ChatService._claim_evidence_check(
        answer=answer,
        evidence_map=evidence,
        query_profile={"question_type": question_type},
        topic="test",
        question=question,
    )


def test_evidence_gate_rejects_answer_without_cited_evidence():
    result = _claim_check(
        "Работодатель обязан провести обучение [SOURCE:NPA_TEST_P1].",
        [],
    )
    assert result["passed"] is False
    assert result["reason"] == "no_cited_evidence"


def test_evidence_gate_rejects_frequency_denial_against_explicit_periodicity():
    result = _claim_check(
        "Периодичность не установлена в законодательстве [SOURCE:NPA_TEST_P1].",
        [{
            "source_id": "NPA_TEST_P1",
            "document": "Правила проверки знаний",
            "point": "Пункт 12",
            "excerpt": "Проверка проводится не реже одного раза в год.",
            "jurisdiction": "BY",
        }],
        question_type="frequency",
        question="Какова периодичность проверки знаний?",
    )
    assert result["passed"] is False
    assert result["reason"] == "answer_denies_frequency_but_cited_evidence_contains_frequency"


def test_evidence_gate_rejects_russian_law_for_belarus_question():
    result = _claim_check(
        "Это предусмотрено Трудовым кодексом [SOURCE:NPA_TEST_P1].",
        [{
            "source_id": "NPA_TEST_P1",
            "document": "Трудовой кодекс Российской Федерации",
            "point": "Статья 214",
            "excerpt": "Работодатель обязан обеспечить безопасные условия труда.",
            "jurisdiction": "RU",
        }],
        question="Какие обязанности работодателя в Беларуси?",
    )
    assert result["passed"] is False
    assert result["reason"] == "non_belarusian_legal_source_used"


def test_evidence_gate_rejects_unverified_currentness_claim():
    result = _claim_check(
        "Норма действует в настоящее время [SOURCE:NPA_TEST_P1].",
        [{
            "source_id": "NPA_TEST_P1",
            "document": "Правила охраны труда",
            "point": "Пункт 12",
            "excerpt": "Работодатель обеспечивает безопасные условия труда.",
            "jurisdiction": "BY",
        }],
    )
    assert result["passed"] is False
    assert result["reason"] == "current_legal_status_not_verified"
