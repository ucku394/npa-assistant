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
