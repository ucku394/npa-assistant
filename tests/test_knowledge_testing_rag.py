import pytest

from rag_query_classifier import (
    detect_knowledge_testing_action,
    detect_knowledge_testing_state,
    detect_topic,
)
from rag_query_generator import build_search_queries
from rag_query_profile import build_universal_query_profile


@pytest.mark.parametrize(
    "query",
    [
        "Если работник не прошел проверку знаний, какие дальнейшие действия",
        "Работник не сдал проверку знаний по охране труда",
        "Что делать при неудовлетворительном результате проверки знаний",
        "Как проводится повторная проверка знаний требований охраны труда",
    ],
)
def test_knowledge_testing_topic(query):
    assert detect_topic(query) == "knowledge_testing"


def test_failed_knowledge_testing_profile():
    query = "Если работник не прошел проверку знаний, какие дальнейшие действия"
    profile = build_universal_query_profile(query)

    assert profile["event"] == "knowledge_testing"
    assert profile["action_state"] == "failed"
    assert profile["action"] == "further_actions"


def test_failed_knowledge_testing_queries_are_specialized():
    query = "Если работник не прошел проверку знаний, какие дальнейшие действия"
    queries = build_search_queries(
        query,
        "knowledge_testing",
        "occupational_safety",
        ["procedure"],
    )
    lowered = "\n".join(queries).lower()

    assert "неудовлетворительные результаты проверки знаний" in lowered
    assert "повторная проверка" in lowered
    assert "нарушение обязанности не выполнено" not in lowered


def test_briefing_query_is_not_knowledge_testing():
    assert detect_topic("Кто проводит вводный инструктаж по охране труда") == "occupational_briefing"


def test_knowledge_testing_action_detection():
    assert detect_knowledge_testing_action(
        "Что делать после того, как работник не сдал проверку знаний"
    ) == "further_actions"
    assert detect_knowledge_testing_action(
        "Нужна ли повторная проверка знаний"
    ) == "repeat_test"


@pytest.mark.parametrize(
    "query, expected_action",
    [
        ("Кто проводит проверку знаний по вопросам ОТ?", "responsible_person"),
        ("Кто проводит проверку знаний по охране труда?", "responsible_person"),
        ("Кто входит в комиссию по проверке знаний?", "responsible_person"),
        ("Когда проводится повторная проверка знаний?", "deadline"),
        ("Сроки проведения проверки знаний", "deadline"),
        ("Какова периодичность проверки знаний требований охраны труда?", "deadline"),
        ("Через какой срок проводится повторная проверка знаний?", "deadline"),
    ],
)
def test_knowledge_testing_who_and_timing_questions(query, expected_action):
    assert detect_topic(query) == "knowledge_testing"
    assert detect_knowledge_testing_action(query) == expected_action


@pytest.mark.parametrize(
    "query, expected_type",
    [
        ("Кто проводит проверку знаний по вопросам ОТ?", "who"),
        ("Когда проводится повторная проверка знаний?", "frequency"),
        ("Какова периодичность проверки знаний требований охраны труда?", "frequency"),
    ],
)
def test_knowledge_testing_question_type(query, expected_type):
    profile = build_universal_query_profile(query)
    assert profile["event"] == "knowledge_testing"
    assert profile["question_type"] == expected_type


def test_knowledge_testing_responsible_search_does_not_leak_briefing_queries():
    query = "Кто проводит проверку знаний по вопросам ОТ?"
    queries = build_search_queries(
        query,
        "knowledge_testing",
        "occupational_safety",
        ["responsible_person"],
    )
    lowered = "\n".join(queries).lower()

    assert "комиссия по проверке знаний требований охраны труда" in lowered
    assert "кто проводит проверку знаний требований охраны труда комиссия" in lowered
    assert "вводный инструктаж" not in lowered


def test_knowledge_testing_timing_search_is_specialized():
    query = "Когда проводится повторная проверка знаний?"
    queries = build_search_queries(
        query,
        "knowledge_testing",
        "occupational_safety",
        [],
    )
    lowered = "\n".join(queries).lower()

    assert "повторная проверка знаний требований охраны труда сроки порядок" in lowered
    assert "периодичность проверки знаний требований охраны труда" in lowered
    assert "вводный инструктаж" not in lowered
