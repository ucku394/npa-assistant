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
