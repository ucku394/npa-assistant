"""Cross-topic routing regression tests for the Belarusian OHS legal assistant.

These tests intentionally cover multiple legal domains and common query forms.
They test routing only; retrieval quality must be measured separately against
known source documents and point numbers.
"""

import pytest

from rag_query_classifier import detect_legal_domain, detect_topic
from rag_query_generator import build_search_queries
from rag_query_profile import build_universal_query_profile


@pytest.mark.parametrize(
    ("query", "expected_domain"),
    [
        ("Какие требования пожарной безопасности соблюдать на предприятии?", "fire_safety"),
        ("Как проверить исправность огнетушителя?", "fire_safety"),
        ("Что делать при аварии на опасном производственном объекте?", "industrial_safety"),
        ("Какие требования предъявляются к эксплуатации электроустановок до 1000 В?", "electrical_safety"),
        ("Как организовать обязательный медицинский осмотр работников?", "occupational_safety"),
        ("Как расследовать несчастный случай на производстве?", "occupational_safety"),
        ("Какие санитарно-эпидемиологические требования действуют на производстве?", "sanitary"),
        ("Какое предельное усилие допускается при остановке падения человека в страховочной системе и какова максимальная длина стропа с амортизатором?", "occupational_safety"),
        ("Какие требования предъявляются к страховочной привязи?", "occupational_safety"),
    ],
)
def test_legal_domain_routing_across_domains(query, expected_domain):
    assert detect_legal_domain(query) == expected_domain


@pytest.mark.parametrize(
    ("query", "expected_topic"),
    [
        ("Нужно ли осматривать приставную лестницу перед использованием?", "portable_ladder"),
        ("Как организовать медицинский осмотр работников?", "medical_examinations"),
        ("Как расследовать несчастный случай на производстве?", "accident_investigation"),
        ("Помоги составить алгоритм обучения работников при работе на высоте 1 группа", "height_work_training"),
        ("Какое предельное усилие допускается при остановке падения человека в страховочной системе и какова максимальная длина стропа с амортизатором?", "fall_protection"),
        ("Какова максимальная длина стропа с амортизатором?", "fall_protection"),
        ("Как провести инструктаж по охране труда для работника?", "occupational_briefing"),
        ("Какие требования безопасности при работе на токарном станке?", "lathe_work"),
        ("Как проводится аттестация рабочих мест по условиям труда?", "workplace_attestation"),
        ("Можно ли заменить молоко денежной компенсацией?", "milk_provision"),
    ],
)
def test_topic_routing_across_common_ohs_tasks(query, expected_topic):
    assert detect_topic(query) == expected_topic


def test_empty_query_falls_back_to_general():
    assert detect_legal_domain("") == "general"
    assert detect_topic("") == "general"



def test_milk_replacement_search_keeps_short_lexical_queries_in_top_eight():
    query = "Можно ли молоко заменить денежной компенсацией или другими видами продукции?"
    queries = build_search_queries(
        query,
        detect_topic(query),
        detect_legal_domain(query),
        [],
    )
    normalized = [item.casefold() for item in queries]

    assert len(queries) <= 8
    assert "денежная компенсация" in normalized
    assert "замена молока" in normalized
    assert "равноценными пищевыми продуктами" in normalized


def test_milk_substitution_is_routed_to_occupational_safety():
    query = "Можно ли молоко заменить денежной компенсацией или другими видами продукции?"
    assert detect_legal_domain(query) == "occupational_safety"
    assert detect_topic(query) == "milk_provision"


def test_accident_selection_preserves_directly_relevant_hybrid_candidate_outside_legacy_act_filter():
    from rag_engine.rag import _select_legal_diverse_chunks

    direct_phrase = "Работник обязан сообщить непосредственному руководителю о несчастном случае."
    candidates = [{
        "doc_name": "Постановление № 260",
        "point_num": "12",
        "content": direct_phrase,
        "_combined_score": 0.9174,
    }]
    candidates.extend(
        {
            "doc_name": "Положение о расследовании несчастных случаев № 30",
            "point_num": str(point),
            "content": direct_phrase,
            "_combined_score": 0.70 - point * 0.01,
        }
        for point in range(1, 6)
    )
    selected = _select_legal_diverse_chunks(
        ranked_chunks=candidates,
        limit=5,
        topic="accident_investigation",
        intents=[],
        cross_reference=False,
        accident_mode="worker_did_not_report",
        user_query="Работник не сообщил о несчастном случае непосредственному руководителю",
    )
    assert any("№ 260" in chunk["doc_name"] for chunk in selected)


def test_fall_protection_search_generates_decree_11_queries():
    query = "Какое предельное усилие допускается при остановке падения человека в страховочной системе и какова максимальная длина стропа с амортизатором?"
    assert detect_legal_domain(query) == "occupational_safety"
    assert detect_topic(query) == "fall_protection"

    profile = build_universal_query_profile(query)
    assert profile["event"] == "fall_protection_equipment"
    assert profile["question_type"] == "limit"

    queries = build_search_queries(
        query,
        detect_topic(query),
        detect_legal_domain(query),
        [],
        profile,
    )
    combined = " ".join(queries).lower()
    assert "11" in combined
    assert "страховочн" in combined
    assert "амортизатор" in combined
    assert "строп" in combined
