"""Cross-topic routing regression tests for the Belarusian OHS legal assistant.

These tests intentionally cover multiple legal domains and common query forms.
They test routing only; retrieval quality must be measured separately against
known source documents and point numbers.
"""

import pytest

from rag_query_classifier import detect_legal_domain, detect_topic


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
