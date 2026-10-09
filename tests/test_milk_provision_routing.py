"""Regression tests for milk/equivalent food product question routing."""

from rag_query_classifier import detect_legal_domain, detect_topic


def test_milk_compensation_question_routes_to_occupational_safety():
    question = "можно ли молоко заменить денежной компенсацией или другими видами продукции"
    assert detect_legal_domain(question) == "occupational_safety"
    assert detect_topic(question) == "milk_provision"


def test_equivalent_food_product_question_routes_to_milk_topic():
    question = "Какие равноценные пищевые продукты выдают вместо молока?"
    assert detect_legal_domain(question) == "occupational_safety"
    assert detect_topic(question) == "milk_provision"
