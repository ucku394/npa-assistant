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


def test_milk_compensation_question_detects_permission_intent():
    from rag_query_classifier import detect_query_intents, detect_primary_intent

    question = (
        "Разрешается ли заменять молоко, выдаваемое работникам за работу "
        "с вредными веществами, денежной компенсацией?"
    )
    intents = detect_query_intents(question)
    assert "permission" in intents
    assert detect_primary_intent(intents, question) == "permission"


def test_milk_norm_question_detects_quantity_intent():
    from rag_query_classifier import detect_query_intents, detect_primary_intent

    question = (
        "Сколько молока положено работнику за смену и в каком пункте "
        "установлена норма выдачи?"
    )
    intents = detect_query_intents(question)
    assert "quantity" in intents
    assert detect_primary_intent(intents, question) == "quantity"


def test_milk_replacement_question_keeps_occupational_domain():
    question = (
        "Какие равноценные пищевые продукты могут выдаваться работнику "
        "вместо молока при работе с вредными веществами?"
    )
    assert detect_legal_domain(question) == "occupational_safety"
    assert detect_topic(question) == "milk_provision"
