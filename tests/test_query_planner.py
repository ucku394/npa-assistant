from rag_engine.query_planner import build_query_plan


def test_query_plan_separates_knowledge_testing_from_briefing():
    plan = build_query_plan("Когда проводится повторная проверка знаний по охране труда?")

    assert plan["topic"] == "knowledge_testing"
    assert plan["action"] == "deadline"
    assert plan["question_type"] == "frequency"
    assert "проверка знаний" in " ".join(plan["search_queries"]).lower()
    assert "вводный инструктаж" not in " ".join(plan["search_queries"]).lower()
    assert "административное правонарушение" in plan["negative_concepts"]


def test_query_plan_contains_explicit_source_constraints():
    plan = build_query_plan("Кто проводит проверку знаний по охране труда?")

    constraints = plan["source_constraints"]

    assert constraints["legal_domain"] == "occupational_safety"
    assert constraints["topic"] == "knowledge_testing"
    assert constraints["preferred_documents"]
    assert constraints["preferred_terms"]


def test_query_plan_assigns_query_roles():
    plan = build_query_plan("Кто проводит проверку знаний по охране труда?")

    assert plan["query_roles"]
    assert plan["query_roles"][0] == "exact"
    assert any(role in {"legal", "semantic", "document"} for role in plan["query_roles"][1:])


def test_query_plan_keeps_domain_and_topic_separate():
    plan = build_query_plan("Как часто проводится проверка знаний по охране труда?")

    assert plan["domain"] == "occupational_safety"
    assert plan["topic"] == "knowledge_testing"
    assert plan["profile"]["event"] == "knowledge_testing"
