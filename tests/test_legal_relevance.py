from rag_engine.legal_relevance import (
    build_second_pass_queries,
    evidence_gate,
    legal_policy,
    legal_relevance_adjustment,
)


def test_knowledge_testing_penalizes_administrative_offence_collision():
    chunk = {
        "doc_name": "Кодекс Республики Беларусь об административных правонарушениях № 91-З",
        "content": "Повторное совершение административного правонарушения признается повторностью правонарушения.",
    }
    result = legal_relevance_adjustment(chunk, "knowledge_testing", {})
    assert result["hard_negative"] is True
    assert result["forbidden_hits"] > 0
    assert result["score"] < 0


def test_knowledge_testing_accepts_actual_ohs_evidence():
    chunk = {
        "doc_name": "Нормативный правовой акт об обучении и проверке знаний требований охраны труда",
        "content": "Проверка знаний требований охраны труда проводится в установленном порядке.",
    }
    result = legal_relevance_adjustment(chunk, "knowledge_testing", {})
    assert result["hard_negative"] is False
    assert result["required_hits"] > 0
    assert result["legal_match"] >= 0.55


def test_evidence_gate_rejects_wrong_domain_candidate():
    chunks = [
        {
            "_combined_score": 2.0,
            "_legal_hard_negative": True,
            "_legal_required_hits": 0,
            "_legal_match": 0.0,
        }
    ]
    result = evidence_gate(chunks, "knowledge_testing", {})
    assert result["sufficient"] is False


def test_evidence_gate_accepts_topic_evidence():
    chunks = [
        {
            "_combined_score": 1.4,
            "_legal_hard_negative": False,
            "_legal_required_hits": 1,
            "_legal_match": 0.85,
        }
    ]
    result = evidence_gate(chunks, "knowledge_testing", {})
    assert result["sufficient"] is True


def test_second_pass_contains_legal_queries_for_timing():
    queries = build_second_pass_queries(
        "С какой периодичностью проводится повторная проверка знаний по вопросам ОТ?",
        "knowledge_testing",
        {"legal_phrases": ["периодичность проверки знаний требований охраны труда"]},
    )
    lowered = "\n".join(queries).lower()
    assert "периодичность проверки знаний требований охраны труда" in lowered
    assert "повторная проверка знаний требований охраны труда сроки порядок" in lowered


def test_knowledge_testing_policy_forbids_coap_collision():
    policy = legal_policy("knowledge_testing")
    assert "административное правонарушение" in policy["forbidden"]
    assert "повторность правонарушения" in policy["forbidden"]


def test_special_accident_investigation_rejects_generic_accident_evidence():
    chunk = {
        "doc_name": "Правила расследования и учета несчастных случаев № 30",
        "content": "Расследование несчастного случая на производстве проводится в установленном порядке.",
    }
    profile = {"qualifiers": ["special_investigation"]}
    result = legal_relevance_adjustment(chunk, "accident_investigation", profile)
    assert result["required_hits"] == 0

    gated = evidence_gate(
        [{
            "_combined_score": 2.0,
            "_legal_hard_negative": False,
            "_legal_required_hits": result["required_hits"],
            "_legal_match": result["legal_match"],
        }],
        "accident_investigation",
        profile,
    )
    assert gated["sufficient"] is False


def test_special_accident_investigation_accepts_explicit_evidence():
    chunk = {
        "doc_name": "Правила расследования и учета несчастных случаев № 30",
        "content": "Специальное расследование несчастного случая проводится государственным инспектором труда.",
    }
    profile = {"qualifiers": ["special_investigation"]}
    result = legal_relevance_adjustment(chunk, "accident_investigation", profile)
    assert result["required_hits"] > 0
    assert result["legal_match"] >= 0.55


def test_special_accident_investigation_queries_include_categories_and_conclusion():
    from rag_query_generator import build_search_queries

    queries = build_search_queries(
        "Какие категории несчастных случаев подлежат специальному расследованию и кто составляет заключение?",
        "accident_investigation",
        "occupational_safety",
        [],
    )
    lowered = "\n".join(queries).lower()
    assert "пункт 40 правил № 30 специальному расследованию подлежат" in lowered
    assert "по результатам специального расследования государственным инспектором труда составляется и подписывается заключение" in lowered


def test_topic_hierarchy_distinguishes_electrical_knowledge_testing():
    from rag_engine.topic_planner import build_topic_hierarchy

    plan = build_topic_hierarchy(
        "Кто проверяет знания электротехнического персонала?",
        "electrical_safety",
    )
    assert plan["subtopic"] == "knowledge_testing"
    assert plan["legal_object"] == "electrical_knowledge_testing"
    assert plan["confidence"] >= 0.9


def test_topic_hierarchy_distinguishes_fire_extinguishers():
    from rag_engine.topic_planner import build_topic_hierarchy

    plan = build_topic_hierarchy(
        "Как часто проверяются огнетушители?",
        "fire_safety",
    )
    assert plan["subtopic"] == "fire_extinguishers"
    assert plan["legal_object"] == "fire_extinguisher"


def test_topic_hierarchy_distinguishes_microclimate_from_general_sanitary():
    from rag_engine.topic_planner import build_topic_hierarchy

    plan = build_topic_hierarchy(
        "Какая температура должна быть на рабочем месте?",
        "sanitary",
    )
    assert plan["subtopic"] == "microclimate"
    assert plan["legal_object"] == "microclimate"


def test_query_plan_exposes_hierarchy():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan("Кто назначает ответственного за электрохозяйство?")
    assert plan["topic"] == "electrical_safety"
    assert plan["subtopic"] == "responsible_person"
    assert plan["legal_object"] == "electrical_responsible_person"
    assert "ответственный за электрохозяйство" in plan["source_constraints"]["preferred_terms"]


def test_query_plan_preserves_kind_intent_for_briefing():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan("Какой инструктаж проводится при приеме на работу?")
    queries = [q.lower() for q in plan["search_queries"]]
    assert any("виды инструктажей" in q for q in queries)
    assert not any("кто проводит вводный инструктаж" in q for q in queries)


def test_query_plan_builds_frequency_anchor_from_hierarchy():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan("Как часто проверяются огнетушители?")
    queries = [q.lower() for q in plan["search_queries"]]
    assert plan["subtopic"] == "fire_extinguishers"
    assert any("периодичность огнетушители" in q for q in queries)


def test_query_plan_builds_electrical_knowledge_testing_anchor():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan("Кто проверяет знания электротехнического персонала?")
    queries = [q.lower() for q in plan["search_queries"]]
    assert any("кто проводит проверку знаний электротехнического персонала" in q for q in queries)
    assert plan["legal_object"] == "electrical_knowledge_testing"


def test_query_plan_exposes_explicit_query_slots():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan("Кто проверяет знания электротехнического персонала?")
    slots = {item["role"]: item["query"].lower() for item in plan["query_slots"]}

    assert slots["exact"] == "кто проверяет знания электротехнического персонала?"
    assert "электротехнический персонал" in slots["object"]
    assert "кто проводит проверку знаний" in slots["intent"]
    assert len(plan["query_slots"]) == len(plan["search_queries"]) <= 8


def test_query_plan_preserves_specialized_recovery_query():
    from rag_engine.query_planner import build_query_plan

    plan = build_query_plan(
        "Какие категории несчастных случаев подлежат специальному расследованию?"
    )
    queries = [q.lower() for q in plan["search_queries"]]

    assert len(queries) <= 8
    assert any("пункт 40 правил № 30" in q for q in queries)
    assert len(plan["query_roles"]) == len(queries)
