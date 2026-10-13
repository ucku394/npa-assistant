"""Regression tests for photo-inspection legal-domain routing."""

from inspection_query import build_finding_query
from rag_query_classifier import detect_legal_domain, detect_topic, is_scaffolding_query


def test_scaffold_and_unfixed_decking_route_to_occupational_safety():
    query = build_finding_query({
        "description": "Риск обрушения незакрепленного деревянного настила",
        "visual_evidence": "Доски уложены на опоры без видимой фиксации",
    })
    assert "средства подмащивания" in query
    assert detect_legal_domain(query) == "occupational_safety"


def test_wooden_support_on_stair_treads_routes_to_work_at_height_rules():
    query = build_finding_query({
        "description": "Использование неинвентарной деревянной стойки для поддержания настила на высоте",
        "visual_evidence": "Деревянный брус упирается в ступень лестницы",
    })
    assert "проект производства работ" in query
    assert detect_legal_domain(query) == "occupational_safety"


def test_obstructed_evacuation_route_routes_to_fire_safety():
    query = build_finding_query({
        "description": "Загромождение пути эвакуации (лестничного марша) строительными лесами и опорами",
        "visual_evidence": "Вышка-тура и брус занимают часть лестничного марша",
    })
    assert "эвакуационная ширина" in query
    assert detect_legal_domain(query) == "fire_safety"


def test_unrelated_ppe_finding_keeps_occupational_safety_routing():
    query = build_finding_query({
        "description": "Отсутствие защитных очков при работе с инструментом",
        "visual_evidence": "Работник работает с болгаркой без очков",
    })
    assert detect_legal_domain(query) == "occupational_safety"



def test_stairwell_and_tower_scaffold_are_not_routed_to_portable_ladder():
    query = build_finding_query({
        "description": "Использование ненадежной деревянной опоры при работах на высоте",
        "visual_evidence": "Вышка-тура, деревянный брус уперт в ступень лестницы для поддержки настила",
    })
    assert is_scaffolding_query(query)
    assert detect_topic(query) == "general"
    assert detect_legal_domain(query) == "occupational_safety"


def test_explicit_portable_ladder_query_keeps_dedicated_topic():
    assert not is_scaffolding_query("Нужно ли осматривать переносную лестницу перед каждым применением?")
    assert detect_topic("Нужно ли осматривать переносную лестницу перед каждым применением?") == "portable_ladder"
