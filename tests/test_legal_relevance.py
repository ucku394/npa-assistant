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
