"""Regression tests for targeted retrieval of Directive No. 1 leader duties."""

from rag_engine.rag import _select_legal_diverse_chunks


def _chunk(point: str, score: float, content: str = ""):
    return {
        "doc_name": "Директива Президента Республики Беларусь № 1 — Общественная безопасность и дисциплина",
        "document_id": "directive-1-current",
        "point_num": point,
        "content": content or f"Нормативное содержание пункта {point}.",
        "_combined_score": score,
    }


def test_leader_duty_query_keeps_directive_points_16_1_16_2_and_17():
    # Generic point 1 has the strongest semantic score, but must not crowd
    # out the operative clauses that answer this specific question.
    chunks = [
        _chunk("1.", 9.0, "Общие положения Директивы."),
        _chunk("17.", 8.0, "Руководитель организации отвечает за общественную безопасность в трудовом коллективе."),
        _chunk("16.", 7.0, "Руководителям организаций в пределах компетенции."),
        _chunk("16.2.", 1.0, "Обеспечить оказание методической, информационной и иной помощи подчиненным организациям."),
        _chunk("16.1.", 0.5, "Принять меры по снижению и предупреждению правонарушений, гибели и травматизма людей."),
    ]

    selected = _select_legal_diverse_chunks(
        ranked_chunks=chunks,
        limit=5,
        topic="general",
        intents=[],
        cross_reference=False,
        user_query="Какие обязанности возлагаются на руководителей организаций по обеспечению дисциплины и безопасности?",
    )

    selected_points = {chunk["point_num"].rstrip(".") for chunk in selected}
    assert {"16", "16.1", "16.2", "17"}.issubset(selected_points)


def test_directive_priority_does_not_apply_to_unrelated_queries():
    chunks = [
        _chunk("1.", 9.0, "Общие положения Директивы."),
        _chunk("17.", 8.0, "Руководитель организации отвечает за общественную безопасность в трудовом коллективе."),
        _chunk("16.", 7.0, "Руководителям организаций в пределах компетенции."),
        _chunk("16.2.", 1.0, "Оказание помощи подчиненным организациям."),
        _chunk("16.1.", 0.5, "Предупреждение правонарушений, гибели и травматизма."),
    ]

    selected = _select_legal_diverse_chunks(
        ranked_chunks=chunks,
        limit=2,
        topic="general",
        intents=[],
        cross_reference=False,
        user_query="Что говорится в общих положениях Директивы?",
    )

    selected_points = [chunk["point_num"].rstrip(".") for chunk in selected]
    assert selected_points[0] == "1"
