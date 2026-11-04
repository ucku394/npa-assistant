from core.answer_presentation import build_answer_presentation, format_telegram_answer
from core.chat_service import ChatService


def test_answer_modes_have_one_canonical_presentation():
    verified = build_answer_presentation("verified_npa")
    hybrid = build_answer_presentation("hybrid_ai")
    fallback = build_answer_presentation("ai_fallback")
    insufficient = build_answer_presentation("insufficient_evidence")

    assert verified["label"] == "ПОДТВЕРЖДЕНО НПА"
    assert verified["ai_generated"] is False
    assert hybrid["label"] == "НОРМЫ НПА + ОТВЕТ ИИ"
    assert hybrid["ai_generated"] is True
    assert fallback["label"] == "ОТВЕТ ИИ — СПРАВОЧНАЯ ИНФОРМАЦИЯ"
    assert fallback["ai_generated"] is True
    assert insufficient["label"] == "НЕДОСТАТОЧНО ДАННЫХ"
    assert insufficient["ai_generated"] is False


def test_unknown_answer_mode_fails_closed():
    result = build_answer_presentation("unexpected_mode")
    assert result["mode"] == "insufficient_evidence"
    assert result["ai_generated"] is False


def test_telegram_uses_canonical_label_and_only_shows_sources_for_npa_modes():
    fallback = build_answer_presentation("ai_fallback")
    rendered = format_telegram_answer(
        "Практический ориентир без нормативных утверждений.",
        [{"document": "Закон № 356-З", "point": "статья 11"}],
        fallback,
    )
    assert "ОТВЕТ ИИ — СПРАВОЧНАЯ ИНФОРМАЦИЯ" in rendered
    assert "не нормативное требование" in rendered
    assert "Закон № 356-З" not in rendered


def test_ai_fallback_rejects_legal_claims_and_citations():
    unsafe_phrases = [
        "Работодатель обязан выдать средства защиты.",
        "Работник должен пройти обучение.",
        "Это запрещено законодательством.",
        "Согласно статье 11 Закона № 356-З, наниматель обязан...",
        "Периодичность проверки составляет один раз в год.",
        "Такие действия не допускаются.",
    ]
    for phrase in unsafe_phrases:
        assert ChatService._is_safe_ai_fallback(phrase) is False


def test_ai_fallback_allows_neutral_practical_guidance():
    assert ChatService._is_safe_ai_fallback(
        "Для анализа ситуации полезно уточнить вид работы, оборудование и фактические условия."
    ) is True
