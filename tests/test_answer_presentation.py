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
    assert fallback["label"] == "ПРЕДВАРИТЕЛЬНЫЙ ОТВЕТ ИИ — ПРОВЕРИТЬ ПО НПА"
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
    assert "ПРЕДВАРИТЕЛЬНЫЙ ОТВЕТ ИИ — ПРОВЕРИТЬ ПО НПА" in rendered
    assert "не подтверждённое правовое заключение" in rendered
    assert "Закон № 356-З" not in rendered


def test_ai_fallback_rejects_legal_claims_and_citations():
    unsafe_phrases = [
        "Работодатель обязан выдать средства защиты.",
        "Работник должен пройти обучение.",
        "Это запрещено законодательством.",
        "Согласно статье 11 Закона № 356-З, наниматель обязан...",
        "Периодичность проверки составляет один раз в год.",
        "Аттестация рабочих мест проводится при наличии вредных факторов.",
        "Основанием для проведения аттестации является наличие вредных условий.",
        "Согласно пункту 4 постановления № 253, аттестация проводится...",
        "Работодатель обязан организовать аттестацию в течение 30 дней.",
        "Такие действия не допускаются.",
        "Наниматель обеспечивает работников средствами индивидуальной защиты.",
        "Работодатель предоставляет работнику специальную одежду.",

    ]
    for phrase in unsafe_phrases:
        assert ChatService._is_safe_ai_fallback(phrase) is False


def test_ai_fallback_allows_neutral_practical_guidance():
    assert ChatService._is_safe_ai_fallback(
        "Для анализа ситуации полезно уточнить вид работы, оборудование и фактические условия."
    ) is True



def test_preliminary_ai_allows_useful_issue_framing_without_legal_conclusions():
    answer = (
        "Чтобы определить, какие обстоятельства имеют значение для вашей ситуации, "
        "полезно собрать сведения о профессиях и рабочих местах, фактических условиях "
        "труда, применяемом оборудовании и результатах предыдущих оценок. "
        "Это предварительный чек-лист, а не подтверждённый перечень правовых оснований. "
        "Конкретный нормативный вывод нужно сверить с актуальными актами Республики Беларусь."
    )
    # A statement that the legal conclusion needs verification is safe; it must
    # not be confused with an assertion that a particular rule exists.
    assert ChatService._is_safe_ai_fallback(answer) is True


def test_preliminary_ai_rejects_unverified_statutory_modality_and_numbers():
    unsafe = [
        "Наниматель должен провести оценку рабочих мест.",
        "Такая процедура обязательна для всех рабочих мест.",
        "Аттестация проводится каждые 5 лет.",
        "Работник вправе отказаться от выполнения работы.",
        "Основанием является наличие вредных условий труда.",
        "Законодательство предусматривает проведение аттестации при таких условиях.",
        "Работодателю предоставляется специальная одежда.",
        "Согласно статье 10, наниматель обязан провести аттестацию.",
    ]
    for answer in unsafe:
        assert ChatService._is_safe_ai_fallback(answer) is False


def test_preliminary_ai_rejects_source_markers_without_evidence_validation():
    assert ChatService._is_safe_ai_fallback(
        "Предварительное пояснение. [SOURCE:NPA_123]"
    ) is False
