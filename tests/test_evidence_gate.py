"""Regression tests for the fail-closed Evidence Gate checks."""
import os

# ChatService creates a Supabase client at module import. These dummy values
# allow unit tests to exercise pure validation methods without network calls.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

from core.chat_service import ChatService


def test_grounding_rejects_unknown_source_id():
    result = ChatService._grounding_check(
        "Требование установлено [SOURCE:NPA_UNKNOWN_1].",
        ["NPA_KNOWN_1"],
    )
    assert result["passed"] is False
    assert result["unknown_source_ids"] == ["NPA_UNKNOWN_1"]


def test_grounding_accepts_known_source_id():
    result = ChatService._grounding_check(
        "Требование установлено [SOURCE:NPA_KNOWN_1].",
        ["NPA_KNOWN_1"],
    )
    assert result["passed"] is True
    assert result["unknown_source_ids"] == []


def test_claim_evidence_rejects_missing_cited_evidence():
    result = ChatService._claim_evidence_check(
        answer="Требование установлено [SOURCE:NPA_MISSING_1].",
        evidence_map=[],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "no_cited_evidence"


def test_claim_evidence_rejects_reference_not_present_in_evidence():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 85 требуется выполнить действие [SOURCE:NPA_RULE_84].",
        evidence_map=[
            {
                "source_id": "NPA_RULE_84",
                "document": "Правила по охране труда",
                "point": "84.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_reference_not_found_in_cited_evidence"
    assert result["unsupported_references"] == ["пункт 85"]


def test_claim_evidence_accepts_reference_matching_source_point():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 84 требуется выполнить действие [SOURCE:NPA_RULE_84].",
        evidence_map=[
            {
                "source_id": "NPA_RULE_84",
                "document": "Правила по охране труда",
                "point": "84.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is True


def test_claim_evidence_does_not_confuse_article_and_paragraph_numbers():
    result = ChatService._claim_evidence_check(
        answer="Согласно пункту 49 требуется выполнить действие [SOURCE:NPA_ARTICLE_49].",
        evidence_map=[
            {
                "source_id": "NPA_ARTICLE_49",
                "document": "Трудовой кодекс",
                "point": "статья 49",
                "excerpt": "Работодатель вправе действовать в предусмотренном порядке.",
            }
        ],
        query_profile={"question_type": "general"},
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_reference_not_found_in_cited_evidence"



def test_claim_evidence_rejects_categorical_absence_claim():
    result = ChatService._claim_evidence_check(
        answer="Законодательство не предусматривает такого требования [SOURCE:NPA_RULE_1].",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила по охране труда",
            "point": "1",
            "excerpt": "Работник обязан соблюдать требования безопасности.",
        }],
        query_profile={"question_type": "general"},
        question="Есть ли такое требование?",
    )
    assert result["passed"] is False
    assert result["reason"] == "categorical_absence_claim_exceeds_retrieved_context"


def test_claim_evidence_rejects_unverified_currentness_claim():
    result = ChatService._claim_evidence_check(
        answer="Это актуальная редакция правил [SOURCE:NPA_RULE_1].",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила по охране труда",
            "point": "1",
            "excerpt": "Работник обязан соблюдать требования безопасности.",
        }],
        query_profile={"question_type": "general"},
        question="Какая редакция действует?",
    )
    assert result["passed"] is False
    assert result["reason"] == "current_legal_status_not_verified"


def test_claim_evidence_rejects_permission_as_duty():
    result = ChatService._claim_evidence_check(
        answer="Работодатель обязан выполнить это действие [SOURCE:NPA_RULE_1].",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила по охране труда",
            "point": "1",
            "excerpt": "Работодатель вправе выполнить это действие.",
        }],
        query_profile={"question_type": "general"},
        question="Что вправе сделать работодатель?",
    )
    assert result["passed"] is False
    assert result["reason"] == "duty_claim_supported_only_by_permission_language"


def test_claim_evidence_rejects_russian_law_for_belarus_question():
    result = ChatService._claim_evidence_check(
        answer="Согласно норме [SOURCE:NPA_RULE_1] применяется следующее правило.",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Трудовой кодекс Российской Федерации",
            "point": "1",
            "excerpt": "Правило.",
        }],
        query_profile={"question_type": "general"},
        question="Каковы требования в Республике Беларусь?",
    )
    assert result["passed"] is False
    assert result["reason"] == "non_belarusian_legal_source_used"


def test_claim_evidence_rejects_unverified_equipment_scope():
    result = ChatService._claim_evidence_check(
        answer="При работе на токарном станке следует учитывать требования безопасности [SOURCE:NPA_RULE_1].",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Общие правила",
            "point": "1",
            "excerpt": "Следует соблюдать общие требования безопасности.",
        }],
        query_profile={"question_type": "general"},
        question="Какие требования при работе с переносной лестницей?",
    )
    assert result["passed"] is False
    assert result["reason"] == "answer_introduces_unverified_work_or_equipment_scope"


def test_claim_evidence_allows_currentness_with_verified_metadata():
    result = ChatService._claim_evidence_check(
        answer="Это актуальная редакция правил [SOURCE:NPA_RULE_1].",
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила по охране труда",
            "point": "1",
            "excerpt": "Работник обязан соблюдать требования безопасности.",
            "is_current": True,
            "last_verified_at": "2026-10-01",
            "source_url": "https://example.by/npa/1",
        }],
        query_profile={"question_type": "general"},
        question="Какая редакция действует?",
    )
    assert result["passed"] is True


def test_claim_evidence_does_not_mix_point_from_another_act():
    result = ChatService._claim_evidence_check(
        answer=(
            "Дополнительный источник [SOURCE:NPA_DOC_B]. "
            "Согласно пункту 84 требуется выполнить действие [SOURCE:NPA_DOC_A]."
        ),
        evidence_map=[
            {
                "source_id": "NPA_DOC_A",
                "document": "Правила по охране труда",
                "point": "85.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            },
            {
                "source_id": "NPA_DOC_B",
                "document": "Другой нормативный акт",
                "point": "84.",
                "excerpt": "Работник обязан соблюдать требования безопасности.",
            },
        ],
        query_profile={"question_type": "general"},
        question="Что установлено пунктом 84?",
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_reference_not_found_in_cited_evidence"

def test_legal_assistant_prompt_includes_evidence_map():
    from prompts import LEGAL_ASSISTANT_PROMPT

    evidence_map_text = "EVIDENCE MAP:\nSOURCE_ID=NPA_TEST_1"
    prompt = LEGAL_ASSISTANT_PROMPT.format(
        user_query="Тестовый вопрос",
        retrieved_text="Фрагмент нормативного акта",
        evidence_map=evidence_map_text,
    )

    assert evidence_map_text in prompt
    assert prompt.index(evidence_map_text) < prompt.rindex("НОРМАТИВНЫЙ КОНТЕКСТ")


def test_grounding_retry_prompt_uses_real_line_breaks():
    prompt = ChatService._build_grounding_retry_prompt("BASE PROMPT", {"reason": "test"})

    assert "\\n\\n===" not in prompt
    assert "\n\n===" in prompt


def test_chat_service_disables_router_source_grounding_for_evidence_gate():
    import inspect
    from services.ai_router import generate_answer

    signature = inspect.signature(generate_answer)
    assert signature.parameters["enforce_source_grounding"].default is True

    # Both initial generation and the single Evidence Gate retry delegate
    # content validation to ChatService; router provider fallbacks remain enabled.
    chat_source = inspect.getsource(ChatService.process_text)
    assert chat_source.count("enforce_source_grounding=False") == 2


def test_legal_system_prompt_is_compact_and_keeps_core_policy():
    from services.ai_router import LEGAL_SYSTEM_PROMPT

    assert len(LEGAL_SYSTEM_PROMPT) < 2000
    assert "Республике Беларусь" in LEGAL_SYSTEM_PROMPT
    assert "Не выдумывай нормы" in LEGAL_SYSTEM_PROMPT
    assert "SOURCE_ID" in LEGAL_SYSTEM_PROMPT
    assert "[SOURCE:NPA_...]" in LEGAL_SYSTEM_PROMPT
    assert "рядом с подтверждаемым утверждением" in LEGAL_SYSTEM_PROMPT


def test_claim_evidence_accepts_source_marker_on_immediately_following_line():
    result = ChatService._claim_evidence_check(
        answer=(
            "Согласно п. 17 Директивы № 1 руководители обеспечивают дисциплину.\n"
            "[SOURCE:NPA_DIRECTIVE_1_P17]\n"
            "Согласно ст. 253 Трудового кодекса применяется соответствующая норма.\n"
            "[SOURCE:NPA_TK_253]"
        ),
        evidence_map=[
            {
                "source_id": "NPA_DIRECTIVE_1_P17",
                "document": "Директива № 1 о мерах по укреплению общественной безопасности и дисциплины",
                "point": "17.",
                "excerpt": "Пункт 17. Руководители организаций обеспечивают общественную безопасность и дисциплину.",
            },
            {
                "source_id": "NPA_TK_253",
                "document": "Трудовой кодекс Республики Беларусь",
                "point": "статья 253",
                "excerpt": "Статья 253. Обязанности нанимателя.",
            },
        ],
        query_profile={"question_type": "general"},
        question="Какие обязанности установлены?",
    )
    assert result["passed"] is True


def test_claim_evidence_rejects_legal_claim_without_local_citation():
    result = ChatService._claim_evidence_check(
        answer=(
            "Работодатель обязан провести обучение. "
            "Обучение проводится в установленном порядке [SOURCE:NPA_RULE_1]."
        ),
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила охраны труда",
            "point": "1",
            "excerpt": "Работодатель обязан провести обучение.",
        }],
        query_profile={"question_type": "general"},
        question="Что обязан сделать работодатель?",
    )
    assert result["passed"] is False
    assert result["reason"] == "legal_claim_missing_local_source_citation"


def test_claim_evidence_accepts_citation_on_immediately_following_line():
    result = ChatService._claim_evidence_check(
        answer=(
            "Работодатель обязан провести обучение.\n"
            "[SOURCE:NPA_RULE_1]"
        ),
        evidence_map=[{
            "source_id": "NPA_RULE_1",
            "document": "Правила охраны труда",
            "point": "1",
            "excerpt": "Работодатель обязан провести обучение.",
        }],
        query_profile={"question_type": "general"},
        question="Что обязан сделать работодатель?",
    )
    assert result["passed"] is True
