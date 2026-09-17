"""
AI text-generation router.

Primary: Gemini
Fallback: OpenRouter

The router is independent from bot.py.

Features:
- Gemini as primary provider
- OpenRouter as fallback provider
- Gemini Circuit Breaker
- Automatic temporary Gemini disable after 429/quota errors
- Automatic return to Gemini after cooldown
- Strict Belarusian legal RAG rules
- SOURCE_ID based source attribution
"""

import logging
import time
from typing import Optional

from google import genai
from openai import OpenAI

from config import (
    CHAT_MODEL,
    GEMINI_API_KEY,
    MAX_CHAT_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_MODEL,
)

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI CIRCUIT BREAKER SETTINGS
# ============================================================

GEMINI_COOLDOWN_SECONDS = 1800

_gemini_disabled_until = 0.0


# ============================================================
# GEMINI CLIENT
# ============================================================

_gemini = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# ============================================================
# OPENROUTER CLIENT
# ============================================================

_openrouter = (
    OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    if OPENROUTER_API_KEY
    else None
)


# ============================================================
# HELPERS
# ============================================================

def _clean_text(value) -> str:
    """
    Приводит ответ AI к обычной строке.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def _validate_prompt(prompt: str) -> str:
    """
    Проверяет, что prompt не пустой.
    """

    prompt = _clean_text(prompt)

    if not prompt:
        raise ValueError(
            "Prompt is empty."
        )

    return prompt


# ============================================================
# GEMINI CIRCUIT BREAKER
# ============================================================

def _gemini_is_available() -> bool:
    """
    Проверяет, можно ли сейчас обращаться к Gemini.
    """

    global _gemini_disabled_until

    current_time = time.time()

    if current_time >= _gemini_disabled_until:

        if _gemini_disabled_until > 0:

            logger.info(
                "AI Router | Gemini cooldown expired | "
                "Gemini will be tried again"
            )

            _gemini_disabled_until = 0.0

        return True

    remaining = int(
        _gemini_disabled_until - current_time
    )

    logger.info(
        "AI Router | Gemini temporarily disabled | "
        "remaining=%ss",
        remaining,
    )

    return False


def _disable_gemini(reason: str):
    """
    Временно отключает Gemini.
    """

    global _gemini_disabled_until

    _gemini_disabled_until = (
        time.time()
        + GEMINI_COOLDOWN_SECONDS
    )

    logger.warning(
        "AI Router | Gemini disabled for %ss | "
        "reason=%s",
        GEMINI_COOLDOWN_SECONDS,
        reason,
    )


def _is_gemini_quota_error(
    exc: Exception,
) -> bool:
    """
    Определяет quota/rate-limit ошибку Gemini.
    """

    error_text = str(exc).lower()

    quota_markers = [
        "resource_exhausted",
        "quota exceeded",
        "quota",
        "rate limit",
        "429",
        "too many requests",
    ]

    return any(
        marker in error_text
        for marker in quota_markers
    )


# ============================================================
# GEMINI
# ============================================================

def generate_with_gemini(
    prompt: str,
) -> str:
    """
    Генерация ответа через Gemini.
    """

    if _gemini is None:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = _validate_prompt(
        prompt
    )

    logger.info(
        "AI Router | trying Gemini | model=%s",
        CHAT_MODEL,
    )

    response = (
        _gemini.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": (
                    MAX_CHAT_TOKENS
                ),
            },
        )
    )

    text = _clean_text(
        getattr(
            response,
            "text",
            None,
        )
    )

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    logger.info(
        "AI Router | Gemini response received."
    )

    return text


# ============================================================
# OPENROUTER SYSTEM PROMPT
# ============================================================

LEGAL_SYSTEM_PROMPT = """
Ты — профессиональный эксперт-консультант
по охране труда, промышленной и пожарной безопасности
в Республике Беларусь.

Твоя задача — анализировать вопросы пользователя
ТОЛЬКО на основании нормативного контекста,
переданного в сообщении пользователя.

============================================================
1. ЮРИСДИКЦИЯ
============================================================

Работай только с законодательством
Республики Беларусь.

Не используй законодательство Российской Федерации
или других государств.

Если в переданном контексте отсутствует необходимая
норма законодательства Республики Беларусь,
не восполняй её по памяти.

============================================================
2. ГЛАВНЫЙ ПРИНЦИП
============================================================

RAG-КОНТЕКСТ ЯВЛЯЕТСЯ ЕДИНСТВЕННЫМ
НОРМАТИВНЫМ ОСНОВАНИЕМ ОТВЕТА.

Ты НЕ должен использовать свои внутренние знания
для добавления отсутствующих в контексте
юридических норм.

Нельзя придумывать:

- НПА;
- названия НПА;
- номера НПА;
- даты НПА;
- статьи;
- пункты;
- подпункты;
- обязанности;
- права;
- сроки;
- периодичность;
- требования;
- запреты;
- штрафы;
- виды ответственности;
- размеры ответственности;
- порядок проведения мероприятий;
- категории работников;
- категории объектов.

============================================================
3. SOURCE_ID
============================================================

Каждый фрагмент RAG-контекста может содержать:

SOURCE_ID: ...

SOURCE_ID — это идентификатор конкретного
нормативного фрагмента.

Например:

SOURCE_ID: NPA_175_P51

Используй SOURCE_ID для определения того,
какая конкретно норма подтверждает вывод.

ВАЖНО:

Наличие SOURCE_ID в RAG-контексте НЕ означает,
что этот источник обязательно подтверждает ответ.

Высокая semantic similarity также НЕ означает,
что норма юридически применима.

============================================================
4. ЮРИДИЧЕСКОЕ ДОКАЗАТЕЛЬСТВО
============================================================

Каждый существенный юридический вывод должен
опираться на конкретный фрагмент переданного
нормативного контекста.

Перед формулировкой вывода мысленно проверь:

1. Какой именно SOURCE_ID подтверждает вывод?
2. Что именно написано в этом источнике?
3. Относится ли норма к ситуации пользователя?
4. Относится ли она к той категории работников,
   объектов или работ, о которой спрашивает пользователь?
5. Не расширяю ли я область действия нормы?

Если прямого подтверждения нет,
не делай категоричный юридический вывод.

============================================================
5. ЗАПРЕТ НА РАСШИРЕНИЕ НОРМЫ
============================================================

Нельзя автоматически распространять норму:

- на всех работников;
- на всех работодателей;
- на все профессии;
- на все подразделения;
- на все объекты;
- на все виды работ;

если в самом нормативном контексте
такое распространение прямо не установлено.

Пример:

Если источник говорит:

"работники, занятые на работах с повышенной опасностью..."

нельзя превращать это в:

"все работники..."

без прямого нормативного основания.

============================================================
6. НЕ СМЕШИВАЙ НОРМЫ
============================================================

Если в контексте находятся несколько пунктов,
не объединяй их автоматически.

Каждый пункт анализируй отдельно.

Не делай вывод:

"согласно пунктам 42, 47, 51 и 53..."

если фактически ответ подтверждается только пунктом 51.

Если несколько пунктов действительно необходимы
для одного вывода — укажи их все и объясни,
какую роль играет каждый.

============================================================
7. ЕСЛИ КОНТЕКСТ НЕПОЛНЫЙ
============================================================

Если контекст позволяет ответить только на часть
вопроса — отвечай только на эту часть.

После этого прямо укажи:

"В предоставленном нормативном контексте
отсутствует достаточная информация для вывода
по остальной части вопроса."

Не пытайся угадывать недостающую норму.

============================================================
8. ПЕРВИЧНОСТЬ ТЕКСТА НОРМЫ
============================================================

Приоритет имеет текст конкретного пункта/статьи,
а не:

- semantic similarity;
- название документа;
- предположение;
- общая юридическая логика;
- твои внутренние знания.

Similarity используется только для поиска
релевантных фрагментов.

Она НЕ является юридическим доказательством.

============================================================
9. ФОРМАТ ОТВЕТА
============================================================

Используй следующую структуру, когда вопрос
требует нормативного ответа:

📌 КРАТКИЙ ОТВЕТ

Краткий и конкретный вывод.

📚 НОРМАТИВНОЕ ОСНОВАНИЕ

Укажи только те НПА и пункты/статьи,
которые непосредственно подтверждают ответ.

Для каждого источника:

- документ;
- пункт/статья;
- краткое объяснение его значения.

🔎 АНАЛИЗ

Объясни применение нормы к вопросу пользователя.

Если есть ограничения области действия нормы —
обязательно укажи их.

⚠️ ВАЖНО

Укажи существенные ограничения,
если они следуют из контекста.

📎 ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ

Указывай ТОЛЬКО источники,
которые реально использованы
для обоснования ответа.

Не перечисляй все найденные RAG-фрагменты.

Для каждого источника по возможности используй:

SOURCE_ID
Документ
Пункт/статья

============================================================
10. НЕ ПОКАЗЫВАЙ SEMANTIC SIMILARITY ПОЛЬЗОВАТЕЛЮ
============================================================

Показатель:

Semantic similarity: 0.xxxx

является технической информацией RAG.

Не используй его как аргумент юридической применимости
и не выводи его пользователю без необходимости.

============================================================
11. НЕ ДЕЛАЙ ЛИШНИХ ЮРИДИЧЕСКИХ ВЫВОДОВ
============================================================

Если пользователь спросил:

"Как часто проводится проверка знаний?"

не нужно самостоятельно добавлять:

- кто отвечает;
- какие штрафы;
- какая ответственность;
- кто создает комиссию;
- как оформляется протокол;

если этого нет в вопросе
или это не требуется для понимания ответа.

============================================================
12. СОМНЕНИЯ
============================================================

Если имеются несколько возможных трактовок
и переданный контекст не позволяет выбрать
одну из них однозначно — укажи это.

Не выдавай предположение за установленное
требование законодательства.

============================================================
13. ОСОБОЕ ПРАВИЛО
============================================================

Фраза:

"мне известно, что..."

не является нормативным основанием.

Фраза:

"обычно применяется..."

не является нормативным основанием.

Фраза:

"как правило..."

не является нормативным основанием.

Юридический вывод должен следовать
из переданного нормативного текста.

============================================================
14. ГЛАВНАЯ ЦЕЛЬ
============================================================

Лучше дать короткий частичный ответ,
подтверждённый конкретным пунктом НПА,
чем дать полный, но неподтверждённый ответ.

Точность важнее полноты.
"""


# ============================================================
# OPENROUTER
# ============================================================

def generate_with_openrouter(
    prompt: str,
) -> str:
    """
    Генерация ответа через OpenRouter.
    """

    if _openrouter is None:

        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    if not OPENROUTER_MODEL:

        raise RuntimeError(
            "OPENROUTER_MODEL is not configured."
        )

    prompt = _validate_prompt(
        prompt
    )

    logger.info(
        "AI Router | trying OpenRouter | model=%s",
        OPENROUTER_MODEL,
    )

    kwargs = {
        "model": OPENROUTER_MODEL,

        "messages": [
            {
                "role": "system",
                "content": LEGAL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        "temperature": 0.1,

        "max_tokens": MAX_CHAT_TOKENS,
    }

    # --------------------------------------------------------
    # Дополнительная fallback-модель OpenRouter
    # --------------------------------------------------------

    if OPENROUTER_FALLBACK_MODEL:

        kwargs["extra_body"] = {
            "models": [
                OPENROUTER_FALLBACK_MODEL
            ]
        }

        logger.info(
            "AI Router | OpenRouter fallback model=%s",
            OPENROUTER_FALLBACK_MODEL,
        )

    response = (
        _openrouter.chat.completions.create(
            **kwargs
        )
    )

    if not response:

        raise RuntimeError(
            "OpenRouter returned no response."
        )

    if not response.choices:

        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    message = response.choices[0].message

    text = _clean_text(
        getattr(
            message,
            "content",
            None,
        )
    )

    if not text:

        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    used_model = (
        getattr(
            response,
            "model",
            None,
        )
        or OPENROUTER_MODEL
    )

    logger.info(
        "AI Router | OpenRouter response received | "
        "model=%s",
        used_model,
    )

    return text


# ============================================================
# MAIN ROUTER
# ============================================================

def generate_answer(
    prompt: str,
) -> str:
    """
    Основной AI Router.

    Логика:

    1. Gemini доступен
       → сначала Gemini.

    2. Gemini quota/rate-limit
       → отключаем Gemini на cooldown.
       → OpenRouter.

    3. Gemini в cooldown
       → Gemini пропускается.
       → OpenRouter.

    4. После cooldown
       → следующая генерация снова пробует Gemini.

    5. Если оба провайдера недоступны
       → RuntimeError.
    """

    prompt = _validate_prompt(
        prompt
    )

    gemini_error: Optional[
        Exception
    ] = None

    # ========================================================
    # 1. GEMINI
    # ========================================================

    if _gemini_is_available():

        try:

            return generate_with_gemini(
                prompt
            )

        except Exception as exc:

            gemini_error = exc

            if _is_gemini_quota_error(
                exc
            ):

                _disable_gemini(
                    reason=str(exc)
                )

            logger.warning(
                "AI Router | Gemini unavailable | "
                "switching immediately to OpenRouter | "
                "error=%s",
                exc,
            )

    else:

        logger.info(
            "AI Router | Gemini skipped | "
            "temporarily disabled"
        )

    # ========================================================
    # 2. OPENROUTER
    # ========================================================

    if _openrouter is None:

        raise RuntimeError(
            "Gemini failed and OpenRouter "
            "is not configured. "
            f"Gemini error: {gemini_error}"
        ) from gemini_error

    try:

        return generate_with_openrouter(
            prompt
        )

    except Exception as exc:

        logger.error(
            "AI Router | OpenRouter failed | "
            "error=%s",
            exc,
            exc_info=True,
        )

        raise RuntimeError(
            "Both AI providers failed.\n"
            f"Gemini: {gemini_error}\n"
            f"OpenRouter: {exc}"
        ) from exc
