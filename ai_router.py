```python
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
- No repeated useless Gemini requests while quota is exhausted
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

# Сколько секунд Gemini считается временно недоступным
# после ошибки quota/rate-limit.
#
# 1800 секунд = 30 минут.
#
# Это не означает, что Gemini восстановится через 30 минут.
# Это защита от постоянных бесполезных запросов к API.
GEMINI_COOLDOWN_SECONDS = 1800


# Время Unix, до которого Gemini считается отключённым.
#
# 0 = Gemini доступен.
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
        raise ValueError("Prompt is empty.")

    return prompt


# ============================================================
# GEMINI CIRCUIT BREAKER
# ============================================================

def _gemini_is_available() -> bool:
    """
    Проверяет, можно ли сейчас обращаться к Gemini.

    Если Gemini временно отключён, возвращает False.
    """

    global _gemini_disabled_until

    current_time = time.time()

    # Gemini снова доступен
    if current_time >= _gemini_disabled_until:

        # Если ранее Gemini был отключён,
        # фиксируем возвращение в рабочее состояние.
        if _gemini_disabled_until > 0:

            logger.info(
                "AI Router | Gemini cooldown expired | "
                "Gemini will be tried again"
            )

            _gemini_disabled_until = 0.0

        return True

    # Gemini ещё находится в cooldown
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

    Используется после quota/rate-limit ошибок.
    """

    global _gemini_disabled_until

    _gemini_disabled_until = (
        time.time() + GEMINI_COOLDOWN_SECONDS
    )

    logger.warning(
        "AI Router | Gemini disabled for %ss | "
        "reason=%s",
        GEMINI_COOLDOWN_SECONDS,
        reason,
    )


def _is_gemini_quota_error(exc: Exception) -> bool:
    """
    Определяет, связана ли ошибка Gemini
    с исчерпанием квоты или rate limit.
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

def generate_with_gemini(prompt: str) -> str:
    """
    Генерация ответа через Gemini.
    """

    if _gemini is None:

        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    prompt = _validate_prompt(prompt)

    logger.info(
        "AI Router | trying Gemini | model=%s",
        CHAT_MODEL,
    )

    response = _gemini.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": MAX_CHAT_TOKENS,
        },
    )

    text = _clean_text(
        getattr(response, "text", None)
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
# OPENROUTER
# ============================================================

def generate_with_openrouter(prompt: str) -> str:
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

    prompt = _validate_prompt(prompt)

    logger.info(
        "AI Router | trying OpenRouter | model=%s",
        OPENROUTER_MODEL,
    )

    kwargs = {
        "model": OPENROUTER_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты профессиональный эксперт-консультант "
                    "по охране труда, промышленной и пожарной "
                    "безопасности в Республике Беларусь.\n\n"

                    "Отвечай только на основании переданного "
                    "нормативного контекста.\n\n"

                    "Не придумывай:\n"
                    "- нормативные правовые акты;\n"
                    "- статьи;\n"
                    "- пункты;\n"
                    "- обязанности;\n"
                    "- сроки;\n"
                    "- штрафы;\n"
                    "- виды ответственности;\n"
                    "- требования законодательства.\n\n"

                    "Если переданного нормативного контекста "
                    "недостаточно для уверенного ответа, "
                    "прямо укажи, что имеющихся данных недостаточно."
                ),
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

    response = _openrouter.chat.completions.create(
        **kwargs
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
        getattr(message, "content", None)
    )

    if not text:

        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    used_model = (
        getattr(response, "model", None)
        or OPENROUTER_MODEL
    )

    logger.info(
        "AI Router | OpenRouter response received | model=%s",
        used_model,
    )

    return text


# ============================================================
# MAIN ROUTER
# ============================================================

def generate_answer(prompt: str) -> str:
    """
    Основной AI Router.

    Логика:

    1. Если Gemini доступен:
       → сначала пробуем Gemini.

    2. Если Gemini возвращает quota/rate-limit:
       → временно отключаем Gemini.
       → сразу переключаемся на OpenRouter.

    3. Пока Gemini находится в cooldown:
       → запросы к Gemini вообще не выполняются.
       → сразу используется OpenRouter.

    4. После окончания cooldown:
       → следующая генерация снова попробует Gemini.

    5. Если OpenRouter также недоступен:
       → возвращается ошибка обоих провайдеров.
    """

    prompt = _validate_prompt(prompt)

    gemini_error: Optional[Exception] = None

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

            # ------------------------------------------------
            # Если ошибка связана с квотой или rate limit
            # ------------------------------------------------

            if _is_gemini_quota_error(exc):

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
            "Gemini failed and OpenRouter is not configured. "
            f"Gemini error: {gemini_error}"
        ) from gemini_error

    try:

        return generate_with_openrouter(
            prompt
        )

    except Exception as exc:

        logger.error(
            "AI Router | OpenRouter failed | error=%s",
            exc,
            exc_info=True,
        )

        # ----------------------------------------------------
        # Оба провайдера не сработали
        # ----------------------------------------------------

        raise RuntimeError(
            "Both AI providers failed.\n"
            f"Gemini: {gemini_error}\n"
            f"OpenRouter: {exc}"
        ) from exc
```
