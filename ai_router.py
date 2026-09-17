"""
AI text-generation router.

Primary: Gemini
Fallback: OpenRouter

The router is independent from bot.py.
"""

import logging
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

    1. Сначала Gemini.
    2. Если Gemini недоступен — OpenRouter.
    3. Gemini при ошибке НЕ повторяется.
    """

    prompt = _validate_prompt(prompt)

    gemini_error: Optional[Exception] = None

    # ========================================================
    # 1. GEMINI
    # ========================================================

    try:

        return generate_with_gemini(
            prompt
        )

    except Exception as exc:

        gemini_error = exc

        logger.warning(
            "AI Router | Gemini unavailable | "
            "switching immediately to OpenRouter | "
            "error=%s",
            exc,
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

        raise RuntimeError(
            "Both AI providers failed.\n"
            f"Gemini: {gemini_error}\n"
            f"OpenRouter: {exc}"
        ) from exc
