"""
AI text generation router.

Primary:
    Gemini

Fallback:
    OpenRouter
"""

import logging

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
# CLIENTS
# ============================================================

_gemini = (
    genai.Client(
        api_key=GEMINI_API_KEY
    )
    if GEMINI_API_KEY
    else None
)


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

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# GEMINI
# ============================================================

def generate_with_gemini(
    prompt: str,
) -> str:

    if _gemini is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not prompt.strip():
        raise ValueError(
            "Prompt is empty."
        )

    logger.info(
        "GEMINI | request started | model=%s",
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
        "GEMINI | response received | chars=%s",
        len(text),
    )

    return text


# ============================================================
# OPENROUTER
# ============================================================

def generate_with_openrouter(
    prompt: str,
) -> str:

    if _openrouter is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    if not OPENROUTER_MODEL:
        raise RuntimeError(
            "OPENROUTER_MODEL is not configured."
        )

    if not prompt.strip():
        raise ValueError(
            "Prompt is empty."
        )

    logger.info(
        "OPENROUTER | request started | model=%s",
        OPENROUTER_MODEL,
    )

    kwargs = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты профессиональный помощник по охране труда "
                    "и промышленной безопасности. "
                    "Отвечай точно, нейтрально и по существу."
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

    if OPENROUTER_FALLBACK_MODEL:

        kwargs["extra_body"] = {
            "models": [
                OPENROUTER_FALLBACK_MODEL
            ]
        }

    response = (
        _openrouter
        .chat
        .completions
        .create(**kwargs)
    )

    if not response.choices:
        raise RuntimeError(
            "OpenRouter returned no choices."
        )

    text = _clean_text(
        response.choices[0]
        .message
        .content
    )

    if not text:
        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    logger.info(
        "OPENROUTER | response received | chars=%s",
        len(text),
    )

    return text


# ============================================================
# MAIN ROUTER
# ============================================================

def generate_answer(
    prompt: str,
) -> str:

    gemini_error = None

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    try:

        return generate_with_gemini(
            prompt
        )

    except Exception as exc:

        gemini_error = exc

        logger.exception(
            "GEMINI | failed, switching to OpenRouter"
        )

    # --------------------------------------------------------
    # OPENROUTER
    # --------------------------------------------------------

    if _openrouter is None:

        raise RuntimeError(
            "Gemini failed and OpenRouter is not configured: "
            f"{gemini_error}"
        ) from gemini_error

    try:

        return generate_with_openrouter(
            prompt
        )

    except Exception as exc:

        logger.exception(
            "OPENROUTER | failed"
        )

        raise RuntimeError(
            "Both AI providers failed. "
            f"Gemini: {gemini_error}; "
            f"OpenRouter: {exc}"
        ) from exc
