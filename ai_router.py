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

_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

_openrouter = (
    OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    if OPENROUTER_API_KEY
    else None
)


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _validate_prompt(prompt: str) -> str:
    prompt = _clean_text(prompt)
    if not prompt:
        raise ValueError("Prompt is empty.")
    return prompt


def generate_with_gemini(prompt: str) -> str:
    if _gemini is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    prompt = _validate_prompt(prompt)

    response = _gemini.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": MAX_CHAT_TOKENS,
        },
    )

    text = _clean_text(getattr(response, "text", None))
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    logger.info("AI Router | Gemini response received.")
    return text


def generate_with_openrouter(prompt: str) -> str:
    if _openrouter is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    if not OPENROUTER_MODEL:
        raise RuntimeError("OPENROUTER_MODEL is not configured.")

    prompt = _validate_prompt(prompt)

    kwargs = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты профессиональный помощник по охране труда, "
                    "промышленной и пожарной безопасности в Республике Беларусь. "
                    "Отвечай только на основании переданного нормативного контекста. "
                    "Не придумывай нормы, статьи, пункты, сроки, штрафы или обязанности."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_CHAT_TOKENS,
    }

    if OPENROUTER_FALLBACK_MODEL:
        kwargs["extra_body"] = {"models": [OPENROUTER_FALLBACK_MODEL]}

    response = _openrouter.chat.completions.create(**kwargs)

    if not response or not response.choices:
        raise RuntimeError("OpenRouter returned no choices.")

    text = _clean_text(response.choices[0].message.content)
    if not text:
        raise RuntimeError("OpenRouter returned an empty response.")

    used_model = getattr(response, "model", None) or OPENROUTER_MODEL
    logger.info("AI Router | OpenRouter response received | model=%s", used_model)
    return text


def generate_answer(prompt: str) -> str:
    """
    Gemini first. OpenRouter fallback on any Gemini failure.
    """
    gemini_error: Optional[Exception] = None

    try:
        return generate_with_gemini(prompt)
    except Exception as exc:
        gemini_error = exc
        logger.warning(
            "AI Router | Gemini failed; trying OpenRouter | error=%s",
            exc,
            exc_info=True,
        )

    if _openrouter is None:
        raise RuntimeError(
            f"Gemini failed and OpenRouter is not configured: {gemini_error}"
        ) from gemini_error

    try:
        return generate_with_openrouter(prompt)
    except Exception as exc:
        logger.error(
            "AI Router | OpenRouter failed | error=%s",
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "Both AI providers failed. "
            f"Gemini: {gemini_error}; OpenRouter: {exc}"
        ) from exc
