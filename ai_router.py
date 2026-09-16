import logging
import time
from typing import Optional
from google import genai
from openai import OpenAI
from config import CHAT_MODEL, GEMINI_API_KEY, MAX_CHAT_TOKENS, OPENROUTER_API_KEY, OPENROUTER_FALLBACK_MODEL, OPENROUTER_MODEL

logger = logging.getLogger(__name__)
_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
_openrouter = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_API_KEY else None

GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAYS = (1.5, 3.0, 6.0)

def _clean_text(value) -> str:
    return str(value).strip() if value is not None else ""

def _extract_gemini_text(response) -> str:
    text = _clean_text(getattr(response, "text", None))
    if text:
        return text
    pieces = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = _clean_text(getattr(part, "text", None))
            if part_text:
                pieces.append(part_text)
    return "\n".join(pieces).strip()

def _is_transient(error: Exception) -> bool:
    s = str(error).upper()
    return any(x in s for x in ("429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "500", "INTERNAL", "503", "UNAVAILABLE", "TIMEOUT", "DEADLINE", "TEMPORAR"))

def generate_with_gemini(prompt: str) -> str:
    if _gemini is None:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    last = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            logger.info("AI Router | Gemini attempt=%d/%d | model=%s", attempt, GEMINI_MAX_RETRIES, CHAT_MODEL)
            response = _gemini.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": MAX_CHAT_TOKENS},
            )
            text = _extract_gemini_text(response)
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            return text
        except Exception as exc:
            last = exc
            if attempt >= GEMINI_MAX_RETRIES or not _is_transient(exc):
                raise
            delay = GEMINI_RETRY_DELAYS[attempt - 1]
            logger.warning("AI Router | Gemini transient error | retry_in=%.1fs | error=%s", delay, exc)
            time.sleep(delay)
    raise RuntimeError("Gemini generation failed.") from last

def generate_with_openrouter(prompt: str) -> str:
    if _openrouter is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    model = OPENROUTER_MODEL or OPENROUTER_FALLBACK_MODEL
    if not model:
        raise RuntimeError("OpenRouter is enabled, but OPENROUTER_MODEL is not configured.")
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты профессиональный помощник по охране труда и промышленной безопасности в Республике Беларусь. Отвечай точно, нейтрально и только на основании предоставленного нормативного контекста."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": MAX_CHAT_TOKENS,
    }
    if OPENROUTER_FALLBACK_MODEL and OPENROUTER_FALLBACK_MODEL != model:
        kwargs["extra_body"] = {"models": [OPENROUTER_FALLBACK_MODEL]}
    response = _openrouter.chat.completions.create(**kwargs)
    if not response.choices:
        raise RuntimeError("OpenRouter returned no choices.")
    text = _clean_text(response.choices[0].message.content)
    if not text:
        raise RuntimeError("OpenRouter returned an empty response.")
    logger.info("AI Router | OpenRouter response received | model=%s", model)
    return text

def generate_answer(prompt: str) -> str:
    gemini_error: Optional[Exception] = None
    try:
        return generate_with_gemini(prompt)
    except Exception as exc:
        gemini_error = exc
        logger.warning("AI Router | Gemini unavailable after retries | error=%s", exc)
    if _openrouter is None:
        raise RuntimeError("AI generation temporarily unavailable. Gemini failed and OpenRouter fallback is not configured.") from gemini_error
    try:
        return generate_with_openrouter(prompt)
    except Exception as exc:
        logger.error("AI Router | OpenRouter failed | error=%s", exc)
        raise RuntimeError("Both AI providers failed to generate an answer.") from exc
