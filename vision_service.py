"""Structured multimodal inspection using OpenRouter free vision models."""

import base64
import json
import logging
import re
from typing import Any, Dict

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_VISION_MODEL,
    OPENROUTER_VISION_FALLBACK_MODEL,
    VISION_MAX_OUTPUT_TOKENS,
)
from prompts import VISION_STRUCTURED_PROMPT

logger = logging.getLogger(__name__)

_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    max_retries=0,
    default_headers={
        "HTTP-Referer": "https://openrouter.ai/",
        "X-Title": "Belarus OHS Safety Assistant - Vision",
    },
) if OPENROUTER_API_KEY else None


def _extract_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content or "").strip()


def _parse_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("Vision model did not return JSON.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Vision response must be an object.")
    return value


def _fallback_from_text(text: str) -> Dict[str, Any]:
    """Convert a free-form vision response into safe visual findings.

    Some free OpenRouter vision models may ignore a JSON-only instruction.
    Keep their text as unverified visual evidence; legal verification happens
    later against the Belarus NPA RAG context.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Vision model returned an empty response.")

    cleaned = re.sub(r"^\s*\`\`\`(?:text|markdown)?\s*", "", raw, flags=re.I)
    cleaned = re.sub(r"\s*\`\`\`\s*$", "", cleaned).strip()

    return {
        "scene": cleaned[:500],
        "category": "unknown",
        "observations": [{
            "description": cleaned[:1500],
            "confidence": 0.5,
            "evidence": "Свободный текстовый ответ модели по изображению; требуется проверка специалистом.",
        }],
        "potential_findings": [{
            "description": cleaned[:1500],
            "risk_level": "medium",
            "confidence": 0.5,
            "visual_evidence": cleaned[:1500],
            "verification_needed": [
                "Проверить описанный факт непосредственно на месте.",
                "Сопоставить его с применимыми НПА Республики Беларусь.",
            ],
        }],
    }


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    observations = []
    for item in data.get("observations") or []:
        if isinstance(item, dict) and str(item.get("description") or "").strip():
            observations.append({
                "description": str(item.get("description")).strip(),
                "confidence": _confidence(item.get("confidence")),
                "evidence": str(item.get("evidence") or "").strip(),
            })

    findings = []
    for item in data.get("potential_findings") or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        level = item.get("risk_level")
        if level not in {"low", "medium", "high", "critical"}:
            level = "medium"
        findings.append({
            "description": description,
            "risk_level": level,
            "confidence": _confidence(item.get("confidence")),
            "visual_evidence": str(item.get("visual_evidence") or "").strip(),
            "verification_needed": [
                str(x).strip()
                for x in (item.get("verification_needed") or [])
                if str(x).strip()
            ],
        })

    return {
        "scene": str(data.get("scene") or "").strip(),
        "category": str(data.get("category") or "unknown").strip(),
        "observations": observations,
        "potential_findings": findings,
    }


def analyze_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    user_caption: str = "",
) -> Dict[str, Any]:
    if _client is None:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = VISION_STRUCTURED_PROMPT.replace("{user_caption}", user_caption or "не указан")

    models = []
    for model in (
        OPENROUTER_VISION_MODEL,
        OPENROUTER_VISION_FALLBACK_MODEL,
        "openrouter/free",
    ):
        model = str(model or "").strip()
        if model and model not in models:
            models.append(model)

    last_error = None
    for model in models:
        try:
            logger.info("VISION | OpenRouter model=%s", model)
            request = {
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                }],
                "temperature": 0.0,
                "max_tokens": VISION_MAX_OUTPUT_TOKENS,
            }

            try:
                response = _client.chat.completions.create(
                    **request,
                    response_format={"type": "json_object"},
                )
            except Exception as structured_exc:
                logger.warning(
                    "VISION | structured output unavailable model=%s | error=%s",
                    model,
                    structured_exc,
                )
                response = _client.chat.completions.create(**request)

            raw_text = _extract_text(response)
            try:
                result = _normalize(_parse_json(raw_text))
            except ValueError:
                logger.warning(
                    "VISION | non-JSON response model=%s | preview=%r",
                    model,
                    raw_text[:1200],
                )
                result = _fallback_from_text(raw_text)
            logger.info(
                "VISION | category=%s | findings=%s",
                result["category"],
                len(result["potential_findings"]),
            )
            return result
        except Exception as exc:
            last_error = exc
            logger.warning(
                "VISION | model failed=%s | error=%s",
                model,
                exc,
            )

    raise RuntimeError(f"Vision analysis failed: {last_error}")
