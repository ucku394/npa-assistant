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
    prompt = VISION_STRUCTURED_PROMPT.format(
        user_caption=user_caption or "не указан"
    )

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
            response = _client.chat.completions.create(
                model=model,
                messages=[{
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
                temperature=0.0,
                max_tokens=VISION_MAX_OUTPUT_TOKENS,
            )
            result = _normalize(_parse_json(_extract_text(response)))
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
