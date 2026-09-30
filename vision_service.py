"""Structured multimodal inspection for Belarus OHS/fire/industrial safety.

Переработано:
- пустой potential_findings больше не считается успехом;
- OSH-чеклист + уточняющий запрос при пустом результате;
- постобработка с пометкой needs_review;
- расширенное логирование.
"""

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List

from openai import OpenAI

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_VISION_MODEL,
    OPENROUTER_VISION_FALLBACK_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_VISION_MODEL,
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

_deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    max_retries=0,
    default_headers={"X-Title": "Belarus OHS Safety Assistant - Vision"},
) if DEEPSEEK_API_KEY else None


# --- OSH-чеклист, который подмешивается в промпт -----------------------------
OSH_CHECKLIST = """
Обязательно проверь по изображению КАЖДЫЙ пункт и, если признак виден,
добавь его в potential_findings:

1. СИЗ: защитные очки / щиток, каска, перчатки, спецобувь, спецодежда,
   защита слуха, респиратор — при работе, где они требуются.
2. Ручной инструмент: болгарка/УШМ, дрель, пила — наличие боковой рукоятки,
   штатного защитного кожуха, правильный хват (не за кожух, не за диск).
3. Электроинструмент: целостность кабеля, наличие заземления, кабель не
   на проходе/в воде, подключение через УЗО.
4. Рабочее место: захламлённость, разлитые жидкости, скользкий пол,
   посторонние предметы в зоне работы.
5. Опасные зоны: работа на высоте, движение техники (погрузчики,
   автомобили) рядом с людьми, отсутствие ограждений и сигнальных знаков.
6. Пожарная безопасность: огнетушитель в зоне сварки/резки, искры рядом
   с горючими материалами.
7. Порядок хранения: инструмент и СИЗ не на рабочих поверхностях
   оборудования, не на полу.
8. Освещение: достаточность света в рабочей зоне.

Для каждого пункта, который подтверждается визуально, верни отдельный
элемент potential_findings с полями:
- description (что именно нарушено),
- risk_level (low|medium|high|critical),
- confidence (0..1),
- visual_evidence (что видно на фото),
- verification_needed (список проверок на месте и по НПА РБ).
""".strip()


# --- Извлечение текста из ответа ---------------------------------------------
def _extract_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
        if text:
            return text
    for field in ("output_text", "text"):
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# --- Парсинг JSON ------------------------------------------------------------
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


# --- Fallback из свободного текста -------------------------------------------
def _fallback_from_text(text: str) -> Dict[str, Any]:
    """Превращает свободный текст модели в безопасные визуальные findings."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Vision model returned an empty response.")

    cleaned = re.sub(r"^\s*```(?:text|markdown)?\s*", "", raw, flags=re.I)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    return {
        "scene": cleaned[:500],
        "category": "unknown",
        "observations": [{
            "description": cleaned[:1500],
            "confidence": 0.5,
            "evidence": "Свободный текстовый ответ модели; требуется проверка специалистом.",
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
        "needs_review": True,
    }


# --- Вспомогательные нормализаторы -------------------------------------------
def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


_VALID_LEVELS = {"low", "medium", "high", "critical"}


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []
    for item in data.get("observations") or []:
        if isinstance(item, dict) and str(item.get("description") or "").strip():
            observations.append({
                "description": str(item.get("description")).strip(),
                "confidence": _confidence(item.get("confidence")),
                "evidence": str(item.get("evidence") or "").strip(),
            })

    findings: List[Dict[str, Any]] = []
    for item in data.get("potential_findings") or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        level = item.get("risk_level")
        if level not in _VALID_LEVELS:
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
        "needs_review": bool(data.get("needs_review", False)),
    }


# --- Постобработка: не даём «пустоте» уйти наверх ----------------------------
def _postprocess(result: Dict[str, Any]) -> Dict[str, Any]:
    """Добавляет findings, если модель явно «промолчала» там, где не должна.

    Логика мягкая: мы не выдумываем нарушение, а помечаем сцену как
    требующую ручной проверки, чтобы downstream не отбрасывал её молча.
    """
    scene_text = " ".join([
        str(result.get("scene") or ""),
        " ".join(o.get("description", "") for o in result.get("observations", [])),
    ]).lower()

    # Признаки ручного инструмента и человека в кадре.
    tool_markers = (
        "болгарк", "ушм", "grinder", "дрел", "пил", "сварк", "welding",
        "инструмент", "tool", "резак", "cutter",
    )
    person_markers = ("человек", "работник", "man", "worker", "person", "мужчина")

    has_tool = any(m in scene_text for m in tool_markers)
    has_person = any(m in scene_text for m in person_markers)

    if has_person and has_tool and not result["potential_findings"]:
        result["potential_findings"].append({
            "description": (
                "В кадре человек работает с ручным/электроинструментом, "
                "но модель не выделила конкретных нарушений ОТ. "
                "Сцена требует ручной проверки."
            ),
            "risk_level": "medium",
            "confidence": 0.3,
            "visual_evidence": "Человек и инструмент в рабочей зоне.",
            "verification_needed": [
                "Проверить наличие и применение СИЗ (очки/щиток, каска, перчатки).",
                "Проверить комплектность инструмента (боковая рукоятка, кожух).",
                "Проверить правильность хвата и позы работника.",
                "Проверить состояние кабеля и подключения.",
                "Сопоставить с НПА Республики Беларусь по охране труда.",
            ],
        })
        result["needs_review"] = True

    return result


# --- Один вызов модели -------------------------------------------------------
def _is_openrouter_daily_free_quota_error(exc: Exception) -> bool:
    """True, если OpenRouter сообщил об исчерпании суточной free-моделей квоты."""
    text = str(exc or "").lower()
    return any(marker in text for marker in (
        "free-models-per-day",
        "openrouter_free_tier_daily",
        "rate limit exceeded: free",
    ))


def _call_model(
    model: str,
    prompt: str,
    image_b64: str,
    mime_type: str,
    *,
    extra_user_text: str = "",
) -> str:
    """Делает запрос к модели и возвращает сырой текст ответа."""
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": prompt},
    ]
    if extra_user_text:
        content.append({"type": "text", "text": extra_user_text})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
    })

    request = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": VISION_MAX_OUTPUT_TOKENS,
    }

    raw_text = ""
    client = _deepseek_client if model.startswith("deepseek-") else _client
    if client is None:
        raise RuntimeError(f"Vision provider is not configured for model={model}")

    for attempt in range(2):
        try:
            try:
                response = client.chat.completions.create(
                    **request,
                    response_format={"type": "json_object"},
                )
            except Exception as structured_exc:
                logger.warning(
                    "VISION | structured output unavailable model=%s attempt=%s | error=%s",
                    model, attempt + 1, structured_exc,
                )
                response = client.chat.completions.create(**request)

            raw_text = _extract_text(response)
            if raw_text:
                break
            logger.warning("VISION | empty response model=%s attempt=%s", model, attempt + 1)
        except Exception as request_exc:
            if _is_openrouter_daily_free_quota_error(request_exc):
                logger.error(
                    "VISION | OpenRouter Free Tier daily quota exhausted; skip retries model=%s",
                    model,
                )
                raise
            logger.warning(
                "VISION | request failed model=%s attempt=%s | error=%s",
                model, attempt + 1, request_exc,
            )
            if attempt == 1:
                raise
        if attempt < 1:
            time.sleep(1.5)

    if not raw_text:
        raise ValueError("Vision model returned an empty response after retries.")
    return raw_text


# --- Парсинг + нормализация + постобработка одного ответа --------------------
def _interpret(model: str, raw_text: str) -> Dict[str, Any]:
    try:
        result = _normalize(_parse_json(raw_text))
    except ValueError:
        logger.warning(
            "VISION | non-JSON response model=%s | preview=%r",
            model, raw_text[:1200],
        )
        result = _normalize(_fallback_from_text(raw_text))

    result = _postprocess(result)
    logger.info(
        "VISION | model=%s category=%s findings=%s needs_review=%s",
        model, result["category"], len(result["potential_findings"]),
        result.get("needs_review"),
    )
    return result


# --- Основная функция --------------------------------------------------------
def analyze_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    user_caption: str = "",
) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    base_prompt = VISION_STRUCTURED_PROMPT.replace(
        "{user_caption}", user_caption or "не указан"
    )
    prompt_with_checklist = f"{base_prompt}\n\n{OSH_CHECKLIST}"

    # Vision: сначала прямой DeepSeek (если API-ключ настроен),
    # затем OpenRouter fallback. Не начинаем с openrouter/free:
    # его суточная free-квота не должна блокировать всю инспекцию.
    models: List[str] = []
    for model in (
        DEEPSEEK_VISION_MODEL if DEEPSEEK_API_KEY else "",
        OPENROUTER_VISION_MODEL,
        OPENROUTER_VISION_FALLBACK_MODEL,
    ):
        model = str(model or "").strip()
        if model and model not in models:
            models.append(model)

    last_error: Exception | None = None
    best_empty: Dict[str, Any] | None = None
    free_tier_exhausted = False

    for model in models:
        if free_tier_exhausted and (
            model == "openrouter/free" or model.endswith(":free")
        ):
            logger.warning(
                "VISION | skip free model after daily quota exhaustion model=%s",
                model,
            )
            continue
        try:
            logger.info("VISION | trying model=%s | provider=%s", model, "deepseek" if model.startswith("deepseek-") else "openrouter")

            # Первая попытка — расширенный промпт с OSH-чеклистом.
            raw_text = _call_model(
                model, prompt_with_checklist, image_b64, mime_type,
            )
            result = _interpret(model, raw_text)

            if result["potential_findings"]:
                return result

            # Findings пусты — пробуем уточняющий запрос к той же модели.
            logger.warning(
                "VISION | empty findings model=%s — уточняющий запрос", model,
            )
            refine_text = (
                "Ты не нашёл нарушений охраны труда. Пересмотри изображение "
                "и ОБЯЗАТЕЛЬНО проверь по чеклисту: СИЗ (очки/щиток, каска, "
                "перчатки), комплектность инструмента (боковая рукоятка, "
                "защитный кожух), хват инструмента, состояние кабеля, "
                "захламлённость, ограждение опасных зон. Если хотя бы один "
                "признак виден — верни его в potential_findings."
            )
            raw_text = _call_model(
                model, prompt_with_checklist, image_b64, mime_type,
                extra_user_text=refine_text,
            )
            result = _interpret(model, raw_text)

            if result["potential_findings"]:
                return result

            # Запоминаем лучший «пустой» результат и идём к следующей модели.
            if best_empty is None or len(result["observations"]) > len(best_empty["observations"]):
                best_empty = result

        except Exception as exc:
            last_error = exc
            if _is_openrouter_daily_free_quota_error(exc):
                free_tier_exhausted = True
                logger.error(
                    "VISION | daily OpenRouter Free Tier quota exhausted; free models disabled for this request"
                )
            logger.warning("VISION | model failed=%s | provider=%s | error=%s", model, "deepseek" if model.startswith("deepseek-") else "openrouter", exc)

    # Ни одна модель не дала findings. Возвращаем лучший «пустой» результат,
    # но помечаем его как требующий ручной проверки.
    if best_empty is not None:
        if not best_empty["potential_findings"]:
            best_empty["needs_review"] = True
        return best_empty

    if free_tier_exhausted:
        raise RuntimeError(
            "Vision analysis unavailable: OpenRouter Free Tier daily quota "
            "is exhausted. Configure a non-free OPENROUTER_VISION_MODEL "
            "or OPENROUTER_VISION_FALLBACK_MODEL."
        ) from last_error

    raise RuntimeError(f"Vision analysis failed: {last_error}")
