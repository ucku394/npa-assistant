"""Canonical answer classification and transport presentation.

The classification is produced by the backend once and reused by both
Telegram and the web client. Unknown states fail closed to insufficient data.
"""
from __future__ import annotations

import html
import re
from typing import Any


ANSWER_PRESENTATIONS: dict[str, dict[str, str]] = {
    "verified_npa": {
        "label": "ПОДТВЕРЖДЕНО НПА",
        "notice": "Ответ подтверждён найденными фрагментами нормативных правовых актов.",
        "css_class": "verified_npa",
    },
    "hybrid_ai": {
        "label": "НОРМЫ НПА + ОТВЕТ ИИ",
        "notice": "Нормативная часть подтверждена источниками. Пояснение ИИ не является нормативным требованием.",
        "css_class": "hybrid_ai",
    },
    "ai_fallback": {
        "label": "ПРЕДВАРИТЕЛЬНЫЙ ОТВЕТ ИИ — ПРОВЕРИТЬ ПО НПА",
        "notice": "Это ориентир для анализа ситуации, а не подтверждённое правовое заключение. Нормативные основания, условия и сроки требуют сверки по актуальным НПА.",
        "css_class": "ai_fallback",
    },
    "insufficient_evidence": {
        "label": "НЕДОСТАТОЧНО ДАННЫХ",
        "notice": "Надёжно подтвердить ответ не удалось. Неподтверждённые требования не выдаются.",
        "css_class": "insufficient_evidence",
    },
}


def build_answer_presentation(mode: str | None) -> dict[str, Any]:
    """Return the canonical, safe presentation metadata for a result mode."""
    normalized = str(mode or "").strip().lower()
    if normalized not in ANSWER_PRESENTATIONS:
        normalized = "insufficient_evidence"
    item = ANSWER_PRESENTATIONS[normalized]
    return {
        "mode": normalized,
        "label": item["label"],
        "notice": item["notice"],
        "css_class": item["css_class"],
        "ai_generated": normalized in {"hybrid_ai", "ai_fallback"},
    }


def format_telegram_answer(
    answer: str,
    sources: list[dict[str, Any]] | None,
    presentation: dict[str, Any],
) -> str:
    """Render Telegram header from the same metadata used by the web UI."""
    label = html.escape(str(presentation.get("label") or "НЕДОСТАТОЧНО ДАННЫХ"))
    notice = html.escape(str(presentation.get("notice") or ""))
    body = str(answer or "").strip()
    body = re.sub(r"^\s*⚠️\s*<b>.*?</b>\s*", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"^\s*⚠️\s*\*\*.*?\*\*\s*", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n +", "\n", body)

    result = f"🏷 <b>{label}</b>\n<i>{notice}</i>\n\n{body}".strip()
    if sources and presentation.get("mode") in {"verified_npa", "hybrid_ai"}:
        source_lines = []
        for source in sources[:6]:
            document = html.escape(str(source.get("document") or "Неизвестный НПА").strip())
            point = html.escape(str(source.get("point") or "").strip())
            source_lines.append(f"• {document}" + (f" — пункт/статья {point}" if point else ""))
        result += "\n\n📎 <b>Источники</b>\n" + "\n".join(source_lines)
    return result
