"""Telegram bot for Belarus occupational / industrial safety questions."""

import asyncio
import base64
import html
import logging
import re
from io import BytesIO

from openai import OpenAI
from supabase import create_client
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ai_router import generate_answer
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_VISION_MODEL,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MESSAGE_LIMIT,
)
from prompts import LEGAL_ASSISTANT_PROMPT, VISION_ANALYSIS_PROMPT
from rag import retrieve_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

deepseek_client = (
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_API_KEY
    else None
)


# ============================================================
# TELEGRAM FORMATTING
# ============================================================


def normalize_whitespace(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_telegram_html(text: str) -> str:
    """Convert the model's lightweight formatting to safe Telegram HTML."""
    text = normalize_whitespace(text)
    text = html.escape(text, quote=False)

    # Section headings become visual cards/headings.
    heading_patterns = [
        r"(?m)^\s*(📌\s*КРАТКИЙ ОТВЕТ)\s*$",
        r"(?m)^\s*(📚\s*НОРМАТИВНОЕ ОБОСНОВАНИЕ)\s*$",
        r"(?m)^\s*(🔎\s*АНАЛИЗ)\s*$",
        r"(?m)^\s*(⚠️\s*ВАЖНО)\s*$",
        r"(?m)^\s*(📎\s*ИСТОЧНИКИ)\s*$",
        r"(?m)^\s*(🔎\s*ВИЗУАЛЬНЫЙ АНАЛИЗ)\s*$",
        r"(?m)^\s*(📷\s*Объект)\s*$",
        r"(?m)^\s*(⚠️\s*ОГРАНИЧЕНИЯ АНАЛИЗА)\s*$",
    ]
    for pattern in heading_patterns:
        text = re.sub(pattern, r"<b>\1</b>", text)

    # Labels in photo analysis and source lines.
    for label in (
        "Что видно:",
        "Потенциальная опасность:",
        "Риск:",
        "Проверить:",
    ):
        text = text.replace(label, f"<b>{label}</b>")

    text = re.sub(r"(?m)^\s*#{1,6}\s*(.+?)\s*$", r"<b>\1</b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    text = re.sub(r"(?m)^\s*[-_](?:\s*[-_]){2,}\s*$", "────────", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    return text.strip()


def split_text_smart(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text.strip()
    while len(remaining) > limit:
        cut = remaining.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)
    return parts


async def send_long_message(update: Update, text: str, use_html: bool = True):
    if not update.effective_message:
        return

    for chunk in split_text_smart(text):
        if use_html:
            rendered = to_telegram_html(chunk)
            try:
                await update.effective_message.reply_text(
                    rendered,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
                continue
            except Exception:
                logger.exception("HTML send failed; retrying as plain text.")

        await update.effective_message.reply_text(
            html.unescape(chunk),
            disable_web_page_preview=True,
        )


# ============================================================
# COMMANDS
# ============================================================

async def debug_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info(
        "TELEGRAM | update=%s user=%s chat=%s text=%r photo=%s",
        update.update_id,
        update.effective_user.id if update.effective_user else None,
        update.effective_chat.id if update.effective_chat else None,
        update.effective_message.text if update.effective_message else None,
        bool(update.effective_message and update.effective_message.photo),
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🛡 <b>Помощник по охране труда</b>\n\n"
        "Задайте вопрос по охране труда, промышленной или пожарной безопасности.\n\n"
        "📷 Можно также отправить фотографию рабочего места — я проведу визуальный анализ."
    )
    await update.effective_message.reply_text(text, parse_mode="HTML")


# ============================================================
# SOURCES / ANSWER
# ============================================================


def build_sources(chunks: list[dict]) -> str:
    sources = []
    seen = set()

    for chunk in chunks:
        doc = str(chunk.get("doc_name") or "НПА").strip()
        point = str(
            chunk.get("point_num")
            or chunk.get("article")
            or chunk.get("section")
            or ""
        ).strip()
        key = (doc, point)
        if key in seen:
            continue
        seen.add(key)

        if point:
            sources.append(f"• <b>{doc}</b> — {point}")
        else:
            sources.append(f"• <b>{doc}</b>")

    if not sources:
        return ""
    return "\n\n📎 ИСТОЧНИКИ\n" + "\n".join(sources)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message:
        return

    question = (update.effective_message.text or "").strip()
    if not question:
        return

    try:
        await update.effective_chat.send_action(ChatAction.TYPING)

        rag_result = await retrieve_context(question, supabase)
        chunks = rag_result["chunks"]
        npa_context = rag_result["retrieved_text"]

        if not npa_context or not chunks:
            await update.effective_message.reply_text(
                "🔎 <b>Нормативное основание не найдено</b>\n\n"
                "В доступной базе не нашлось достаточно релевантного фрагмента НПА. "
                "Я не буду придумывать требование.",
                parse_mode="HTML",
            )
            return

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=npa_context,
            user_query=question,
        )

        answer = await asyncio.to_thread(generate_answer, prompt)
        answer = normalize_whitespace(answer)
        answer += build_sources(chunks)

        await send_long_message(update, answer, use_html=True)

    except Exception:
        logger.exception("Text handler failed.")
        await update.effective_message.reply_text(
            "⚠️ Не удалось обработать запрос. Попробуйте ещё раз через несколько секунд."
        )


# ============================================================
# PHOTO / VISION
# ============================================================


def _extract_vision_text(response) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text") or ""))
        return "\n".join(pieces).strip()

    return str(content or "").strip()


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.photo:
        return

    if deepseek_client is None:
        await update.effective_message.reply_text(
            "📷 Анализ фотографий сейчас недоступен: не настроен DEEPSEEK_API_KEY."
        )
        return

    try:
        await update.effective_chat.send_action(ChatAction.TYPING)

        photo = update.effective_message.photo[-1]
        telegram_file = await context.bot.get_file(photo.file_id)

        buffer = BytesIO()
        await telegram_file.download_to_memory(out=buffer)
        image_bytes = buffer.getvalue()

        if len(image_bytes) > 32 * 1024 * 1024:
            await update.effective_message.reply_text(
                "📷 Фотография слишком большая для анализа."
            )
            return

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        caption = update.effective_message.caption or ""
        prompt = VISION_ANALYSIS_PROMPT.format(user_caption=caption)

        response = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model=DEEPSEEK_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=1400,
            )
        )

        result = _extract_vision_text(response)
        if not result:
            raise RuntimeError("Vision model returned an empty response.")

        await send_long_message(update, result, use_html=True)

    except Exception:
        logger.exception("Photo handler failed.")
        await update.effective_message.reply_text(
            "⚠️ Не удалось проанализировать фотографию. Попробуйте отправить её ещё раз."
        )


# ============================================================
# MAIN
# ============================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(
        "UNHANDLED TELEGRAM ERROR: %s",
        context.error,
        exc_info=context.error,
    )


def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(
        MessageHandler(filters.ALL, debug_update),
        group=-100,
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler)
    )
    application.add_error_handler(error_handler)

    logger.info("Bot started.")
    application.run_polling(
        drop_pending_updates=False,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
