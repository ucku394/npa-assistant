"""
Telegram bot for Belarus occupational / industrial safety questions.

Architecture:
Telegram -> local E5 query embedding -> Supabase vector search
        -> Gemini -> OpenRouter fallback -> Telegram

Photo questions:
Telegram -> DeepSeek Vision -> Telegram
"""

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
    SUPABASE_MATCH_COUNT,
    SUPABASE_MATCH_THRESHOLD,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MESSAGE_LIMIT,
)
from embedding import get_query_embedding


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured.")

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)

deepseek_client = (
    OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    if DEEPSEEK_API_KEY
    else None
)

# Avoid concurrent inference against the shared SentenceTransformer instance.
_embedding_async_lock = asyncio.Lock()


LEGAL_ASSISTANT_PROMPT = """Ты — профессиональный помощник по охране труда и
промышленной безопасности в Республике Беларусь.

Отвечай только по существу вопроса. Основой ответа являются найденные ниже
фрагменты НПА. Не придумывай требования, номера пунктов, статьи, сроки,
штрафы или обязанности, которых нет в контексте.

Правила:
1. Если контекст подтверждает ответ — объясни его простым языком.
2. Если точного ответа в контексте нет — прямо скажи, что в предоставленных
   фрагментах недостаточно данных, и не выдумывай норму.
3. Разделяй обязательное требование НПА и практическую рекомендацию.
4. Если приводишь ссылку на НПА, используй только название и пункт/статью,
   которые есть в контексте.
5. Не утверждай наличие нарушения только на основании предположения.
6. Не добавляй фиктивные источники.
7. Для нумерованных пунктов каждый пункт начинай с нового абзаца.
8. Не используй Markdown-таблицы.
9. Ответ должен быть компактным, но достаточным для практического применения.

КОНТЕКСТ НПА:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
"""


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_telegram_html(text: str) -> str:
    """
    Conservative Markdown-ish -> Telegram HTML conversion.

    We escape HTML first and only then add our own tags. This substantially
    reduces Telegram parse errors caused by raw '<', '>' or '&' in AI output.
    """
    text = normalize_whitespace(text)
    text = html.escape(text, quote=False)

    # Headers
    text = re.sub(
        r"(?m)^\s*#{1,6}\s*(.+?)\s*$",
        r"<b>\1</b>",
        text,
    )

    # Bold **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)

    # Markdown bullets -> Telegram-friendly bullet.
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)

    # Horizontal rule
    text = re.sub(r"(?m)^\s*([-_])(?:\s*\1){2,}\s*$", "────────", text)

    # Italic *text* only when not part of **
    text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<i>\1</i>", text)

    return text.strip()


def split_text_smart(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT):
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


async def send_long_message(
    update: Update,
    text: str,
    use_html: bool = True,
):
    if not update.effective_message:
        return

    chunks = split_text_smart(text)

    for chunk in chunks:
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
                logger.exception("HTML Telegram send failed; retrying plain text.")

        await update.effective_message.reply_text(
            html.unescape(chunk),
            disable_web_page_preview=True,
        )


async def debug_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.info(
        "TELEGRAM UPDATE RECEIVED | update_id=%s | user_id=%s | chat_id=%s | text=%r | photo=%s",
        update.update_id,
        update.effective_user.id if update.effective_user else None,
        update.effective_chat.id if update.effective_chat else None,
        update.effective_message.text if update.effective_message else None,
        bool(update.effective_message and update.effective_message.photo),
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "Здравствуйте! Я помощник по охране труда и промышленной безопасности "
        "в Республике Беларусь.\n\n"
        "Задайте вопрос текстом или отправьте фотографию — я помогу "
        "разобрать ситуацию."
    )


async def _match_npa(query_embedding):
    def call():
        return (
            supabase.rpc(
                "match_npa_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": SUPABASE_MATCH_THRESHOLD,
                    "match_count": SUPABASE_MATCH_COUNT,
                },
            )
            .execute()
        )

    response = await asyncio.to_thread(call)
    return response.data or []


def build_context(rows) -> str:
    blocks = []

    for idx, row in enumerate(rows, 1):
        doc_name = str(row.get("doc_name") or "НПА").strip()
        point_num = str(row.get("point_num") or "Без номера").strip()
        content = str(row.get("content") or "").strip()

        if not content:
            continue

        blocks.append(
            f"[Источник {idx}]\n"
            f"Документ: {doc_name}\n"
            f"Пункт/статья: {point_num}\n"
            f"Текст: {content}"
        )

    return "\n\n".join(blocks)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message:
        return

    question = (update.effective_message.text or "").strip()
    if not question:
        return

    try:
        await update.effective_chat.send_action(ChatAction.TYPING)

        async with _embedding_async_lock:
            query_embedding = await asyncio.to_thread(
                get_query_embedding,
                question,
            )

        rows = await _match_npa(query_embedding)
        npa_context = build_context(rows)

        if not npa_context:
            await update.effective_message.reply_text(
                "Я не нашёл достаточно релевантных фрагментов НПА в базе, "
                "поэтому не буду придумывать нормативное требование."
            )
            return

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            context=npa_context,
            question=question,
        )

        answer = await asyncio.to_thread(generate_answer, prompt)

        sources = []
        seen = set()
        for row in rows:
            doc_name = str(row.get("doc_name") or "НПА").strip()
            point_num = str(row.get("point_num") or "").strip()
            key = (doc_name, point_num)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                f"• {doc_name}" + (f" — пункт {point_num}" if point_num else "")
            )

        if sources:
            answer = (
                answer.rstrip()
                + "\n\n<b>📄 Источники:</b>\n"
                + "\n".join(sources)
            )

        await send_long_message(update, answer, use_html=True)

    except Exception:
        logger.exception("Text handler failed.")
        await update.effective_message.reply_text(
            "Произошла ошибка при обработке запроса. "
            "Попробуйте ещё раз через несколько секунд."
        )


def _extract_vision_text(response) -> str:
    if not response.choices:
        return ""
    content = response.choices[0].message.content

    if isinstance(content, str):
        return content.strip()

    # Some OpenAI-compatible APIs can return content parts.
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
            "Анализ фотографий сейчас недоступен: не настроен DEEPSEEK_API_KEY."
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
                "Фотография слишком большая для анализа."
            )
            return

        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        prompt = """Проанализируй фотографию с точки зрения охраны труда и
промышленной безопасности в Беларуси.

Опиши только то, что реально видно на изображении.
Укажи:
1. Что изображено.
2. Какие потенциально опасные факторы визуально заметны.
3. Какие меры безопасности разумно проверить или принять.
4. Что по фотографии определить невозможно.

Не утверждай, что конкретная норма НПА нарушена, если для этого недостаточно
данных. Не придумывай номера НПА, пунктов или статей.
Для нумерованных пунктов каждый пункт начинай с нового абзаца.
"""

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
                max_tokens=1200,
            )
        )

        result = _extract_vision_text(response)
        if not result:
            raise RuntimeError("Vision model returned an empty response.")

        await send_long_message(update, result, use_html=False)

    except Exception:
        logger.exception("Photo handler failed.")
        await update.effective_message.reply_text(
            "Не удалось проанализировать фотографию. Попробуйте отправить "
            "её ещё раз."
        )



async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
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
    application.add_handler(
        MessageHandler(filters.PHOTO, photo_handler)
    )
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
