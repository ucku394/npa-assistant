"""
Telegram bot for Belarus occupational / industrial safety questions.

Architecture:

TEXT:
Telegram
    ↓
local multilingual-e5-base
    ↓
Supabase vector search
    ↓
Gemini
    ↓
OpenRouter fallback
    ↓
Telegram

PHOTO:
Telegram
    ↓
DeepSeek Vision
    ↓
Telegram
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

from embedding import (
    get_query_embedding,
    warmup_model,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not configured."
    )

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not configured."
    )

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY is not configured."
    )


# ============================================================
# CLIENTS
# ============================================================

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


# ============================================================
# EMBEDDING LOCK
# ============================================================

_embedding_async_lock = asyncio.Lock()


# ============================================================
# LEGAL PROMPT
# ============================================================

LEGAL_ASSISTANT_PROMPT = """Ты — профессиональный помощник по охране труда и
промышленной безопасности в Республике Беларусь.

Отвечай только по существу вопроса.

Основой ответа являются найденные ниже фрагменты НПА.

Не придумывай:
- требования;
- номера пунктов;
- номера статей;
- сроки;
- штрафы;
- обязанности;
- документы;
- нормативные акты,

если этого нет в предоставленном контексте.

ПРАВИЛА:

1. Если контекст подтверждает ответ — объясни его простым языком.

2. Если точного ответа в контексте нет — прямо скажи, что в предоставленных
фрагментах недостаточно данных.

3. Не выдумывай нормативные требования.

4. Разделяй обязательное требование НПА и практическую рекомендацию.

5. Если приводишь ссылку на НПА, используй только название и пункт/статью,
которые реально присутствуют в контексте.

6. Не утверждай наличие нарушения, если для этого недостаточно данных.

7. Не добавляй фиктивные источники.

8. Для нумерованных списков каждый пункт начинай с нового абзаца.

9. Не используй Markdown-таблицы.

10. Ответ должен быть понятным, компактным и практически полезным.

КОНТЕКСТ НПА:

{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:

{question}
"""


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_whitespace(text: str) -> str:
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def to_telegram_html(text: str) -> str:
    text = normalize_whitespace(text)

    text = html.escape(
        text,
        quote=False,
    )

    text = re.sub(
        r"(?m)^\s*#{1,6}\s*(.+?)\s*$",
        r"<b>\1</b>",
        text,
    )

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
        flags=re.DOTALL,
    )

    text = re.sub(
        r"(?m)^\s*[-*]\s+",
        "• ",
        text,
    )

    text = re.sub(
        r"(?m)^\s*([-_])(?:\s*\1){2,}\s*$",
        "────────",
        text,
    )

    text = re.sub(
        r"(?<!\*)\*([^*\n]+?)\*(?!\*)",
        r"<i>\1</i>",
        text,
    )

    return text.strip()


def split_text_smart(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
):
    if len(text) <= limit:
        return [text]

    parts = []

    remaining = text.strip()

    while len(remaining) > limit:

        cut = remaining.rfind(
            "\n\n",
            0,
            limit,
        )

        if cut < limit // 2:
            cut = remaining.rfind(
                "\n",
                0,
                limit,
            )

        if cut < limit // 2:
            cut = remaining.rfind(
                " ",
                0,
                limit,
            )

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
                logger.exception(
                    "TELEGRAM | HTML send failed; "
                    "retrying as plain text"
                )

        await update.effective_message.reply_text(
            html.unescape(chunk),
            disable_web_page_preview=True,
        )


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_message:
        return

    await update.effective_message.reply_text(
        "Здравствуйте! Я помощник по охране труда и "
        "промышленной безопасности в Республике Беларусь.\n\n"
        "Задайте вопрос текстом или отправьте фотографию — "
        "я помогу разобрать ситуацию."
    )


# ============================================================
# SUPABASE SEARCH
# ============================================================

async def _match_npa(
    query_embedding,
):
    logger.info(
        "SUPABASE | vector search started | "
        "threshold=%s | count=%s",
        SUPABASE_MATCH_THRESHOLD,
        SUPABASE_MATCH_COUNT,
    )

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

    rows = response.data or []

    logger.info(
        "SUPABASE | vector search completed | rows=%s",
        len(rows),
    )

    return rows


def build_context(rows) -> str:
    blocks = []

    for idx, row in enumerate(rows, 1):

        doc_name = str(
            row.get("doc_name")
            or "НПА"
        ).strip()

        point_num = str(
            row.get("point_num")
            or "Без номера"
        ).strip()

        content = str(
            row.get("content")
            or ""
        ).strip()

        if not content:
            continue

        blocks.append(
            f"[Источник {idx}]\n"
            f"Документ: {doc_name}\n"
            f"Пункт/статья: {point_num}\n"
            f"Текст: {content}"
        )

    return "\n\n".join(blocks)


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_message:
        return

    question = (
        update.effective_message.text
        or ""
    ).strip()

    if not question:
        return

    logger.info(
        "TEXT | processing started | chars=%s",
        len(question),
    )

    try:

        # ----------------------------------------------------
        # TELEGRAM TYPING
        # ----------------------------------------------------

        await update.effective_chat.send_action(
            ChatAction.TYPING
        )

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        logger.info(
            "EMBEDDING | acquiring async lock"
        )

        async with _embedding_async_lock:

            logger.info(
                "EMBEDDING | creating query vector"
            )

            query_embedding = await asyncio.to_thread(
                get_query_embedding,
                question,
            )

        logger.info(
            "EMBEDDING | vector ready | dimension=%s",
            len(query_embedding),
        )

        # ----------------------------------------------------
        # SUPABASE
        # ----------------------------------------------------

        rows = await _match_npa(
            query_embedding
        )

        npa_context = build_context(rows)

        logger.info(
            "RAG | context length=%s | chunks=%s",
            len(npa_context),
            len(rows),
        )

        if not npa_context:

            logger.warning(
                "RAG | no relevant chunks found"
            )

            await update.effective_message.reply_text(
                "Я не нашёл достаточно релевантных "
                "фрагментов НПА в базе, поэтому не буду "
                "придумывать нормативное требование."
            )

            return

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            context=npa_context,
            question=question,
        )

        logger.info(
            "AI | generating answer"
        )

        # ----------------------------------------------------
        # GEMINI / OPENROUTER
        # ----------------------------------------------------

        answer = await asyncio.to_thread(
            generate_answer,
            prompt,
        )

        if not answer:
            raise RuntimeError(
                "AI returned an empty answer."
            )

        answer = answer.strip()

        logger.info(
            "AI | answer generated | chars=%s",
            len(answer),
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = []
        seen = set()

        for row in rows:

            doc_name = str(
                row.get("doc_name")
                or "НПА"
            ).strip()

            point_num = str(
                row.get("point_num")
                or ""
            ).strip()

            key = (
                doc_name,
                point_num,
            )

            if key in seen:
                continue

            seen.add(key)

            source = f"• {doc_name}"

            if point_num:
                source += (
                    f" — пункт {point_num}"
                )

            sources.append(source)

        if sources:

            answer = (
                answer.rstrip()
                + "\n\n"
                + "<b>📄 Источники:</b>\n"
                + "\n".join(sources)
            )

        # ----------------------------------------------------
        # TELEGRAM
        # ----------------------------------------------------

        logger.info(
            "TELEGRAM | sending answer"
        )

        await send_long_message(
            update,
            answer,
            use_html=True,
        )

        logger.info(
            "TEXT | processing completed"
        )

    except Exception:

        logger.exception(
            "TEXT | handler failed"
        )

        try:

            await update.effective_message.reply_text(
                "Произошла ошибка при обработке запроса.\n\n"
                "Попробуйте ещё раз через несколько секунд."
            )

        except Exception:

            logger.exception(
                "TELEGRAM | failed to send error message"
            )


# ============================================================
# DEEPSEEK VISION
# ============================================================

def _extract_vision_text(
    response,
) -> str:

    if not response.choices:
        return ""

    content = (
        response.choices[0]
        .message
        .content
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        pieces = []

        for part in content:

            if (
                isinstance(part, dict)
                and part.get("type") == "text"
            ):
                pieces.append(
                    str(
                        part.get("text")
                        or ""
                    )
                )

        return "\n".join(
            pieces
        ).strip()

    return str(
        content or ""
    ).strip()


async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if (
        not update.effective_message
        or not update.effective_message.photo
    ):
        return

    if deepseek_client is None:

        await update.effective_message.reply_text(
            "Анализ фотографий сейчас недоступен: "
            "не настроен DEEPSEEK_API_KEY."
        )

        return

    try:

        await update.effective_chat.send_action(
            ChatAction.TYPING
        )

        photo = (
            update.effective_message
            .photo[-1]
        )

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        buffer = BytesIO()

        await telegram_file.download_to_memory(
            out=buffer
        )

        image_bytes = buffer.getvalue()

        if len(image_bytes) > 32 * 1024 * 1024:

            await update.effective_message.reply_text(
                "Фотография слишком большая для анализа."
            )

            return

        image_b64 = base64.b64encode(
            image_bytes
        ).decode("ascii")

        prompt = """Проанализируй фотографию с точки зрения охраны труда и
промышленной безопасности в Беларуси.

Опиши только то, что реально видно на изображении.

Укажи:

1. Что изображено.

2. Какие потенциально опасные факторы визуально заметны.

3. Какие меры безопасности разумно проверить или принять.

4. Что по фотографии определить невозможно.

Не утверждай, что конкретная норма НПА нарушена, если для этого недостаточно
данных.

Не придумывай номера НПА, пунктов или статей.

Для нумерованных пунктов каждый пункт начинай с нового абзаца.
"""

        logger.info(
            "VISION | request started"
        )

        response = await asyncio.to_thread(
            lambda: deepseek_client.chat.completions.create(
                model=DEEPSEEK_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64,"
                                        + image_b64
                                    )
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=1200,
            )
        )

        result = _extract_vision_text(
            response
        )

        if not result:
            raise RuntimeError(
                "Vision model returned empty response."
            )

        logger.info(
            "VISION | response received | chars=%s",
            len(result),
        )

        await send_long_message(
            update,
            result,
            use_html=False,
        )

    except Exception:

        logger.exception(
            "VISION | handler failed"
        )

        await update.effective_message.reply_text(
            "Не удалось проанализировать фотографию. "
            "Попробуйте отправить её ещё раз."
        )


# ============================================================
# STARTUP
# ============================================================

def main():

    logger.info(
        "STARTUP | application initialization"
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Load E5 BEFORE Telegram polling starts.
    # --------------------------------------------------------

    try:

        warmup_model()

    except Exception:

        logger.exception(
            "STARTUP | embedding model warmup FAILED"
        )

        raise

    logger.info(
        "STARTUP | embedding model is ready"
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    application = (
        Application
        .builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            text_handler,
        )
    )

    logger.info(
        "STARTUP | bot polling starting"
    )

    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
