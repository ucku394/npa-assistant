"""
Telegram bot for Belarus occupational / industrial safety questions.

Architecture:
Telegram
    -> local multilingual-e5-small query embedding
    -> Supabase vector search
    -> Gemini reranking
    -> Gemini answer generation
    -> OpenRouter fallback
    -> Telegram

Photo:
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
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_MESSAGE_LIMIT,
)
from prompts import LEGAL_ASSISTANT_PROMPT, VISION_ANALYSIS_PROMPT
from rag import retrieve_context, get_source_references


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG VALIDATION
# ============================================================

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN is not configured."
    )

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured."
    )


# ============================================================
# SUPABASE
# ============================================================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)


# ============================================================
# DEEPSEEK VISION CLIENT
# ============================================================

deepseek_client = (
    OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    if DEEPSEEK_API_KEY
    else None
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_whitespace(text: str) -> str:
    text = str(text or "")

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
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


def clean_ai_markup(text: str) -> str:
    """
    Removes common HTML accidentally emitted by an AI model.

    The actual Telegram HTML is created later by
    to_telegram_html().
    """

    text = str(text or "")

    replacements = {
        "<br>": "\n",
        "<br/>": "\n",
        "<br />": "\n",
        "</p>": "\n\n",
        "<p>": "",
        "<strong>": "",
        "</strong>": "",
        "<b>": "",
        "</b>": "",
        "<i>": "",
        "</i>": "",
        "<em>": "",
        "</em>": "",
    }

    for old, new in replacements.items():
        text = re.sub(
            re.escape(old),
            new,
            text,
            flags=re.IGNORECASE,
        )

    # Remove code fences but keep their content.
    text = re.sub(
        r"```(?:html|markdown|text)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    )

    return normalize_whitespace(text)


def ensure_numbered_list_spacing(text: str) -> str:
    """
    Makes Telegram lists readable:

    1. item

    2. item
    """

    text = normalize_whitespace(text)

    # Normalize numbered lists.
    # Do not add excessive spacing inside decimal numbers.
    text = re.sub(
        r"(?m)(^|\n)(\s*)(\d{1,2})[.)]\s+",
        r"\1\2\3. ",
        text,
    )

    # Add an empty line between numbered items.
    text = re.sub(
        r"(?m)([^\n])\n(\s*\d{1,2}\.\s+)",
        r"\1\n\n\2",
        text,
    )

    return text


def to_telegram_html(text: str) -> str:
    """
    Safe Markdown-ish -> Telegram HTML conversion.

    Important:
    HTML is escaped BEFORE custom Telegram tags are inserted.

    Therefore AI-generated HTML cannot become raw Telegram HTML.
    """

    text = clean_ai_markup(text)

    text = ensure_numbered_list_spacing(
        text
    )

    # Escape all user/AI HTML first.
    text = html.escape(
        text,
        quote=False,
    )

    # --------------------------------------------------------
    # Markdown headings
    # --------------------------------------------------------

    text = re.sub(
        r"(?m)^\s*#{1,6}\s*(.+?)\s*$",
        r"<b>\1</b>",
        text,
    )

    # --------------------------------------------------------
    # Bold
    # --------------------------------------------------------

    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text,
        flags=re.DOTALL,
    )

    # --------------------------------------------------------
    # Bullets
    # --------------------------------------------------------

    text = re.sub(
        r"(?m)^\s*[-*]\s+",
        "• ",
        text,
    )

    # --------------------------------------------------------
    # Horizontal rules
    # --------------------------------------------------------

    text = re.sub(
        r"(?m)^\s*([-_])(?:\s*\1){2,}\s*$",
        "────────",
        text,
    )

    # --------------------------------------------------------
    # Single-star italic
    # --------------------------------------------------------

    text = re.sub(
        r"(?<!\*)\*([^*\n]+?)\*(?!\*)",
        r"<i>\1</i>",
        text,
    )

    return text.strip()


# ============================================================
# TELEGRAM MESSAGE SPLITTING
# ============================================================

def split_text_smart(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:

    text = str(text or "").strip()

    if len(text) <= limit:
        return [text]

    parts = []

    remaining = text

    while len(remaining) > limit:

        # Prefer paragraph boundary.
        cut = remaining.rfind(
            "\n\n",
            0,
            limit,
        )

        # Then normal line boundary.
        if cut < limit // 2:
            cut = remaining.rfind(
                "\n",
                0,
                limit,
            )

        # Then word boundary.
        if cut < limit // 2:
            cut = remaining.rfind(
                " ",
                0,
                limit,
            )

        # Last resort.
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

    for chunk in split_text_smart(text):

        if use_html:

            rendered = to_telegram_html(
                chunk
            )

            try:

                await update.effective_message.reply_text(
                    rendered,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

                continue

            except Exception:

                logger.exception(
                    "Telegram HTML send failed; retrying plain text."
                )

        await update.effective_message.reply_text(
            html.unescape(
                str(chunk)
            ),
            disable_web_page_preview=True,
        )


# ============================================================
# DEBUG UPDATE
# ============================================================

async def debug_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.info(
        "TELEGRAM UPDATE | update_id=%s | user_id=%s | chat_id=%s | "
        "text=%r | photo=%s",
        update.update_id,
        (
            update.effective_user.id
            if update.effective_user
            else None
        ),
        (
            update.effective_chat.id
            if update.effective_chat
            else None
        ),
        (
            update.effective_message.text
            if update.effective_message
            else None
        ),
        bool(
            update.effective_message
            and update.effective_message.photo
        ),
    )


# ============================================================
# /START
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
# CONTEXT LOG
# ============================================================

def build_context_for_log(
    chunks: list[dict],
) -> str:

    return "\n".join(
        str(
            chunk.get("doc_name")
            or "НПА"
        )
        for chunk in chunks
    )


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

    try:

        # ----------------------------------------------------
        # Show typing indicator.
        # ----------------------------------------------------

        await update.effective_chat.send_action(
            ChatAction.TYPING
        )

        # ----------------------------------------------------
        # RAG
        #
        # ВАЖНО:
        # embedding НЕ вызывается здесь отдельно.
        #
        # retrieve_context() сам выполняет:
        #
        # question
        #    ↓
        # E5 embedding
        #    ↓
        # Supabase
        #    ↓
        # Gemini reranking
        #
        # Это устраняет двойную векторизацию.
        # ----------------------------------------------------

        rag_result = await retrieve_context(
            question,
            supabase,
        )

        chunks = rag_result[
            "chunks"
        ]

        npa_context = rag_result[
            "retrieved_text"
        ]

        logger.info(
            "RAG | candidates=%s | final=%s",
            rag_result[
                "candidate_count"
            ],
            rag_result[
                "final_count"
            ],
        )

        # ----------------------------------------------------
        # No relevant NPA found.
        # ----------------------------------------------------

        if (
            not rag_result["found"]
            or not npa_context
        ):

            await update.effective_message.reply_text(
                "Я не нашёл достаточно релевантных фрагментов "
                "НПА в базе, поэтому не буду придумывать "
                "нормативное требование."
            )

            return

        # ----------------------------------------------------
        # Legal assistant prompt.
        # ----------------------------------------------------

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=npa_context,
            user_query=question,
        )

        # ----------------------------------------------------
        # Generate answer.
        #
        # ai_router.py handles Gemini/OpenRouter logic.
        # ----------------------------------------------------

        answer = await asyncio.to_thread(
            generate_answer,
            prompt,
        )

        # ----------------------------------------------------
        # Clean model output.
        # ----------------------------------------------------

        answer = clean_ai_markup(
            answer
        )

        answer = ensure_numbered_list_spacing(
            answer
        )

        # ----------------------------------------------------
        # Sources.
        #
        # ONLY final RAG chunks are used.
        # ----------------------------------------------------

        source_refs = get_source_references(
            chunks
        )

        if source_refs:

            answer += (
                "\n\n📎 ИСТОЧНИКИ\n\n"
            )

            answer += "\n\n".join(
                f"• {source}"
                for source in source_refs
            )

        # ----------------------------------------------------
        # Send answer.
        # ----------------------------------------------------

        await send_long_message(
            update,
            answer,
            use_html=True,
        )

    except Exception:

        logger.exception(
            "Text handler failed."
        )

        await update.effective_message.reply_text(
            "Произошла ошибка при обработке запроса. "
            "Попробуйте ещё раз через несколько секунд."
        )


# ============================================================
# DEEPSEEK VISION RESPONSE PARSER
# ============================================================

def _extract_vision_text(
    response,
) -> str:

    if not response:
        return ""

    if not response.choices:
        return ""

    content = (
        response.choices[0]
        .message
        .content
    )

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    if isinstance(
        content,
        list,
    ):

        pieces = []

        for part in content:

            if (
                isinstance(
                    part,
                    dict,
                )
                and part.get("type")
                == "text"
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


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if (
        not update.effective_message
        or not update.effective_message.photo
    ):
        return

    # --------------------------------------------------------
    # Check DeepSeek configuration.
    # --------------------------------------------------------

    if deepseek_client is None:

        await update.effective_message.reply_text(
            "Анализ фотографий сейчас недоступен: "
            "не настроен DEEPSEEK_API_KEY."
        )

        return

    try:

        # ----------------------------------------------------
        # Typing indicator.
        # ----------------------------------------------------

        await update.effective_chat.send_action(
            ChatAction.TYPING
        )

        # ----------------------------------------------------
        # Get highest-resolution Telegram photo.
        # ----------------------------------------------------

        photo = (
            update.effective_message.photo[-1]
        )

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        # ----------------------------------------------------
        # Download photo to memory.
        # ----------------------------------------------------

        buffer = BytesIO()

        await telegram_file.download_to_memory(
            out=buffer
        )

        image_bytes = (
            buffer.getvalue()
        )

        # ----------------------------------------------------
        # File size protection.
        # ----------------------------------------------------

        if len(image_bytes) > 32 * 1024 * 1024:

            await update.effective_message.reply_text(
                "Фотография слишком большая для анализа."
            )

            return

        # ----------------------------------------------------
        # Convert image to Base64.
        # ----------------------------------------------------

        image_b64 = base64.b64encode(
            image_bytes
        ).decode(
            "ascii"
        )

        # ----------------------------------------------------
        # Optional user caption.
        # ----------------------------------------------------

        user_caption = (
            update.effective_message.caption
            or ""
        ).strip()

        # ----------------------------------------------------
        # Vision prompt.
        # ----------------------------------------------------

        prompt = VISION_ANALYSIS_PROMPT.format(
            user_caption=(
                user_caption
                or "Подпись отсутствует."
            ),
        )

        # ----------------------------------------------------
        # DeepSeek Vision.
        # ----------------------------------------------------

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
                                        f"{image_b64}"
                                    ),
                                },
                            },
                        ],
                    }
                ],

                temperature=0.1,

                max_tokens=1200,
            )
        )

        # ----------------------------------------------------
        # Extract response.
        # ----------------------------------------------------

        result = _extract_vision_text(
            response
        )

        if not result:

            raise RuntimeError(
                "Vision model returned an empty response."
            )

        # ----------------------------------------------------
        # Send result.
        # ----------------------------------------------------

        await send_long_message(
            update,
            result,
            use_html=True,
        )

    except Exception:

        logger.exception(
            "Photo handler failed."
        )

        await update.effective_message.reply_text(
            "Не удалось проанализировать фотографию. "
            "Попробуйте отправить её ещё раз."
        )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.error(
        "UNHANDLED TELEGRAM ERROR: %s",
        context.error,
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application = (
        Application
        .builder()
        .token(
            TELEGRAM_BOT_TOKEN
        )
        .build()
    )

    # --------------------------------------------------------
    # Debug handler.
    #
    # group=-100 ensures it executes before regular handlers.
    # --------------------------------------------------------

    application.add_handler(

        MessageHandler(
            filters.ALL,
            debug_update,
        ),

        group=-100,
    )

    # --------------------------------------------------------
    # /start
    # --------------------------------------------------------

    application.add_handler(

        CommandHandler(
            "start",
            start,
        )
    )

    # --------------------------------------------------------
    # Photos
    # --------------------------------------------------------

    application.add_handler(

        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    application.add_handler(

        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            text_handler,
        )
    )

    # --------------------------------------------------------
    # Global errors
    # --------------------------------------------------------

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Bot started."
    )

    # --------------------------------------------------------
    # Polling
    # --------------------------------------------------------

    application.run_polling(

        drop_pending_updates=False,

        allowed_updates=Update.ALL_TYPES,

        close_loop=False,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()