import os
import re
import html
import logging
import atexit
import io
import base64
import asyncio

from pathlib import Path

from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from google import genai
from google.genai import types

from supabase import create_client, Client

from openai import OpenAI

from prompts import LEGAL_ASSISTANT_PROMPT

from ai_router import generate_answer


# ============================================================
# 1. НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. ЗАГРУЗКА .ENV
# ============================================================

script_dir = Path(__file__).parent

env_path = script_dir / ".env"

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# 3. КЛЮЧИ
# ============================================================

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)


# ============================================================
# 4. ПРОВЕРКА КЛЮЧЕЙ
# ============================================================

if not all([
    TELEGRAM_TOKEN,
    GEMINI_API_KEY,
    DEEPSEEK_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
]):

    raise ValueError(
        "Проверьте наличие всех ключей "
        "в .env или переменные окружения Railway!"
    )


# ============================================================
# 5. ИНИЦИАЛИЗАЦИЯ GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 6. ИНИЦИАЛИЗАЦИЯ DEEPSEEK
# ============================================================

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 7. SUPABASE
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# 8. МОДЕЛИ
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

DEEPSEEK_VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "deepseek-v4-flash-vision-exp"
)


# ============================================================
# 9. TELEGRAM
# ============================================================

TELEGRAM_MESSAGE_LIMIT = 4000


# ============================================================
# 10. ГЛОБАЛЬНАЯ ПЕРЕМЕННАЯ APPLICATION
# ============================================================

app = None


# ============================================================
# ФОРМАТИРОВАНИЕ TELEGRAM
# ============================================================

_BOLD_PATTERN = re.compile(
    r"\*\*(.+?)\*\*",
    re.DOTALL
)


_HEADER_PATTERN = re.compile(
    r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$",
    re.MULTILINE
)


_BULLET_PATTERN = re.compile(
    r"^([ \t]*)[*\-][ \t]+",
    re.MULTILINE
)


_HR_PATTERN = re.compile(
    r"^[ \t]*-{3,}[ \t]*$",
    re.MULTILINE
)


_ITALIC_PATTERN = re.compile(
    r"\*(.+?)\*"
)


def to_telegram_html(text: str) -> str:

    """
    Безопасная конвертация Markdown-подобной
    разметки в Telegram HTML.
    """

    escaped = html.escape(
        text,
        quote=False
    )

    escaped = _HR_PATTERN.sub(
        "",
        escaped
    )

    escaped = _HEADER_PATTERN.sub(
        r"<b>\1</b>",
        escaped
    )

    escaped = _BULLET_PATTERN.sub(
        r"\1• ",
        escaped
    )

    escaped = _BOLD_PATTERN.sub(
        r"<b>\1</b>",
        escaped
    )

    escaped = _ITALIC_PATTERN.sub(
        r"<i>\1</i>",
        escaped
    )

    return escaped


# ============================================================
# РАЗБИВКА ДЛИННОГО ТЕКСТА
# ============================================================

def split_text_smart(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT
) -> list[str]:

    if len(text) <= limit:
        return [text]

    parts = []

    remaining = text

    while len(remaining) > limit:

        cut = remaining.rfind(
            "\n\n",
            0,
            limit
        )

        if cut == -1:

            cut = remaining.rfind(
                "\n",
                0,
                limit
            )

        if cut == -1:

            cut = remaining.rfind(
                " ",
                0,
                limit
            )

        if cut == -1:

            cut = limit

        parts.append(
            remaining[:cut].rstrip()
        )

        remaining = remaining[
            cut:
        ].lstrip()

    if remaining:
        parts.append(
            remaining
        )

    return parts


# ============================================================
# ОТПРАВКА ДЛИННОГО СООБЩЕНИЯ
# ============================================================

async def send_long_message(
    update: Update,
    text: str,
    use_html: bool = True
):

    chunks = split_text_smart(
        text,
        TELEGRAM_MESSAGE_LIMIT
    )

    for chunk in chunks:

        formatted = (
            to_telegram_html(chunk)
            if use_html
            else chunk
        )

        try:

            await update.message.reply_text(
                formatted,
                parse_mode=(
                    "HTML"
                    if use_html
                    else None
                )
            )

        except Exception as parse_err:

            logger.warning(
                f"Ошибка HTML-парсинга Telegram: "
                f"{parse_err}. "
                f"Отправляем обычным текстом."
            )

            await update.message.reply_text(
                chunk
            )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    welcome_text = (
        "Здравствуйте! Я ваш ИИ-ассистент "
        "по охране труда и промышленной безопасности.\n\n"
        "Задайте мне вопрос по законодательству РБ "
        "и нормативным актам (НПА), "
        "и я найду релевантные нормативные положения "
        "и дам развернутый ответ."
    )

    await update.message.reply_text(
        welcome_text
    )


# ============================================================
# АНАЛИЗ ФОТО
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    """
    Анализ фотографии через DeepSeek Vision.
    OpenRouter здесь НЕ используется.
    """

    try:

        await update.message.reply_chat_action(
            "upload_photo"
        )


        # ----------------------------------------------------
        # Получаем фотографию
        # ----------------------------------------------------

        photo = update.message.photo[-1]

        tg_file = await context.bot.get_file(
            photo.file_id
        )


        # ----------------------------------------------------
        # Загружаем в память
        # ----------------------------------------------------

        photo_buffer = io.BytesIO()

        await tg_file.download_to_memory(
            photo_buffer
        )

        image_bytes = photo_buffer.getvalue()


        # ----------------------------------------------------
        # Проверка размера
        # ----------------------------------------------------

        if len(image_bytes) > 32 * 1024 * 1024:

            await update.message.reply_text(
                "⚠️ Фотография слишком большая "
                "для анализа. "
                "Отправьте изображение меньшего размера."
            )

            return


        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        user_caption = (
            update.message.caption or ""
        ).strip()


        # ----------------------------------------------------
        # Base64
        # ----------------------------------------------------

        image_b64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")


        # ----------------------------------------------------
        # PROMPT VISION
        # ----------------------------------------------------

        vision_prompt = f"""

Ты — эксперт по охране труда,
промышленной и пожарной безопасности
в Республике Беларусь
с большим практическим опытом.

Тебе передана фотография производственного объекта,
рабочего места, оборудования или территории.

Твоя задача — провести ТОЛЬКО ВИЗУАЛЬНЫЙ АНАЛИЗ фотографии.

ВАЖНЕЙШЕЕ ПРАВИЛО:

Не утверждай, что действие или объект является
нарушением законодательства, если это невозможно
установить только по фотографии.

Не придумывай номера пунктов НПА, документы,
размеры, характеристики оборудования
или обстоятельства, которых на фото не видно.

Разделяй:

1. что ДОСТОВЕРНО ВИДНО на фотографии;
2. что МОЖЕТ СВИДЕТЕЛЬСТВОВАТЬ
   о потенциальном нарушении;
3. что НЕВОЗМОЖНО определить по фотографии.

Проверь, насколько это возможно:

- СИЗ работников;
- ограждения опасных зон;
- состояние оборудования;
- электрические кабели;
- электрооборудование;
- проходы;
- проезды;
- лестницы;
- ограждения;
- порядок и складирование материалов;
- наличие потенциального падения предметов;
- пожарную безопасность;
- блокировки;
- защитные устройства;
- транспорт;
- движение техники;
- очевидные опасные факторы;
- другие явно видимые небезопасные условия.

Для каждого потенциального нарушения укажи:

• Что видно;
• Почему это потенциально опасно;
• Уровень риска:
  🔴 высокий / 🟠 средний / 🟡 низкий;
• Что необходимо дополнительно проверить.

Не ставь окончательный юридический диагноз
только на основании фотографии.

ФОРМАТ ОТВЕТА:

🔎 ВИЗУАЛЬНЫЙ АНАЛИЗ

Если явных проблем не видно:

🟢 Явных нарушений по фотографии не обнаружено.

Затем укажи, что всё равно невозможно
проверить визуально.

Если проблемы обнаружены:

🔴 1. [краткое название]

Что видно: ...

Риск: ...

Почему требует внимания: ...

Проверить: ...

🟠 2. ...

В конце:

⚠️ ОГРАНИЧЕНИЯ АНАЛИЗА

Укажи 1–3 наиболее важных обстоятельства,
которые невозможно определить по фотографии
и которые могут изменить вывод.

Не ссылайся на конкретные НПА в этом режиме.

Нормативное обоснование будет выполняться
отдельным этапом через базу НПА.

Подпись/комментарий пользователя к фото:

{user_caption if user_caption else "не указан"}

"""


        # ----------------------------------------------------
        # DEEPSEEK VISION
        # ----------------------------------------------------

        def call_deepseek():

            return deepseek_client.chat.completions.create(

                model=DEEPSEEK_VISION_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": [

                            {
                                "type": "text",
                                "text": vision_prompt
                            },

                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":
                                        f"data:image/jpeg;base64,{image_b64}"
                                }
                            }

                        ]
                    }
                ],

                max_tokens=1800,

                temperature=0.1
            )


        response = await asyncio.to_thread(
            call_deepseek
        )


        # ----------------------------------------------------
        # Получение результата
        # ----------------------------------------------------

        result = (
            response.choices[0].message.content
            if response.choices
            else None
        )


        if not result:

            result = (
                "Не удалось получить результат "
                "визуального анализа."
            )


        # ----------------------------------------------------
        # Отправка
        # ----------------------------------------------------

        for chunk in split_text_smart(
            result,
            TELEGRAM_MESSAGE_LIMIT
        ):

            await update.message.reply_text(
                chunk
            )


    except Exception as e:

        error_text = str(e)

        logger.error(
            f"Ошибка DeepSeek Vision: {e}",
            exc_info=True
        )


        if (
            "429" in error_text
            or "insufficient" in error_text.lower()
            or "balance" in error_text.lower()
        ):

            await update.message.reply_text(

                "⚠️ DeepSeek API вернул ошибку "
                "429/лимита.\n\n"
                "Проверьте баланс и доступность "
                "API-ключа DeepSeek.\n"
                "После пополнения баланса "
                "повторите отправку фотографии."
            )


        elif (
            "401" in error_text
            or "403" in error_text
        ):

            await update.message.reply_text(

                "⚠️ DeepSeek API не принял "
                "ключ доступа.\n\n"
                "Проверьте переменную "
                "DEEPSEEK_API_KEY в Railway."
            )


        else:

            await update.message.reply_text(

                "Не удалось проанализировать "
                "фотографию через DeepSeek. "
                "Попробуйте отправить изображение ещё раз."
            )


# ============================================================
# ОБРАБОТКА ТЕКСТОВОГО ВОПРОСА
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    """
    Основной RAG-поиск.

    ВАЖНО:

    Embedding пока остаётся на Gemini.

    Генерация ответа теперь идёт через ai_router:
    
    Gemini
       ↓
    ошибка / quota
       ↓
    OpenRouter
       ↓
    резервная модель
    """

    user_query = (
        update.message.text or ""
    ).strip()


    if not user_query:

        return


    await update.message.reply_chat_action(
        "typing"
    )


    try:

        # ====================================================
        # 1. EMBEDDING
        # ====================================================

        def call_embed():

            return gemini_client.models.embed_content(

                model=EMBEDDING_MODEL,

                contents=user_query,

                config=types.EmbedContentConfig(

                    task_type="RETRIEVAL_QUERY",

                    output_dimensionality=768,
                )
            )


        try:

            emb_response = await asyncio.to_thread(
                call_embed
            )


        except Exception as embed_err:

            error_text = str(
                embed_err
            )

            logger.error(
                f"Ошибка получения embedding: "
                f"{embed_err}",
                exc_info=True
            )


            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED"
                in error_text
            ):

                await update.message.reply_text(

                    "⚠️ Достигнут дневной лимит "
                    "Gemini API для эмбеддингов.\n\n"

                    "Сейчас поиск по базе НПА "
                    "не может выполнить векторизацию "
                    "запроса.\n\n"

                    "Важно: OpenRouter на этом этапе "
                    "не заменяет Gemini embedding. "
                    "Это будет отдельным этапом "
                    "настройки."
                )

            else:

                await update.message.reply_text(

                    "Не удалось обработать запрос "
                    "(ошибка сервиса эмбеддингов). "
                    "Попробуйте повторить позже."
                )

            return


        if (
            not emb_response
            or not emb_response.embeddings
        ):

            await update.message.reply_text(

                "Gemini не вернул embedding "
                "для вашего запроса."
            )

            return


        query_vector = (
            emb_response
            .embeddings[0]
            .values
        )


        # ====================================================
        # 2. ПОИСК SUPABASE
        # ====================================================

        rpc_response = supabase.rpc(

            "match_npa_chunks",

            {
                "query_embedding": query_vector,

                "match_threshold": 0.3,

                "match_count": 4
            }

        ).execute()


        context_chunks = (
            rpc_response.data
            or []
        )


        # ====================================================
        # 3. ФОРМИРОВАНИЕ КОНТЕКСТА
        # ====================================================

        if context_chunks:

            retrieved_text = (
                "\n\n---\n\n".join(

                    [
                        (
                            f"Источник: "
                            f"{c.get('doc_name', 'НПА')}, "
                            f"ст./п. "
                            f"{c.get('point_num', '-')}\n"
                            f"Текст: "
                            f"{c.get('content', '')}"
                        )

                        for c in context_chunks
                    ]
                )
            )


        else:

            retrieved_text = (
                "Релевантные нормативные акты "
                "в базе не найдены."
            )


        # ====================================================
        # 4. ФОРМИРОВАНИЕ ПРОМПТА
        # ====================================================

        prompt = LEGAL_ASSISTANT_PROMPT.format(

            retrieved_text=retrieved_text,

            user_query=user_query
        )


        # ====================================================
        # 5. AI ROUTER
        # ====================================================

        text, ai_provider = await generate_answer(
            prompt
        )


        if not text:

            await update.message.reply_text(

                "Сервис ИИ временно не вернул "
                "ответ. Попробуйте повторить вопрос."
            )

            return


        text = text.strip()


        # ====================================================
        # 6. ИСТОЧНИКИ
        # ====================================================

        if context_chunks:

            sources = sorted(

                {
                    c.get(
                        "doc_name",
                        "НПА"
                    )

                    for c in context_chunks
                }
            )


            text += (

                "\n\n📄 **Источники:** "
                + "; ".join(sources)
            )


        # ====================================================
        # 7. ОТПРАВКА
        # ====================================================

        await send_long_message(

            update,

            text,

            use_html=True
        )


        # ====================================================
        # 8. ЛОГ
        # ====================================================

        logger.info(
            f"Ответ пользователю сформирован "
            f"через {ai_provider}."
        )


    except Exception as e:

        logger.error(
            f"Ошибка при обработке запроса: {e}",
            exc_info=True
        )


        await update.message.reply_text(

            "Произошла ошибка при поиске ответа.\n\n"
            "Попробуйте сформулировать вопрос иначе "
            "или повторите запрос немного позже."
        )


# ============================================================
# CLEANUP
# ============================================================

def cleanup():

    global app


    if app:

        logger.info(
            "Остановка бота..."
        )


        try:

            app.stop()

        except Exception as e:

            logger.warning(
                f"Ошибка при остановке бота: {e}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    global app


    app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )


    # /start
    app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )


    # Фото
    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )


    # Текст
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )


    logger.info(
        "Бот по охране труда запущен!"
    )


    # Корректное завершение
    atexit.register(
        cleanup
    )


    # Запуск
    app.run_polling(

        drop_pending_updates=True,

        allowed_updates=Update.ALL_TYPES,

        close_loop=False
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
