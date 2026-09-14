import logging
import asyncio
import io
import base64

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from openai import OpenAI
from google import genai
from supabase import create_client, Client

from config import (
    TELEGRAM_TOKEN,
    GEMINI_API_KEY,
    DEEPSEEK_API_KEY,
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    VISION_MODEL,
    MAX_IMAGE_SIZE_MB,
    MAX_VISION_TOKENS,
    MAX_CHAT_TOKENS,
    TELEGRAM_MESSAGE_LIMIT,
    validate_config,
)

from prompts import VISION_ANALYSIS_PROMPT
from rag import retrieve_context, get_source_names


# ============================================================
# 1. ЛОГИРОВАНИЕ
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. ПРОВЕРКА КОНФИГУРАЦИИ
# ============================================================

validate_config()


# ============================================================
# 3. ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
)


# DeepSeek нужен только для анализа фотографий.
deepseek_client = None

if DEEPSEEK_API_KEY:
    deepseek_client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


# ============================================================
# 4. ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ============================================================

app = None


# ============================================================
# 5. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

async def send_long_message(
    message,
    text: str,
    parse_mode=None,
):
    """
    Telegram ограничивает размер одного сообщения.
    Поэтому длинный ответ разбиваем на части.
    """

    if not text:
        return

    limit = TELEGRAM_MESSAGE_LIMIT

    chunks = [
        text[i:i + limit]
        for i in range(0, len(text), limit)
    ]

    for chunk in chunks:

        try:
            if parse_mode:
                await message.reply_text(
                    chunk,
                    parse_mode=parse_mode,
                )
            else:
                await message.reply_text(chunk)

        except Exception as error:

            logger.warning(
                "Ошибка отправки сообщения: %s",
                error,
            )

            # Если Markdown сломался —
            # отправляем обычным текстом.
            try:
                await message.reply_text(chunk)

            except Exception:
                logger.exception(
                    "Не удалось отправить сообщение Telegram"
                )


# ============================================================
# 6. /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Приветственное сообщение.
    """

    welcome_text = (
        "Здравствуйте! 👋\n\n"
        "Я — ИИ-ассистент по охране труда, "
        "промышленной и пожарной безопасности "
        "в Республике Беларусь.\n\n"

        "📚 Я могу:\n"
        "• отвечать на вопросы по НПА;\n"
        "• искать нормативное обоснование в базе;\n"
        "• анализировать фотографии рабочих мест;\n"
        "• выявлять потенциально опасные факторы;\n"
        "• помогать специалисту по охране труда "
        "проводить предварительную проверку.\n\n"

        "⚠️ Нормативные ответы формируются "
        "на основании доступной базы НПА Республики Беларусь.\n\n"

        "Просто задайте вопрос или отправьте фотографию."
    )

    await update.message.reply_text(welcome_text)


# ============================================================
# 7. АНАЛИЗ ФОТОГРАФИИ
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Временная версия Vision-модуля.

    На следующем этапе будет вынесена
    в отдельный vision.py.
    """

    if deepseek_client is None:

        await update.message.reply_text(
            "⚠️ Анализ фотографий сейчас недоступен: "
            "не настроен DEEPSEEK_API_KEY."
        )

        return

    try:

        await update.message.reply_chat_action(
            "upload_photo"
        )

        # ----------------------------------------------------
        # Получаем фотографию максимального качества
        # ----------------------------------------------------

        if not update.message.photo:

            await update.message.reply_text(
                "Не удалось получить фотографию."
            )

            return

        photo = update.message.photo[-1]

        tg_file = await context.bot.get_file(
            photo.file_id
        )

        # ----------------------------------------------------
        # Скачиваем фотографию в память
        # ----------------------------------------------------

        photo_buffer = io.BytesIO()

        await tg_file.download_to_memory(
            photo_buffer
        )

        image_bytes = photo_buffer.getvalue()

        # ----------------------------------------------------
        # Проверяем размер
        # ----------------------------------------------------

        max_size = (
            MAX_IMAGE_SIZE_MB
            * 1024
            * 1024
        )

        if len(image_bytes) > max_size:

            await update.message.reply_text(
                f"⚠️ Фотография слишком большая.\n\n"
                f"Максимальный размер: "
                f"{MAX_IMAGE_SIZE_MB} МБ."
            )

            return

        # ----------------------------------------------------
        # Комментарий пользователя
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
        # Формируем Vision prompt
        # ----------------------------------------------------

        prompt = VISION_ANALYSIS_PROMPT.format(
            user_caption=(
                user_caption
                if user_caption
                else "не указан"
            )
        )

        # ----------------------------------------------------
        # Запрос DeepSeek
        # ----------------------------------------------------

        def call_deepseek():

            return deepseek_client.chat.completions.create(
                model=VISION_MODEL,

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
                                    )
                                },
                            },

                        ],
                    }
                ],

                max_tokens=MAX_VISION_TOKENS,

                temperature=0.1,
            )

        response = await asyncio.to_thread(
            call_deepseek
        )

        # ----------------------------------------------------
        # Получаем результат
        # ----------------------------------------------------

        result = None

        if response and response.choices:

            result = (
                response
                .choices[0]
                .message
                .content
            )

        if not result:

            result = (
                "Не удалось получить результат "
                "визуального анализа."
            )

        # ----------------------------------------------------
        # Отправляем результат
        # ----------------------------------------------------

        await send_long_message(
            update.message,
            result,
        )

    except Exception as error:

        error_text = str(error)

        logger.error(
            "Ошибка Vision: %s",
            error,
            exc_info=True,
        )

        # ----------------------------------------------------
        # Ошибка API / баланс
        # ----------------------------------------------------

        if (
            "429" in error_text
            or "insufficient" in error_text.lower()
            or "balance" in error_text.lower()
        ):

            await update.message.reply_text(
                "⚠️ DeepSeek API сообщил "
                "об ограничении запроса или баланса.\n\n"
                "Проверьте DEEPSEEK_API_KEY "
                "и доступность API DeepSeek."
            )

            return

        # ----------------------------------------------------
        # Ошибка авторизации
        # ----------------------------------------------------

        if (
            "401" in error_text
            or "403" in error_text
        ):

            await update.message.reply_text(
                "⚠️ DeepSeek API не принял ключ доступа.\n\n"
                "Проверьте DEEPSEEK_API_KEY."
            )

            return

        # ----------------------------------------------------
        # Остальные ошибки
        # ----------------------------------------------------

        await update.message.reply_text(
            "❌ Не удалось выполнить "
            "визуальный анализ.\n\n"
            "Попробуйте отправить фотографию ещё раз."
        )


# ============================================================
# 8. ОБРАБОТКА ТЕКСТОВОГО ВОПРОСА
# ============================================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Основной текстовый режим.

    Архитектура:

    Пользователь
        ↓
    retrieve_context()
        ↓
    Gemini Embedding
        ↓
    Supabase
        ↓
    Reranking
        ↓
    Context
        ↓
    Gemini
        ↓
    Ответ
    """

    if not update.message:
        return

    user_query = (
        update.message.text or ""
    ).strip()

    if not user_query:
        return

    await update.message.reply_chat_action(
        "typing"
    )

    try:

        logger.info(
            "Новый запрос пользователя: %s",
            user_query[:300],
        )

        # ====================================================
        # 1. RAG
        # ====================================================

        rag_result = await retrieve_context(
            user_query=user_query,
            supabase=supabase,
        )

        # ----------------------------------------------------
        # Проверяем результат
        # ----------------------------------------------------

        if not rag_result.get("found"):

            await update.message.reply_text(
                "⚠️ В базе НПА не найдено "
                "достаточно релевантной информации "
                "для уверенного ответа.\n\n"

                "Я не буду придумывать нормативное "
                "обоснование.\n\n"

                "Попробуйте:\n"
                "• уточнить вопрос;\n"
                "• указать конкретную профессию;\n"
                "• указать оборудование;\n"
                "• указать вид работ;\n"
                "• назвать известный вам НПА."
            )

            return

        # ====================================================
        # 2. Получаем контекст
        # ====================================================

        retrieved_text = (
            rag_result.get(
                "retrieved_text",
                "",
            )
        )

        context_chunks = (
            rag_result.get(
                "chunks",
                [],
            )
        )

        # ====================================================
        # 3. Формируем prompt
        # ====================================================

        from prompts import LEGAL_ASSISTANT_PROMPT

        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=retrieved_text,
            user_query=user_query,
        )

        # ====================================================
        # 4. Генерация ответа
        # ====================================================

        def generate_answer():

            return gemini_client.models.generate_content(
                model=__import__(
                    "config"
                ).CHAT_MODEL,

                contents=prompt,

                config={
                    "temperature": 0.1,
                    "max_output_tokens": MAX_CHAT_TOKENS,
                },
            )

        # Gemini синхронный → выносим из event loop
        response = await asyncio.to_thread(
            generate_answer
        )

        # ====================================================
        # 5. Проверяем результат
        # ====================================================

        if not response:

            await update.message.reply_text(
                "⚠️ Сервис временно недоступен."
            )

            return

        text = getattr(
            response,
            "text",
            None,
        )

        if not text:

            await update.message.reply_text(
                "⚠️ Не удалось сформировать ответ."
            )

            return

        text = text.strip()

        # ====================================================
        # 6. Добавляем информацию о найденных источниках
        # ====================================================

        source_names = get_source_names(
            context_chunks
        )

        if source_names:

            text += (
                "\n\n📚 **Источники, использованные "
                "при поиске:**\n"
            )

            for source in source_names:

                text += (
                    f"• {source}\n"
                )

        # ====================================================
        # 7. Отправка
        # ====================================================

        await send_long_message(
            update.message,
            text,
            parse_mode="Markdown",
        )

        logger.info(
            "Ответ успешно сформирован. "
            "Найдено кандидатов: %s, "
            "итоговых чанков: %s",
            rag_result.get("candidate_count", 0),
            rag_result.get("final_count", 0),
        )

    except Exception as error:

        logger.error(
            "Ошибка обработки вопроса: %s",
            error,
            exc_info=True,
        )

        error_text = str(error)

        # ====================================================
        # Gemini quota
        # ====================================================

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED"
            in error_text
        ):

            await update.message.reply_text(
                "⚠️ Превышен лимит API Gemini.\n\n"
                "Попробуйте повторить запрос позже."
            )

            return

        # ====================================================
        # Gemini 503
        # ====================================================

        if (
            "503" in error_text
            or "UNAVAILABLE"
            in error_text
        ):

            await update.message.reply_text(
                "⚠️ Сервис ИИ временно перегружен.\n\n"
                "Повторите вопрос через несколько секунд."
            )

            return

        # ====================================================
        # Общая ошибка
        # ====================================================

        await update.message.reply_text(
            "❌ Произошла ошибка при обработке вопроса.\n\n"
            "Попробуйте сформулировать вопрос иначе."
        )


# ============================================================
# 9. ОБРАБОТКА ОШИБОК TELEGRAM
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    """
    Центральный обработчик ошибок Telegram.
    """

    logger.error(
        "Ошибка Telegram:",
        exc_info=context.error,
    )


# ============================================================
# 10. ЗАПУСК
# ============================================================

def main():

    global app

    logger.info(
        "Запуск ИИ-ассистента по охране труда..."
    )

    # --------------------------------------------------------
    # Создаём Telegram Application
    # --------------------------------------------------------

    app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    # --------------------------------------------------------
    # Команды
    # --------------------------------------------------------

    app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    # --------------------------------------------------------
    # Фото
    # --------------------------------------------------------

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo,
        )
    )

    # --------------------------------------------------------
    # Текст
    # --------------------------------------------------------

    app.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            handle_message,
        )
    )

    # --------------------------------------------------------
    # Глобальный обработчик ошибок
    # --------------------------------------------------------

    app.add_error_handler(
        error_handler
    )

    logger.info(
        "======================================"
    )

    logger.info(
        "ИИ-АССИСТЕНТ ПО ОХРАНЕ ТРУДА ЗАПУЩЕН"
    )

    logger.info(
        "RAG: ENABLED"
    )

    logger.info(
        "VISION: %s",
        "ENABLED"
        if deepseek_client
        else "DISABLED",
    )

    logger.info(
        "======================================"
    )

    # --------------------------------------------------------
    # Запускаем polling
    # --------------------------------------------------------

    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
