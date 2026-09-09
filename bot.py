import os
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from supabase import create_client, Client

# 1. Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 2. Загрузка переменных окружения
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([TELEGRAM_TOKEN, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Проверьте наличие всех ключей в файле .env или переменные окружения в Railway!")

# 3. Инициализация клиентов
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Актуальные рабочие модели Gemini API
EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветственное сообщение по команде /start"""
    welcome_text = (
        "Здравствуйте! Я ваш ИИ-ассистент по охране труда и промышленной безопасности.\n\n"
        "Задайте мне вопрос по законодательству РБ и нормативным актам (НПА), "
        "и я найду точные статьи и дам развернутый ответ."
    )
    await update.message.reply_text(welcome_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых вопросов пользователя"""
    user_query = update.message.text
    await update.message.reply_chat_action("typing")

    try:
        # 1. Векторизация запроса через Gemini API
        emb_response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=user_query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            ),
        )
        query_vector = emb_response.embedding.values

        # 2. Поиск релевантных чанков в Supabase
        rpc_response = supabase.rpc(
            "match_npa_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.3,
                "match_count": 4
            }
        ).execute()

        context_chunks = rpc_response.data

        # 3. Сборка контекста из найденных фрагментов
        if context_chunks:
            retrieved_text = "\n\n---\n\n".join(
                [f"Источник: {c.get('doc_name', 'НПА')}, ст./п. {c.get('point_num', '-')}\nТекст: {c.get('content', '')}" for c in context_chunks]
            )
        else:
            retrieved_text = "Релевантные нормативные акты в базе не найдены."

        # 4. Формирование инструкции для Gemini
        prompt = f"""Ты — квалифицированный эксперт и консультант по охране труда и промышленной безопасности Беларуси.
Ответь на вопрос пользователя, строго опираясь на предоставленный ниже контекст из нормативных правовых актов (НПА).

Правила ответа:
1. Обязательно ссылайся на конкретные статьи, пункты и названия документов из контекста.
2. Ответ должен быть точным, структурированным и профессиональным.
3. Если в контексте нет прямого ответа, честно скажи об этом.

Контекст из базы НПА:
{retrieved_text}

Вопрос пользователя:
{user_query}
"""

        # 5. Генерация ответа через gemini-2.5-flash с защитой от сбоев 503 (высокая нагрузка)
        response = None
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model=CHAT_MODEL,
                    contents=prompt,
                )
                break
            except Exception as gen_err:
                if "503" in str(gen_err) or "UNAVAILABLE" in str(gen_err):
                    time.sleep(2)  # Пауза перед повторной попыткой при перегрузке
                else:
                    raise gen_err

        if response and response.text:
            await update.message.reply_text(response.text)
        else:
            await update.message.reply_text("Сервис временного перегружен. Пожалуйста, повторите вопрос через несколько секунд.")

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при поиске ответа. Попробуйте сформулировать вопрос иначе.")


def main():
    """Запуск Telegram-бота"""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот по охране труда запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()