import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from supabase import create_client, Client

# Заглушка для Render Web Service (чтобы не закрывал бесплатный процесс)
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is alive!")

def run_health_check_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Запускаем веб-сервер в отдельном потоке
threading.Thread(target=run_health_check_server, daemon=True).start()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Загрузка переменных окружения
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([TELEGRAM_TOKEN, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Проверьте наличие всех ключей в файле .env!")

# Инициализация клиентов
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

EMBEDDING_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-3.6-flash"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    welcome_text = (
        "Здравствуйте! Я ваш ИИ-ассистент по охране труда.\n\n"
        "Задайте мне вопрос по законодательству и нормативным актам (НПА), "
        "и я найду точные статьи и дам развернутый ответ."
    )
    await update.message.reply_text(welcome_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых вопросов пользователя"""
    user_query = update.message.text
    await update.message.reply_chat_action("typing")

    try:
        # 1. Получаем вектор для вопроса пользователя
        emb_response = gemini_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=user_query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            ),
        )
        query_vector = emb_response.embedding.values

        # 2. Ищем похожие статьи в Supabase (RPC match_npa_chunks)
        rpc_response = supabase.rpc(
            "match_npa_chunks",
            {
                "query_embedding": query_vector,
                "match_threshold": 0.3,
                "match_count": 4
            }
        ).execute()

        context_chunks = rpc_response.data

        # 3. Формируем контекст из найденных документов
        if context_chunks:
            retrieved_text = "\n\n---\n\n".join(
                [f"Источник: {c['doc_name']}, ст./п. {c['point_num']}\nТекст: {c['content']}" for c in context_chunks]
            )
        else:
            retrieved_text = "Релевантные нормативные акты в базе не найдены."

        # 4. Формируем системный промпт для Gemini
        prompt = f"""Ты — квалифицированный эксперт и консультант по охране труда и промышленной безопасности.
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

        # 5. Генерируем ответ
        response = gemini_client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )

        await update.message.reply_text(response.text)

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text("Произошла ошибка при поиске ответа. Попробуйте сформулировать вопрос иначе.")

def main():
    """Запуск бота"""
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(" Бот по охране труда запущен!")
    app.run_polling()

if __name__ == "__main__":
    main()