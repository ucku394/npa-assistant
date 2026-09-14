import os
import re
import html
import logging
import time
import atexit
import io
import base64
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types
from supabase import create_client, Client
from openai import OpenAI

from prompts import LEGAL_ASSISTANT_PROMPT

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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not all([TELEGRAM_TOKEN, GEMINI_API_KEY, DEEPSEEK_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Проверьте наличие всех ключей в файле .env или переменные окружения в Railway!")

# 3. Инициализация клиентов
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Модели
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"
DEEPSEEK_VISION_MODEL = "deepseek-v4-flash-vision-exp"

TELEGRAM_MESSAGE_LIMIT = 4000

# Глобальная переменная для приложения
app = None


# ============================================================
# ФОРМАТИРОВАНИЕ ДЛЯ TELEGRAM
# ============================================================

# Regex для **жирный текст** из ответа модели — конвертируем в <b>.
# Компилируем один раз на уровне модуля.
_BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def to_telegram_html(text: str) -> str:
    """
    Безопасно конвертирует ответ модели (с разметкой **bold**)
    в HTML, понятный Telegram (parse_mode="HTML").

    В отличие от parse_mode="Markdown" (legacy), HTML-режим Telegram
    не ломается на одиночных символах _, *, [, ` внутри юридического
    текста (номера пунктов, скобки, тире и т.д.) — такие символы
    просто отображаются как есть, а не пытаются закрыть несуществующий
    тег форматирования.
    """

    # 1. Сначала экранируем спецсимволы HTML, чтобы "<", ">", "&"
    #    из исходного текста НПА не сломали разметку.
    escaped = html.escape(text, quote=False)

    # 2. Возвращаем нашу разметку **bold** поверх экранированного
    #    текста (экранирование не трогает "*", так что паттерн
    #    по-прежнему находит пары звёздочек).
    return _BOLD_PATTERN.sub(r"<b>\1</b>", escaped)


def split_text_smart(text: str, limit: int = 4000) -> list[str]:
    """
    Разбивает длинный текст на части не длиннее `limit` символов,
    стараясь резать по границам абзацев (\\n\\n), а не посреди слова
    или посреди тега форматирования.

    В отличие от простого text[i:i+limit], это не разрывает
    **bold**-конструкции пополам — иначе после конвертации в HTML
    получился бы незакрытый <b>, и Telegram вернул бы ошибку парсинга.
    """

    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text

    while len(remaining) > limit:
        # Ищем последний разрыв абзаца в пределах лимита.
        cut = remaining.rfind("\n\n", 0, limit)

        if cut == -1:
            # Абзацев не нашли — ищем хотя бы конец строки.
            cut = remaining.rfind("\n", 0, limit)

        if cut == -1:
            # Совсем без переносов — режем по пробелу, чтобы не
            # разорвать слово.
            cut = remaining.rfind(" ", 0, limit)

        if cut == -1:
            # Совсем без пробелов (редкий случай) — режем жёстко.
            cut = limit

        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


async def send_long_message(update: Update, text: str, use_html: bool = True):
    """
    Отправляет длинный текст частями, с безопасным фолбэком:
    если HTML-парсинг части почему-то не проходит (например, из-за
    непредвиденной конструкции в ответе модели), отправляем эту
    конкретную часть обычным текстом, а не роняем всё сообщение.
    """

    chunks = split_text_smart(text, TELEGRAM_MESSAGE_LIMIT)

    for chunk in chunks:
        formatted = to_telegram_html(chunk) if use_html else chunk

        try:
            await update.message.reply_text(
                formatted,
                parse_mode="HTML" if use_html else None,
            )
        except Exception as parse_err:
            logging.warning(
                f"Ошибка HTML-парсинга в Telegram: {parse_err}. "
                f"Отправка части простым текстом."
            )
            await update.message.reply_text(chunk)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Приветственное сообщение по команде /start"""
    welcome_text = (
        "Здравствуйте! Я ваш ИИ-ассистент по охране труда и промышленной безопасности.\n\n"
        "Задайте мне вопрос по законодательству РБ и нормативным актам (НПА), "
        "и я найду точные статьи и дам развернутый ответ."
    )
    await update.message.reply_text(welcome_text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Анализ фотографии рабочего места/производственного объекта через DeepSeek Vision."""
    try:
        await update.message.reply_chat_action("upload_photo")

        # Берём фотографию максимального доступного размера.
        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)

        # Скачиваем фото в память, без записи на диск.
        photo_buffer = io.BytesIO()
        await tg_file.download_to_memory(photo_buffer)
        image_bytes = photo_buffer.getvalue()

        # Защита от слишком большого изображения.
        if len(image_bytes) > 32 * 1024 * 1024:
            await update.message.reply_text(
                "⚠️ Фотография слишком большая для анализа. "
                "Отправьте изображение меньшего размера."
            )
            return

        user_caption = (update.message.caption or "").strip()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        vision_prompt = f"""
Ты — эксперт по охране труда, промышленной и пожарной безопасности
в Республике Беларусь с большим практическим опытом.

Тебе передана фотография производственного объекта, рабочего места,
оборудования или территории.

Твоя задача — провести ТОЛЬКО ВИЗУАЛЬНЫЙ АНАЛИЗ фотографии.

ВАЖНЕЙШЕЕ ПРАВИЛО:
Не утверждай, что действие или объект является нарушением законодательства,
если это невозможно установить только по фотографии.
Не придумывай номера пунктов НПА, документы, размеры, характеристики
оборудования или обстоятельства, которых на фото не видно.

Разделяй:
1. что ДОСТОВЕРНО ВИДНО на фотографии;
2. что МОЖЕТ СВИДЕТЕЛЬСТВОВАТЬ о потенциальном нарушении;
3. что НЕВОЗМОЖНО определить по фотографии.

Проверь, насколько это возможно по изображению:
- СИЗ работников;
- ограждения опасных зон;
- состояние оборудования;
- электрические кабели и электрооборудование;
- проходы, проезды, лестницы и ограждения;
- порядок и складирование материалов;
- наличие потенциальных источников падения предметов;
- пожарную безопасность;
- блокировки и защитные устройства, если они визуально доступны;
- транспорт и движение техники;
- наличие очевидных опасных факторов;
- другие явно видимые небезопасные условия.

Для каждого потенциального нарушения укажи:
• Что видно;
• Почему это потенциально опасно;
• Уровень риска: 🔴 высокий / 🟠 средний / 🟡 низкий;
• Что необходимо дополнительно проверить.

Не ставь окончательный юридический диагноз только на основании фото.

ФОРМАТ ОТВЕТА:

🔎 ВИЗУАЛЬНЫЙ АНАЛИЗ

Если явных проблем не видно:
🟢 Явных нарушений по фотографии не обнаружено.
Затем укажи, что всё равно невозможно проверить визуально.

Если проблемы обнаружены:

🔴 1. [краткое название]
Что видно: ...
Риск: ...
Почему требует внимания: ...
Проверить: ...

🟠 2. ...

В конце:

⚠️ ОГРАНИЧЕНИЯ АНАЛИЗА
Укажи 1–3 наиболее важных обстоятельства, которые невозможно определить
по фотографии и которые могут изменить вывод.

Не ссылайся на конкретные НПА в этом режиме.
Нормативное обоснование будет выполняться отдельным этапом через базу НПА.

Подпись/комментарий пользователя к фото:
{user_caption if user_caption else "не указан"}
"""

        # DeepSeek использует OpenAI-совместимый Chat Completions API.
        # Изображение передаём как base64 data URL.
        def call_deepseek():
            return deepseek_client.chat.completions.create(
                model=DEEPSEEK_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1800,
                temperature=0.1,
            )

        response = await asyncio.to_thread(call_deepseek)
        result = response.choices[0].message.content if response.choices else None

        if not result:
            result = "Не удалось получить результат визуального анализа."

        # Для фото отправляем обычным текстом — без риска Markdown-разметки.
        # Резка теперь идёт по границам абзацев/строк, а не посимвольно.
        for chunk in split_text_smart(result, TELEGRAM_MESSAGE_LIMIT):
            await update.message.reply_text(chunk)

    except Exception as e:
        error_text = str(e)
        logging.error(f"Ошибка при анализе фотографии через DeepSeek: {e}", exc_info=True)

        # Отдельно обрабатываем нехватку баланса/лимита или HTTP 429.
        if "429" in error_text or "insufficient" in error_text.lower() or "balance" in error_text.lower():
            await update.message.reply_text(
                "⚠️ DeepSeek API вернул ошибку 429/лимита.\n\n"
                "Проверьте баланс и доступность API-ключа DeepSeek.\n"
                "После пополнения баланса повторите отправку фотографии."
            )
        elif "401" in error_text or "403" in error_text:
            await update.message.reply_text(
                "⚠️ DeepSeek API не принял ключ доступа.\n\n"
                "Проверьте переменную DEEPSEEK_API_KEY в Railway."
            )
        else:
            await update.message.reply_text(
                "Не удалось проанализировать фотографию через DeepSeek. "
                "Попробуйте отправить изображение ещё раз."
            )


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
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        query_vector = emb_response.embeddings[0].values

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
                [f"Источник: {c.get('doc_name', 'НПА')}, ст./п. {c.get('point_num', '-')}\nТекст: {c.get('content', '')}"
                 for c in context_chunks]
            )
        else:
            retrieved_text = "Релевантные нормативные акты в базе не найдены."

        # 4. Формирование инструкции для Gemini
        # Используем единый структурированный промпт из prompts.py
        # (📌 Краткий ответ / 📚 Обоснование / 🔎 Анализ / ⚠️ Важно),
        # а не отдельную упрощённую копию — чтобы формат ответа
        # не расходился между файлами при будущих правках.
        prompt = LEGAL_ASSISTANT_PROMPT.format(
            retrieved_text=retrieved_text,
            user_query=user_query,
        )

        # 5. Генерация ответа через gemini-3.6-flash с защитой от сбоев 503
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
                    time.sleep(2)
                else:
                    raise gen_err

        # 6. Безопасная отправка ответа (HTML вместо хрупкого legacy Markdown,
        #    резка по абзацам, источники — отдельным визуальным блоком)
        if response and response.text:
            text = response.text.strip()

            # Гарантированно показываем, из каких документов реально взят контекст —
            # независимо от того, упомянула ли модель их все в тексте ответа.
            # Блок источников отделён двойным переносом строки, чтобы при
            # разбиении на части он не "слипался" с последним абзацем ответа.
            if context_chunks:
                sources = sorted({c.get("doc_name", "НПА") for c in context_chunks})
                # Используем ту же разметку **bold**, что и модель в основном
                # тексте — она пройдёт через тот же escape+convert пайплайн
                # в to_telegram_html(), без риска сломать HTML вручную.
                text += "\n\n📄 **Источники:** " + "; ".join(sources)

            await send_long_message(update, text, use_html=True)
        else:
            await update.message.reply_text("Сервис временно перегружен. Пожалуйста, повторите вопрос через несколько секунд.")

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при поиске ответа. Попробуйте сформулировать вопрос иначе.")


def cleanup():
    """Корректное завершение работы бота"""
    global app
    if app:
        logging.info("Остановка бота...")
        app.stop()


def main():
    """Запуск Telegram-бота"""
    global app

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    # Фотографии анализируются отдельным Vision-обработчиком.
    # Текстовый RAG-режим остаётся без изменений.
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logging.info("Бот по охране труда запущен!")

    # Регистрация обработчика корректного завершения
    atexit.register(cleanup)

    # Запуск polling
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
        close_loop=False
    )


if __name__ == "__main__":
    main()
