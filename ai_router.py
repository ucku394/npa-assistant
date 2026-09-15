import os
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from openai import OpenAI


# ============================================================
# ЗАГРУЗКА .ENV
# ============================================================

script_dir = Path(__file__).parent
env_path = script_dir / ".env"

load_dotenv(dotenv_path=env_path)


# ============================================================
# НАСТРОЙКИ
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.6-flash"
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "google/gemini-3.1-flash-lite"
).strip()

OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "deepseek/deepseek-chat"
).strip()

MAX_CHAT_TOKENS = int(
    os.getenv("MAX_CHAT_TOKENS", "3000")
)


# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# OPENROUTER CLIENT
# ============================================================

openrouter_client = None

if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


# ============================================================
# ОПРЕДЕЛЕНИЕ ОШИБОК GEMINI
# ============================================================

def is_gemini_fallback_error(error: Exception) -> bool:
    """
    Определяет, нужно ли после ошибки Gemini
    переключаться на OpenRouter.
    """

    error_text = str(error).upper()

    fallback_markers = [
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA",
        "RATE LIMIT",
        "RATE_LIMIT",
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
        "TIMEOUT",
        "DEADLINE",
        "SERVICE UNAVAILABLE",
    ]

    return any(
        marker in error_text
        for marker in fallback_markers
    )


# ============================================================
# ГЕНЕРАЦИЯ ЧЕРЕЗ GEMINI
# ============================================================

async def generate_with_gemini(prompt: str) -> str:

    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY не настроен."
        )

    logger.info(
        f"Попытка генерации ответа через Gemini: {CHAT_MODEL}"
    )

    def call_gemini():

        return gemini_client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": MAX_CHAT_TOKENS,
            }
        )

    response = await asyncio.to_thread(
        call_gemini
    )

    if not response:
        raise RuntimeError(
            "Gemini вернул пустой ответ."
        )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini не вернул текст ответа."
        )

    logger.info(
        "Gemini успешно сформировал ответ."
    )

    return text.strip()


# ============================================================
# ГЕНЕРАЦИЯ ЧЕРЕЗ OPENROUTER
# ============================================================

async def generate_with_openrouter(prompt: str) -> str:

    if openrouter_client is None:
        raise RuntimeError(
            "OPENROUTER_API_KEY не настроен."
        )

    logger.warning(
        "Переключение на OpenRouter."
    )

    logger.info(
        f"OpenRouter primary model: {OPENROUTER_MODEL}"
    )

    logger.info(
        f"OpenRouter fallback model: "
        f"{OPENROUTER_FALLBACK_MODEL}"
    )

    def call_openrouter():

        return openrouter_client.chat.completions.create(

            # Основная модель.
            model=OPENROUTER_MODEL,

            # OpenRouter использует models как
            # список резервных моделей.
            extra_body={
                "models": [
                    OPENROUTER_FALLBACK_MODEL
                ]
            },

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            max_tokens=MAX_CHAT_TOKENS,

            # Необязательные заголовки.
            extra_headers={
                "HTTP-Referer": "https://t.me/",
                "X-Title": "Ассистент по охране труда РБ",
            },
        )

    response = await asyncio.to_thread(
        call_openrouter
    )

    if not response:
        raise RuntimeError(
            "OpenRouter вернул пустой ответ."
        )

    if not response.choices:
        raise RuntimeError(
            "OpenRouter не вернул choices."
        )

    message = response.choices[0].message

    text = getattr(
        message,
        "content",
        None
    )

    if not text:
        raise RuntimeError(
            "OpenRouter не вернул текст ответа."
        )

    used_model = getattr(
        response,
        "model",
        "неизвестно"
    )

    logger.info(
        f"OpenRouter успешно сформировал ответ. "
        f"Фактически использованная модель: {used_model}"
    )

    return text.strip()


# ============================================================
# ОСНОВНОЙ AI ROUTER
# ============================================================

async def generate_answer(prompt: str):

    """
    Главный маршрутизатор генерации.

    1. Сначала Gemini.
    2. Если Gemini работает — возвращаем ответ.
    3. Если Gemini получил quota/rate-limit/503 и т.п. —
       переключаемся на OpenRouter.
    4. OpenRouter сам пытается использовать резервную модель.
    """

    # --------------------------------------------------------
    # 1. GEMINI
    # --------------------------------------------------------

    try:

        text = await generate_with_gemini(
            prompt
        )

        return text, "Gemini"

    except Exception as gemini_error:

        logger.error(
            f"Ошибка Gemini: {gemini_error}",
            exc_info=True
        )

        # ----------------------------------------------------
        # Проверяем, можно ли переключаться
        # ----------------------------------------------------

        if not is_gemini_fallback_error(
            gemini_error
        ):

            logger.error(
                "Ошибка Gemini не относится "
                "к типу, для которого включён fallback."
            )

            raise

        logger.warning(
            "Gemini недоступен. "
            "Переходим на OpenRouter."
        )


    # --------------------------------------------------------
    # 2. OPENROUTER
    # --------------------------------------------------------

    if not OPENROUTER_API_KEY:

        logger.error(
            "OPENROUTER_API_KEY не задан. "
            "Fallback невозможен."
        )

        raise RuntimeError(
            "Gemini недоступен, а OPENROUTER_API_KEY "
            "не настроен."
        )


    try:

        text = await generate_with_openrouter(
            prompt
        )

        return text, "OpenRouter"

    except Exception as openrouter_error:

        logger.error(
            f"Ошибка OpenRouter: {openrouter_error}",
            exc_info=True
        )

        raise RuntimeError(
            "Не удалось получить ответ ни через Gemini, "
            "ни через OpenRouter."
        ) from openrouter_error
