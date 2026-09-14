import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# Загрузка .env
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# Telegram
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


# ============================================================
# AI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


# ============================================================
# Supabase
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


# ============================================================
# Модели
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.6-flash"
)

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "deepseek-v4-flash-vision-exp"
)


# ============================================================
# RAG
# ============================================================

RAG_MATCH_THRESHOLD = float(
    os.getenv("RAG_MATCH_THRESHOLD", "0.30")
)

RAG_MATCH_COUNT = int(
    os.getenv("RAG_MATCH_COUNT", "10")
)

RAG_FINAL_COUNT = int(
    os.getenv("RAG_FINAL_COUNT", "5")
)


# ============================================================
# Ограничения
# ============================================================

MAX_IMAGE_SIZE_MB = int(
    os.getenv("MAX_IMAGE_SIZE_MB", "32")
)

MAX_TELEGRAM_MESSAGE_LENGTH = 4000

MAX_VISION_TOKENS = int(
    os.getenv("MAX_VISION_TOKENS", "1800")
)

MAX_CHAT_TOKENS = int(
    os.getenv("MAX_CHAT_TOKENS", "3000")
)


# ============================================================
# Rate limit
# ============================================================

REQUESTS_PER_MINUTE = int(
    os.getenv("REQUESTS_PER_MINUTE", "10")
)

REQUESTS_PER_HOUR = int(
    os.getenv("REQUESTS_PER_HOUR", "50")
)


# ============================================================
# Проверка конфигурации
# ============================================================

def validate_config():
    required = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Не заданы обязательные переменные окружения: "
            + ", ".join(missing)
        )