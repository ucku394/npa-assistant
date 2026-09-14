import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
).strip()

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.6-flash"
).strip()


# ============================================================
# DEEPSEEK
# ============================================================

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    ""
).strip()

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "deepseek-v4-flash-vision-exp"
).strip()


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    ""
).strip()

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    ""
).strip()


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
# LIMITS
# ============================================================

MAX_IMAGE_SIZE_MB = int(
    os.getenv("MAX_IMAGE_SIZE_MB", "32")
)

MAX_VISION_TOKENS = int(
    os.getenv("MAX_VISION_TOKENS", "1800")
)

MAX_CHAT_TOKENS = int(
    os.getenv("MAX_CHAT_TOKENS", "3000")
)

TELEGRAM_MESSAGE_LIMIT = 4000


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    required = {
        "TELEGRAM_BOT_TOKEN": TELEGRAM_TOKEN,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Не заполнены обязательные переменные окружения: "
            + ", ".join(missing)
        )
