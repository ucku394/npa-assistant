import os

from dotenv import load_dotenv


load_dotenv()


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.6-flash",
).strip()

GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-3.8-flash",
).strip()


# ============================================================
# DEEPSEEK
# ============================================================

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY",
    "",
).strip()

DEEPSEEK_VISION_MODEL = os.getenv(
    "DEEPSEEK_VISION_MODEL",
    "deepseek-flash",
).strip()


# ============================================================
# OPENROUTER
# ============================================================

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "",
).strip()

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "",
).strip()

OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "",
).strip()


# ============================================================
# OPENROUTER VISION / INSPECTION
# ============================================================

OPENROUTER_VISION_MODEL = os.getenv(
    "OPENROUTER_VISION_MODEL",
    "openrouter/free",
).strip()

OPENROUTER_VISION_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_VISION_FALLBACK_MODEL",
    "",
).strip()

VISION_MAX_OUTPUT_TOKENS = int(os.getenv("VISION_MAX_OUTPUT_TOKENS", "1800"))

PRESCRIPTION_DEFAULT_DEADLINE_DAYS = int(
    os.getenv("PRESCRIPTION_DEFAULT_DEADLINE_DAYS", "10")
)

PRESCRIPTION_ORGANIZATION = os.getenv(
    "PRESCRIPTION_ORGANIZATION",
    'ОАО "Организация"',
).strip()

PRESCRIPTION_SERVICE_NAME = os.getenv(
    "PRESCRIPTION_SERVICE_NAME",
    "Служба ОТиПрБ",
).strip()

PRESCRIPTION_DEFAULT_RESPONSIBLE = os.getenv(
    "PRESCRIPTION_DEFAULT_RESPONSIBLE",
    "",
).strip()


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "",
).strip()

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "",
).strip()

SUPABASE_MATCH_THRESHOLD = float(
    os.getenv(
        "SUPABASE_MATCH_THRESHOLD",
        "0.30",
    )
)

SUPABASE_MATCH_COUNT = int(
    os.getenv(
        "SUPABASE_MATCH_COUNT",
        "6",
    )
)


# ============================================================
# AI
# ============================================================

MAX_CHAT_TOKENS = int(
    os.getenv(
        "MAX_CHAT_TOKENS",
        "1800",
    )
)


# ============================================================
# EMBEDDING
# ============================================================

EMBEDDING_CPU_THREADS = int(
    os.getenv(
        "EMBEDDING_CPU_THREADS",
        "1",
    )
)


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_MESSAGE_LIMIT = 4000
