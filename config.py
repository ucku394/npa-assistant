# Shared configuration for the Telegram occupational-safety assistant.
# All secrets are read from environment variables / .env.

import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "").strip()

DEEPSEEK_VISION_MODEL = os.getenv(
    "DEEPSEEK_VISION_MODEL",
    "deepseek-v4-flash-vision-exp",
).strip()

MAX_CHAT_TOKENS = int(os.getenv("MAX_CHAT_TOKENS", "1800"))
SUPABASE_MATCH_THRESHOLD = float(os.getenv("SUPABASE_MATCH_THRESHOLD", "0.30"))
SUPABASE_MATCH_COUNT = int(os.getenv("SUPABASE_MATCH_COUNT", "6"))

TELEGRAM_MESSAGE_LIMIT = 4000

if not CHAT_MODEL:
    # Keep startup explicit rather than silently choosing a possibly obsolete model.
    CHAT_MODEL = "gemini-3.6-flash"
