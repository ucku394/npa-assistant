"""Unified entry point to run both the FastAPI Web API and the Telegram Bot concurrently.

This allows hosting both services on a single container / server instance,
sharing memory for models and respecting the $PORT environment variable.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from api import app as web_app
from bot import build_application
from config import TELEGRAM_BOT_TOKEN

logger = logging.getLogger("npa_runner")


async def run_services():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    # Configure Uvicorn Web Server
    config = uvicorn.Config(
        web_app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Initialize Telegram Bot if token is present
    bot_app = None
    if TELEGRAM_BOT_TOKEN:
        try:
            bot_app = build_application()
            await bot_app.initialize()
            await bot_app.start()
            await bot_app.updater.start_polling(drop_pending_updates=False)
            logger.info("Telegram Bot polling started successfully.")
        except Exception as exc:
            logger.error("Failed to start Telegram Bot: %s", exc)
            bot_app = None
    else:
        logger.warning(
            "TELEGRAM_BOT_TOKEN is not configured; running Web API only."
        )

    logger.info("Starting Web API on http://%s:%s", host, port)

    try:
        await server.serve()
    finally:
        if bot_app:
            logger.info("Stopping Telegram Bot...")
            try:
                if bot_app.updater and bot_app.updater.running:
                    await bot_app.updater.stop()
                if bot_app.running:
                    await bot_app.stop()
                await bot_app.shutdown()
                logger.info("Telegram Bot stopped gracefully.")
            except Exception as exc:
                logger.warning("Error during bot shutdown: %s", exc)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # httpx/httpcore can leak credentials in logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        asyncio.run(run_services())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Process terminated.")


if __name__ == "__main__":
    main()
