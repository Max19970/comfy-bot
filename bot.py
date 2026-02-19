"""
Telegram bot for ComfyUI image generation.

Features:
- /generate — prompt editor with arbitrary field editing order, smart prompt, and reference images
- /repeat — repeat last generation
- /presets — prompt library (save/load/delete per user)
- /download — search & download models from CivitAI / HuggingFace
- /models — refresh & list available models
- /queue — show ComfyUI queue status
- /settings — view defaults
- /training (/learn) — print text onboarding guide
- /cancel — abort current operation
- Whitelist-based access control
"""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher

from app_context import AppContext, create_app_context
from comfyui_client import ComfyUIClient
from config import Config
from handlers.registry import register_handlers
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_bot(
    cfg: Config,
) -> tuple[Bot, Dispatcher, ComfyUIClient, ModelDownloader, SmartPromptService]:
    app = create_app(cfg)
    return app.bot, app.dispatcher, app.client, app.downloader, app.smart_prompt


def create_app(cfg: Config) -> AppContext:
    app = create_app_context(cfg)
    register_handlers(
        app.router,
        bot=app.bot,
        cfg=cfg,
        client=app.client,
        downloader=app.downloader,
        runtime=app.runtime,
        smart_prompt=app.smart_prompt,
    )
    return app


async def main() -> None:
    cfg = Config.from_env()
    if not cfg.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    app = create_app(cfg)

    if await app.client.check_connection():
        logger.info("ComfyUI is reachable at %s", cfg.comfyui_url)
        await app.client.refresh_info()
    else:
        logger.warning("ComfyUI is NOT reachable at %s", cfg.comfyui_url)

    try:
        logger.info("Bot starting...")
        await app.dispatcher.start_polling(app.bot)
    finally:
        await app.close()


if __name__ == "__main__":
    asyncio.run(main())
