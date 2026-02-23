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
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand

from app_context import AppContext, create_app_context, create_app_services
from comfyui_client import ComfyUIClient
from config import Config
from domain.localization import LocalizationService
from handlers.registry import HandlerRegistryDeps, register_handlers_with_deps
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BOT_COMMAND_SPECS: tuple[tuple[str, str, str], ...] = (
    ("start", "bot.command.start", "Главное меню"),
    ("help", "bot.command.help", "Помощь"),
    ("generate", "bot.command.generate", "Новая генерация"),
    ("repeat", "bot.command.repeat", "Повтор последней"),
    ("presets", "bot.command.presets", "Мои пресеты"),
    ("download", "bot.command.download", "Скачать модель"),
    ("models", "bot.command.models", "Список моделей"),
    ("queue", "bot.command.queue", "Очередь ComfyUI"),
    ("jobs", "bot.command.jobs", "Активные задачи"),
    ("settings", "bot.command.settings", "Настройки"),
    ("training", "bot.command.training", "Обучение"),
    ("cancel", "bot.command.cancel", "Отменить операцию"),
)


async def configure_bot_commands(
    bot: Bot,
    localization: LocalizationService | None = None,
) -> None:
    if localization is None:
        await bot.set_my_commands(default_bot_commands())
        return

    default_locale = localization.default_locale()
    await bot.set_my_commands(default_bot_commands(localization, locale=default_locale))

    for locale_code in localization.available_locales():
        if locale_code == default_locale:
            continue
        await bot.set_my_commands(
            default_bot_commands(localization, locale=locale_code),
            language_code=locale_code,
        )


def default_bot_commands(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
) -> list[BotCommand]:
    commands: list[BotCommand] = []
    for command, key, default_text in BOT_COMMAND_SPECS:
        description = (
            localization.t(key, locale=locale, default=default_text)
            if localization is not None
            else default_text
        )
        commands.append(BotCommand(command=command, description=description))
    return commands


def create_bot(
    cfg: Config,
) -> tuple[Bot, Dispatcher, ComfyUIClient, ModelDownloader, SmartPromptService]:
    app = create_app(cfg)
    return app.bot, app.dispatcher, app.client, app.downloader, app.smart_prompt


def create_app(cfg: Config) -> AppContext:
    services = create_app_services(cfg)
    app = create_app_context(cfg, services=services)
    register_handlers_with_deps(
        app.router,
        HandlerRegistryDeps(
            cfg=cfg,
            client=app.client,
            downloader=app.downloader,
            runtime=app.runtime,
            smart_prompt=app.smart_prompt,
            localization=app.localization,
        ),
    )
    return app


async def main() -> None:
    cfg = Config.from_env()
    if not cfg.telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    app = create_app(cfg)

    try:
        await configure_bot_commands(app.bot, app.localization)
    except TelegramAPIError:
        logger.warning("Failed to configure Telegram commands", exc_info=True)

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
