from __future__ import annotations

from dataclasses import dataclass

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties

from comfyui_client import ComfyUIClient
from config import Config
from core.runtime import RuntimeStore, load_runtime_store
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService


@dataclass(slots=True)
class AppServices:
    client: ComfyUIClient
    downloader: ModelDownloader
    smart_prompt: SmartPromptService
    runtime: RuntimeStore


@dataclass(slots=True)
class AppContext:
    cfg: Config
    bot: Bot
    dispatcher: Dispatcher
    router: Router
    client: ComfyUIClient
    downloader: ModelDownloader
    smart_prompt: SmartPromptService
    runtime: RuntimeStore

    async def close(self) -> None:
        self.runtime.persist()
        await self.smart_prompt.close()
        await self.client.close()
        await self.downloader.close()
        await self.bot.session.close()


def create_app_services(cfg: Config) -> AppServices:
    return AppServices(
        client=ComfyUIClient(cfg),
        downloader=ModelDownloader(cfg),
        smart_prompt=SmartPromptService(cfg),
        runtime=load_runtime_store(),
    )


def create_telegram_stack(cfg: Config) -> tuple[Bot, Dispatcher, Router]:
    bot = Bot(
        token=cfg.telegram_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    dispatcher = Dispatcher()
    router = Router()
    dispatcher.include_router(router)
    return bot, dispatcher, router


def create_app_context(cfg: Config, *, services: AppServices | None = None) -> AppContext:
    active_services = services or create_app_services(cfg)
    bot, dispatcher, router = create_telegram_stack(cfg)

    return AppContext(
        cfg=cfg,
        bot=bot,
        dispatcher=dispatcher,
        router=router,
        client=active_services.client,
        downloader=active_services.downloader,
        smart_prompt=active_services.smart_prompt,
        runtime=active_services.runtime,
    )
