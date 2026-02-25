from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties

from application.localization_service import DefaultLocalizationService
from application.model_downloader import ModelDownloader
from application.smart_prompt_service import SmartPromptService
from application.ui_text_service import DefaultUITextService
from core.config import Config
from core.runtime import RuntimeStore, load_runtime_store
from core.runtime_gateway_extension_loader import (
    RuntimeGatewayExtensionLoaderError,
    load_runtime_gateway_extensions,
)
from core.runtime_gateways import RuntimeGateways, create_runtime_gateways
from domain.localization import LocalizationService
from domain.ui_text import UITextService
from infrastructure.comfy_workflow_builder import set_generation_node_packages
from infrastructure.comfyui_client import ComfyUIClient
from infrastructure.localization_catalog import FileSystemTranslationCatalog
from infrastructure.ui_text_pipeline import build_ui_text_modifiers
from infrastructure.ui_text_resources import (
    FileSystemUICopyProfileCatalog,
    FileSystemUITextRegistry,
)


@dataclass(slots=True)
class AppServices:
    client: ComfyUIClient
    downloader: ModelDownloader
    smart_prompt: SmartPromptService
    runtime: RuntimeStore
    runtime_gateways: RuntimeGateways
    localization: LocalizationService
    ui_text: UITextService


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
    runtime_gateways: RuntimeGateways
    localization: LocalizationService
    ui_text: UITextService

    async def close(self) -> None:
        self.runtime.persist()
        await self.smart_prompt.close()
        await self.client.close()
        await self.downloader.close()
        await self.bot.session.close()


def create_app_services(cfg: Config) -> AppServices:
    set_generation_node_packages(cfg.comfy_node_packages)
    runtime = load_runtime_store()
    runtime_gateways = create_runtime_gateways(runtime)
    try:
        load_runtime_gateway_extensions(
            cfg.runtime_gateway_extension_packages,
            gateways=runtime_gateways,
        )
    except RuntimeGatewayExtensionLoaderError as exc:
        raise RuntimeError(f"Failed to load runtime gateway extensions: {exc}") from exc

    locales_root = Path(__file__).resolve().parent.parent / "locales"
    catalog = FileSystemTranslationCatalog(str(locales_root))
    localization = DefaultLocalizationService(catalog)

    ui_text_root = Path(__file__).resolve().parent.parent / "ui_text"
    ui_text_registry = FileSystemUITextRegistry(str(ui_text_root / "registry.json"))
    ui_text_profiles = FileSystemUICopyProfileCatalog(str(ui_text_root / "profiles"))
    ui_text_modifiers = build_ui_text_modifiers(
        cfg.ui_text_modifier_factories,
        dependencies={
            "localization": localization,
            "copy_profiles": ui_text_profiles,
        },
    )
    ui_text = DefaultUITextService(
        registry=ui_text_registry,
        profiles=ui_text_profiles,
        modifiers=ui_text_modifiers,
    )

    return AppServices(
        client=ComfyUIClient(cfg),
        downloader=ModelDownloader(cfg),
        smart_prompt=SmartPromptService(cfg),
        runtime=runtime,
        runtime_gateways=runtime_gateways,
        localization=localization,
        ui_text=ui_text,
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
        runtime_gateways=active_services.runtime_gateways,
        localization=active_services.localization,
        ui_text=active_services.ui_text,
    )
