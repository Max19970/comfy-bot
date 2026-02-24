from __future__ import annotations

from dataclasses import dataclass

from aiogram import Router

from application.model_downloader import ModelDownloader
from application.smart_prompt_service import SmartPromptService
from application.ui_text_localization_bridge import UITextLocalizationBridge
from core.config import Config
from core.runtime import RuntimeStore
from domain.localization import LocalizationService
from domain.ui_text import UITextService
from infrastructure.comfyui_client import ComfyUIClient
from plugins.contracts import HANDLER_CAPABILITY_REGISTRATION, HandlerPluginContext
from plugins.loader import HandlerPluginLoaderError, load_handler_plugins_from_packages

from .common import register_common_handlers
from .download import register_download_handlers
from .presets import register_preset_handlers
from .prompt_editor import register_prompt_editor_handlers

_HANDLER_LOCALIZATION_BRIDGE_KEY = "handlers.localization.bridge"


@dataclass(slots=True)
class HandlerRegistryDeps:
    cfg: Config
    client: ComfyUIClient
    downloader: ModelDownloader
    runtime: RuntimeStore
    smart_prompt: SmartPromptService
    localization: LocalizationService
    ui_text: UITextService


def register_handlers_with_deps(router: Router, deps: HandlerRegistryDeps) -> None:
    handler_localization = UITextLocalizationBridge(
        localization=deps.localization,
        ui_text=deps.ui_text,
    )

    try:
        plugins = load_handler_plugins_from_packages(deps.cfg.handler_plugin_packages)
    except HandlerPluginLoaderError as exc:
        raise RuntimeError(f"Failed to load handler plugins: {exc}") from exc

    if not plugins:
        _register_handlers_legacy(router, deps, handler_localization)
        return

    context = HandlerPluginContext(router=router, deps=deps)
    context.shared[_HANDLER_LOCALIZATION_BRIDGE_KEY] = handler_localization

    for plugin in plugins:
        descriptor = plugin.descriptor
        if not descriptor.enabled_by_default:
            continue
        if HANDLER_CAPABILITY_REGISTRATION not in descriptor.capabilities:
            continue
        plugin.register(context)


def _register_handlers_legacy(
    router: Router,
    deps: HandlerRegistryDeps,
    handler_localization: UITextLocalizationBridge,
) -> None:
    register_common_handlers(
        router,
        deps.cfg,
        deps.client,
        deps.downloader,
        deps.runtime,
        handler_localization,
    )
    prompt_editor = register_prompt_editor_handlers(
        router=router,
        cfg=deps.cfg,
        client=deps.client,
        downloader=deps.downloader,
        runtime=deps.runtime,
        localization=handler_localization,
        ui_text=deps.ui_text,
        smart_prompt=deps.smart_prompt,
    )
    register_preset_handlers(router, deps.runtime, prompt_editor, handler_localization)
    register_download_handlers(
        router,
        deps.client,
        deps.downloader,
        deps.runtime,
        handler_localization,
    )


def register_handlers(
    router: Router,
    *,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    smart_prompt: SmartPromptService,
    localization: LocalizationService,
    ui_text: UITextService,
) -> None:
    register_handlers_with_deps(
        router,
        HandlerRegistryDeps(
            cfg=cfg,
            client=client,
            downloader=downloader,
            runtime=runtime,
            smart_prompt=smart_prompt,
            localization=localization,
            ui_text=ui_text,
        ),
    )
