from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

_DEFAULT_HANDLER_PLUGIN_PACKAGES = "handlers.plugins.builtin"

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


def _load_plugins_with_default_fallback(packages_csv: str):
    configured_packages = (packages_csv or "").strip()
    try:
        plugins = load_handler_plugins_from_packages(configured_packages)
    except HandlerPluginLoaderError as exc:
        raise RuntimeError(f"Failed to load handler plugins: {exc}") from exc

    if plugins:
        return plugins

    if configured_packages and configured_packages != _DEFAULT_HANDLER_PLUGIN_PACKAGES:
        logger.warning(
            "No handler plugins discovered from '%s'; falling back to '%s'",
            configured_packages,
            _DEFAULT_HANDLER_PLUGIN_PACKAGES,
        )

    try:
        fallback_plugins = load_handler_plugins_from_packages(_DEFAULT_HANDLER_PLUGIN_PACKAGES)
    except HandlerPluginLoaderError as exc:
        raise RuntimeError(f"Failed to load default handler plugins: {exc}") from exc

    if not fallback_plugins:
        raise RuntimeError("No handler plugins available after fallback to default plugin package")
    return fallback_plugins


def register_handlers_with_deps(router: Router, deps: HandlerRegistryDeps) -> None:
    handler_localization = UITextLocalizationBridge(
        localization=deps.localization,
        ui_text=deps.ui_text,
    )

    plugins = _load_plugins_with_default_fallback(deps.cfg.handler_plugin_packages)

    context = HandlerPluginContext(router=router, deps=deps)
    context.shared[_HANDLER_LOCALIZATION_BRIDGE_KEY] = handler_localization

    discovered = ", ".join(
        f"{plugin.descriptor.plugin_id}@{plugin.descriptor.api_version}" for plugin in plugins
    )
    logger.info("Discovered handler plugins: %s", discovered)

    registered_ids: list[str] = []

    for plugin in plugins:
        descriptor = plugin.descriptor
        if not descriptor.enabled_by_default:
            logger.info("Skipping disabled handler plugin: %s", descriptor.plugin_id)
            continue
        if HANDLER_CAPABILITY_REGISTRATION not in descriptor.capabilities:
            logger.info(
                "Skipping plugin without '%s' capability: %s",
                HANDLER_CAPABILITY_REGISTRATION,
                descriptor.plugin_id,
            )
            continue
        plugin.register(context)
        registered_ids.append(descriptor.plugin_id)

    if not registered_ids:
        raise RuntimeError("No handler plugins with registration capability were enabled/loaded")

    logger.info("Registered handler plugins: %s", ", ".join(registered_ids))


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
