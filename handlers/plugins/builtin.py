from __future__ import annotations

from typing import Protocol, cast

from aiogram import Router

from application.model_downloader import ModelDownloader
from application.smart_prompt_service import SmartPromptService
from application.ui_text_localization_bridge import UITextLocalizationBridge
from core.config import Config
from core.runtime import RuntimeStore
from domain.localization import LocalizationService
from domain.ui_text import UITextService
from infrastructure.comfyui_client import ComfyUIClient
from plugins.contracts import HandlerPluginContext, PluginDescriptor, PluginRegistry

from ..common import register_common_handlers
from ..download import register_download_handlers
from ..presets import register_preset_handlers
from ..prompt_editor import PromptEditorService, register_prompt_editor_handlers

HANDLER_LOCALIZATION_BRIDGE_KEY = "handlers.localization.bridge"
PROMPT_EDITOR_SERVICE_KEY = "handlers.prompt_editor.service"


class HandlerPluginDeps(Protocol):
    cfg: Config
    client: ComfyUIClient
    downloader: ModelDownloader
    runtime: RuntimeStore
    smart_prompt: SmartPromptService
    localization: LocalizationService
    ui_text: UITextService


def _deps(context: HandlerPluginContext) -> HandlerPluginDeps:
    return cast(HandlerPluginDeps, context.deps)


def _handler_localization_bridge(context: HandlerPluginContext) -> UITextLocalizationBridge:
    shared_value = context.shared.get(HANDLER_LOCALIZATION_BRIDGE_KEY)
    if isinstance(shared_value, UITextLocalizationBridge):
        return shared_value

    deps = _deps(context)
    bridge = UITextLocalizationBridge(localization=deps.localization, ui_text=deps.ui_text)
    context.shared[HANDLER_LOCALIZATION_BRIDGE_KEY] = bridge
    return bridge


def _prompt_editor_service(context: HandlerPluginContext) -> PromptEditorService:
    shared_value = context.shared.get(PROMPT_EDITOR_SERVICE_KEY)
    if isinstance(shared_value, PromptEditorService):
        return shared_value
    raise RuntimeError(
        "Prompt editor service is not registered. "
        "Ensure 'handlers.prompt_editor' plugin is loaded before 'handlers.presets'."
    )


class CommonHandlersPlugin:
    descriptor = PluginDescriptor(
        plugin_id="handlers.common",
        display_name="Common handlers",
        order=100,
    )

    def register(self, context: HandlerPluginContext) -> None:
        deps = _deps(context)
        register_common_handlers(
            cast(Router, context.router),
            deps.cfg,
            deps.client,
            deps.downloader,
            deps.runtime,
            _handler_localization_bridge(context),
        )


class PromptEditorHandlersPlugin:
    descriptor = PluginDescriptor(
        plugin_id="handlers.prompt_editor",
        display_name="Prompt editor handlers",
        order=200,
    )

    def register(self, context: HandlerPluginContext) -> PromptEditorService:
        deps = _deps(context)
        service = register_prompt_editor_handlers(
            router=cast(Router, context.router),
            cfg=deps.cfg,
            client=deps.client,
            downloader=deps.downloader,
            runtime=deps.runtime,
            localization=_handler_localization_bridge(context),
            ui_text=deps.ui_text,
            smart_prompt=deps.smart_prompt,
        )
        context.shared[PROMPT_EDITOR_SERVICE_KEY] = service
        return service


class PresetHandlersPlugin:
    descriptor = PluginDescriptor(
        plugin_id="handlers.presets",
        display_name="Preset handlers",
        order=300,
    )

    def register(self, context: HandlerPluginContext) -> None:
        deps = _deps(context)
        register_preset_handlers(
            cast(Router, context.router),
            deps.runtime,
            _prompt_editor_service(context),
            _handler_localization_bridge(context),
        )


class DownloadHandlersPlugin:
    descriptor = PluginDescriptor(
        plugin_id="handlers.download",
        display_name="Download handlers",
        order=400,
    )

    def register(self, context: HandlerPluginContext) -> None:
        deps = _deps(context)
        register_download_handlers(
            cast(Router, context.router),
            deps.client,
            deps.downloader,
            deps.runtime,
            _handler_localization_bridge(context),
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.register(CommonHandlersPlugin())
    registry.register(PromptEditorHandlersPlugin())
    registry.register(PresetHandlersPlugin())
    registry.register(DownloadHandlersPlugin())
