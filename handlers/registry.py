from __future__ import annotations

from dataclasses import dataclass

from aiogram import Router

from comfyui_client import ComfyUIClient
from config import Config
from core.runtime import RuntimeStore
from domain.localization import LocalizationService
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

from .common import register_common_handlers
from .download import register_download_handlers
from .presets import register_preset_handlers
from .prompt_editor import register_prompt_editor_handlers


@dataclass(slots=True)
class HandlerRegistryDeps:
    cfg: Config
    client: ComfyUIClient
    downloader: ModelDownloader
    runtime: RuntimeStore
    smart_prompt: SmartPromptService
    localization: LocalizationService


def register_handlers_with_deps(router: Router, deps: HandlerRegistryDeps) -> None:
    register_common_handlers(
        router,
        deps.cfg,
        deps.client,
        deps.downloader,
        deps.runtime,
        deps.localization,
    )
    prompt_editor = register_prompt_editor_handlers(
        router,
        deps.cfg,
        deps.client,
        deps.downloader,
        deps.runtime,
        deps.smart_prompt,
    )
    register_preset_handlers(router, deps.runtime, prompt_editor)
    register_download_handlers(router, deps.client, deps.downloader, deps.runtime)


def register_handlers(
    router: Router,
    *,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    smart_prompt: SmartPromptService,
    localization: LocalizationService,
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
        ),
    )
