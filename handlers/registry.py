from __future__ import annotations

from aiogram import Router

from comfyui_client import ComfyUIClient
from config import Config
from core.runtime import RuntimeStore
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

from .common import register_common_handlers
from .download import register_download_handlers
from .presets import register_preset_handlers
from .prompt_editor import register_prompt_editor_handlers


def register_handlers(
    router: Router,
    *,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    smart_prompt: SmartPromptService,
) -> None:
    register_common_handlers(router, cfg, client, downloader, runtime)
    prompt_editor = register_prompt_editor_handlers(
        router,
        cfg,
        client,
        downloader,
        runtime,
        smart_prompt,
    )
    register_preset_handlers(router, runtime, prompt_editor)
    register_download_handlers(router, client, downloader, runtime)
