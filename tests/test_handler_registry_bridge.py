from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from aiogram import Router

from application.model_downloader import ModelDownloader
from application.smart_prompt_service import SmartPromptService
from application.ui_text_localization_bridge import UITextLocalizationBridge
from core.config import Config
from core.runtime import RuntimeStore
from domain.localization import LocalizationService
from domain.ui_text import UITextService
from handlers.prompt_editor import PromptEditorService
from handlers.registry import HandlerRegistryDeps, register_handlers_with_deps
from infrastructure.comfyui_client import ComfyUIClient
from plugins.contracts import PluginDescriptor


def test_registry_wraps_handler_localization_with_ui_text_bridge(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_common(router, cfg, client, downloader, runtime, localization):
        captured["common"] = localization

    def _fake_prompt_editor(
        *,
        router,
        cfg,
        client,
        downloader,
        runtime,
        localization,
        ui_text,
        smart_prompt,
    ):
        async def _ensure_models(_message):
            return True

        async def _show_prompt_editor(*_args, **_kwargs):
            return None

        captured["prompt_editor"] = localization
        return PromptEditorService(
            ensure_models=_ensure_models,
            normalize_params=lambda params: params,
            show_prompt_editor=_show_prompt_editor,
        )

    def _fake_presets(router, runtime, editor, localization):
        captured["presets"] = localization

    def _fake_download(router, client, downloader, runtime, localization):
        captured["download"] = localization

    monkeypatch.setattr("handlers.plugins.builtin.register_common_handlers", _fake_common)
    monkeypatch.setattr(
        "handlers.plugins.builtin.register_prompt_editor_handlers",
        _fake_prompt_editor,
    )
    monkeypatch.setattr("handlers.plugins.builtin.register_preset_handlers", _fake_presets)
    monkeypatch.setattr("handlers.plugins.builtin.register_download_handlers", _fake_download)

    deps = HandlerRegistryDeps(
        cfg=cast(Config, SimpleNamespace(handler_plugin_packages="handlers.plugins.builtin")),
        client=cast(ComfyUIClient, SimpleNamespace()),
        downloader=cast(ModelDownloader, SimpleNamespace()),
        runtime=cast(RuntimeStore, SimpleNamespace()),
        smart_prompt=cast(SmartPromptService, SimpleNamespace()),
        localization=cast(
            LocalizationService,
            SimpleNamespace(
                t=lambda *args, **kwargs: "ok",
                default_locale=lambda: "ru",
                available_locales=lambda: ("ru", "en"),
            ),
        ),
        ui_text=cast(
            UITextService,
            SimpleNamespace(
                text=lambda *args, **kwargs: "ok",
                default_profile=lambda: "base",
                available_profiles=lambda: ("base",),
            ),
        ),
    )

    register_handlers_with_deps(Router(), deps)

    assert isinstance(captured["common"], UITextLocalizationBridge)
    assert isinstance(captured["prompt_editor"], UITextLocalizationBridge)
    assert isinstance(captured["presets"], UITextLocalizationBridge)
    assert isinstance(captured["download"], UITextLocalizationBridge)
    assert captured["common"] is captured["download"]
    assert captured["common"] is captured["presets"]


def test_registry_falls_back_to_default_plugins_when_plugin_list_is_empty(monkeypatch) -> None:
    calls: list[str] = []

    class _FallbackPlugin:
        descriptor = PluginDescriptor(
            plugin_id="handlers.fallback",
            display_name="Fallback plugin",
            order=10,
        )

        def register(self, context):
            context.shared["fallback.registered"] = True

    def _fake_load_plugins(packages_csv: str):
        calls.append(packages_csv)
        if not packages_csv:
            return ()
        if packages_csv == "handlers.plugins.builtin":
            return (_FallbackPlugin(),)
        return ()

    monkeypatch.setattr("handlers.registry.load_handler_plugins_from_packages", _fake_load_plugins)

    deps = HandlerRegistryDeps(
        cfg=cast(Config, SimpleNamespace(handler_plugin_packages="")),
        client=cast(ComfyUIClient, SimpleNamespace()),
        downloader=cast(ModelDownloader, SimpleNamespace()),
        runtime=cast(RuntimeStore, SimpleNamespace()),
        smart_prompt=cast(SmartPromptService, SimpleNamespace()),
        localization=cast(
            LocalizationService,
            SimpleNamespace(
                t=lambda *args, **kwargs: "ok",
                default_locale=lambda: "ru",
                available_locales=lambda: ("ru", "en"),
            ),
        ),
        ui_text=cast(
            UITextService,
            SimpleNamespace(
                text=lambda *args, **kwargs: "ok",
                default_profile=lambda: "base",
                available_profiles=lambda: ("base",),
            ),
        ),
    )

    register_handlers_with_deps(Router(), deps)

    assert calls == ["", "handlers.plugins.builtin"]


def test_registry_raises_when_no_registration_plugins_available(monkeypatch) -> None:
    class _NonHandlerPlugin:
        descriptor = PluginDescriptor(
            plugin_id="plugins.non_handler",
            display_name="Non-handler plugin",
            capabilities=("other.capability",),
            order=10,
        )

        def register(self, context):
            raise AssertionError("register should not be called for non-handler capability")

    monkeypatch.setattr(
        "handlers.registry.load_handler_plugins_from_packages",
        lambda _: (_NonHandlerPlugin(),),
    )

    deps = HandlerRegistryDeps(
        cfg=cast(Config, SimpleNamespace(handler_plugin_packages="handlers.plugins.builtin")),
        client=cast(ComfyUIClient, SimpleNamespace()),
        downloader=cast(ModelDownloader, SimpleNamespace()),
        runtime=cast(RuntimeStore, SimpleNamespace()),
        smart_prompt=cast(SmartPromptService, SimpleNamespace()),
        localization=cast(
            LocalizationService,
            SimpleNamespace(
                t=lambda *args, **kwargs: "ok",
                default_locale=lambda: "ru",
                available_locales=lambda: ("ru", "en"),
            ),
        ),
        ui_text=cast(
            UITextService,
            SimpleNamespace(
                text=lambda *args, **kwargs: "ok",
                default_profile=lambda: "base",
                available_profiles=lambda: ("base",),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="No handler plugins with registration capability"):
        register_handlers_with_deps(Router(), deps)
