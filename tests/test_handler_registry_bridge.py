from __future__ import annotations

from types import SimpleNamespace

from aiogram import Router

from application.ui_text_localization_bridge import UITextLocalizationBridge
from handlers.registry import HandlerRegistryDeps, register_handlers_with_deps


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
        captured["prompt_editor"] = localization
        return SimpleNamespace(name="prompt_editor_service")

    def _fake_presets(router, runtime, editor, localization):
        captured["presets"] = localization

    def _fake_download(router, client, downloader, runtime, localization):
        captured["download"] = localization

    monkeypatch.setattr("handlers.registry.register_common_handlers", _fake_common)
    monkeypatch.setattr("handlers.registry.register_prompt_editor_handlers", _fake_prompt_editor)
    monkeypatch.setattr("handlers.registry.register_preset_handlers", _fake_presets)
    monkeypatch.setattr("handlers.registry.register_download_handlers", _fake_download)

    deps = HandlerRegistryDeps(
        cfg=SimpleNamespace(),
        client=SimpleNamespace(),
        downloader=SimpleNamespace(),
        runtime=SimpleNamespace(),
        smart_prompt=SimpleNamespace(),
        localization=SimpleNamespace(
            t=lambda *args, **kwargs: "ok",
            default_locale=lambda: "ru",
            available_locales=lambda: ("ru", "en"),
        ),
        ui_text=SimpleNamespace(text=lambda *args, **kwargs: "ok"),
    )

    register_handlers_with_deps(Router(), deps)

    assert isinstance(captured["common"], UITextLocalizationBridge)
    assert isinstance(captured["prompt_editor"], UITextLocalizationBridge)
    assert isinstance(captured["presets"], UITextLocalizationBridge)
    assert isinstance(captured["download"], UITextLocalizationBridge)
    assert captured["common"] is captured["download"]
    assert captured["common"] is captured["presets"]
