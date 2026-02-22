from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup

from core.models import GenerationParams
from core.runtime import PromptRequest
from handlers.download import (
    _build_base_filter_keyboard,
    _build_filter_keyboard,
    _build_source_keyboard,
    _build_type_keyboard,
)
from handlers.prompt_editor_send import generation_result_keyboard, preview_image_keyboard
from handlers.prompt_editor_ui import editor_keyboard


def _callback_payloads(markup: InlineKeyboardMarkup) -> set[str]:
    payloads: set[str] = set()
    for row in markup.inline_keyboard:
        for item in row:
            if item.callback_data is not None:
                payloads.add(item.callback_data)
    return payloads


def test_download_type_and_source_callback_prefixes_are_stable() -> None:
    type_payloads = _callback_payloads(_build_type_keyboard())
    assert {
        "dltype:checkpoint",
        "dltype:lora",
        "dltype:upscaler",
        "dltype:embedding",
        "dltype:controlnet",
        "dltype:vae",
        "dltype:cancel",
    } <= type_payloads
    assert all(item.startswith("dltype:") for item in type_payloads)

    source_payloads = _callback_payloads(_build_source_keyboard())
    assert {
        "dlsrc:civitai",
        "dlsrc:huggingface",
        "dlsrc:all",
        "dlsrc:back",
    } <= source_payloads
    assert all(item.startswith("dlsrc:") for item in source_payloads)


def test_download_filter_callback_payload_contracts_are_stable() -> None:
    filter_payloads = _callback_payloads(
        _build_filter_keyboard(
            model_type="checkpoint",
            source="all",
            sort_code="downloads",
            period_code="all",
            base_code="all",
            include_nsfw=False,
            author_nick="",
            page_size=8,
        )
    )

    expected_filter_payloads = {
        "dlflt:sort:downloads",
        "dlflt:sort:rating",
        "dlflt:sort:newest",
        "dlflt:period:all",
        "dlflt:period:month",
        "dlflt:period:week",
        "dlflt:base_menu",
        "dlflt:nsfw:toggle",
        "dlflt:author",
        "dlflt:profile:popular",
        "dlflt:profile:fresh",
        "dlflt:profile:quality",
        "dlflt:profile:anime",
        "dlflt:pagesize:5",
        "dlflt:pagesize:8",
        "dlflt:pagesize:10",
        "dlflt:go",
        "dlflt:back",
    }
    assert expected_filter_payloads <= filter_payloads
    assert all(item.startswith("dlflt:") for item in filter_payloads)

    base_payloads = _callback_payloads(_build_base_filter_keyboard(base_code="all"))
    assert "dlbase:back" in base_payloads
    assert any(item.startswith("dlflt:base:") for item in base_payloads)


def test_prompt_editor_keyboard_callbacks_are_stable_for_simple_and_pro_modes() -> None:
    request = PromptRequest(params=GenerationParams(), operation="generate")

    simple_payloads = _callback_payloads(
        editor_keyboard(request, smart_prompt_enabled=True, pro_mode=False)
    )
    assert {
        "pe:edit:checkpoint",
        "pe:edit:lora",
        "pe:edit:positive",
        "pe:edit:negative",
        "pe:smart:start",
        "pe:edit:size",
        "pe:sub:more",
        "pe:exchange",
        "pe:proceed",
        "pe:toggle:mode",
        "pe:cancel",
        "menu:root",
    } <= simple_payloads
    assert "pe:sub:sampling" not in simple_payloads

    pro_payloads = _callback_payloads(
        editor_keyboard(request, smart_prompt_enabled=True, pro_mode=True)
    )
    assert {
        "pe:sub:sampling",
        "pe:sub:image",
        "pe:sub:enhancements",
        "pe:save",
        "pe:presets",
        "pe:proceed",
        "pe:toggle:mode",
        "pe:cancel",
        "menu:root",
    } <= pro_payloads
    assert "pe:sub:more" not in pro_payloads


def test_send_keyboards_callback_contracts_are_stable() -> None:
    result_payloads = _callback_payloads(generation_result_keyboard())
    assert result_payloads == {"save_preset", "send:new", "send:cancel", "menu:root"}

    preview_payloads = _callback_payloads(preview_image_keyboard("art-1"))
    assert preview_payloads == {"img:png:art-1", "img:open:art-1", "menu:root"}

    preview_with_parent_payloads = _callback_payloads(
        preview_image_keyboard("art-1", parent_artifact_id="art-parent")
    )
    assert preview_with_parent_payloads == {
        "img:png:art-1",
        "img:open:art-1",
        "img:goto_parent:art-1",
        "menu:root",
    }
