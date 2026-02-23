from __future__ import annotations

from handlers.presets_ui import empty_presets_keyboard, presets_list_keyboard


def test_presets_list_keyboard_shows_editor_back_when_request_active() -> None:
    keyboard = presets_list_keyboard(
        1,
        ["alpha"],
        has_active_prompt_request=True,
    )

    rows = keyboard.inline_keyboard
    assert rows[0][0].callback_data == "preset_load:0"
    assert rows[1][0].callback_data == "pe:back"


def test_presets_list_keyboard_shows_delete_confirmation_label() -> None:
    keyboard = presets_list_keyboard(
        1,
        ["alpha", "beta"],
        has_active_prompt_request=False,
        confirm_delete_index=1,
    )

    rows = keyboard.inline_keyboard
    assert rows[1][1].callback_data == "preset_del:1"
    assert "Удалить" in (rows[1][1].text or "")
    assert rows[2][0].callback_data == "menu:generation"


def test_empty_presets_keyboard_contains_generation_and_menu() -> None:
    keyboard = empty_presets_keyboard()
    rows = keyboard.inline_keyboard

    assert rows[0][0].callback_data == "menu:generation"
    assert rows[1][0].callback_data == "menu:root"
