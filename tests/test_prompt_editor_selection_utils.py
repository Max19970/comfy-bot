from __future__ import annotations

from handlers.prompt_editor_selection_utils import scalar_choice_keyboard


def test_scalar_choice_keyboard_layout() -> None:
    keyboard = scalar_choice_keyboard(
        prefix="pe_x",
        values_rows=[["1", "2"], ["3"]],
        back_callback="pe:back",
    )
    rows = keyboard.inline_keyboard
    assert rows[0][0].callback_data == "pe_x:1"
    assert rows[0][1].callback_data == "pe_x:2"
    assert rows[1][0].callback_data == "pe_x:3"
    assert rows[2][0].callback_data == "pe_x:custom"
    assert rows[3][0].callback_data == "pe:back"
