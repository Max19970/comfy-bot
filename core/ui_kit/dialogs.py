from __future__ import annotations

from aiogram.types import InlineKeyboardButton

from .buttons import back_button, button, menu_root_button


def confirm_cancel_rows(
    *,
    confirm_callback: str,
    cancel_callback: str,
    confirm_text: str = "✅ Подтвердить",
    cancel_text: str = "❌ Отмена",
) -> list[list[InlineKeyboardButton]]:
    return [[button(confirm_text, confirm_callback)], [button(cancel_text, cancel_callback)]]


def back_to_menu_rows(
    back_callback: str,
    *,
    root_callback: str = "menu:root",
) -> list[list[InlineKeyboardButton]]:
    return [[back_button(back_callback)], [menu_root_button(root_callback)]]
