from __future__ import annotations

from aiogram.types import InlineKeyboardButton

BACK_TEXT = "⬅️ Назад"
CANCEL_TEXT = "❌ Отмена"
MENU_ROOT_TEXT = "🏠 В меню"
CUSTOM_VALUE_TEXT = "✏️ Ввести своё значение"


def button(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def back_button(callback_data: str, text: str = BACK_TEXT) -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)


def cancel_button(
    callback_data: str = "menu:cancel", text: str = CANCEL_TEXT
) -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)


def menu_root_button(
    callback_data: str = "menu:root",
    text: str = MENU_ROOT_TEXT,
) -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)


def noop_button(text: str, callback_data: str = "noop") -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)


def custom_value_button(
    callback_data: str,
    text: str = CUSTOM_VALUE_TEXT,
) -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)
