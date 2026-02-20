from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

START_TEXT = (
    "🎨 <b>ComfyUI Generator Bot</b>\n"
    "\n"
    "Выберите раздел в меню ниже.\n"
    "Команды тоже работают, но кнопочная навигация быстрее."
)

FALLBACK_TEXT = "Я не ожидаю текст. Откройте меню кнопками ниже."


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🎨 Генерация", callback_data="menu:generation"),
                InlineKeyboardButton(text="📦 Модели", callback_data="menu:models"),
            ],
            [
                InlineKeyboardButton(text="⚙️ Сервис", callback_data="menu:service"),
                InlineKeyboardButton(text="🛑 Отмена", callback_data="menu:cancel"),
            ],
        ]
    )
