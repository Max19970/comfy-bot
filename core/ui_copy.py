from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup

from core.ui_kit import build_keyboard
from core.ui_kit.buttons import button

START_TEXT = (
    "🎨 <b>ComfyUI Generator Bot</b>\n"
    "\n"
    "Выберите раздел в меню ниже.\n"
    "Команды тоже работают, но кнопочная навигация быстрее."
)

FALLBACK_TEXT = "Я не ожидаю текст. Откройте меню кнопками ниже."


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("🎨 Генерация", "menu:generation"), button("📦 Модели", "menu:models")],
            [button("⚙️ Сервис", "menu:service"), button("🛑 Отмена", "menu:cancel")],
        ]
    )
