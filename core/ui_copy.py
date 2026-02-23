from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup

from core.ui_kit import build_keyboard
from core.ui_kit.buttons import button
from domain.localization import LocalizationService

START_TEXT = (
    "🎨 <b>ComfyUI Generator Bot</b>\n"
    "\n"
    "Выберите раздел в меню ниже.\n"
    "Команды тоже работают, но кнопочная навигация быстрее."
)

FALLBACK_TEXT = "Я не ожидаю текст. Откройте меню кнопками ниже."


def start_text(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
) -> str:
    if localization is None:
        return START_TEXT
    return localization.t("ui.start", locale=locale, default=START_TEXT)


def fallback_text(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
) -> str:
    if localization is None:
        return FALLBACK_TEXT
    return localization.t("ui.fallback", locale=locale, default=FALLBACK_TEXT)


def main_menu_keyboard(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    generation_text = (
        localization.t("ui.menu.generation", locale=locale, default="🎨 Генерация")
        if localization is not None
        else "🎨 Генерация"
    )
    models_text = (
        localization.t("ui.menu.models", locale=locale, default="📦 Модели")
        if localization is not None
        else "📦 Модели"
    )
    service_text = (
        localization.t("ui.menu.service", locale=locale, default="⚙️ Сервис")
        if localization is not None
        else "⚙️ Сервис"
    )
    cancel_text = (
        localization.t("ui.menu.cancel", locale=locale, default="🛑 Отмена")
        if localization is not None
        else "🛑 Отмена"
    )

    return build_keyboard(
        [
            [
                button(generation_text, "menu:generation"),
                button(models_text, "menu:models"),
            ],
            [button(service_text, "menu:service"), button(cancel_text, "menu:cancel")],
        ]
    )
