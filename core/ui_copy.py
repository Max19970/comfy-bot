from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup

from core.ui_kit import build_keyboard, resolve_ui_text, text_button, ui_text
from core.ui_kit.buttons import button
from domain.localization import LocalizationService
from domain.ui_text import UITextService

START_TEXT = (
    "🎨 <b>ComfyUI Generator Bot</b>\n"
    "\n"
    "Выберите раздел в меню ниже.\n"
    "Команды тоже работают, но кнопочная навигация быстрее."
)

FALLBACK_TEXT = "Я не ожидаю текст. Откройте меню кнопками ниже."

START_COPY = ui_text("ui.start", START_TEXT)
FALLBACK_COPY = ui_text("ui.fallback", FALLBACK_TEXT)
MENU_GENERATION_COPY = ui_text("ui.menu.generation", "🎨 Генерация")
MENU_MODELS_COPY = ui_text("ui.menu.models", "📦 Модели")
MENU_SERVICE_COPY = ui_text("ui.menu.service", "⚙️ Сервис")
MENU_CANCEL_COPY = ui_text("ui.menu.cancel", "🛑 Отмена")


def start_text(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> str:
    if text_service is not None:
        return resolve_ui_text(text_service, START_COPY, locale=locale, profile=profile)
    if localization is None:
        return START_TEXT
    return localization.t("ui.start", locale=locale, default=START_TEXT)


def fallback_text(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> str:
    if text_service is not None:
        return resolve_ui_text(text_service, FALLBACK_COPY, locale=locale, profile=profile)
    if localization is None:
        return FALLBACK_TEXT
    return localization.t("ui.fallback", locale=locale, default=FALLBACK_TEXT)


def main_menu_keyboard(
    localization: LocalizationService | None = None,
    *,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> InlineKeyboardMarkup:
    if text_service is not None:
        return build_keyboard(
            [
                [
                    text_button(
                        text_service,
                        MENU_GENERATION_COPY,
                        "menu:generation",
                        locale=locale,
                        profile=profile,
                    ),
                    text_button(
                        text_service,
                        MENU_MODELS_COPY,
                        "menu:models",
                        locale=locale,
                        profile=profile,
                    ),
                ],
                [
                    text_button(
                        text_service,
                        MENU_SERVICE_COPY,
                        "menu:service",
                        locale=locale,
                        profile=profile,
                    ),
                    text_button(
                        text_service,
                        MENU_CANCEL_COPY,
                        "menu:cancel",
                        locale=locale,
                        profile=profile,
                    ),
                ],
            ]
        )

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
