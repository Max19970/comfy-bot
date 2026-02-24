from __future__ import annotations

from aiogram.types import InlineKeyboardButton

from domain.localization import LocalizationService
from domain.ui_text import UITextService

BACK_TEXT = "⬅️ Назад"
CANCEL_TEXT = "❌ Отмена"
MENU_ROOT_TEXT = "🏠 В меню"
CUSTOM_VALUE_TEXT = "✏️ Ввести своё значение"


def _t(
    localization: LocalizationService | None,
    key: str,
    *,
    locale: str | None,
    default: str,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> str:
    if text_service is not None:
        return text_service.text(key, locale=locale, profile=profile, default=default)
    if localization is None:
        return default
    return localization.t(key, locale=locale, default=default)


def button(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def back_button(
    callback_data: str,
    text: str | None = None,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> InlineKeyboardButton:
    resolved_text = (
        text
        if text is not None
        else _t(
            localization,
            "common.action.back",
            locale=locale,
            default=BACK_TEXT,
            text_service=text_service,
            profile=profile,
        )
    )
    return button(text=resolved_text, callback_data=callback_data)


def cancel_button(
    callback_data: str = "menu:cancel",
    text: str | None = None,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> InlineKeyboardButton:
    resolved_text = (
        text
        if text is not None
        else _t(
            localization,
            "common.action.cancel",
            locale=locale,
            default=CANCEL_TEXT,
            text_service=text_service,
            profile=profile,
        )
    )
    return button(text=resolved_text, callback_data=callback_data)


def menu_root_button(
    callback_data: str = "menu:root",
    text: str | None = None,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> InlineKeyboardButton:
    resolved_text = (
        text
        if text is not None
        else _t(
            localization,
            "common.menu.root",
            locale=locale,
            default=MENU_ROOT_TEXT,
            text_service=text_service,
            profile=profile,
        )
    )
    return button(text=resolved_text, callback_data=callback_data)


def noop_button(text: str, callback_data: str = "noop") -> InlineKeyboardButton:
    return button(text=text, callback_data=callback_data)


def custom_value_button(
    callback_data: str,
    text: str | None = None,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> InlineKeyboardButton:
    resolved_text = (
        text
        if text is not None
        else _t(
            localization,
            "common.action.enter_manual",
            locale=locale,
            default=CUSTOM_VALUE_TEXT,
            text_service=text_service,
            profile=profile,
        )
    )
    return button(text=resolved_text, callback_data=callback_data)
