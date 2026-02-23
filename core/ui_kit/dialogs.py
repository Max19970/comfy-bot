from __future__ import annotations

from aiogram.types import InlineKeyboardButton

from domain.localization import LocalizationService

from .buttons import back_button, button, menu_root_button


def _t(
    localization: LocalizationService | None,
    key: str,
    *,
    locale: str | None,
    default: str,
) -> str:
    if localization is None:
        return default
    return localization.t(key, locale=locale, default=default)


def confirm_cancel_rows(
    *,
    confirm_callback: str,
    cancel_callback: str,
    confirm_text: str | None = None,
    cancel_text: str | None = None,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    resolved_confirm_text = (
        confirm_text
        if confirm_text is not None
        else _t(
            localization,
            "core.ui_kit.dialogs.confirm",
            locale=locale,
            default="✅ Подтвердить",
        )
    )
    resolved_cancel_text = (
        cancel_text
        if cancel_text is not None
        else _t(localization, "common.action.cancel", locale=locale, default="❌ Отмена")
    )
    return [
        [button(resolved_confirm_text, confirm_callback)],
        [button(resolved_cancel_text, cancel_callback)],
    ]


def back_to_menu_rows(
    back_callback: str,
    *,
    root_callback: str = "menu:root",
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    return [
        [back_button(back_callback, localization=localization, locale=locale)],
        [menu_root_button(root_callback, localization=localization, locale=locale)],
    ]
