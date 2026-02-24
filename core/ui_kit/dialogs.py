from __future__ import annotations

from aiogram.types import InlineKeyboardButton

from domain.localization import LocalizationService
from domain.ui_text import UITextService

from .buttons import back_button, button, menu_root_button


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


def confirm_cancel_rows(
    *,
    confirm_callback: str,
    cancel_callback: str,
    confirm_text: str | None = None,
    cancel_text: str | None = None,
    localization: LocalizationService | None = None,
    locale: str | None = None,
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    resolved_confirm_text = (
        confirm_text
        if confirm_text is not None
        else _t(
            localization,
            "core.ui_kit.dialogs.confirm",
            locale=locale,
            default="✅ Подтвердить",
            text_service=text_service,
            profile=profile,
        )
    )
    resolved_cancel_text = (
        cancel_text
        if cancel_text is not None
        else _t(
            localization,
            "common.action.cancel",
            locale=locale,
            default="❌ Отмена",
            text_service=text_service,
            profile=profile,
        )
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
    text_service: UITextService | None = None,
    profile: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    return [
        [
            back_button(
                back_callback,
                localization=localization,
                locale=locale,
                text_service=text_service,
                profile=profile,
            )
        ],
        [
            menu_root_button(
                root_callback,
                localization=localization,
                locale=locale,
                text_service=text_service,
                profile=profile,
            )
        ],
    ]
