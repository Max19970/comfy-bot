from __future__ import annotations

from collections.abc import Mapping

from aiogram.types import InlineKeyboardButton

from domain.ui_text import UIText, UITextService

from .buttons import button


def ui_text(
    text_id: str,
    default_text: str,
    *,
    params: Mapping[str, object] | None = None,
    profile: str | None = None,
) -> UIText:
    return UIText(
        text_id=text_id,
        default_text=default_text,
        params=params,
        profile=profile,
    )


def resolve_ui_text(
    text_service: UITextService,
    value: UIText,
    *,
    locale: str | None = None,
    profile: str | None = None,
) -> str:
    resolved_profile = profile if profile is not None else value.profile
    return text_service.text(
        value.text_id,
        locale=locale,
        profile=resolved_profile,
        params=value.params,
        default=value.default_text,
    )


def text_button(
    text_service: UITextService,
    value: UIText,
    callback_data: str,
    *,
    locale: str | None = None,
    profile: str | None = None,
) -> InlineKeyboardButton:
    return button(
        resolve_ui_text(
            text_service,
            value,
            locale=locale,
            profile=profile,
        ),
        callback_data,
    )
