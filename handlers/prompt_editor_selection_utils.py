from __future__ import annotations

from collections.abc import Callable

from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from core.callbacks import IndexedSelectionCallback, ValueSelectionCallback
from core.ui import custom_btn

TranslateText = Callable[[str, str | None, str], str]


def _tx(translate: TranslateText | None, key: str, locale: str | None, default: str) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


async def parse_indexed_selection(
    cb: CallbackQuery,
    *,
    prefix: str,
    invalid_text: str = "❌ Некорректный запрос.",
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> int | None:
    parsed = IndexedSelectionCallback.parse(cb.data or "", prefix=prefix)
    if parsed is None:
        await cb.answer(
            _tx(
                translate,
                "common.alert.invalid_request",
                locale,
                invalid_text,
            ),
            show_alert=True,
        )
        return None
    return parsed.index


async def parse_value_selection(
    cb: CallbackQuery,
    *,
    prefix: str,
    invalid_text: str = "❌ Некорректный запрос.",
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str | None:
    parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
    if parsed is None:
        await cb.answer(
            _tx(
                translate,
                "common.alert.invalid_request",
                locale,
                invalid_text,
            ),
            show_alert=True,
        )
        return None
    return parsed.value


def scalar_choice_keyboard(
    *,
    prefix: str,
    values_rows: list[list[str]],
    back_callback: str,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text=value, callback_data=f"{prefix}:{value}")
            for value in values_row
        ]
        for values_row in values_rows
    ]
    rows.append(custom_btn(f"{prefix}:custom"))
    rows.append(
        [
            InlineKeyboardButton(
                text=_tx(translate, "common.action.back", locale, "⬅️ Назад"),
                callback_data=back_callback,
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
