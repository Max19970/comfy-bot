from __future__ import annotations

from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

from core.callbacks import IndexedSelectionCallback, ValueSelectionCallback
from core.ui import custom_btn


async def parse_indexed_selection(
    cb: CallbackQuery,
    *,
    prefix: str,
    invalid_text: str = "❌ Некорректный запрос.",
) -> int | None:
    parsed = IndexedSelectionCallback.parse(cb.data or "", prefix=prefix)
    if parsed is None:
        await cb.answer(invalid_text, show_alert=True)
        return None
    return parsed.index


async def parse_value_selection(
    cb: CallbackQuery,
    *,
    prefix: str,
    invalid_text: str = "❌ Некорректный запрос.",
) -> str | None:
    parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
    if parsed is None:
        await cb.answer(invalid_text, show_alert=True)
        return None
    return parsed.value


def scalar_choice_keyboard(
    *,
    prefix: str,
    values_rows: list[list[str]],
    back_callback: str,
) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(text=value, callback_data=f"{prefix}:{value}")
            for value in values_row
        ]
        for values_row in values_rows
    ]
    rows.append(custom_btn(f"{prefix}:custom"))
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=back_callback)])
    return InlineKeyboardMarkup(inline_keyboard=rows)
