from __future__ import annotations

from collections.abc import Iterable, Sequence

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def build_row(*buttons: InlineKeyboardButton) -> list[InlineKeyboardButton]:
    return [*buttons]


def build_keyboard(rows: Iterable[Sequence[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[list(row) for row in rows])
