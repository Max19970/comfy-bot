from __future__ import annotations

from dataclasses import dataclass

from aiogram.types import InlineKeyboardButton

from .buttons import back_button, cancel_button, menu_root_button


@dataclass(frozen=True)
class MenuNavSpec:
    back_callback: str | None = None
    cancel_callback: str | None = None
    root_callback: str | None = None


def menu_nav_row(spec: MenuNavSpec) -> list[InlineKeyboardButton]:
    row: list[InlineKeyboardButton] = []
    if spec.back_callback:
        row.append(back_button(spec.back_callback))
    if spec.cancel_callback:
        row.append(cancel_button(spec.cancel_callback))
    if spec.root_callback:
        row.append(menu_root_button(spec.root_callback))
    return row
