from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from aiogram.types import InlineKeyboardButton

from .buttons import button, noop_button

_ItemT = TypeVar("_ItemT")


@dataclass(frozen=True)
class PageWindow(Generic[_ItemT]):
    items: list[_ItemT]
    page: int
    total_pages: int
    start_index: int


def build_page_window(items: list[_ItemT], page: int, page_size: int) -> PageWindow[_ItemT]:
    total_pages = max(1, math.ceil(len(items) / page_size))
    normalized_page = max(0, min(page, total_pages - 1))
    start = normalized_page * page_size
    return PageWindow(
        items=items[start : start + page_size],
        page=normalized_page,
        total_pages=total_pages,
        start_index=start,
    )


def item_callback(prefix: str, index: int) -> str:
    return f"{prefix}:{index}"


def page_callback(prefix: str, page: int) -> str:
    return f"{prefix}_page:{page}"


def page_nav_row(prefix: str, page: int, total_pages: int) -> list[InlineKeyboardButton]:
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(button(text="◀️", callback_data=page_callback(prefix, page - 1)))
    nav.append(noop_button(text=f"· {page + 1}/{total_pages} ·"))
    if page < total_pages - 1:
        nav.append(button(text="▶️", callback_data=page_callback(prefix, page + 1)))
    return nav
