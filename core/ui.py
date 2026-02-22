from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.models import GenerationParams
from core.ui_kit import (
    build_keyboard,
    build_page_window,
    custom_value_button,
    item_callback,
    page_nav_row,
)
from core.ui_summary import MAX_REFERENCE_IMAGES_DEFAULT
from core.ui_summary import loras_text as _loras_text
from core.ui_summary import params_summary_full as _params_summary_full
from core.ui_summary import params_summary_simple as _params_summary_simple

SIZES = [
    ("512x512", 512, 512),
    ("512x768", 512, 768),
    ("768x512", 768, 512),
    ("768x768", 768, 768),
    ("768x1024", 768, 1024),
    ("1024x768", 1024, 768),
    ("1024x1024", 1024, 1024),
    ("1024x1536", 1024, 1536),
    ("1536x1024", 1536, 1024),
]

PAGE_SIZE = 8
MAX_REFERENCE_IMAGES = MAX_REFERENCE_IMAGES_DEFAULT


def paginated_keyboard(
    items: list[str],
    page: int,
    prefix: str,
    *,
    extra: list[list[InlineKeyboardButton]] | None = None,
) -> InlineKeyboardMarkup:
    window = build_page_window(items, page, PAGE_SIZE)

    rows: list[list[InlineKeyboardButton]] = []
    for i, item in enumerate(window.items):
        short = item if len(item) <= 40 else item[:37] + "..."
        rows.append(
            [
                InlineKeyboardButton(
                    text=short,
                    callback_data=item_callback(prefix, window.start_index + i),
                )
            ]
        )

    rows.append(page_nav_row(prefix, window.page, window.total_pages))
    if extra:
        rows.extend(extra)
    return build_keyboard(rows)


def custom_btn(callback_data: str) -> list[InlineKeyboardButton]:
    return [custom_value_button(callback_data)]


def loras_text(loras: list[tuple[str, float]]) -> str:
    return _loras_text(loras)


def params_summary(params: GenerationParams) -> str:
    return params_summary_for_mode(params, pro_mode=True)


def params_summary_simple(params: GenerationParams) -> str:
    return _params_summary_simple(params)


def params_summary_for_mode(params: GenerationParams, *, pro_mode: bool) -> str:
    if pro_mode:
        return params_summary_full(params)
    return params_summary_simple(params)


def params_summary_full(params: GenerationParams) -> str:
    return _params_summary_full(params, max_reference_images=MAX_REFERENCE_IMAGES)
