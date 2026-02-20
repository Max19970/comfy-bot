from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.html_utils import h


def prompt_input_text(
    field_name: str,
    current_value: str,
    *,
    allow_clear: bool = False,
) -> str:
    label = "Positive" if field_name == "positive" else "Negative"
    base = f"✏️ Введите <b>{label}</b>"
    if allow_clear:
        base += " (или <code>-</code> для очистки)"
    base += ":"

    value = current_value.strip()
    if not value:
        return base

    return f"Текущее значение {label}:\n<code>{h(value)}</code>\n\n{base}"


def merge_prompt_text(base: str, addon: str) -> str:
    base_text = base.strip()
    addon_text = addon.strip()
    if not addon_text:
        return base_text
    if not base_text:
        return addon_text
    if addon_text.casefold() in base_text.casefold():
        return base_text
    return f"{base_text}, {addon_text}"


def prompt_preview(text: str, max_len: int = 1200) -> str:
    value = text.strip()
    if not value:
        return "—"
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def smart_prompt_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Принять",
                    callback_data="pe:smart:apply",
                ),
                InlineKeyboardButton(
                    text="✏️ Редактировать",
                    callback_data="pe:smart:edit",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📝 Объединить",
                    callback_data="pe:smart:merge",
                ),
                InlineKeyboardButton(
                    text="↩️ Вернуть прежний",
                    callback_data="pe:smart:restore",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ В редактор",
                    callback_data="pe:smart:cancel",
                )
            ],
        ]
    )
