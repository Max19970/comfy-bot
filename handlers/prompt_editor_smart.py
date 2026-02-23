from __future__ import annotations

from collections.abc import Callable

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.html_utils import h

TranslateText = Callable[[str, str | None, str], str]


def _tx(translate: TranslateText | None, key: str, locale: str | None, default: str) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


def prompt_input_text(
    field_name: str,
    current_value: str,
    *,
    allow_clear: bool = False,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    if field_name == "positive":
        label = _tx(
            translate,
            "prompt_editor.smart.label.positive",
            locale,
            "Positive",
        )
    else:
        label = _tx(
            translate,
            "prompt_editor.smart.label.negative",
            locale,
            "Negative",
        )

    base = (
        _tx(
            translate,
            "prompt_editor.smart.input.enter",
            locale,
            "✏️ Введите <b>{label}</b>",
        )
    ).format(label=label)
    if allow_clear:
        base += _tx(
            translate,
            "prompt_editor.smart.input.clear_hint",
            locale,
            " (или <code>-</code> для очистки)",
        )
    base += ":"

    value = current_value.strip()
    if not value:
        return base

    current_prefix = (
        _tx(
            translate,
            "prompt_editor.smart.input.current_value",
            locale,
            "Текущее значение {label}:",
        )
    ).format(label=label)
    return f"{current_prefix}\n<code>{h(value)}</code>\n\n{base}"


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


def prompt_preview(
    text: str,
    max_len: int = 1200,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    value = text.strip()
    if not value:
        return _tx(translate, "prompt_editor.smart.preview.empty", locale, "—")
    if len(value) <= max_len:
        return value
    return value[: max_len - 1] + "…"


def smart_prompt_result_keyboard(
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_tx(
                        translate,
                        "prompt_editor.smart_result.button.apply",
                        locale,
                        "✅ Принять",
                    ),
                    callback_data="pe:smart:apply",
                ),
                InlineKeyboardButton(
                    text=_tx(
                        translate,
                        "prompt_editor.smart_result.button.edit",
                        locale,
                        "✏️ Редактировать",
                    ),
                    callback_data="pe:smart:edit",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=_tx(
                        translate,
                        "prompt_editor.smart_result.button.merge",
                        locale,
                        "📝 Объединить",
                    ),
                    callback_data="pe:smart:merge",
                ),
                InlineKeyboardButton(
                    text=_tx(
                        translate,
                        "prompt_editor.smart_result.button.restore",
                        locale,
                        "↩️ Вернуть прежний",
                    ),
                    callback_data="pe:smart:restore",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=_tx(
                        translate,
                        "prompt_editor.smart_result.button.back_to_editor",
                        locale,
                        "⬅️ В редактор",
                    ),
                    callback_data="pe:smart:cancel",
                )
            ],
        ]
    )
