from __future__ import annotations

from collections.abc import Callable

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, menu_root_button


def _tx(
    translate: Callable[[str, str | None, str], str] | None,
    key: str,
    locale: str | None,
    default: str,
) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


def presets_title_text(
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> str:
    return _tx(
        translate,
        "presets.title",
        locale,
        "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:",
    )


def save_name_keyboard(
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    back_text = _tx(translate, "presets.back", locale, "⬅️ Назад")
    return build_keyboard([[back_button("preset:save:back", text=back_text)]])


def overwrite_confirmation_keyboard(
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    overwrite_text = _tx(translate, "presets.overwrite_button", locale, "✅ Перезаписать")
    cancel_text = _tx(translate, "presets.cancel_button", locale, "❌ Отмена")
    return build_keyboard(
        [
            [
                button(overwrite_text, "preset:overwrite:yes"),
                cancel_button("preset:overwrite:no", text=cancel_text),
            ]
        ]
    )


def result_save_keyboard(
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    save_text = _tx(translate, "presets.save_button", locale, "💾 Сохранить как пресет")
    return build_keyboard([[button(save_text, "save_preset")]])


def empty_presets_keyboard(
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    back_text = _tx(translate, "presets.back_generation", locale, "⬅️ Генерация")
    return build_keyboard([[back_button("menu:generation", text=back_text)], [menu_root_button()]])


def presets_list_keyboard(
    uid: int,
    names: list[str],
    *,
    has_active_prompt_request: bool,
    confirm_delete_index: int | None = None,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            button(f"📂 {name}", f"preset_load:{index}"),
            button(
                _delete_button_text(
                    name,
                    index,
                    confirm_delete_index,
                    translate=translate,
                    locale=locale,
                ),
                f"preset_del:{index}",
            ),
        ]
        for index, name in enumerate(names)
    ]
    del uid
    if has_active_prompt_request:
        rows.append(
            [
                back_button(
                    "pe:back",
                    text=_tx(translate, "presets.back_editor", locale, "⬅️ В редактор"),
                )
            ]
        )
    else:
        rows.append(
            [
                back_button(
                    "menu:generation",
                    text=_tx(translate, "presets.back_generation", locale, "⬅️ Генерация"),
                )
            ]
        )
    rows.append([menu_root_button()])
    return build_keyboard(rows)


def _delete_button_text(
    name: str,
    index: int,
    confirm_delete_index: int | None,
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> str:
    if confirm_delete_index != index:
        return "🗑"

    short_name = name if len(name) <= 18 else f"{name[:15]}..."
    template = _tx(
        translate,
        "presets.delete_confirm",
        locale,
        "⚠️ Удалить «{name}»?",
    )
    try:
        return template.format(name=short_name)
    except (KeyError, ValueError):
        return template
