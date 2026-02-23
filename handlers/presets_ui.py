from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, menu_root_button


def presets_title_text() -> str:
    return "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:"


def save_name_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard([[back_button("preset:save:back")]])


def overwrite_confirmation_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [[button("✅ Перезаписать", "preset:overwrite:yes"), cancel_button("preset:overwrite:no")]]
    )


def result_save_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard([[button("💾 Сохранить как пресет", "save_preset")]])


def empty_presets_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [[back_button("menu:generation", text="⬅️ Генерация")], [menu_root_button()]]
    )


def presets_list_keyboard(
    uid: int,
    names: list[str],
    *,
    has_active_prompt_request: bool,
    confirm_delete_index: int | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            button(f"📂 {name}", f"preset_load:{index}"),
            button(_delete_button_text(name, index, confirm_delete_index), f"preset_del:{index}"),
        ]
        for index, name in enumerate(names)
    ]
    del uid
    if has_active_prompt_request:
        rows.append([back_button("pe:back", text="⬅️ В редактор")])
    else:
        rows.append([back_button("menu:generation", text="⬅️ Генерация")])
    rows.append([menu_root_button()])
    return build_keyboard(rows)


def _delete_button_text(name: str, index: int, confirm_delete_index: int | None) -> str:
    if confirm_delete_index != index:
        return "🗑"

    short_name = name if len(name) <= 18 else f"{name[:15]}..."
    return f"⚠️ Удалить «{short_name}»?"
