from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.runtime import PromptRequest
from core.ui_kit import MenuNavSpec, back_button, build_keyboard, build_row, menu_nav_row
from core.ui_kit.buttons import button, cancel_button

from .prompt_editor_enhancements import enhancements_menu_label


@dataclass(frozen=True)
class PromptEditorShell:
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    editor_keyboard: Callable[..., InlineKeyboardMarkup]


def build_prompt_editor_shell() -> PromptEditorShell:
    return PromptEditorShell(
        back_keyboard=back_keyboard,
        editor_keyboard=editor_keyboard,
    )


def operation_action_text(operation: str) -> str:
    if operation == "generate":
        return "Генерировать"
    return "Продолжить"


def back_keyboard(
    callback_data: str = "pe:back",
    text: str = "⬅️ Назад",
) -> InlineKeyboardMarkup:
    return build_keyboard([build_row(back_button(callback_data, text=text))])


def _editor_header_rows(
    *,
    lora_count: int,
    smart_label: str,
) -> list[list[InlineKeyboardButton]]:
    return [
        [
            button("🧪 Checkpoint", "pe:edit:checkpoint"),
            button(f"🧲 LoRA ({lora_count})", "pe:edit:lora"),
        ],
        [
            button("🟢 Positive", "pe:edit:positive"),
            button("🔴 Negative", "pe:edit:negative"),
        ],
        [button(smart_label, "pe:smart:start")],
    ]


def _editor_shared_footer_rows(
    req: PromptRequest, *, mode_btn_text: str
) -> list[list[InlineKeyboardButton]]:
    return [
        [button("📋 Копировать/вставить", "pe:exchange")],
        menu_nav_row(MenuNavSpec(root_callback="menu:root")),
        [
            button(f"▶️ {operation_action_text(req.operation)}", "pe:proceed"),
            button(mode_btn_text, "pe:toggle:mode"),
            cancel_button("pe:cancel"),
        ],
    ]


def editor_keyboard(
    req: PromptRequest,
    *,
    smart_prompt_enabled: bool,
    pro_mode: bool = False,
) -> InlineKeyboardMarkup:
    params = req.params
    smart_label = "🧠 Smart" if smart_prompt_enabled else "🧠 Smart (off)"

    enh_label = enhancements_menu_label(params)

    mode_btn_text = "🟢 Простой" if pro_mode else "🔧 Про"
    rows = _editor_header_rows(lora_count=len(params.loras), smart_label=smart_label)

    if pro_mode:
        rows.extend(
            [
                [
                    button("📐 Сэмплинг", "pe:sub:sampling"),
                    button("🖼 Изображение", "pe:sub:image"),
                    button(enh_label, "pe:sub:enhancements"),
                ],
                [
                    button(f"📏 {params.width}×{params.height}", "pe:edit:size"),
                    button(f"🔢 Steps {params.steps}", "pe:edit:steps"),
                    button(f"CFG {params.cfg}", "pe:edit:cfg"),
                ],
                [
                    button("💾 Сохранить", "pe:save"),
                    button("📚 Пресеты", "pe:presets"),
                ],
            ]
        )
    else:
        rows.append(
            [
                button(f"📏 {params.width}×{params.height}", "pe:edit:size"),
                button("⚙️ Ещё настроек", "pe:sub:more"),
            ]
        )

    rows.extend(_editor_shared_footer_rows(req, mode_btn_text=mode_btn_text))
    return build_keyboard(rows)
