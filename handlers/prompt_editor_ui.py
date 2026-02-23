from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.runtime import PromptRequest
from core.ui_kit import MenuNavSpec, back_button, build_keyboard, build_row, menu_nav_row
from core.ui_kit.buttons import button, cancel_button

from .prompt_editor_enhancements import enhancements_menu_label


def _tx(
    translate: Callable[[str, str | None, str], str] | None,
    key: str,
    locale: str | None,
    default: str,
) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


@dataclass(frozen=True)
class PromptEditorShell:
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    editor_keyboard: Callable[..., InlineKeyboardMarkup]


def build_prompt_editor_shell() -> PromptEditorShell:
    return PromptEditorShell(
        back_keyboard=back_keyboard,
        editor_keyboard=editor_keyboard,
    )


def operation_action_text(
    operation: str,
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> str:
    if operation == "generate":
        return _tx(translate, "prompt_editor.ui.action.generate", locale, "Generate")
    return _tx(translate, "prompt_editor.ui.action.continue", locale, "Continue")


def back_keyboard(
    callback_data: str = "pe:back",
    text: str | None = None,
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    back_text = text or _tx(translate, "common.action.back", locale, "⬅️ Back")
    return build_keyboard([build_row(back_button(callback_data, text=back_text))])


def _editor_header_rows(
    *,
    lora_count: int,
    smart_label: str,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    return [
        [
            button(
                _tx(translate, "prompt_editor.ui.button.checkpoint", locale, "🧪 Checkpoint"),
                "pe:edit:checkpoint",
            ),
            button(
                _tx(
                    translate,
                    "prompt_editor.ui.button.lora",
                    locale,
                    "🧲 LoRA ({count})",
                ).format(count=lora_count),
                "pe:edit:lora",
            ),
        ],
        [
            button(
                _tx(translate, "prompt_editor.ui.button.positive", locale, "🟢 Positive"),
                "pe:edit:positive",
            ),
            button(
                _tx(translate, "prompt_editor.ui.button.negative", locale, "🔴 Negative"),
                "pe:edit:negative",
            ),
        ],
        [button(smart_label, "pe:smart:start")],
    ]


def _editor_shared_footer_rows(
    req: PromptRequest,
    *,
    mode_btn_text: str,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> list[list[InlineKeyboardButton]]:
    return [
        [
            button(
                _tx(
                    translate,
                    "prompt_editor.ui.button.exchange",
                    locale,
                    "📋 Copy/Paste",
                ),
                "pe:exchange",
            )
        ],
        menu_nav_row(MenuNavSpec(root_callback="menu:root")),
        [
            button(
                f"▶️ {operation_action_text(req.operation, translate=translate, locale=locale)}",
                "pe:proceed",
            ),
            button(mode_btn_text, "pe:toggle:mode"),
            cancel_button("pe:cancel"),
        ],
    ]


def editor_keyboard(
    req: PromptRequest,
    *,
    smart_prompt_enabled: bool,
    pro_mode: bool = False,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    params = req.params
    size_btn_text = _tx(
        translate,
        "prompt_editor.ui.button.size",
        locale,
        "📏 {width}×{height}",
    ).format(width=params.width, height=params.height)
    steps_btn_text = _tx(
        translate,
        "prompt_editor.ui.button.steps",
        locale,
        "🔢 Steps {steps}",
    ).format(steps=params.steps)
    cfg_btn_text = _tx(
        translate,
        "prompt_editor.ui.button.cfg",
        locale,
        "CFG {cfg}",
    ).format(cfg=params.cfg)
    smart_label = (
        _tx(translate, "prompt_editor.ui.button.smart.on", locale, "🧠 Smart")
        if smart_prompt_enabled
        else _tx(translate, "prompt_editor.ui.button.smart.off", locale, "🧠 Smart (off)")
    )

    enh_label = enhancements_menu_label(params, translate=translate, locale=locale)

    mode_btn_text = (
        _tx(translate, "prompt_editor.ui.button.mode.simple", locale, "🟢 Simple")
        if pro_mode
        else _tx(translate, "prompt_editor.ui.button.mode.pro", locale, "🔧 Pro")
    )
    rows = _editor_header_rows(
        lora_count=len(params.loras),
        smart_label=smart_label,
        translate=translate,
        locale=locale,
    )

    if pro_mode:
        rows.extend(
            [
                [
                    button(
                        _tx(translate, "prompt_editor.ui.button.sampling", locale, "📐 Sampling"),
                        "pe:sub:sampling",
                    ),
                    button(
                        _tx(translate, "prompt_editor.ui.button.image", locale, "🖼 Image"),
                        "pe:sub:image",
                    ),
                    button(enh_label, "pe:sub:enhancements"),
                ],
                [
                    button(size_btn_text, "pe:edit:size"),
                    button(steps_btn_text, "pe:edit:steps"),
                    button(cfg_btn_text, "pe:edit:cfg"),
                ],
                [
                    button(
                        _tx(translate, "prompt_editor.ui.button.save", locale, "💾 Save"), "pe:save"
                    ),
                    button(
                        _tx(translate, "prompt_editor.ui.button.presets", locale, "📚 Presets"),
                        "pe:presets",
                    ),
                ],
            ]
        )
    else:
        rows.append(
            [
                button(size_btn_text, "pe:edit:size"),
                button(
                    _tx(
                        translate,
                        "prompt_editor.ui.button.more_settings",
                        locale,
                        "⚙️ More settings",
                    ),
                    "pe:sub:more",
                ),
            ]
        )

    rows.extend(
        _editor_shared_footer_rows(
            req,
            mode_btn_text=mode_btn_text,
            translate=translate,
            locale=locale,
        )
    )
    return build_keyboard(rows)
