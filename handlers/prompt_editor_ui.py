from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.runtime import PromptRequest


def operation_action_text(operation: str) -> str:
    if operation == "generate":
        return "Генерировать"
    return "Продолжить"


def back_keyboard(
    callback_data: str = "pe:back",
    text: str = "⬅️ Назад",
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=text,
                    callback_data=callback_data,
                )
            ]
        ]
    )


def editor_keyboard(
    req: PromptRequest,
    *,
    smart_prompt_enabled: bool,
    pro_mode: bool = False,
) -> InlineKeyboardMarkup:
    params = req.params
    smart_label = "🧠 Smart" if smart_prompt_enabled else "🧠 Smart (off)"

    enh_count = sum(
        [
            params.enable_hires_fix,
            params.enable_freeu,
            params.enable_pag,
            bool(params.upscale_model),
            params.enable_tiled_diffusion,
        ]
    )
    enh_label = f"✨ Улучшения ({enh_count})" if enh_count else "✨ Улучшения"

    mode_btn_text = "🟢 Простой" if pro_mode else "🔧 Про"

    if pro_mode:
        rows = [
            [
                InlineKeyboardButton(
                    text="🧪 Checkpoint",
                    callback_data="pe:edit:checkpoint",
                ),
                InlineKeyboardButton(
                    text=f"🧲 LoRA ({len(params.loras)})",
                    callback_data="pe:edit:lora",
                ),
            ],
            [
                InlineKeyboardButton(text="🟢 Positive", callback_data="pe:edit:positive"),
                InlineKeyboardButton(text="🔴 Negative", callback_data="pe:edit:negative"),
            ],
            [
                InlineKeyboardButton(text=smart_label, callback_data="pe:smart:start"),
            ],
            [
                InlineKeyboardButton(
                    text="📐 Сэмплинг",
                    callback_data="pe:sub:sampling",
                ),
                InlineKeyboardButton(
                    text="🖼 Изображение",
                    callback_data="pe:sub:image",
                ),
                InlineKeyboardButton(
                    text=enh_label,
                    callback_data="pe:sub:enhancements",
                ),
            ],
            [
                InlineKeyboardButton(
                    text=f"📏 {params.width}×{params.height}",
                    callback_data="pe:edit:size",
                ),
                InlineKeyboardButton(
                    text=f"🔢 Steps {params.steps}",
                    callback_data="pe:edit:steps",
                ),
                InlineKeyboardButton(text=f"CFG {params.cfg}", callback_data="pe:edit:cfg"),
            ],
            [
                InlineKeyboardButton(
                    text="💾 Сохранить",
                    callback_data="pe:save",
                ),
                InlineKeyboardButton(
                    text="📚 Пресеты",
                    callback_data="pe:presets",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📋 Копировать/вставить",
                    callback_data="pe:exchange",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ В меню",
                    callback_data="menu:root",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"▶️ {operation_action_text(req.operation)}",
                    callback_data="pe:proceed",
                ),
                InlineKeyboardButton(
                    text=mode_btn_text,
                    callback_data="pe:toggle:mode",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="pe:cancel",
                ),
            ],
        ]
    else:
        rows = [
            [
                InlineKeyboardButton(
                    text="🧪 Checkpoint",
                    callback_data="pe:edit:checkpoint",
                ),
                InlineKeyboardButton(
                    text=f"🧲 LoRA ({len(params.loras)})",
                    callback_data="pe:edit:lora",
                ),
            ],
            [
                InlineKeyboardButton(text="🟢 Positive", callback_data="pe:edit:positive"),
                InlineKeyboardButton(text="🔴 Negative", callback_data="pe:edit:negative"),
            ],
            [
                InlineKeyboardButton(text=smart_label, callback_data="pe:smart:start"),
            ],
            [
                InlineKeyboardButton(
                    text=f"📏 {params.width}×{params.height}",
                    callback_data="pe:edit:size",
                ),
                InlineKeyboardButton(
                    text="⚙️ Ещё настроек",
                    callback_data="pe:sub:more",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="📋 Копировать/вставить",
                    callback_data="pe:exchange",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ В меню",
                    callback_data="menu:root",
                )
            ],
            [
                InlineKeyboardButton(
                    text=f"▶️ {operation_action_text(req.operation)}",
                    callback_data="pe:proceed",
                ),
                InlineKeyboardButton(
                    text=mode_btn_text,
                    callback_data="pe:toggle:mode",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="pe:cancel",
                ),
            ],
        ]

    return InlineKeyboardMarkup(inline_keyboard=rows)
