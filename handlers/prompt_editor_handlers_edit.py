from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient
from core.callbacks import IndexedSelectionCallback, ValueSelectionCallback
from core.interaction import callback_message as interaction_callback_message
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui import custom_btn


@dataclass
class PromptEditorEditHandlersDeps:
    runtime: RuntimeStore
    client: ComfyUIClient
    sizes: list[tuple[str, int, int]]
    callback_user_id: Callable[[CallbackQuery], int]
    get_user_pro_mode: Callable[[RuntimeStore, int], bool]
    set_user_pro_mode: Callable[[RuntimeStore, int, bool], None]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    open_paginated_choice: Callable[..., Awaitable[None]]
    change_paginated_choice_page: Callable[..., Awaitable[None]]
    set_prompt_param_from_callback: Callable[..., Awaitable[None]]
    show_prompt_editor: Callable[..., Awaitable[None]]
    prompt_input_text: Callable[..., str]
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    incompatible_loras: Callable[[GenerationParams], list[tuple[str, str, str]]]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def register_prompt_editor_edit_handlers(
    router: Router,
    deps: PromptEditorEditHandlersDeps,
) -> None:
    _callback_message = interaction_callback_message

    async def _selected_index(cb: CallbackQuery, *, prefix: str) -> int | None:
        parsed = IndexedSelectionCallback.parse(cb.data or "", prefix=prefix)
        if parsed is None:
            await cb.answer("❌ Некорректный запрос.", show_alert=True)
            return None
        return parsed.index

    async def _selected_value(cb: CallbackQuery, *, prefix: str) -> str | None:
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
        if parsed is None:
            await cb.answer("❌ Некорректный запрос.", show_alert=True)
            return None
        return parsed.value

    @router.callback_query(F.data == "pe:edit:positive")
    async def pe_edit_positive(cb: CallbackQuery, state: FSMContext):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        await state.set_state(PromptEditorStates.entering_positive)
        await message.edit_text(
            deps.prompt_input_text("positive", req.params.positive),
            reply_markup=deps.back_keyboard(),
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_positive, F.text)
    async def pe_positive_entered(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        req.params.positive = (msg.text or "").strip()
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Positive обновлён.")

    @router.callback_query(F.data == "pe:edit:negative")
    async def pe_edit_negative(cb: CallbackQuery, state: FSMContext):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        await state.set_state(PromptEditorStates.entering_negative)
        await message.edit_text(
            deps.prompt_input_text(
                "negative",
                req.params.negative,
                allow_clear=True,
            ),
            reply_markup=deps.back_keyboard(),
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_negative, F.text)
    async def pe_negative_entered(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        value = (msg.text or "").strip()
        req.params.negative = "" if value == "-" else value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Negative обновлён.")

    @router.callback_query(F.data == "pe:edit:checkpoint")
    async def pe_edit_checkpoint(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите checkpoint:",
            items=deps.client.info.checkpoints,
            prefix="pe_ckpt",
        )

    @router.callback_query(F.data.startswith("pe_ckpt_page:"))
    async def pe_ckpt_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=deps.client.info.checkpoints,
            prefix="pe_ckpt",
        )

    @router.callback_query(F.data.startswith("pe_ckpt:"))
    async def pe_ckpt_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_ckpt")
        if idx is None:
            return
        items = deps.client.info.checkpoints
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return

        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        req.params.checkpoint = items[idx]

        notice = "✅ Checkpoint обновлён."
        bad_loras = deps.incompatible_loras(req.params)
        if bad_loras:
            names = ", ".join(name for name, _, _ in bad_loras[:3])
            suffix = "" if len(bad_loras) <= 3 else f" и ещё {len(bad_loras) - 3}"
            notice += f" ⚠️ Несовместимые LoRA: {names}{suffix}."

        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        await deps.show_prompt_editor(message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:edit:sampler")
    async def pe_edit_sampler(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите sampler:",
            items=deps.client.info.samplers or ["euler"],
            prefix="pe_smpl",
        )

    @router.callback_query(F.data.startswith("pe_smpl_page:"))
    async def pe_sampler_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=deps.client.info.samplers or ["euler"],
            prefix="pe_smpl",
        )

    @router.callback_query(F.data.startswith("pe_smpl:"))
    async def pe_sampler_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_smpl")
        if idx is None:
            return
        items = deps.client.info.samplers or ["euler"]
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="sampler",
            value=items[idx],
            notice="✅ Sampler обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:scheduler")
    async def pe_edit_scheduler(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите scheduler:",
            items=deps.client.info.schedulers or ["normal"],
            prefix="pe_sched",
        )

    @router.callback_query(F.data.startswith("pe_sched_page:"))
    async def pe_sched_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=deps.client.info.schedulers or ["normal"],
            prefix="pe_sched",
        )

    @router.callback_query(F.data.startswith("pe_sched:"))
    async def pe_sched_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_sched")
        if idx is None:
            return
        items = deps.client.info.schedulers or ["normal"]
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="scheduler",
            value=items[idx],
            notice="✅ Scheduler обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:upscaler")
    async def pe_edit_upscaler(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите upscaler:",
            items=["(без апскейла)"] + deps.client.info.upscale_models,
            prefix="pe_upsc",
        )

    @router.callback_query(F.data.startswith("pe_upsc_page:"))
    async def pe_upsc_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=["(без апскейла)"] + deps.client.info.upscale_models,
            prefix="pe_upsc",
        )

    @router.callback_query(F.data.startswith("pe_upsc:"))
    async def pe_upsc_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_upsc")
        if idx is None:
            return
        items = ["(без апскейла)"] + deps.client.info.upscale_models
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        chosen = items[idx]
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="upscale_model",
            value="" if chosen == "(без апскейла)" else chosen,
            notice="✅ Upscaler обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:vae")
    async def pe_edit_vae(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите VAE:",
            items=["(из checkpoint)"] + deps.client.info.vaes,
            prefix="pe_vae",
        )

    @router.callback_query(F.data.startswith("pe_vae_page:"))
    async def pe_vae_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=["(из checkpoint)"] + deps.client.info.vaes,
            prefix="pe_vae",
        )

    @router.callback_query(F.data.startswith("pe_vae:"))
    async def pe_vae_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_vae")
        if idx is None:
            return
        items = ["(из checkpoint)"] + deps.client.info.vaes
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        chosen = items[idx]
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="vae_name",
            value="" if chosen == "(из checkpoint)" else chosen,
            notice="✅ VAE обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:controlnet")
    async def pe_edit_controlnet(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите ControlNet:",
            items=["(выкл)"] + deps.client.info.controlnets,
            prefix="pe_cn",
        )

    @router.callback_query(F.data.startswith("pe_cn_page:"))
    async def pe_cn_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=["(выкл)"] + deps.client.info.controlnets,
            prefix="pe_cn",
        )

    @router.callback_query(F.data.startswith("pe_cn:"))
    async def pe_cn_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_cn")
        if idx is None:
            return
        items = ["(выкл)"] + deps.client.info.controlnets
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        chosen = items[idx]
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="controlnet_name",
            value="" if chosen == "(выкл)" else chosen,
            notice="✅ ControlNet обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:embedding")
    async def pe_edit_embedding(cb: CallbackQuery):
        await deps.open_paginated_choice(
            cb,
            title="Выберите embedding:",
            items=["(без embedding)"] + deps.client.info.embeddings,
            prefix="pe_emb",
        )

    @router.callback_query(F.data.startswith("pe_emb_page:"))
    async def pe_emb_page(cb: CallbackQuery):
        await deps.change_paginated_choice_page(
            cb,
            items=["(без embedding)"] + deps.client.info.embeddings,
            prefix="pe_emb",
        )

    @router.callback_query(F.data.startswith("pe_emb:"))
    async def pe_emb_chosen(cb: CallbackQuery, state: FSMContext):
        idx = await _selected_index(cb, prefix="pe_emb")
        if idx is None:
            return
        items = ["(без embedding)"] + deps.client.info.embeddings
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return
        chosen = items[idx]
        await deps.set_prompt_param_from_callback(
            cb,
            state,
            field="embedding_name",
            value="" if chosen == "(без embedding)" else chosen,
            notice="✅ Embedding обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:controlnet_strength")
    async def pe_edit_controlnet_strength(cb: CallbackQuery):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="0.4", callback_data="pe_cns:0.4"),
                    InlineKeyboardButton(text="0.7", callback_data="pe_cns:0.7"),
                    InlineKeyboardButton(text="1.0", callback_data="pe_cns:1.0"),
                ],
                [
                    InlineKeyboardButton(text="1.2", callback_data="pe_cns:1.2"),
                    InlineKeyboardButton(text="1.5", callback_data="pe_cns:1.5"),
                ],
                custom_btn("pe_cns:custom"),
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("ControlNet strength (0.0-2.0):", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_cns:"))
    async def pe_controlnet_strength_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_cns")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_controlnet_strength)
            await message.edit_text(
                "ControlNet strength (0.0-2.0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.controlnet_strength = float(value)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="✅ ControlNet strength обновлён.",
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_controlnet_strength, F.text)
    async def pe_custom_controlnet_strength(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            if not 0.0 <= value <= 2.0:
                raise ValueError
        except ValueError:
            await msg.answer("Число 0.0-2.0:")
            return

        req.params.controlnet_strength = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(
            msg,
            state,
            uid,
            notice="✅ ControlNet strength обновлён.",
        )

    @router.callback_query(F.data == "pe:edit:size")
    async def pe_edit_size(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        rows: list[list[InlineKeyboardButton]] = []
        row: list[InlineKeyboardButton] = []
        for label, width, height in deps.sizes:
            row.append(InlineKeyboardButton(text=label, callback_data=f"pe_size:{width}:{height}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton(text="Свой размер...", callback_data="pe_size:custom")])
        rows.append(
            [
                InlineKeyboardButton(
                    text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                    callback_data="pe:back",
                )
            ]
        )
        await message.edit_text(
            "Размер изображения:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_size:"))
    async def pe_size_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix="pe_size")
        if parsed is None:
            await cb.answer("❌ Некорректный запрос.", show_alert=True)
            return
        if parsed.value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_size)
            await message.edit_text(
                "Введите размер ШИРИНАxВЫСОТА (например 640x960):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        size_parts = parsed.value.split(":", 1)
        if len(size_parts) != 2:
            await cb.answer("❌ Некорректный размер.", show_alert=True)
            return
        try:
            req.params.width, req.params.height = int(size_parts[0]), int(size_parts[1])
        except ValueError:
            await cb.answer("❌ Некорректный размер.", show_alert=True)
            return
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ Размер обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_size, F.text)
    async def pe_custom_size(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        text = (msg.text or "").strip().lower().replace(" ", "")
        if "x" not in text:
            await msg.answer("Формат: ШИРИНАxВЫСОТА")
            return

        try:
            width_s, height_s = text.split("x")
            width_i, height_i = int(width_s), int(height_s)
            if not (64 <= width_i <= 4096 and 64 <= height_i <= 4096):
                raise ValueError
        except ValueError:
            await msg.answer("Размеры 64-4096. Попробуйте ещё:")
            return

        req.params.width, req.params.height = width_i, height_i
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Размер обновлён.")

    @router.callback_query(F.data == "pe:edit:steps")
    async def pe_edit_steps(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_steps:{value}")
                    for value in (10, 15, 20)
                ],
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_steps:{value}")
                    for value in (25, 30, 40)
                ],
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_steps:{value}")
                    for value in (50, 60, 80)
                ],
                custom_btn("pe_steps:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("Steps:", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_steps:"))
    async def pe_steps_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_steps")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_steps)
            await message.edit_text(
                "Steps (1-200):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.steps = int(value)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ Steps обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_steps, F.text)
    async def pe_custom_steps(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = int((msg.text or "").strip())
            if not 1 <= value <= 200:
                raise ValueError
        except ValueError:
            await msg.answer("Целое число 1-200:")
            return

        req.params.steps = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Steps обновлён.")

    @router.callback_query(F.data == "pe:edit:cfg")
    async def pe_edit_cfg(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_cfg:{value}")
                    for value in (1.0, 3.0, 5.0)
                ],
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_cfg:{value}")
                    for value in (7.0, 7.5, 8.0)
                ],
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_cfg:{value}")
                    for value in (10.0, 12.0, 15.0)
                ],
                custom_btn("pe_cfg:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("CFG:", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_cfg:"))
    async def pe_cfg_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_cfg")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_cfg)
            await message.edit_text(
                "CFG (0.0-30.0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.cfg = float(value)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ CFG обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_cfg, F.text)
    async def pe_custom_cfg(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            if not 0.0 <= value <= 30.0:
                raise ValueError
        except ValueError:
            await msg.answer("Число 0.0-30.0:")
            return

        req.params.cfg = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ CFG обновлён.")

    @router.callback_query(F.data == "pe:edit:denoise")
    async def pe_edit_denoise(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_dn:{value}")
                    for value in (0.3, 0.5, 0.7)
                ],
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_dn:{value}")
                    for value in (0.8, 0.9, 1.0)
                ],
                custom_btn("pe_dn:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("Denoise:", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_dn:"))
    async def pe_dn_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_dn")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_denoise)
            await message.edit_text(
                "Denoise (0.0-1.0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.denoise = float(value)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ Denoise обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_denoise, F.text)
    async def pe_custom_dn(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            if not 0.0 <= value <= 1.0:
                raise ValueError
        except ValueError:
            await msg.answer("Число 0.0-1.0:")
            return

        req.params.denoise = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Denoise обновлён.")

    @router.callback_query(F.data == "pe:edit:ref_strength")
    async def pe_edit_ref_strength(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="0.3", callback_data="pe_rstr:0.3"),
                    InlineKeyboardButton(text="0.5", callback_data="pe_rstr:0.5"),
                    InlineKeyboardButton(text="0.7", callback_data="pe_rstr:0.7"),
                ],
                [
                    InlineKeyboardButton(text="0.9", callback_data="pe_rstr:0.9"),
                    InlineKeyboardButton(text="1.0", callback_data="pe_rstr:1.0"),
                    InlineKeyboardButton(text="1.2", callback_data="pe_rstr:1.2"),
                ],
                [
                    InlineKeyboardButton(text="1.5", callback_data="pe_rstr:1.5"),
                    InlineKeyboardButton(text="1.8", callback_data="pe_rstr:1.8"),
                    InlineKeyboardButton(text="2.0", callback_data="pe_rstr:2.0"),
                ],
                custom_btn("pe_rstr:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text(
            "Сила референса (0.0-2.0):\n"
            "- Для IP-Adapter: больше = сильнее похожесть\n"
            "- Для img2img fallback: больше = ниже denoise",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_rstr:"))
    async def pe_ref_strength_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_rstr")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_reference_strength)
            await message.edit_text(
                "Сила референса (0.0-2.0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.reference_strength = float(value)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="✅ Сила референса обновлена.",
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_reference_strength, F.text)
    async def pe_custom_ref_strength(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            if not 0.0 <= value <= 2.0:
                raise ValueError
        except ValueError:
            await msg.answer("Число 0.0-2.0:")
            return

        req.params.reference_strength = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Сила референса обновлена.")

    @router.callback_query(F.data == "pe:edit:seed")
    async def pe_edit_seed(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="Случайный", callback_data="pe_seed:random")],
                custom_btn("pe_seed:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("Seed:", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_seed:"))
    async def pe_seed_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_seed")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_seed)
            await message.edit_text(
                "Seed (>= 0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.seed = -1
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ Seed обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_seed, F.text)
    async def pe_custom_seed(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = int((msg.text or "").strip())
            if value < 0:
                raise ValueError
        except ValueError:
            await msg.answer("Целое число >= 0:")
            return

        req.params.seed = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(msg, state, uid, notice="✅ Seed обновлён.")

    @router.callback_query(F.data == "pe:edit:batch")
    async def pe_edit_batch(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(value), callback_data=f"pe_batch:{value}")
                    for value in (1, 2, 3, 4)
                ],
                custom_btn("pe_batch:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text("Batch size:", reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_batch:"))
    async def pe_batch_chosen(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = await _selected_value(cb, prefix="pe_batch")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_batch)
            await message.edit_text(
                "Batch (1-16):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        req.params.batch_size = int(value)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice="✅ Batch обновлён.")
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_batch, F.text)
    async def pe_custom_batch(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = int((msg.text or "").strip())
            if not 1 <= value <= 16:
                raise ValueError
        except ValueError:
            await msg.answer("Число 1-16:")
            return

        req.params.batch_size = value
        await deps.cleanup_user_message(msg)
        await deps.show_prompt_editor(
            msg,
            state,
            uid,
            notice="\u2705 Batch \u043e\u0431\u043d\u043e\u0432\u043b\u0451\u043d.",
        )

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    @router.callback_query(F.data == "pe:toggle:mode")
    async def pe_toggle_mode(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, _ = payload
        current = deps.get_user_pro_mode(deps.runtime, uid)
        deps.set_user_pro_mode(deps.runtime, uid, not current)
        mode_name = "\U0001f527 Про" if not current else "\U0001f7e2 Простой"
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=f"Режим переключён: {mode_name}",
        )
        await cb.answer()
