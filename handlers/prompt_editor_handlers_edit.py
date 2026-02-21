from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui import custom_btn

from .prompt_editor_selection_utils import (
    parse_indexed_selection,
    parse_value_selection,
    scalar_choice_keyboard,
)


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


@dataclass
class PaginatedParamConfig:
    open_callback: str
    prefix: str
    title: str
    field: str
    notice: str
    items_getter: Callable[[], list[str]]
    transform_value: Callable[[str], str] = lambda value: value


@dataclass
class ScalarParamConfig:
    open_callback: str
    prefix: str
    custom_state: Any
    menu_text: str
    custom_prompt: str
    invalid_input_text: str
    notice: str
    values_rows: list[list[str]]
    set_value: Callable[[GenerationParams, int | float], None]
    parse_value: Callable[[str], int | float]
    validate_value: Callable[[int | float], bool]
    value_aliases: dict[str, int | float] = field(default_factory=dict)


def register_prompt_editor_edit_handlers(
    router: Router,
    deps: PromptEditorEditHandlersDeps,
) -> None:
    def _register_paginated_param(config: PaginatedParamConfig) -> None:
        @router.callback_query(F.data == config.open_callback)
        async def _open(cb: CallbackQuery):
            await deps.open_paginated_choice(
                cb,
                title=config.title,
                items=config.items_getter(),
                prefix=config.prefix,
            )

        @router.callback_query(F.data.startswith(f"{config.prefix}_page:"))
        async def _page(cb: CallbackQuery):
            await deps.change_paginated_choice_page(
                cb,
                items=config.items_getter(),
                prefix=config.prefix,
            )

        @router.callback_query(F.data.startswith(f"{config.prefix}:"))
        async def _chosen(cb: CallbackQuery, state: FSMContext):
            idx = await parse_indexed_selection(cb, prefix=config.prefix)
            if idx is None:
                return
            items = config.items_getter()
            if idx < 0 or idx >= len(items):
                await cb.answer("❌ Неверный индекс.", show_alert=True)
                return
            await deps.set_prompt_param_from_callback(
                cb,
                state,
                field=config.field,
                value=config.transform_value(items[idx]),
                notice=config.notice,
            )

    def _register_scalar_param(config: ScalarParamConfig) -> None:
        @router.callback_query(F.data == config.open_callback)
        async def _open(cb: CallbackQuery):
            message = await require_callback_message(cb)
            if message is None:
                return
            await message.edit_text(
                config.menu_text,
                reply_markup=scalar_choice_keyboard(
                    prefix=config.prefix,
                    values_rows=config.values_rows,
                    back_callback="pe:back",
                ),
            )
            await cb.answer()

        @router.callback_query(F.data.startswith(f"{config.prefix}:"))
        async def _selected(cb: CallbackQuery, state: FSMContext):
            message = await require_callback_message(cb)
            if message is None:
                return
            payload = await deps.require_prompt_request_for_callback(cb)
            if not payload:
                return

            uid, req = payload
            raw_value = await parse_value_selection(cb, prefix=config.prefix)
            if raw_value is None:
                return

            if raw_value == "custom":
                await state.set_state(cast(Any, config.custom_state))
                await message.edit_text(
                    config.custom_prompt,
                    reply_markup=deps.back_keyboard(),
                )
                await cb.answer()
                return

            try:
                parsed = config.value_aliases.get(raw_value)
                if parsed is None:
                    parsed = config.parse_value(raw_value)
                if not config.validate_value(parsed):
                    raise ValueError
            except ValueError:
                await cb.answer("❌ Некорректное значение.", show_alert=True)
                return

            config.set_value(req.params, parsed)
            await deps.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice=config.notice,
            )
            await cb.answer()

        @router.message(cast(Any, config.custom_state), F.text)
        async def _custom(msg: Message, state: FSMContext):
            payload = await deps.require_prompt_request_for_message(msg, state)
            if not payload:
                return

            uid, req = payload
            try:
                parsed = config.parse_value((msg.text or "").strip())
                if not config.validate_value(parsed):
                    raise ValueError
            except ValueError:
                await msg.answer(config.invalid_input_text)
                return

            config.set_value(req.params, parsed)
            await deps.cleanup_user_message(msg)
            await deps.show_prompt_editor(msg, state, uid, notice=config.notice)

    @router.callback_query(F.data == "pe:edit:positive")
    async def pe_edit_positive(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
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
        message = await require_callback_message(cb)
        if message is None:
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
        idx = await parse_indexed_selection(cb, prefix="pe_ckpt")
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

        message = await require_callback_message(cb)
        if message is None:
            return
        await deps.show_prompt_editor(message, state, uid, edit=True, notice=notice)
        await cb.answer()

    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:sampler",
            prefix="pe_smpl",
            title="Выберите sampler:",
            field="sampler",
            notice="✅ Sampler обновлён.",
            items_getter=lambda: deps.client.info.samplers or ["euler"],
        )
    )
    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:scheduler",
            prefix="pe_sched",
            title="Выберите scheduler:",
            field="scheduler",
            notice="✅ Scheduler обновлён.",
            items_getter=lambda: deps.client.info.schedulers or ["normal"],
        )
    )
    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:upscaler",
            prefix="pe_upsc",
            title="Выберите upscaler:",
            field="upscale_model",
            notice="✅ Upscaler обновлён.",
            items_getter=lambda: ["(без апскейла)"] + deps.client.info.upscale_models,
            transform_value=lambda value: "" if value == "(без апскейла)" else value,
        )
    )
    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:vae",
            prefix="pe_vae",
            title="Выберите VAE:",
            field="vae_name",
            notice="✅ VAE обновлён.",
            items_getter=lambda: ["(из checkpoint)"] + deps.client.info.vaes,
            transform_value=lambda value: "" if value == "(из checkpoint)" else value,
        )
    )
    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:controlnet",
            prefix="pe_cn",
            title="Выберите ControlNet:",
            field="controlnet_name",
            notice="✅ ControlNet обновлён.",
            items_getter=lambda: ["(выкл)"] + deps.client.info.controlnets,
            transform_value=lambda value: "" if value == "(выкл)" else value,
        )
    )
    _register_paginated_param(
        PaginatedParamConfig(
            open_callback="pe:edit:embedding",
            prefix="pe_emb",
            title="Выберите embedding:",
            field="embedding_name",
            notice="✅ Embedding обновлён.",
            items_getter=lambda: ["(без embedding)"] + deps.client.info.embeddings,
            transform_value=lambda value: "" if value == "(без embedding)" else value,
        )
    )

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:controlnet_strength",
            prefix="pe_cns",
            custom_state=PromptEditorStates.entering_custom_controlnet_strength,
            menu_text="ControlNet strength (0.0-2.0):",
            custom_prompt="ControlNet strength (0.0-2.0):",
            invalid_input_text="Число 0.0-2.0:",
            notice="✅ ControlNet strength обновлён.",
            values_rows=[["0.4", "0.7", "1.0"], ["1.2", "1.5"]],
            set_value=lambda params, value: setattr(params, "controlnet_strength", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: 0.0 <= float(value) <= 2.0,
        )
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
        value = await parse_value_selection(cb, prefix="pe_size")
        if value is None:
            return
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_size)
            await message.edit_text(
                "Введите размер ШИРИНАxВЫСОТА (например 640x960):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        size_parts = value.split(":", 1)
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

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:steps",
            prefix="pe_steps",
            custom_state=PromptEditorStates.entering_custom_steps,
            menu_text="Steps:",
            custom_prompt="Steps (1-200):",
            invalid_input_text="Целое число 1-200:",
            notice="✅ Steps обновлён.",
            values_rows=[["10", "15", "20"], ["25", "30", "40"], ["50", "60", "80"]],
            set_value=lambda params, value: setattr(params, "steps", int(value)),
            parse_value=lambda raw: int(raw),
            validate_value=lambda value: 1 <= int(value) <= 200,
        )
    )

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:cfg",
            prefix="pe_cfg",
            custom_state=PromptEditorStates.entering_custom_cfg,
            menu_text="CFG:",
            custom_prompt="CFG (0.0-30.0):",
            invalid_input_text="Число 0.0-30.0:",
            notice="✅ CFG обновлён.",
            values_rows=[["1.0", "3.0", "5.0"], ["7.0", "7.5", "8.0"], ["10.0", "12.0", "15.0"]],
            set_value=lambda params, value: setattr(params, "cfg", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: 0.0 <= float(value) <= 30.0,
        )
    )

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:denoise",
            prefix="pe_dn",
            custom_state=PromptEditorStates.entering_custom_denoise,
            menu_text="Denoise:",
            custom_prompt="Denoise (0.0-1.0):",
            invalid_input_text="Число 0.0-1.0:",
            notice="✅ Denoise обновлён.",
            values_rows=[["0.3", "0.5", "0.7"], ["0.8", "0.9", "1.0"]],
            set_value=lambda params, value: setattr(params, "denoise", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: 0.0 <= float(value) <= 1.0,
        )
    )

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:ref_strength",
            prefix="pe_rstr",
            custom_state=PromptEditorStates.entering_custom_reference_strength,
            menu_text=(
                "Сила референса (0.0-2.0):\n"
                "- Для IP-Adapter: больше = сильнее похожесть\n"
                "- Для img2img fallback: больше = ниже denoise"
            ),
            custom_prompt="Сила референса (0.0-2.0):",
            invalid_input_text="Число 0.0-2.0:",
            notice="✅ Сила референса обновлена.",
            values_rows=[
                ["0.3", "0.5", "0.7"],
                ["0.9", "1.0", "1.2"],
                ["1.5", "1.8", "2.0"],
            ],
            set_value=lambda params, value: setattr(params, "reference_strength", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: 0.0 <= float(value) <= 2.0,
        )
    )

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
        value = await parse_value_selection(cb, prefix="pe_seed")
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

    _register_scalar_param(
        ScalarParamConfig(
            open_callback="pe:edit:batch",
            prefix="pe_batch",
            custom_state=PromptEditorStates.entering_custom_batch,
            menu_text="Batch size:",
            custom_prompt="Batch (1-16):",
            invalid_input_text="Число 1-16:",
            notice="✅ Batch обновлён.",
            values_rows=[["1", "2", "3", "4"]],
            set_value=lambda params, value: setattr(params, "batch_size", int(value)),
            parse_value=lambda raw: int(raw),
            validate_value=lambda value: 1 <= int(value) <= 16,
        )
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
