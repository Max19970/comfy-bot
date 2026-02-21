from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery

from core.callbacks import PagedSelectionCallback
from core.interaction import callback_message
from core.models import GenerationParams
from core.ui_kit import back_button


async def open_paginated_choice(
    cb: CallbackQuery,
    *,
    title: str,
    items: list[str],
    prefix: str,
    back_callback: str = "pe:back",
    paginated_keyboard: Callable[..., Any],
) -> None:
    kb = paginated_keyboard(
        items,
        0,
        prefix,
        extra=[[back_button(back_callback)]],
    )
    message = callback_message(cb)
    if message is None:
        await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
        return
    await message.edit_text(title, reply_markup=kb)
    await cb.answer()


async def change_paginated_choice_page(
    cb: CallbackQuery,
    *,
    items: list[str],
    prefix: str,
    back_callback: str = "pe:back",
    paginated_keyboard: Callable[..., Any],
) -> None:
    page_cb = PagedSelectionCallback.parse(cb.data or "", prefix=prefix)
    if page_cb is None:
        await cb.answer("❌ Некорректный запрос.", show_alert=True)
        return
    page = page_cb.page
    kb = paginated_keyboard(
        items,
        page,
        prefix,
        extra=[[back_button(back_callback)]],
    )
    message = callback_message(cb)
    if message is None:
        await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
        return
    await message.edit_reply_markup(reply_markup=kb)
    await cb.answer()


async def set_prompt_param_from_callback(
    cb: CallbackQuery,
    state: FSMContext,
    *,
    field: str,
    value: Any,
    notice: str,
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, Any] | None]
    ],
    show_prompt_editor: Callable[..., Awaitable[None]],
) -> None:
    payload = await require_prompt_request_for_callback(cb)
    if not payload:
        return
    uid, req = payload
    setattr(req.params, field, value)
    message = callback_message(cb)
    if message is None:
        await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
        return
    await show_prompt_editor(message, state, uid, edit=True, notice=notice)
    await cb.answer()


def changed_params_count(
    params: GenerationParams,
    *,
    normalize_params: Callable[[GenerationParams], GenerationParams],
    default_params: Callable[[], GenerationParams],
) -> int:
    current = normalize_params(GenerationParams(**asdict(params)))
    defaults = normalize_params(default_params())

    changed = 0
    if current.positive.strip() != defaults.positive.strip():
        changed += 1
    if current.negative.strip() != defaults.negative.strip():
        changed += 1
    if current.checkpoint != defaults.checkpoint:
        changed += 1
    if list(current.loras) != list(defaults.loras):
        changed += 1
    if current.upscale_model != defaults.upscale_model:
        changed += 1
    if current.vae_name != defaults.vae_name:
        changed += 1
    if current.controlnet_name != defaults.controlnet_name:
        changed += 1
    if abs(float(current.controlnet_strength) - float(defaults.controlnet_strength)) > 1e-6:
        changed += 1
    if current.embedding_name != defaults.embedding_name:
        changed += 1
    if current.width != defaults.width or current.height != defaults.height:
        changed += 1
    if current.steps != defaults.steps:
        changed += 1
    if abs(float(current.cfg) - float(defaults.cfg)) > 1e-6:
        changed += 1
    if current.sampler != defaults.sampler:
        changed += 1
    if current.scheduler != defaults.scheduler:
        changed += 1
    if abs(float(current.denoise) - float(defaults.denoise)) > 1e-6:
        changed += 1
    if current.seed != defaults.seed:
        changed += 1
    if current.batch_size != defaults.batch_size:
        changed += 1
    if abs(float(current.reference_strength) - float(defaults.reference_strength)) > 1e-6:
        changed += 1
    if current.reference_images != defaults.reference_images:
        changed += 1
    if current.enable_hires_fix != defaults.enable_hires_fix:
        changed += 1
    if current.enable_freeu != defaults.enable_freeu:
        changed += 1
    if current.enable_pag != defaults.enable_pag:
        changed += 1
    return changed
