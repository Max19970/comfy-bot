from __future__ import annotations

import asyncio
import uuid

import aiohttp
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from comfyui_client import ComfyUIClient
from config import Config
from core.html_utils import h
from core.interaction import (
    edit_message_by_anchor,
    edit_or_answer,
    require_callback_message,
)
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.telegram import callback_user_id, message_user_id
from core.user_preferences import read_generation_defaults


def remember_prompt_panel(runtime: RuntimeStore, req: PromptRequest, panel_msg: Message) -> None:
    if panel_msg.chat is None:
        return
    req.ui_chat_id = panel_msg.chat.id
    req.ui_message_id = panel_msg.message_id

    uid: int | None = None
    for user_id, active_req in runtime.active_prompt_requests.items():
        if active_req is req:
            uid = user_id
            break
    if uid is not None:
        runtime.user_ui_panels[uid] = {
            "chat_id": panel_msg.chat.id,
            "message_id": panel_msg.message_id,
        }


async def edit_prompt_panel_by_anchor(
    runtime: RuntimeStore,
    req: PromptRequest,
    source_message: Message,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
) -> Message | None:
    edited = await edit_message_by_anchor(
        source_message,
        chat_id=req.ui_chat_id,
        message_id=req.ui_message_id,
        text=text,
        reply_markup=reply_markup,
        not_modified_fallback=source_message,
    )
    if edited is not None:
        remember_prompt_panel(runtime, req, edited)
    return edited


async def show_prompt_panel(
    runtime: RuntimeStore,
    message: Message,
    req: PromptRequest,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
    *,
    prefer_edit: bool,
) -> Message:
    anchored = await edit_prompt_panel_by_anchor(runtime, req, message, text, reply_markup)
    if anchored is not None:
        return anchored
    panel_msg = await edit_or_answer(
        message,
        text=text,
        reply_markup=reply_markup,
        prefer_edit=prefer_edit,
    )
    remember_prompt_panel(runtime, req, panel_msg)
    return panel_msg


async def move_prompt_panel_to_bottom(
    runtime: RuntimeStore,
    message: Message,
    req: PromptRequest,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
) -> Message:
    prev_chat_id = req.ui_chat_id
    prev_message_id = req.ui_message_id

    sent = await message.answer(text, reply_markup=reply_markup)
    remember_prompt_panel(runtime, req, sent)

    if (
        prev_chat_id is not None
        and prev_message_id is not None
        and (prev_chat_id != sent.chat.id or prev_message_id != sent.message_id)
        and message.bot is not None
    ):
        try:
            await message.bot.delete_message(
                chat_id=prev_chat_id,
                message_id=prev_message_id,
            )
        except TelegramBadRequest:
            pass

    return sent


async def ensure_models_available(client: ComfyUIClient, message: Message) -> bool:
    if not client.info.checkpoints:
        await message.answer("⏳ Загружаю список моделей…")
        try:
            await client.refresh_info()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            await message.answer(
                f"❌ <b>Не удалось подключиться к ComfyUI</b>\n<code>{h(exc)}</code>"
            )
            return False
    if not client.info.checkpoints:
        await message.answer("❌ На сервере ComfyUI нет ни одного checkpoint.")
        return False
    return True


def build_default_params(cfg: Config) -> GenerationParams:
    return GenerationParams(
        width=cfg.default_width,
        height=cfg.default_height,
        steps=cfg.default_steps,
        cfg=cfg.default_cfg,
        sampler=cfg.default_sampler,
        scheduler=cfg.default_scheduler,
        denoise=cfg.default_denoise,
    )


def build_default_params_for_user(cfg: Config, runtime: RuntimeStore, uid: int) -> GenerationParams:
    params = build_default_params(cfg)
    prefs = runtime.user_preferences.get(uid, {})

    defaults = read_generation_defaults(
        prefs,
        default_width=params.width,
        default_height=params.height,
        default_steps=params.steps,
        default_cfg=params.cfg,
        default_denoise=params.denoise,
        default_sampler=params.sampler,
        default_scheduler=params.scheduler,
    )
    params.width = defaults["width"]
    params.height = defaults["height"]
    params.steps = defaults["steps"]
    params.cfg = defaults["cfg"]
    params.denoise = defaults["denoise"]
    params.seed = defaults["seed"]
    params.batch_size = defaults["batch"]
    params.sampler = defaults["sampler"]
    params.scheduler = defaults["scheduler"]
    return params


def normalize_generation_params(
    params: GenerationParams,
    client: ComfyUIClient,
    *,
    max_reference_images: int,
) -> GenerationParams:
    if not params.checkpoint and client.info.checkpoints:
        params.checkpoint = client.info.checkpoints[0]
    if client.info.samplers and params.sampler not in client.info.samplers:
        params.sampler = client.info.samplers[0]
    if client.info.schedulers and params.scheduler not in client.info.schedulers:
        params.scheduler = client.info.schedulers[0]
    if params.upscale_model and params.upscale_model not in client.info.upscale_models:
        params.upscale_model = ""
    if params.vae_name and params.vae_name not in client.info.vaes:
        params.vae_name = ""
    if params.controlnet_name and params.controlnet_name not in client.info.controlnets:
        params.controlnet_name = ""

    try:
        params.controlnet_strength = float(params.controlnet_strength)
    except (TypeError, ValueError):
        params.controlnet_strength = 1.0
    params.controlnet_strength = max(0.0, min(2.0, params.controlnet_strength))

    if (
        client.info.embeddings
        and params.embedding_name
        and params.embedding_name not in client.info.embeddings
    ):
        params.embedding_name = ""
    if client.info.loras:
        params.loras = [item for item in params.loras if item[0] in client.info.loras]
    else:
        params.loras = []

    normalized_refs: list[dict[str, str]] = []
    for item in params.reference_images:
        if isinstance(item, dict):
            file_id = str(item.get("file_id", "")).strip()
            if not file_id:
                continue
            ref_id = str(item.get("id") or uuid.uuid4().hex)
            normalized_refs.append({"id": ref_id, "file_id": file_id})
            continue

        if isinstance(item, str):
            file_id = item.strip()
            if file_id:
                normalized_refs.append({"id": uuid.uuid4().hex, "file_id": file_id})

    params.reference_images = normalized_refs[:max_reference_images]
    try:
        params.reference_strength = float(params.reference_strength)
    except (TypeError, ValueError):
        params.reference_strength = 0.8
    params.reference_strength = max(0.0, min(2.0, params.reference_strength))
    return params


async def require_prompt_request_for_message(
    runtime: RuntimeStore,
    msg: Message,
    state: FSMContext,
) -> tuple[int, PromptRequest] | None:
    uid = message_user_id(msg)
    req = runtime.active_prompt_requests.get(uid)
    if req:
        return uid, req
    await state.clear()
    await msg.answer("❌ Активный запрос не найден. Используйте /generate.")
    return None


async def require_prompt_request_for_callback(
    runtime: RuntimeStore,
    cb: CallbackQuery,
) -> tuple[int, PromptRequest] | None:
    message = await require_callback_message(cb)
    if message is None:
        return None

    uid = callback_user_id(cb)
    req = runtime.active_prompt_requests.get(uid)
    if req:
        return uid, req
    await cb.answer("❌ Нет активного запроса.", show_alert=True)
    return None


async def safe_delete_user_message(message: Message) -> None:
    try:
        await message.delete()
    except TelegramBadRequest:
        return
