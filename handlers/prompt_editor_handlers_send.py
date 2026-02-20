from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import cast

import aiohttp
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient
from core.html_utils import h
from core.image_utils import image_dimensions
from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore
from core.ui_copy import START_TEXT, main_menu_keyboard


@dataclass
class PromptEditorSendHandlersDeps:
    logger: logging.Logger
    runtime: RuntimeStore
    client: ComfyUIClient
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    generation_result_keyboard: Callable[[], InlineKeyboardMarkup]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]


def register_prompt_editor_send_handlers(
    router: Router,
    deps: PromptEditorSendHandlersDeps,
) -> None:
    def _callback_message(cb: CallbackQuery) -> Message | None:
        message = cb.message
        if message is None or not hasattr(message, "answer"):
            return None
        return cast(Message, message)

    def _user_artifact(uid: int, artifact_id: str) -> PreviewArtifact | None:
        artifact = deps.runtime.preview_artifacts.get(artifact_id)
        if artifact and artifact.owner_uid == uid:
            return artifact
        return None

    def _has_pending_input(msg: Message) -> bool:
        uid = msg.from_user.id if msg.from_user else 0
        return uid > 0 and uid in deps.runtime.pending_image_inputs

    async def _move_main_panel_to_bottom(
        uid: int,
        source_message: Message,
        text: str,
    ) -> None:
        req = deps.runtime.active_prompt_requests.get(uid)
        prev_chat_id = req.ui_chat_id if req else None
        prev_message_id = req.ui_message_id if req else None

        sent = await source_message.answer(
            text,
            reply_markup=deps.generation_result_keyboard(),
        )

        if req:
            req.ui_chat_id = sent.chat.id
            req.ui_message_id = sent.message_id
        deps.runtime.user_ui_panels[uid] = {
            "chat_id": sent.chat.id,
            "message_id": sent.message_id,
        }

        if (
            prev_chat_id is not None
            and prev_message_id is not None
            and (prev_chat_id != sent.chat.id or prev_message_id != sent.message_id)
            and source_message.bot is not None
        ):
            try:
                await source_message.bot.delete_message(
                    chat_id=prev_chat_id,
                    message_id=prev_message_id,
                )
            except TelegramBadRequest:
                pass

    def _artifact_menu_keyboard(artifact: PreviewArtifact) -> InlineKeyboardMarkup:
        params = artifact.params
        sampler_status = "✅" if artifact.enable_sampler_pass else "❌"
        hires_status = "✅" if params.enable_hires_fix else "❌"
        freeu_status = "✅" if params.enable_freeu else "❌"
        pag_status = "✅" if params.enable_pag else "❌"
        upsc_status = "✅" if params.upscale_model else "❌"

        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"🔁 Сэмплер {sampler_status}",
                        callback_data=f"img:tgl:smp:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=f"🔢 Steps {params.steps}",
                        callback_data=f"img:menu:steps:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=f"CFG {params.cfg}",
                        callback_data=f"img:menu:cfg:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"Denoise {params.denoise}",
                        callback_data=f"img:menu:den:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="⚙️ Sampler",
                        callback_data=f"img:menu:sampler:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text="📈 Scheduler",
                        callback_data=f"img:menu:scheduler:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"🛠 Hi-res {hires_status}",
                        callback_data=f"img:tgl:hi:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=f"⚡ FreeU {freeu_status}",
                        callback_data=f"img:tgl:fu:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"🎯 PAG {pag_status}",
                        callback_data=f"img:tgl:pag:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=f"🔍 Upscaler {upsc_status}",
                        callback_data=f"img:menu:up:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"Hi-res scale ×{params.hires_scale}",
                        callback_data=f"img:menu:hrs:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=f"Hi-res den {params.hires_denoise}",
                        callback_data=f"img:menu:hrd:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"PAG scale {params.pag_scale}",
                        callback_data=f"img:menu:pags:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🧬 В редактор",
                        callback_data=f"img:to_editor:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="▶️ Запустить улучшения",
                        callback_data=f"img:run:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад",
                        callback_data=f"img:back:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="⬅️ В меню",
                        callback_data="menu:root",
                    )
                ],
            ]
        )

    def _artifact_menu_caption(artifact: PreviewArtifact) -> str:
        params = artifact.params
        lines = [
            "✨ <b>Улучшения для этой картинки</b>",
            f"Шаг цепочки: <code>{artifact.generation_step}</code>",
            f"Seed: <code>{artifact.used_seed}</code>",
        ]
        modes: list[str] = []
        if artifact.enable_sampler_pass:
            modes.append("🔁 sampler")
        if params.enable_hires_fix:
            modes.append("🛠 hi-res")
        if params.enable_freeu:
            modes.append("⚡ freeu")
        if params.enable_pag:
            modes.append("🎯 pag")
        if params.upscale_model:
            modes.append("🔍 upscale")
        lines.append("Режимы: " + (", ".join(modes) if modes else "<i>не выбраны</i>"))
        return "\n".join(lines)

    async def _edit_preview_message(
        cb: CallbackQuery,
        *,
        caption: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        message = _callback_message(cb)
        if message is None:
            return
        try:
            await message.edit_caption(caption=caption, reply_markup=reply_markup)
            return
        except TelegramBadRequest:
            pass
        try:
            await message.edit_text(caption, reply_markup=reply_markup)
        except TelegramBadRequest:
            await message.answer(caption, reply_markup=reply_markup)

    def _simple_value_keyboard(
        *,
        artifact_id: str,
        key: str,
        values: list[str],
    ) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        row: list[InlineKeyboardButton] = []
        for value in values:
            row.append(
                InlineKeyboardButton(
                    text=value,
                    callback_data=f"img:set:{key}:{artifact_id}:{value}",
                )
            )
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append(
            [
                InlineKeyboardButton(
                    text="✏️ Ввести свое",
                    callback_data=f"img:custom:{key}:{artifact_id}",
                )
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=f"img:open:{artifact_id}",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _paginated_pick_keyboard(
        *,
        artifact_id: str,
        menu: str,
        items: list[str],
        page: int,
    ) -> InlineKeyboardMarkup:
        page_size = 8
        total_pages = max(1, (len(items) + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size

        rows: list[list[InlineKeyboardButton]] = []
        for idx, label in enumerate(items[start : start + page_size], start=start):
            rows.append(
                [
                    InlineKeyboardButton(
                        text=label if len(label) <= 40 else label[:37] + "...",
                        callback_data=f"img:pick:{menu}:{artifact_id}:{idx}",
                    )
                ]
            )

        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀️",
                    callback_data=f"img:page:{menu}:{artifact_id}:{page - 1}",
                )
            )
        nav.append(InlineKeyboardButton(text=f"· {page + 1}/{total_pages} ·", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="▶️",
                    callback_data=f"img:page:{menu}:{artifact_id}:{page + 1}",
                )
            )
        rows.append(nav)
        rows.append(
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=f"img:open:{artifact_id}",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _custom_field_meta(field: str) -> tuple[str, float | int, float | int]:
        if field == "steps":
            return ("Steps", 1, 150)
        if field == "cfg":
            return ("CFG", 0.0, 30.0)
        if field == "denoise":
            return ("Denoise", 0.0, 1.0)
        if field == "hires_scale":
            return ("Hi-res scale", 1.0, 3.0)
        if field == "hires_denoise":
            return ("Hi-res denoise", 0.0, 1.0)
        if field == "pag_scale":
            return ("PAG scale", 0.5, 10.0)
        raise ValueError("unknown field")

    def _apply_field_value(
        artifact: PreviewArtifact,
        *,
        field: str,
        value: float | int,
    ) -> bool:
        if field == "steps":
            artifact.params.steps = int(value)
            artifact.enable_sampler_pass = True
            return True
        if field == "cfg":
            artifact.params.cfg = float(value)
            artifact.enable_sampler_pass = True
            return True
        if field == "denoise":
            artifact.params.denoise = float(value)
            artifact.enable_sampler_pass = True
            return True
        if field == "hires_scale":
            artifact.params.hires_scale = float(value)
            return True
        if field == "hires_denoise":
            artifact.params.hires_denoise = float(value)
            return True
        if field == "pag_scale":
            artifact.params.pag_scale = float(value)
            return True
        return False

    @router.callback_query(F.data.startswith("send:"))
    async def send_images(cb: CallbackQuery, state: FSMContext):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return

        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 2:
            await cb.answer("❌ Некорректный режим.", show_alert=True)
            return
        mode = parts[1]

        if mode == "new":
            payload = await deps.require_prompt_request_for_callback(cb)
            if not payload:
                return

            uid, _ = payload
            await deps.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice="🔄 Возвращаемся к настройкам для новой генерации.",
            )
            await cb.answer()
            return

        if mode == "cancel":
            await state.clear()
            try:
                await message.edit_text(
                    START_TEXT,
                    reply_markup=main_menu_keyboard(),
                )
            except TelegramBadRequest:
                await message.answer(
                    START_TEXT,
                    reply_markup=main_menu_keyboard(),
                )
            await cb.answer()
            return

        await cb.answer("❌ Неизвестный режим отправки.", show_alert=True)

    @router.callback_query(F.data.startswith("img:png:"))
    async def image_send_png(cb: CallbackQuery):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return

        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact_id = parts[2]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена или недоступна.", show_alert=True)
            return

        image_bytes = deps.runtime.artifact_bytes(artifact)
        if not image_bytes:
            await cb.answer("⚠️ Исходные данные картинки не найдены.", show_alert=True)
            return

        await deps.deliver_generated_images(
            message,
            [image_bytes],
            used_seed=artifact.used_seed,
            mode="file",
        )
        await cb.answer("📄 PNG отправлен")

    @router.callback_query(F.data.startswith("img:back:"))
    async def image_back(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact_id = parts[2]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return
        await _edit_preview_message(
            cb,
            caption=(
                f"🖼 Шаг {artifact.generation_step} | Seed: {artifact.used_seed}\n"
                "Выберите действие для этой картинки."
            ),
            reply_markup=deps.preview_image_keyboard(
                artifact.artifact_id,
                artifact.parent_artifact_id,
            ),
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("img:goto_parent:"))
    async def image_goto_parent(cb: CallbackQuery):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact = _user_artifact(uid, parts[2])
        if not artifact or not artifact.parent_artifact_id:
            await cb.answer("⚠️ Исходник не найден.", show_alert=True)
            return
        parent = _user_artifact(uid, artifact.parent_artifact_id)
        if not parent or parent.preview_message_id is None or parent.preview_chat_id is None:
            await cb.answer("⚠️ Ссылка на исходник недоступна.", show_alert=True)
            return
        if parent.preview_chat_id != message.chat.id:
            await cb.answer("⚠️ Исходник в другом чате.", show_alert=True)
            return
        await message.answer(
            "↩️ Исходная картинка",
            reply_to_message_id=parent.preview_message_id,
        )
        await cb.answer("Готово")

    @router.callback_query(F.data.startswith("img:open:"))
    async def image_open_enhancements(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact_id = parts[2]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        await _edit_preview_message(
            cb,
            caption=_artifact_menu_caption(artifact),
            reply_markup=_artifact_menu_keyboard(artifact),
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("img:tgl:"))
    async def image_toggle(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        toggle_key = parts[2]
        artifact_id = parts[3]

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        if toggle_key == "smp":
            artifact.enable_sampler_pass = not artifact.enable_sampler_pass
        elif toggle_key == "hi":
            artifact.params.enable_hires_fix = not artifact.params.enable_hires_fix
            if artifact.params.enable_hires_fix:
                artifact.enable_sampler_pass = True
        elif toggle_key == "fu":
            if not deps.client.info.freeu_supported:
                await cb.answer("⚠️ FreeU не поддерживается сервером.", show_alert=True)
                return
            artifact.params.enable_freeu = not artifact.params.enable_freeu
            if artifact.params.enable_freeu:
                artifact.enable_sampler_pass = True
        elif toggle_key == "pag":
            if not deps.client.info.pag_supported:
                await cb.answer("⚠️ PAG не поддерживается сервером.", show_alert=True)
                return
            artifact.params.enable_pag = not artifact.params.enable_pag
            if artifact.params.enable_pag:
                artifact.enable_sampler_pass = True
        else:
            await cb.answer("⚠️ Неизвестный переключатель.", show_alert=True)
            return

        await _edit_preview_message(
            cb,
            caption=_artifact_menu_caption(artifact),
            reply_markup=_artifact_menu_keyboard(artifact),
        )
        await cb.answer("✅ Обновлено")

    @router.callback_query(F.data.startswith("img:menu:"))
    async def image_menu_open(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        menu_key = parts[2]
        artifact_id = parts[3]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        if menu_key == "steps":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="steps",
                values=["10", "15", "20", "25", "30", "40"],
            )
            await _edit_preview_message(
                cb,
                caption="Выберите Steps:",
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "cfg":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="cfg",
                values=["4.0", "5.0", "6.0", "7.0", "8.0", "10.0"],
            )
            await _edit_preview_message(cb, caption="Выберите CFG:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "den":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="denoise",
                values=["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
            )
            await _edit_preview_message(cb, caption="Выберите Denoise:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "hrs":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="hires_scale",
                values=["1.25", "1.5", "1.75", "2.0"],
            )
            await _edit_preview_message(cb, caption="Выберите Hi-res scale:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "hrd":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="hires_denoise",
                values=["0.3", "0.4", "0.5", "0.6", "0.7"],
            )
            await _edit_preview_message(cb, caption="Выберите Hi-res denoise:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "pags":
            kb = _simple_value_keyboard(
                artifact_id=artifact_id,
                key="pag_scale",
                values=["1.0", "2.0", "3.0", "4.0", "5.0"],
            )
            await _edit_preview_message(cb, caption="Выберите PAG scale:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "sampler":
            samplers = deps.client.info.samplers or ["euler"]
            kb = _paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="sampler",
                items=samplers,
                page=0,
            )
            await _edit_preview_message(cb, caption="Выберите sampler:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "scheduler":
            schedulers = deps.client.info.schedulers or ["normal"]
            kb = _paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="scheduler",
                items=schedulers,
                page=0,
            )
            await _edit_preview_message(cb, caption="Выберите scheduler:", reply_markup=kb)
            await cb.answer()
            return
        if menu_key == "up":
            upscalers = ["(без апскейла)"] + deps.client.info.upscale_models
            kb = _paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="up",
                items=upscalers,
                page=0,
            )
            await _edit_preview_message(cb, caption="Выберите upscaler:", reply_markup=kb)
            await cb.answer()
            return

        await cb.answer("⚠️ Неизвестное меню.", show_alert=True)

    @router.callback_query(F.data.startswith("img:page:"))
    async def image_menu_page(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        menu = parts[2]
        artifact_id = parts[3]
        page = int(parts[4])
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        if menu == "sampler":
            items = deps.client.info.samplers or ["euler"]
            caption = "Выберите sampler:"
        elif menu == "scheduler":
            items = deps.client.info.schedulers or ["normal"]
            caption = "Выберите scheduler:"
        elif menu == "up":
            items = ["(без апскейла)"] + deps.client.info.upscale_models
            caption = "Выберите upscaler:"
        else:
            await cb.answer("⚠️ Неизвестная страница.", show_alert=True)
            return

        kb = _paginated_pick_keyboard(
            artifact_id=artifact_id,
            menu=menu,
            items=items,
            page=page,
        )
        await _edit_preview_message(cb, caption=caption, reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("img:pick:"))
    async def image_menu_pick(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        menu = parts[2]
        artifact_id = parts[3]
        idx = int(parts[4])

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        if menu == "sampler":
            items = deps.client.info.samplers or ["euler"]
            if idx < 0 or idx >= len(items):
                await cb.answer("❌ Неверный выбор.", show_alert=True)
                return
            artifact.params.sampler = items[idx]
            artifact.enable_sampler_pass = True
        elif menu == "scheduler":
            items = deps.client.info.schedulers or ["normal"]
            if idx < 0 or idx >= len(items):
                await cb.answer("❌ Неверный выбор.", show_alert=True)
                return
            artifact.params.scheduler = items[idx]
            artifact.enable_sampler_pass = True
        elif menu == "up":
            items = ["(без апскейла)"] + deps.client.info.upscale_models
            if idx < 0 or idx >= len(items):
                await cb.answer("❌ Неверный выбор.", show_alert=True)
                return
            chosen = items[idx]
            artifact.params.upscale_model = "" if chosen == "(без апскейла)" else chosen
        else:
            await cb.answer("⚠️ Неизвестный выбор.", show_alert=True)
            return

        await _edit_preview_message(
            cb,
            caption=_artifact_menu_caption(artifact),
            reply_markup=_artifact_menu_keyboard(artifact),
        )
        await cb.answer("✅ Обновлено")

    @router.callback_query(F.data.startswith("img:set:"))
    async def image_set_value(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        field = parts[2]
        artifact_id = parts[3]
        raw_value = parts[4]

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        try:
            parsed_value: float | int
            if field == "steps":
                parsed_value = int(raw_value)
            else:
                parsed_value = float(raw_value)
            if not _apply_field_value(artifact, field=field, value=parsed_value):
                await cb.answer("⚠️ Неизвестный параметр.", show_alert=True)
                return
        except ValueError:
            await cb.answer("⚠️ Не удалось применить значение.", show_alert=True)
            return

        await _edit_preview_message(
            cb,
            caption=_artifact_menu_caption(artifact),
            reply_markup=_artifact_menu_keyboard(artifact),
        )
        await cb.answer("✅ Параметр обновлен")

    @router.callback_query(F.data.startswith("img:custom:"))
    async def image_custom_start(cb: CallbackQuery):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return

        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        field = parts[2]
        artifact_id = parts[3]

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        try:
            label, min_val, max_val = _custom_field_meta(field)
        except ValueError:
            await cb.answer("⚠️ Неизвестный параметр.", show_alert=True)
            return

        deps.runtime.pending_image_inputs[uid] = {
            "artifact_id": artifact_id,
            "field": field,
        }
        await message.answer(
            f"✏️ Введите {label} ({min_val}..{max_val}).\nМожно использовать точку или запятую."
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("img:to_editor:"))
    async def image_to_editor(cb: CallbackQuery, state: FSMContext):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return

        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact_id = parts[2]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return

        deps.runtime.active_prompt_requests[uid] = PromptRequest(
            params=GenerationParams(**asdict(artifact.params)),
            operation="generate",
        )
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=False,
            notice="🧬 Параметры картинки перенесены в главный редактор.",
        )
        await cb.answer()

    @router.message(_has_pending_input, F.text)
    async def image_custom_value_input(msg: Message):
        uid = msg.from_user.id if msg.from_user else 0
        if uid <= 0:
            return

        pending = deps.runtime.pending_image_inputs.get(uid)
        if not pending:
            return

        raw = (msg.text or "").strip()
        if raw.lower() in {"cancel", "/cancel", "отмена"}:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer("↩️ Ввод параметра отменен.")
            return

        artifact_id = str(pending.get("artifact_id") or "")
        field = str(pending.get("field") or "")
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer("⚠️ Картинка уже недоступна.")
            return

        try:
            label, min_val, max_val = _custom_field_meta(field)
        except ValueError:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer("⚠️ Неизвестный параметр.")
            return

        value_raw = raw.replace(",", ".")
        try:
            value: float | int
            if field == "steps":
                value = int(float(value_raw))
            else:
                value = float(value_raw)
            if value < min_val or value > max_val:
                raise ValueError("out of range")
        except ValueError:
            await msg.answer(f"⚠️ Введите {label} в диапазоне {min_val}..{max_val}.")
            return

        _apply_field_value(artifact, field=field, value=value)

        deps.runtime.pending_image_inputs.pop(uid, None)
        await msg.answer(
            f"✅ {label} обновлен. Вернитесь к превью-картинке и нажмите «✨ Улучшить»."
        )

    @router.callback_query(F.data.startswith("img:run:"))
    async def image_run_enhancements(cb: CallbackQuery):
        message = _callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return

        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":", 2)
        if len(parts) != 3:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        artifact_id = parts[2]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer("⚠️ Картинка не найдена.", show_alert=True)
            return
        artifact_item = artifact

        if not artifact.enable_sampler_pass and not artifact.params.upscale_model:
            await cb.answer(
                "⚠️ Включите сэмплер или выберите upscaler.",
                show_alert=True,
            )
            return

        status_msg = await message.answer("⏳ Запускаю улучшение...")
        await cb.answer("🚀 Улучшение запущено")
        generation_id = f"enh_{uuid.uuid4().hex}"
        enhancement_cancel_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="❌ Отменить улучшение",
                        callback_data=f"pe:gen:cancel:{generation_id}",
                    )
                ]
            ]
        )
        try:
            await status_msg.edit_reply_markup(reply_markup=enhancement_cancel_kb)
        except TelegramBadRequest:
            pass

        async def _progress(current: int, total: int, text: str) -> None:
            line = f"⏳ {h(text)}"
            if total > 0:
                line = f"⏳ {h(text)} ({current}/{total})"
            try:
                await status_msg.edit_text(line, reply_markup=enhancement_cancel_kb)
            except TelegramBadRequest:
                deps.logger.debug("Image enhancement progress update failed", exc_info=True)

        async def _run() -> None:
            try:
                source_bytes = deps.runtime.artifact_bytes(artifact_item)
                if not source_bytes:
                    await status_msg.edit_text(
                        "❌ Не найдены данные исходной картинки.",
                        reply_markup=None,
                    )
                    return

                run_params = GenerationParams(**asdict(artifact_item.params))
                run_params.batch_size = 1
                run_params.reference_images = []
                run_params.reference_strength = 0.8
                if run_params.seed < 0:
                    run_params.seed = random.randint(0, 2**63 - 1)

                async def _prompt_id_cb(prompt_id: str) -> None:
                    active = deps.runtime.active_generations.get(generation_id)
                    if active is not None:
                        active.prompt_id = prompt_id
                        deps.runtime.persist()

                if artifact_item.enable_sampler_pass:
                    images = await deps.client.generate_from_image(
                        run_params,
                        image_bytes=source_bytes,
                        progress_cb=_progress,
                        prompt_id_cb=_prompt_id_cb,
                    )
                else:
                    images = await deps.client.upscale_image_only(
                        image_bytes=source_bytes,
                        upscale_model=run_params.upscale_model,
                        progress_cb=_progress,
                        prompt_id_cb=_prompt_id_cb,
                    )

                if not images:
                    await status_msg.edit_text(
                        "❌ ComfyUI не вернул изображение.", reply_markup=None
                    )
                    return

                result_image = images[0]
                next_params = GenerationParams(**asdict(run_params))
                try:
                    next_w, next_h = image_dimensions(result_image)
                    next_params.width = next_w
                    next_params.height = next_h
                except (OSError, ValueError):
                    pass

                next_artifact_id = uuid.uuid4().hex
                next_artifact = PreviewArtifact(
                    artifact_id=next_artifact_id,
                    owner_uid=artifact_item.owner_uid,
                    image_bytes=result_image,
                    params=next_params,
                    used_seed=int(run_params.seed),
                    parent_artifact_id=artifact_item.artifact_id,
                    generation_step=artifact_item.generation_step + 1,
                    enable_sampler_pass=artifact_item.enable_sampler_pass,
                )
                deps.runtime.register_preview_artifact(next_artifact)
                deps.runtime.prune_preview_artifacts(artifact_item.owner_uid)

                deps.runtime.last_params[artifact_item.owner_uid] = GenerationParams(
                    **asdict(next_params)
                )
                deps.runtime.last_seeds[artifact_item.owner_uid] = int(run_params.seed)
                deps.runtime.persist()

                sent_previews = await deps.deliver_generated_images(
                    status_msg,
                    [result_image],
                    used_seed=run_params.seed,
                    mode="photo",
                    preview_keyboards=[
                        deps.preview_image_keyboard(
                            next_artifact_id,
                            artifact_item.artifact_id,
                        )
                    ],
                )
                if sent_previews:
                    next_artifact.preview_chat_id = sent_previews[0].chat.id
                    next_artifact.preview_message_id = sent_previews[0].message_id
                await _move_main_panel_to_bottom(
                    artifact_item.owner_uid,
                    status_msg,
                    "✅ Улучшение завершено. Отправил новую превью.\n"
                    "Для каждой превью доступны: отправка PNG и меню улучшений.",
                )
            except asyncio.CancelledError:
                await status_msg.edit_text("❌ Улучшение отменено.", reply_markup=None)
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                deps.logger.exception("Image enhancement failed")
                await status_msg.edit_text(
                    f"❌ Ошибка улучшения: <code>{h(exc)}</code>",
                    reply_markup=None,
                )
            finally:
                deps.runtime.active_image_jobs.pop(temp_job_id, None)
                deps.runtime.active_generations.pop(generation_id, None)
                deps.runtime.persist()

        temp_job_id = f"job_{time.time_ns()}"
        task = asyncio.create_task(_run())
        deps.runtime.active_image_jobs[temp_job_id] = task
        deps.runtime.active_generations[generation_id] = ActiveGeneration(
            owner_uid=uid,
            generation_id=generation_id,
            task=task,
            kind="enhancement",
            title="Улучшение",
            status_msg=status_msg,
            status_chat_id=status_msg.chat.id,
            status_message_id=status_msg.message_id,
        )
        deps.runtime.persist()
