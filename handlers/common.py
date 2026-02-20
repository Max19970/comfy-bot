from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.types.base import TelegramObject

from comfyui_client import ComfyUIClient
from config import Config
from core.html_utils import h, truncate
from core.panels import render_user_panel
from core.queue_utils import queue_item_prompt_id
from core.runtime import RuntimeStore
from core.telegram import callback_user_id, message_user_id
from core.ui_copy import FALLBACK_TEXT, START_TEXT, main_menu_keyboard
from model_downloader import ModelDownloader

from .common_core_handlers import CommonCoreDeps, register_common_core_handlers
from .common_ops_handlers import (
    CommonDeleteDeps,
    CommonJobsDeps,
    register_common_delete_handlers,
    register_common_jobs_handlers,
)

logger = logging.getLogger(__name__)


def _user_generations(runtime: RuntimeStore, uid: int):
    items = [item for item in runtime.active_generations.values() if item.owner_uid == uid]
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


def _queue_prompt_state(queue: dict[str, Any], prompt_id: str) -> str:
    if not prompt_id:
        return "локальная задача"
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    if isinstance(running, list):
        for item in running:
            if queue_item_prompt_id(item) == prompt_id:
                return "выполняется"
    if isinstance(pending, list):
        for index, item in enumerate(pending, start=1):
            if queue_item_prompt_id(item) == prompt_id:
                return f"в очереди ({index})"
    return "не найдена в очереди"


TRAINING_TEXT = (
    "\U0001f393 <b>\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 ComfyBot (\u0434\u043b\u044f \u043d\u043e\u0432\u0438\u0447\u043a\u043e\u0432)</b>\n"
    "\n"
    "1) \u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u00ab\U0001f4e6 \u041c\u043e\u0434\u0435\u043b\u0438\u00bb -> \u00ab\U0001f9ea \u041e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u0441\u043f\u0438\u0441\u043e\u043a\u00bb.\n"
    "2) \u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u00ab\U0001f3a8 \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\u00bb -> \u00ab\u2728 \u041d\u043e\u0432\u0430\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\u00bb.\n"
    "3) \u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 Checkpoint, \u0437\u0430\u0434\u0430\u0439\u0442\u0435 Positive/Negative.\n"
    "4) \u041f\u0440\u0438 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u0438 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u0442\u0435 Steps, CFG, Seed, \u0440\u0430\u0437\u043c\u0435\u0440.\n"
    "5) \u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u00ab\u0413\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c\u00bb.\n"
    "\n"
    "\u041f\u043e\u043b\u0435\u0437\u043d\u043e: \u00ab\U0001f501 \u041f\u043e\u0432\u0442\u043e\u0440\u00bb, \u00ab\U0001f4c2 \u041f\u0440\u0435\u0441\u0435\u0442\u044b\u00bb, \u00ab\u2b07\ufe0f \u0421\u043a\u0430\u0447\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c\u00bb, \u00ab\U0001f6d1 \u041e\u0442\u043c\u0435\u043d\u0430\u00bb.\n"
    "\u0415\u0449\u0451: \U0001f9e0 Smart (TIPO), \U0001f4ce LoRA, \U0001f5bc \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u044b, \u2728 \u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f.\n"
    "\n"
    "\u041f\u043e\u043b\u043d\u044b\u0439 \u0433\u0430\u0439\u0434: <code>docs/COMMANDS.md</code>"
)

DELETE_MODEL_TYPES = {
    "checkpoint": "Checkpoint",
    "lora": "LoRA",
    "upscaler": "Upscaler",
    "embedding": "Embedding",
    "controlnet": "ControlNet",
    "vae": "VAE",
}
DELETE_PAGE_SIZE = 8


def _generation_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✨ Новая генерация",
                    callback_data="menu:generate",
                )
            ],
            [InlineKeyboardButton(text="🔁 Повтор", callback_data="menu:repeat")],
            [InlineKeyboardButton(text="📂 Пресеты", callback_data="menu:presets")],
            [
                InlineKeyboardButton(
                    text="⬅️ В меню",
                    callback_data="menu:root",
                )
            ],
        ]
    )


def _models_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⬇️ Скачать модель",
                    callback_data="menu:download",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🧪 Обновить список",
                    callback_data="menu:models_refresh",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Удалить локальную",
                    callback_data="menu:delete_model",
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


def _service_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🧵 Мои задачи",
                    callback_data="menu:jobs",
                )
            ],
            [
                InlineKeyboardButton(text="📊 Очередь", callback_data="menu:queue"),
                InlineKeyboardButton(text="⚙️ Настройки", callback_data="menu:settings"),
            ],
            [InlineKeyboardButton(text="🎓 Обучение", callback_data="menu:training")],
            [
                InlineKeyboardButton(
                    text="⬅️ В меню",
                    callback_data="menu:root",
                )
            ],
        ]
    )


def _jobs_keyboard(
    items: list,
    *,
    page: int,
    page_size: int = 6,
) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    chunk = items[start : start + page_size]

    rows: list[list[InlineKeyboardButton]] = []
    for offset, item in enumerate(chunk, start=1):
        kind_icon = "🎨" if item.kind != "enhancement" else "✨"
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{start + offset}. {kind_icon} {truncate(item.title, 24)}",
                    callback_data=f"menu:jobs:open:{item.generation_id}",
                )
            ]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="◀️",
                callback_data=f"menu:jobs:page:{page - 1}",
            )
        )
    nav.append(
        InlineKeyboardButton(
            text=f"· {page + 1}/{total_pages} ·",
            callback_data="noop",
        )
    )
    if page < total_pages - 1:
        nav.append(
            InlineKeyboardButton(
                text="▶️",
                callback_data=f"menu:jobs:page:{page + 1}",
            )
        )
    rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                text="🔄 Обновить",
                callback_data=f"menu:jobs:page:{page}",
            ),
            InlineKeyboardButton(
                text="❌ Отменить все",
                callback_data="menu:cancel",
            ),
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text="⬅️ Сервис",
                callback_data="menu:service",
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _job_detail_keyboard(generation_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Обновить",
                    callback_data=f"menu:jobs:open:{generation_id}",
                ),
                InlineKeyboardButton(
                    text="📌 К источнику",
                    callback_data=f"menu:jobs:goto:{generation_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🛑 Отменить задачу",
                    callback_data=f"menu:jobs:cancel:{generation_id}",
                ),
                InlineKeyboardButton(
                    text="⬅️ К списку",
                    callback_data="menu:jobs",
                ),
            ],
        ]
    )


def _models_section(title: str, emoji: str, items: list[str], limit: int = 15) -> str:
    count = len(items)
    header = f"{emoji} <b>{title}</b> ({count})"
    if not items:
        return header
    shown = [f"  <code>{h(item)}</code>" for item in items[:limit]]
    if count > limit:
        shown.append(f"  <i>…и ещё {count - limit}</i>")
    return header + "\n" + "\n".join(shown)


def _delete_types_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for model_type, label in DELETE_MODEL_TYPES.items():
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"dmdl:type:{model_type}",
                )
            ]
        )
    rows.append([InlineKeyboardButton(text="❌ Отмена", callback_data="dmdl:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _delete_files_keyboard(
    *,
    model_type: str,
    files: list[str],
    page: int,
) -> InlineKeyboardMarkup:
    total_pages = max(1, (len(files) + DELETE_PAGE_SIZE - 1) // DELETE_PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * DELETE_PAGE_SIZE
    chunk = files[start : start + DELETE_PAGE_SIZE]

    rows: list[list[InlineKeyboardButton]] = []
    for idx, filename in enumerate(chunk, start=start):
        short = filename if len(filename) <= 42 else filename[:39] + "..."
        rows.append(
            [
                InlineKeyboardButton(
                    text=short,
                    callback_data=f"dmdl:file:{idx}",
                )
            ]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="◀️", callback_data=f"dmdl:page:{page - 1}"))
    nav.append(
        InlineKeyboardButton(
            text=f"· {page + 1}/{total_pages} ·",
            callback_data="noop",
        )
    )
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton(text="▶️", callback_data=f"dmdl:page:{page + 1}"))
    rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(text="⬅️ Тип", callback_data="dmdl:back:type"),
            InlineKeyboardButton(text="❌ Отмена", callback_data="dmdl:cancel"),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _delete_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🗑 Удалить", callback_data="dmdl:confirm:yes"),
                InlineKeyboardButton(text="⬅️ Назад", callback_data="dmdl:confirm:no"),
            ],
            [InlineKeyboardButton(text="❌ Отмена", callback_data="dmdl:cancel")],
        ]
    )


def register_common_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
) -> None:
    async def wl_msg(
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)
        command_text = (event.text or "").strip()
        should_delete_command = command_text.startswith("/")
        if cfg.allowed_users and event.from_user and event.from_user.id not in cfg.allowed_users:
            await event.answer(
                "\u26d4 <b>\u0414\u043e\u0441\u0442\u0443\u043f \u0437\u0430\u043f\u0440\u0435\u0449\u0451\u043d</b>\n\u0412\u0430\u0448 ID \u043d\u0435 \u0432 \u0441\u043f\u0438\u0441\u043a\u0435 \u0440\u0430\u0437\u0440\u0435\u0448\u0451\u043d\u043d\u044b\u0445."
            )
            return
        try:
            return await handler(event, data)
        finally:
            if should_delete_command:
                try:
                    await event.delete()
                except TelegramBadRequest:
                    pass
            runtime.persist()

    async def wl_cb(
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, CallbackQuery):
            return await handler(event, data)
        if cfg.allowed_users and event.from_user and event.from_user.id not in cfg.allowed_users:
            await event.answer(
                "\u26d4 \u0414\u043e\u0441\u0442\u0443\u043f \u0437\u0430\u043f\u0440\u0435\u0449\u0451\u043d.",
                show_alert=True,
            )
            return
        try:
            return await handler(event, data)
        finally:
            runtime.persist()

    router.message.outer_middleware(wl_msg)
    router.callback_query.outer_middleware(wl_cb)

    def _callback_message(cb: CallbackQuery) -> Message | None:
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            return None
        return cast(Message, cb.message)

    register_common_core_handlers(
        CommonCoreDeps(
            router=router,
            cfg=cfg,
            runtime=runtime,
            client=client,
            callback_message=_callback_message,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            render_user_panel=render_user_panel,
            start_text=START_TEXT,
            training_text=TRAINING_TEXT,
            fallback_text=FALLBACK_TEXT,
            main_menu_keyboard=main_menu_keyboard,
            generation_menu_keyboard=_generation_menu_keyboard,
            models_menu_keyboard=_models_menu_keyboard,
            service_menu_keyboard=_service_menu_keyboard,
            models_section=_models_section,
            user_generations=_user_generations,
            truncate=truncate,
            h=h,
        )
    )

    register_common_delete_handlers(
        CommonDeleteDeps(
            router=router,
            runtime=runtime,
            client=client,
            downloader=downloader,
            callback_message=_callback_message,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            render_user_panel=render_user_panel,
            delete_model_types=DELETE_MODEL_TYPES,
            delete_types_keyboard=_delete_types_keyboard,
            delete_files_keyboard=_delete_files_keyboard,
            delete_confirm_keyboard=_delete_confirm_keyboard,
            h=h,
        )
    )

    register_common_jobs_handlers(
        CommonJobsDeps(
            router=router,
            runtime=runtime,
            client=client,
            callback_message=_callback_message,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            render_user_panel=render_user_panel,
            user_generations=_user_generations,
            queue_prompt_state=_queue_prompt_state,
            jobs_keyboard=_jobs_keyboard,
            job_detail_keyboard=_job_detail_keyboard,
            truncate=truncate,
            h=h,
        )
    )
