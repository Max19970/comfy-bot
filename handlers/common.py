from __future__ import annotations

import logging
from typing import Any

from aiogram import Router
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from comfyui_client import ComfyUIClient
from config import Config
from core.html_utils import h, truncate
from core.panels import render_user_panel
from core.queue_utils import queue_item_prompt_id
from core.runtime import RuntimeStore
from core.telegram import callback_user_id, message_user_id
from core.ui_copy import FALLBACK_TEXT, START_TEXT, main_menu_keyboard
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, noop_button
from model_downloader import ModelDownloader

from .common_core_handlers import CommonCoreDeps, register_common_core_handlers
from .common_middleware import register_access_middlewares
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
    return build_keyboard(
        [
            [button("✨ Новая генерация", "menu:generate")],
            [button("🔁 Повтор", "menu:repeat")],
            [button("📂 Пресеты", "menu:presets")],
            [back_button("menu:root", text="⬅️ В меню")],
        ]
    )


def _models_menu_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("⬇️ Скачать модель", "menu:download")],
            [button("🧪 Обновить список", "menu:models_refresh")],
            [button("🗑 Удалить локальную", "menu:delete_model")],
            [back_button("menu:root", text="⬅️ В меню")],
        ]
    )


def _service_menu_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("🧵 Мои задачи", "menu:jobs")],
            [button("📊 Очередь", "menu:queue"), button("⚙️ Настройки", "menu:settings")],
            [button("🎓 Обучение", "menu:training")],
            [back_button("menu:root", text="⬅️ В меню")],
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
        nav.append(button("◀️", f"menu:jobs:page:{page - 1}"))
    nav.append(noop_button(f"· {page + 1}/{total_pages} ·"))
    if page < total_pages - 1:
        nav.append(button("▶️", f"menu:jobs:page:{page + 1}"))
    rows.append(nav)

    rows.append(
        [
            button("🔄 Обновить", f"menu:jobs:page:{page}"),
            cancel_button("menu:jobs:cancel_all", text="❌ Отменить все"),
        ]
    )
    rows.append([back_button("menu:service", text="⬅️ Сервис")])
    return build_keyboard(rows)


def _job_detail_keyboard(generation_id: str) -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [
                button("🔄 Обновить", f"menu:jobs:open:{generation_id}"),
                button("📌 К источнику", f"menu:jobs:goto:{generation_id}"),
            ],
            [
                button("🛑 Отменить задачу", f"menu:jobs:cancel:{generation_id}"),
                back_button("menu:jobs", text="⬅️ К списку"),
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
    rows.append([cancel_button("dmdl:cancel")])
    return build_keyboard(rows)


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
        nav.append(button("◀️", f"dmdl:page:{page - 1}"))
    nav.append(noop_button(f"· {page + 1}/{total_pages} ·"))
    if page < total_pages - 1:
        nav.append(button("▶️", f"dmdl:page:{page + 1}"))
    rows.append(nav)

    rows.append(
        [
            back_button("dmdl:back:type", text="⬅️ Тип"),
            cancel_button("dmdl:cancel"),
        ]
    )
    return build_keyboard(rows)


def _delete_confirm_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("🗑 Удалить", "dmdl:confirm:yes"), back_button("dmdl:confirm:no")],
            [cancel_button("dmdl:cancel")],
        ]
    )


def register_common_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
) -> None:
    register_access_middlewares(
        router,
        allowed_users=cfg.allowed_users,
        runtime_persist=runtime.persist,
    )

    register_common_core_handlers(
        CommonCoreDeps(
            router=router,
            cfg=cfg,
            runtime=runtime,
            client=client,
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
