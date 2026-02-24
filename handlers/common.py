from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from aiogram import Router
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from application.model_downloader import ModelDownloader
from application.user_locale_resolver import DefaultUserLocaleResolver
from core.config import Config
from core.html_utils import h, truncate
from core.panels import render_user_panel
from core.queue_utils import queue_item_prompt_id
from core.runtime import RuntimeStore
from core.telegram import callback_user_id, message_user_id
from core.ui_copy import main_menu_keyboard
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, noop_button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService
from infrastructure.comfyui_client import ComfyUIClient

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


def _queue_prompt_state(
    queue: dict[str, Any],
    prompt_id: str,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    if not prompt_id:
        return _tr(
            localization,
            locale,
            "common.jobs.queue_state.local_task",
            "локальная задача",
        )
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    if isinstance(running, list):
        for item in running:
            if queue_item_prompt_id(item) == prompt_id:
                return _tr(
                    localization,
                    locale,
                    "common.jobs.queue_state.running",
                    "выполняется",
                )
    if isinstance(pending, list):
        for index, item in enumerate(pending, start=1):
            if queue_item_prompt_id(item) == prompt_id:
                return _tr(
                    localization,
                    locale,
                    "common.jobs.queue_state.pending_index",
                    f"в очереди ({index})",
                    params={"index": index},
                )
    return _tr(
        localization,
        locale,
        "common.jobs.queue_state.not_found",
        "не найдена в очереди",
    )


def build_training_text(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    title = _tr(
        localization,
        locale,
        "common.training.legacy.title",
        "🎓 <b>Обучение ComfyBot (для новичков)</b>",
    )
    step_1 = _tr(
        localization,
        locale,
        "common.training.legacy.step_1",
        "1) Нажмите «📦 Модели» -> «🧪 Обновить список».",
    )
    step_2 = _tr(
        localization,
        locale,
        "common.training.legacy.step_2",
        "2) Нажмите «🎨 Генерация» -> «✨ Новая генерация».",
    )
    step_3 = _tr(
        localization,
        locale,
        "common.training.legacy.step_3",
        "3) Выберите Checkpoint, задайте Positive/Negative.",
    )
    step_4 = _tr(
        localization,
        locale,
        "common.training.legacy.step_4",
        "4) При необходимости настройте Steps, CFG, Seed, размер.",
    )
    step_5 = _tr(
        localization,
        locale,
        "common.training.legacy.step_5",
        "5) Нажмите «Генерировать».",
    )
    useful = _tr(
        localization,
        locale,
        "common.training.legacy.useful",
        "Полезно: «🔁 Повтор», «📂 Пресеты», «⬇️ Скачать модель», «🛑 Отмена».",
    )
    extra = _tr(
        localization,
        locale,
        "common.training.legacy.extra",
        "Ещё: 🧠 Smart (TIPO), 📎 LoRA, 🖼 референсы, ✨ улучшения.",
    )
    full_guide = _tr(
        localization,
        locale,
        "common.training.legacy.full_guide",
        "Полный гайд: <code>docs/COMMANDS.md</code>",
    )

    return "\n".join(
        [
            title,
            "",
            step_1,
            step_2,
            step_3,
            step_4,
            step_5,
            "",
            useful,
            extra,
            "",
            full_guide,
        ]
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


def _tr(
    localization: LocalizationService | None,
    locale: str | None,
    key: str,
    default: str,
    *,
    params: Mapping[str, object] | None = None,
) -> str:
    if localization is None:
        return default
    return localization.t(key, locale=locale, params=params, default=default)


TRAINING_TEXT = build_training_text()


def _generation_menu_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    new_label = _tr(localization, locale, "common.menu.generation.new", "✨ Новая генерация")
    repeat_label = _tr(localization, locale, "common.menu.generation.repeat", "🔁 Повтор")
    presets_label = _tr(localization, locale, "common.menu.generation.presets", "📂 Пресеты")
    back_label = _tr(localization, locale, "common.menu.back_to_menu", "⬅️ В меню")

    return build_keyboard(
        [
            [button(new_label, "menu:generate")],
            [button(repeat_label, "menu:repeat")],
            [button(presets_label, "menu:presets")],
            [back_button("menu:root", text=back_label)],
        ]
    )


def _models_menu_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    download_label = _tr(localization, locale, "common.menu.models.download", "⬇️ Скачать модель")
    refresh_label = _tr(localization, locale, "common.menu.models.refresh", "🧪 Обновить список")
    delete_label = _tr(
        localization,
        locale,
        "common.menu.models.delete_local",
        "🗑 Удалить локальную",
    )
    back_label = _tr(localization, locale, "common.menu.back_to_menu", "⬅️ В меню")

    return build_keyboard(
        [
            [button(download_label, "menu:download")],
            [button(refresh_label, "menu:models_refresh")],
            [button(delete_label, "menu:delete_model")],
            [back_button("menu:root", text=back_label)],
        ]
    )


def _service_menu_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    jobs_label = _tr(localization, locale, "common.menu.service.jobs", "🧵 Мои задачи")
    queue_label = _tr(localization, locale, "common.menu.service.queue", "📊 Очередь")
    settings_label = _tr(localization, locale, "common.menu.service.settings", "⚙️ Настройки")
    training_label = _tr(localization, locale, "common.menu.service.training", "🎓 Обучение")
    back_label = _tr(localization, locale, "common.menu.back_to_menu", "⬅️ В меню")

    return build_keyboard(
        [
            [button(jobs_label, "menu:jobs")],
            [button(queue_label, "menu:queue"), button(settings_label, "menu:settings")],
            [button(training_label, "menu:training")],
            [back_button("menu:root", text=back_label)],
        ]
    )


def _jobs_keyboard(
    items: list,
    *,
    page: int,
    page_size: int = 6,
    localization: LocalizationService | None = None,
    locale: str | None = None,
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

    refresh_label = _tr(localization, locale, "common.jobs.keyboard.refresh", "🔄 Обновить")
    cancel_all_label = _tr(
        localization,
        locale,
        "common.jobs.keyboard.cancel_all",
        "❌ Отменить все",
    )
    back_label = _tr(localization, locale, "common.menu.back_to_service", "⬅️ Сервис")

    rows.append(
        [
            button(refresh_label, f"menu:jobs:page:{page}"),
            cancel_button("menu:jobs:cancel_all", text=cancel_all_label),
        ]
    )
    rows.append([back_button("menu:service", text=back_label)])
    return build_keyboard(rows)


def _job_detail_keyboard(
    generation_id: str,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    refresh_label = _tr(localization, locale, "common.jobs.keyboard.refresh", "🔄 Обновить")
    source_label = _tr(localization, locale, "common.jobs.keyboard.goto_source", "📌 К источнику")
    cancel_label = _tr(
        localization, locale, "common.jobs.keyboard.cancel_job", "🛑 Отменить задачу"
    )
    back_label = _tr(localization, locale, "common.menu.back_to_list", "⬅️ К списку")

    return build_keyboard(
        [
            [
                button(refresh_label, f"menu:jobs:open:{generation_id}"),
                button(source_label, f"menu:jobs:goto:{generation_id}"),
            ],
            [
                button(cancel_label, f"menu:jobs:cancel:{generation_id}"),
                back_button("menu:jobs", text=back_label),
            ],
        ]
    )


def _models_section(
    title: str,
    emoji: str,
    items: list[str],
    limit: int = 15,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    count = len(items)
    header = f"{emoji} <b>{title}</b> ({count})"
    if not items:
        return header
    shown = [f"  <code>{h(item)}</code>" for item in items[:limit]]
    if count > limit:
        more_label = _tr(
            localization,
            locale,
            "common.models.section.more",
            f"…и ещё {count - limit}",
            params={"count": count - limit},
        )
        shown.append(f"  <i>{more_label}</i>")
    return header + "\n" + "\n".join(shown)


def _delete_types_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for model_type, label in DELETE_MODEL_TYPES.items():
        localized_label = _tr(
            localization,
            locale,
            f"common.delete.model_type.{model_type}",
            label,
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text=localized_label,
                    callback_data=f"dmdl:type:{model_type}",
                )
            ]
        )
    cancel_label = _tr(localization, locale, "common.action.cancel", "❌ Отмена")
    rows.append([cancel_button("dmdl:cancel", text=cancel_label)])
    return build_keyboard(rows)


def _delete_files_keyboard(
    *,
    model_type: str,
    files: list[str],
    page: int,
    localization: LocalizationService | None = None,
    locale: str | None = None,
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

    back_type_label = _tr(localization, locale, "common.menu.back_to_type", "⬅️ Тип")
    cancel_label = _tr(localization, locale, "common.action.cancel", "❌ Отмена")
    rows.append(
        [
            back_button("dmdl:back:type", text=back_type_label),
            cancel_button("dmdl:cancel", text=cancel_label),
        ]
    )
    return build_keyboard(rows)


def _delete_confirm_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    delete_label = _tr(localization, locale, "common.delete.action.delete", "🗑 Удалить")
    back_label = _tr(localization, locale, "common.action.back", "⬅️ Назад")
    cancel_label = _tr(localization, locale, "common.action.cancel", "❌ Отмена")

    return build_keyboard(
        [
            [
                button(delete_label, "dmdl:confirm:yes"),
                back_button("dmdl:confirm:no", text=back_label),
            ],
            [cancel_button("dmdl:cancel", text=cancel_label)],
        ]
    )


def register_common_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    localization: LocalizationService,
) -> None:
    locale_resolver = DefaultUserLocaleResolver(localization)

    def _resolve_locale_for_user(user_id: int | None, telegram_locale: str | None) -> str:
        if user_id is None or user_id <= 0:
            return locale_resolver.resolve(
                user_locale=None,
                telegram_locale=telegram_locale,
            )
        prefs = runtime.user_preferences.get(user_id, {})
        selected_locale = read_user_locale(
            prefs,
            default_locale=localization.default_locale(),
        )
        return locale_resolver.resolve(
            user_locale=selected_locale,
            telegram_locale=telegram_locale,
        )

    def _localized_models_section(
        title: str,
        emoji: str,
        items: list[str],
        limit: int = 15,
    ) -> str:
        return _models_section(
            title,
            emoji,
            items,
            limit,
            localization=localization,
        )

    register_access_middlewares(
        router,
        allowed_users=cfg.allowed_users,
        runtime_persist=runtime.persist,
        localization=localization,
        resolve_locale_for_user=_resolve_locale_for_user,
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
            localization=localization,
            resolve_user_locale=locale_resolver.resolve,
            main_menu_keyboard=lambda locale: main_menu_keyboard(localization, locale=locale),
            generation_menu_keyboard=(
                lambda locale: _generation_menu_keyboard(localization=localization, locale=locale)
            ),
            models_menu_keyboard=(
                lambda locale: _models_menu_keyboard(localization=localization, locale=locale)
            ),
            service_menu_keyboard=(
                lambda locale: _service_menu_keyboard(localization=localization, locale=locale)
            ),
            models_section=_localized_models_section,
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
            localization=localization,
            resolve_user_locale=locale_resolver.resolve,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            render_user_panel=render_user_panel,
            delete_model_types=DELETE_MODEL_TYPES,
            delete_types_keyboard=lambda locale: _delete_types_keyboard(
                localization=localization,
                locale=locale,
            ),
            delete_files_keyboard=(
                lambda model_type, files, page, locale: _delete_files_keyboard(
                    model_type=model_type,
                    files=files,
                    page=page,
                    localization=localization,
                    locale=locale,
                )
            ),
            delete_confirm_keyboard=lambda locale: _delete_confirm_keyboard(
                localization=localization,
                locale=locale,
            ),
            h=h,
        )
    )

    register_common_jobs_handlers(
        CommonJobsDeps(
            router=router,
            runtime=runtime,
            client=client,
            localization=localization,
            resolve_user_locale=locale_resolver.resolve,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            render_user_panel=render_user_panel,
            user_generations=_user_generations,
            queue_prompt_state=lambda queue, prompt_id, locale: _queue_prompt_state(
                queue,
                prompt_id,
                localization=localization,
                locale=locale,
            ),
            jobs_keyboard=lambda items, page, locale: _jobs_keyboard(
                items,
                page=page,
                localization=localization,
                locale=locale,
            ),
            job_detail_keyboard=lambda generation_id, locale: _job_detail_keyboard(
                generation_id,
                localization=localization,
                locale=locale,
            ),
            truncate=truncate,
            h=h,
        )
    )
