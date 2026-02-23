from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from core.download_filters import (
    DOWNLOAD_FILTER_PROFILES,
    download_base_label,
)
from core.interaction import require_callback_message
from core.states import ServiceSettingsStates
from core.ui_copy import FALLBACK_TEXT, START_TEXT
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, menu_root_button, noop_button
from core.user_preferences import read_download_defaults, read_generation_defaults, read_user_locale
from domain.localization import normalize_locale_code

from .common_core_utils import (
    get_training_mode,
    get_training_page,
    set_pref,
    set_training_mode,
    set_training_page,
    training_advanced,
    training_pages,
)
from .common_settings_logic import (
    SettingsParseError,
    parse_download_author,
    parse_download_callback_value,
    parse_generation_callback_value,
    parse_generation_manual_value,
)


@dataclass
class CommonCoreDeps:
    router: Router
    cfg: Any
    runtime: Any
    client: Any
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    localization: Any
    resolve_user_locale: Callable[..., str]
    main_menu_keyboard: Callable[[str | None], Any]
    generation_menu_keyboard: Callable[[str | None], Any]
    models_menu_keyboard: Callable[[str | None], Any]
    service_menu_keyboard: Callable[[str | None], Any]
    models_section: Callable[[str, str, list[str], int], str]
    user_generations: Callable[[Any, int], list[Any]]
    truncate: Callable[[str, int], str]
    h: Callable[[Any], str]


def register_common_core_handlers(deps: CommonCoreDeps) -> None:
    router = deps.router

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

    def _service_back_keyboard() -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [back_button("menu:service", text="⬅️ Сервис")],
                [menu_root_button()],
            ]
        )

    def _models_back_keyboard() -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [back_button("menu:models", text="⬅️ Модели")],
                [menu_root_button()],
            ]
        )

    def _training_keyboard(*, page: int, total: int, mode: str) -> InlineKeyboardMarkup:
        rows: list[list[InlineKeyboardButton]] = []
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(button("◀️", f"menu:training:page:{page - 1}"))
        nav.append(noop_button(f"· {page + 1}/{total} ·"))
        if page < total - 1:
            nav.append(button("▶️", f"menu:training:page:{page + 1}"))
        rows.append(nav)
        rows.append(
            [
                button(
                    ("✅ Простой" if mode == "simple" else "Простой"),
                    "menu:training:mode:simple",
                ),
                button(
                    ("✅ Расширенный" if mode == "advanced" else "Расширенный"),
                    "menu:training:mode:advanced",
                ),
            ]
        )
        rows.append([back_button("menu:service", text="⬅️ Сервис")])
        rows.append([menu_root_button()])
        return build_keyboard(rows)

    async def _show_training(message: Message, uid: int, *, page: int | None = None) -> None:
        pages = training_pages()
        mode = get_training_mode(deps.runtime, uid)
        total = len(pages)
        current_page = get_training_page(deps.runtime, uid) if page is None else page
        current_page = max(0, min(current_page, total - 1))
        set_training_page(deps.runtime, uid, current_page)

        title, simple_text = pages[current_page]
        lines = [
            "🎓 <b>Обучение ComfyBot</b>",
            f"<b>Тема:</b> {deps.h(title)}",
            "",
            simple_text,
        ]
        if mode == "advanced":
            lines.extend(["", "<b>Подробно:</b>", training_advanced()[current_page]])

        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "\n".join(lines),
            reply_markup=_training_keyboard(page=current_page, total=total, mode=mode),
        )

    def _gen_defaults(uid: int) -> dict[str, Any]:
        prefs = deps.runtime.user_preferences.get(uid, {})
        return read_generation_defaults(
            prefs,
            default_width=deps.cfg.default_width,
            default_height=deps.cfg.default_height,
            default_steps=deps.cfg.default_steps,
            default_cfg=deps.cfg.default_cfg,
            default_denoise=deps.cfg.default_denoise,
            default_sampler=deps.cfg.default_sampler,
            default_scheduler=deps.cfg.default_scheduler,
        )

    def _download_defaults(uid: int) -> dict[str, Any]:
        prefs = deps.runtime.user_preferences.get(uid, {})
        return read_download_defaults(prefs)

    def _apply_download_profile(uid: int, profile_code: str) -> bool:
        profile = DOWNLOAD_FILTER_PROFILES.get(profile_code)
        if not profile:
            return False
        set_pref(deps.runtime, uid, "dl_default_source", profile["source"])
        set_pref(deps.runtime, uid, "dl_default_sort", profile["sort"])
        set_pref(deps.runtime, uid, "dl_default_period", profile["period"])
        set_pref(deps.runtime, uid, "dl_default_base", profile["base"])
        set_pref(deps.runtime, uid, "dl_default_nsfw", profile["nsfw"])
        return True

    def _base_label(code: str) -> str:
        return download_base_label(code)

    def _resolved_locale(uid: int, *, telegram_locale: str | None) -> str:
        prefs = deps.runtime.user_preferences.get(uid, {})
        user_locale = read_user_locale(
            prefs,
            default_locale=deps.localization.default_locale(),
        )
        return deps.resolve_user_locale(
            user_locale=user_locale,
            telegram_locale=telegram_locale,
        )

    def _locale_label(locale: str) -> str:
        locale_code = normalize_locale_code(locale, default="")
        if not locale_code:
            return "Unknown"
        localized_name = deps.localization.t(
            "system.language_name",
            locale=locale_code,
            default=locale_code.upper(),
        )
        return f"{localized_name} ({locale_code})"

    def _start_text_for_uid(uid: int, *, telegram_locale: str | None) -> str:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        return deps.localization.t("ui.start", locale=locale, default=START_TEXT)

    def _fallback_text_for_uid(uid: int, *, telegram_locale: str | None) -> str:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        return deps.localization.t("ui.fallback", locale=locale, default=FALLBACK_TEXT)

    async def _show_locale_settings(
        message: Message,
        uid: int,
        *,
        telegram_locale: str | None,
    ) -> None:
        current_locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        selected_locale = read_user_locale(
            deps.runtime.user_preferences.get(uid, {}),
            default_locale="",
        )

        rows = [
            [
                InlineKeyboardButton(
                    text=(
                        f"✅ {_locale_label(locale_code)}"
                        if locale_code == current_locale
                        else _locale_label(locale_code)
                    ),
                    callback_data=f"menu:settings:set:locale:{locale_code}",
                )
            ]
            for locale_code in deps.localization.available_locales()
        ]
        rows.append([InlineKeyboardButton(text="⬅️ К настройкам", callback_data="menu:settings")])
        rows.append([InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")])

        explicit_line = (
            f"<b>Явный выбор пользователя:</b> <code>{deps.h(_locale_label(selected_locale))}</code>"
            if selected_locale
            else "<b>Явный выбор пользователя:</b> <code>не задан</code>"
        )

        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🌐 <b>Язык интерфейса</b>\n\n"
            f"<b>Текущий язык:</b> <code>{deps.h(_locale_label(current_locale))}</code>\n"
            f"{explicit_line}\n"
            "\nВыберите язык для сохранения в персональных настройках.",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )

    async def _show_settings(
        message: Message,
        uid: int,
        *,
        telegram_locale: str | None,
    ) -> None:
        pro_mode = deps.runtime.user_preferences.get(uid, {}).get("pro_mode", False)
        mode_label = "🔧 Про" if pro_mode else "🟢 Простой"
        locale_label = _locale_label(_resolved_locale(uid, telegram_locale=telegram_locale))
        smart_prompt_status = "❌ выключен"
        if deps.cfg.smart_prompt_enabled:
            model = deps.cfg.smart_prompt_model or "(модель не задана)"
            smart_prompt_status = f"✅ TIPO | <code>{deps.h(model)}</code>"

        gen = _gen_defaults(uid)
        dl = _download_defaults(uid)
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="🎨 Генерация", callback_data="menu:settings:gen"),
                    InlineKeyboardButton(text="📦 Поиск моделей", callback_data="menu:settings:dl"),
                ],
                [
                    InlineKeyboardButton(
                        text=("🟢 Простой" if pro_mode else "🔧 Про"),
                        callback_data="menu:settings:toggle_mode",
                    ),
                    InlineKeyboardButton(text="🌐 Язык", callback_data="menu:settings:locale"),
                ],
                [InlineKeyboardButton(text="🔄 Обновить", callback_data="menu:settings")],
                [InlineKeyboardButton(text="⬅️ Сервис", callback_data="menu:service")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
            ]
        )

        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "⚙️ <b>Настройки</b>\n\n"
            f"<b>Режим интерфейса:</b> {mode_label}\n"
            f"<b>Язык интерфейса:</b> <code>{deps.h(locale_label)}</code>\n"
            f"<b>Генерация по умолчанию:</b> <code>{gen['width']}×{gen['height']}</code> | "
            f"Steps <code>{gen['steps']}</code> | CFG <code>{gen['cfg']}</code> | "
            f"Denoise <code>{gen['denoise']}</code>\n"
            f"<b>Seed/Batch:</b> <code>{gen['seed']}</code> / <code>{gen['batch']}</code>\n"
            f"<b>Sampler/Scheduler:</b> <code>{deps.h(gen['sampler'])}</code> / "
            f"<code>{deps.h(gen['scheduler'])}</code>\n"
            f"<b>Поиск моделей по умолчанию:</b> {deps.h(dl['source'])}, {deps.h(dl['sort'])}, "
            f"{deps.h(dl['period'])}, base={deps.h(_base_label(dl['base']))}, nsfw={'on' if dl['nsfw'] else 'off'}\n"
            f"<b>Автор(ы) CivitAI:</b> <code>{deps.h(dl['author'] or 'любой')}</code>\n"
            f"<b>Smart Prompt:</b> {smart_prompt_status}\n"
            f"<b>Папка моделей:</b> <code>{deps.h(deps.cfg.comfyui_models_path)}</code>",
            reply_markup=kb,
        )

    async def _show_generation_settings(message: Message, uid: int) -> None:
        gen = _gen_defaults(uid)
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"🖼 Размер: {gen['width']}×{gen['height']}",
                        callback_data="menu:settings:gen:menu:size",
                    ),
                    InlineKeyboardButton(
                        text=f"🔢 Steps: {gen['steps']}",
                        callback_data="menu:settings:gen:menu:steps",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"🎚 CFG: {gen['cfg']}",
                        callback_data="menu:settings:gen:menu:cfg",
                    ),
                    InlineKeyboardButton(
                        text=f"🌫 Denoise: {gen['denoise']}",
                        callback_data="menu:settings:gen:menu:denoise",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=("🎲 Seed: random" if gen["seed"] < 0 else f"🎲 Seed: {gen['seed']}"),
                        callback_data="menu:settings:gen:menu:seed",
                    ),
                    InlineKeyboardButton(
                        text=f"🗂 Batch: {gen['batch']}",
                        callback_data="menu:settings:gen:menu:batch",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"⚙️ Sampler: {gen['sampler']}",
                        callback_data="menu:settings:gen:menu:sampler:0",
                    ),
                    InlineKeyboardButton(
                        text=f"📈 Scheduler: {gen['scheduler']}",
                        callback_data="menu:settings:gen:menu:scheduler:0",
                    ),
                ],
                [InlineKeyboardButton(text="↺ Сбросить", callback_data="menu:settings:reset:gen")],
                [InlineKeyboardButton(text="⬅️ К настройкам", callback_data="menu:settings")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
            ]
        )
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🎨 <b>Настройки генерации по умолчанию</b>\n\n"
            f"<b>Размер:</b> <code>{gen['width']}×{gen['height']}</code>\n"
            f"<b>Steps:</b> <code>{gen['steps']}</code>\n"
            f"<b>CFG:</b> <code>{gen['cfg']}</code>\n"
            f"<b>Denoise:</b> <code>{gen['denoise']}</code>\n"
            f"<b>Seed:</b> <code>{gen['seed']}</code>\n"
            f"<b>Batch:</b> <code>{gen['batch']}</code>\n"
            f"<b>Sampler:</b> <code>{deps.h(gen['sampler'])}</code>\n"
            f"<b>Scheduler:</b> <code>{deps.h(gen['scheduler'])}</code>",
            reply_markup=kb,
        )

    async def _show_generation_picker(
        message: Message,
        uid: int,
        *,
        field: str,
        page: int = 0,
    ) -> None:
        label_map = {
            "size": "Размер",
            "steps": "Steps",
            "cfg": "CFG",
            "denoise": "Denoise",
            "seed": "Seed",
            "batch": "Batch",
            "sampler": "Sampler",
            "scheduler": "Scheduler",
        }
        current = _gen_defaults(uid)

        if field == "size":
            values = [
                "512x512",
                "512x768",
                "768x512",
                "768x768",
                "768x1024",
                "1024x1024",
                "1024x1536",
                "1536x1024",
            ]
            rows = [
                [
                    InlineKeyboardButton(
                        text=value, callback_data=f"menu:settings:set:gen:size:{value}"
                    )
                    for value in values[i : i + 2]
                ]
                for i in range(0, len(values), 2)
            ]
        elif field == "steps":
            values = ["10", "15", "20", "25", "30", "40", "50", "60", "80", "100"]
            rows = [
                [
                    InlineKeyboardButton(
                        text=value, callback_data=f"menu:settings:set:gen:steps:{value}"
                    )
                    for value in values[i : i + 3]
                ]
                for i in range(0, len(values), 3)
            ]
        elif field == "cfg":
            values = ["3", "4", "5", "6", "7", "8", "9", "10", "12", "14"]
            rows = [
                [
                    InlineKeyboardButton(
                        text=value, callback_data=f"menu:settings:set:gen:cfg:{value}"
                    )
                    for value in values[i : i + 3]
                ]
                for i in range(0, len(values), 3)
            ]
        elif field == "denoise":
            values = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
            rows = [
                [
                    InlineKeyboardButton(
                        text=value, callback_data=f"menu:settings:set:gen:denoise:{value}"
                    )
                    for value in values[i : i + 3]
                ]
                for i in range(0, len(values), 3)
            ]
        elif field == "seed":
            values = ["-1", "0", "42", "123", "777", "1337", "2024"]
            rows = [
                [
                    InlineKeyboardButton(
                        text=("random" if value == "-1" else value),
                        callback_data=f"menu:settings:set:gen:seed:{value}",
                    )
                    for value in values[i : i + 3]
                ]
                for i in range(0, len(values), 3)
            ]
        elif field == "batch":
            values = ["1", "2", "3", "4", "6", "8", "12", "16"]
            rows = [
                [
                    InlineKeyboardButton(
                        text=value, callback_data=f"menu:settings:set:gen:batch:{value}"
                    )
                    for value in values[i : i + 4]
                ]
                for i in range(0, len(values), 4)
            ]
        elif field in {"sampler", "scheduler"}:
            items = deps.client.info.samplers if field == "sampler" else deps.client.info.schedulers
            if not items:
                items = ["euler"] if field == "sampler" else ["normal"]
            page_size = 8
            total_pages = max(1, (len(items) + page_size - 1) // page_size)
            current_page = max(0, min(page, total_pages - 1))
            start = current_page * page_size
            chunk = items[start : start + page_size]
            rows = [
                [
                    InlineKeyboardButton(
                        text=(item if len(item) <= 36 else item[:33] + "..."),
                        callback_data=f"menu:settings:set:gen:{field}:{item}",
                    )
                ]
                for item in chunk
            ]
            nav: list[InlineKeyboardButton] = []
            if current_page > 0:
                nav.append(
                    InlineKeyboardButton(
                        text="◀️",
                        callback_data=f"menu:settings:gen:menu:{field}:{current_page - 1}",
                    )
                )
            nav.append(
                InlineKeyboardButton(
                    text=f"· {current_page + 1}/{total_pages} ·", callback_data="noop"
                )
            )
            if current_page < total_pages - 1:
                nav.append(
                    InlineKeyboardButton(
                        text="▶️",
                        callback_data=f"menu:settings:gen:menu:{field}:{current_page + 1}",
                    )
                )
            rows.append(nav)
        else:
            return

        rows.append(
            [
                InlineKeyboardButton(
                    text="✏️ Ввести вручную",
                    callback_data=f"menu:settings:input:{field}",
                )
            ]
        )
        rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="menu:settings:gen")])

        current_value = current.get(field)
        current_text = (
            "random"
            if field == "seed" and isinstance(current_value, int) and current_value < 0
            else str(current_value)
        )
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            f"🎛 <b>Настройка: {label_map.get(field, field)}</b>\n"
            f"<b>Текущее значение:</b> <code>{deps.h(current_text)}</code>",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )

    async def _show_download_settings(message: Message, uid: int) -> None:
        dl = _download_defaults(uid)
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="Source: All", callback_data="menu:settings:set:dl:source:all"
                    ),
                    InlineKeyboardButton(
                        text="CivitAI", callback_data="menu:settings:set:dl:source:civitai"
                    ),
                    InlineKeyboardButton(
                        text="HF", callback_data="menu:settings:set:dl:source:huggingface"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Sort: Downloads", callback_data="menu:settings:set:dl:sort:downloads"
                    ),
                    InlineKeyboardButton(
                        text="Rating", callback_data="menu:settings:set:dl:sort:rating"
                    ),
                    InlineKeyboardButton(
                        text="Newest", callback_data="menu:settings:set:dl:sort:newest"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="Period: All", callback_data="menu:settings:set:dl:period:all"
                    ),
                    InlineKeyboardButton(
                        text="Month", callback_data="menu:settings:set:dl:period:month"
                    ),
                    InlineKeyboardButton(
                        text="Week", callback_data="menu:settings:set:dl:period:week"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"🧬 Базовые модели: {_base_label(dl['base'])}",
                        callback_data="menu:settings:dl:base",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=("🔞 NSFW: ON" if dl["nsfw"] else "🛡 NSFW: OFF"),
                        callback_data="menu:settings:set:dl:nsfw:toggle",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🔥 Популярные",
                        callback_data="menu:settings:set:dl:profile:popular",
                    ),
                    InlineKeyboardButton(
                        text="🆕 Новые",
                        callback_data="menu:settings:set:dl:profile:fresh",
                    ),
                    InlineKeyboardButton(
                        text="⭐ Рейтинг",
                        callback_data="menu:settings:set:dl:profile:quality",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="🎎 Anime",
                        callback_data="menu:settings:set:dl:profile:anime",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=f"👤 Автор: {dl['author'] or 'любой'}",
                        callback_data="menu:settings:input:dl_author",
                    )
                ],
                [InlineKeyboardButton(text="↺ Сбросить", callback_data="menu:settings:reset:dl")],
                [InlineKeyboardButton(text="⬅️ К настройкам", callback_data="menu:settings")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
            ]
        )
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "📦 <b>Настройки поиска моделей по умолчанию</b>\n\n"
            f"<b>Source:</b> <code>{deps.h(dl['source'])}</code>\n"
            f"<b>Sort:</b> <code>{deps.h(dl['sort'])}</code>\n"
            f"<b>Period:</b> <code>{deps.h(dl['period'])}</code>\n"
            f"<b>Base:</b> <code>{deps.h(_base_label(dl['base']))}</code>\n"
            f"<b>NSFW:</b> <code>{'on' if dl['nsfw'] else 'off'}</code>\n"
            f"<b>Автор(ы) CivitAI:</b> <code>{deps.h(dl['author'] or 'любой')}</code>",
            reply_markup=kb,
        )

    async def _show_download_base_settings(message: Message, uid: int) -> None:
        dl = _download_defaults(uid)
        current = dl["base"]

        def _mk(code: str, label: str) -> InlineKeyboardButton:
            prefix = "✅ " if current == code else ""
            return InlineKeyboardButton(
                text=f"{prefix}{label}",
                callback_data=f"menu:settings:set:dl:base:{code}",
            )

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [_mk("all", "Все")],
                [_mk("sd15", "SD 1.5"), _mk("sd2", "SD 2.x")],
                [_mk("sdxl09", "SDXL 0.9"), _mk("sdxl", "SDXL 1.0")],
                [_mk("sd3", "SD 3"), _mk("sd35", "SD 3.5")],
                [_mk("pony", "Pony"), _mk("flux", "Flux")],
                [_mk("illustrious", "Illustrious"), _mk("noobai", "NoobAI")],
                [
                    InlineKeyboardButton(
                        text="⬅️ К настройкам поиска", callback_data="menu:settings:dl"
                    )
                ],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
            ]
        )
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🧬 <b>Базовые модели по умолчанию</b>\n\n"
            "Выберите семейство, которое будет применяться в фильтрах скачивания.\n"
            f"<b>Текущий выбор:</b> <code>{deps.h(_base_label(current))}</code>",
            reply_markup=kb,
        )

    async def _show_models_report(message: Message, uid: int) -> None:
        status_msg = await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "⏳ Обновляю список моделей…",
            reply_markup=_models_back_keyboard(),
        )
        try:
            info = await deps.client.refresh_info()
            parts = [
                deps.models_section("Checkpoints", "🧪", info.checkpoints, 10),
                deps.models_section("LoRA", "🧲", info.loras, 10),
                deps.models_section("Embeddings", "🔤", info.embeddings, 10),
                deps.models_section("Upscale", "🔍", info.upscale_models, 10),
                deps.models_section("VAE", "🧬", info.vaes, 10),
                deps.models_section("ControlNet", "🧷", info.controlnets, 10),
                deps.models_section("CLIP Vision", "👁", info.clip_vision_models, 5),
                deps.models_section("IP-Adapter", "🖼", info.ipadapter_models, 5),
            ]
            ipa_icon = "✅" if info.ipadapter_supported else "❌"
            parts.append(
                f"\nIP-Adapter: {ipa_icon} {'доступен' if info.ipadapter_supported else 'недоступен'}"
            )
            await status_msg.edit_text(
                "✅ <b>Список моделей обновлён</b>\n\n📦 <b>Модели ComfyUI</b>\n\n"
                + "\n\n".join(parts),
                reply_markup=_models_back_keyboard(),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            await status_msg.edit_text(
                f"❌ <b>Ошибка подключения</b>\n<code>{deps.h(exc)}</code>",
                reply_markup=_models_back_keyboard(),
            )

    async def _show_queue(message: Message, uid: int) -> None:
        try:
            queue = await deps.client.get_queue_status()
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
            if running == 0 and pending == 0:
                status = "🟢 Очередь пуста"
            else:
                status = f"▶️ <b>Выполняется:</b> {running}\n⏳ <b>Ожидает:</b> {pending}"
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                f"📊 <b>Очередь ComfyUI</b>\n{status}",
                reply_markup=_service_back_keyboard(),
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                f"❌ Не удалось получить статус: <code>{deps.h(exc)}</code>",
                reply_markup=_service_back_keyboard(),
            )

    @router.message(CommandStart())
    async def cmd_start(message: Message):
        uid = deps.message_user_id(message)
        telegram_locale = message.from_user.language_code if message.from_user else None
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            _start_text_for_uid(uid, telegram_locale=telegram_locale),
            reply_markup=deps.main_menu_keyboard(locale),
            prefer_edit=False,
        )

    @router.message(Command("help"))
    async def cmd_help(msg: Message):
        await cmd_start(msg)

    @router.callback_query(F.data == "menu:root")
    async def menu_root(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            _start_text_for_uid(uid, telegram_locale=cb.from_user.language_code),
            reply_markup=deps.main_menu_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(F.data == "menu:generation")
    async def menu_generation(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            deps.localization.t(
                "common.menu.generation.title",
                locale=locale,
                default="🎨 <b>Генерация</b>\nВыберите действие:",
            ),
            reply_markup=deps.generation_menu_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(F.data == "menu:models")
    async def menu_models(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            deps.localization.t(
                "common.menu.models.title",
                locale=locale,
                default="📦 <b>Модели</b>\nВыберите действие:",
            ),
            reply_markup=deps.models_menu_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(F.data == "menu:service")
    async def menu_service(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            deps.localization.t(
                "common.menu.service.title",
                locale=locale,
                default="⚙️ <b>Сервис</b>\nВыберите действие:",
            ),
            reply_markup=deps.service_menu_keyboard(locale),
        )
        await cb.answer()

    @router.message(Command("training"))
    @router.message(Command("learn"))
    async def cmd_training(msg: Message):
        uid = deps.message_user_id(msg)
        await _show_training(msg, uid)

    async def _cancel_for_user(msg: Message, state: FSMContext, *, uid: int) -> None:
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        root_keyboard = deps.main_menu_keyboard(locale)

        state_name = await state.get_state()
        had_state = state_name is not None
        prompt_req = deps.runtime.active_prompt_requests.get(uid)
        had_prompt = prompt_req is not None

        user_gens = deps.user_generations(deps.runtime, uid)
        had_generation = bool(user_gens)
        gen_progress = ""
        if user_gens and user_gens[0].status_msg:
            status_text = (
                user_gens[0].status_msg.text or user_gens[0].status_msg.caption or ""
            ).strip()
            match = re.search(r"(\d{1,3})%", status_text)
            if match:
                pct = min(100, max(0, int(match.group(1))))
                gen_progress = f" ({pct}% готово)"

        dl_task = deps.runtime.active_downloads.get(uid)
        had_download = bool(dl_task and not dl_task.done())

        cancelled_items: list[str] = []
        if prompt_req:
            pos = prompt_req.params.positive.strip() or "—"
            cancelled_items.append(
                f"Редактор промптов (Positive: «{deps.h(deps.truncate(pos, 36))}»)"
            )
        if had_generation:
            count = len(user_gens)
            suffix = "" if count == 1 else f" ({count} задач)"
            cancelled_items.append(f"Генерация{gen_progress or ' (в процессе)'}{suffix}")
        if had_download:
            cancelled_items.append("Скачивание модели")
        if had_state and not cancelled_items:
            state_short = state_name.split(":")[-1] if isinstance(state_name, str) else ""
            cancelled_items.append(f"Текущее состояние ({deps.h(state_short) or 'активно'})")

        if had_state:
            await state.clear()
        deps.runtime.active_prompt_requests.pop(uid, None)

        if dl_task and not dl_task.done():
            dl_task.cancel()

        for gen in user_gens:
            if gen.task is not None and not gen.task.done():
                gen.task.cancel()
            if gen.prompt_id:
                asyncio.create_task(deps.client.cancel_prompt(gen.prompt_id))

        if cancelled_items:
            await deps.render_user_panel(
                msg,
                deps.runtime,
                uid,
                deps.localization.t(
                    "common.cancel.summary_title",
                    locale=locale,
                    default="❌ <b>Отменено:</b>",
                )
                + "\n"
                + "\n".join(f"• {item}" for item in cancelled_items),
                reply_markup=root_keyboard,
            )
        elif had_state or had_prompt or had_generation or had_download:
            await deps.render_user_panel(
                msg,
                deps.runtime,
                uid,
                deps.localization.t(
                    "common.cancel.done",
                    locale=locale,
                    default="❌ Операция отменена.",
                ),
                reply_markup=root_keyboard,
            )
        else:
            await deps.render_user_panel(
                msg,
                deps.runtime,
                uid,
                deps.localization.t(
                    "common.cancel.nothing",
                    locale=locale,
                    default="Нечего отменять.",
                ),
                reply_markup=root_keyboard,
            )

    @router.message(Command("cancel"))
    async def cmd_cancel(msg: Message, state: FSMContext):
        await _cancel_for_user(msg, state, uid=deps.message_user_id(msg))

    @router.callback_query(F.data == "menu:cancel")
    async def menu_cancel(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        await _cancel_for_user(message, state, uid=deps.callback_user_id(cb))
        await cb.answer()

    @router.message(Command("models"))
    async def cmd_models(message: Message):
        uid = deps.message_user_id(message)
        await _show_models_report(message, uid)

    @router.callback_query(F.data == "menu:models_refresh")
    async def menu_models_refresh(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_models_report(message, deps.callback_user_id(cb))
        await cb.answer("✅ Список обновлён")

    @router.message(Command("queue"))
    async def cmd_queue(msg: Message):
        uid = deps.message_user_id(msg)
        await _show_queue(msg, uid)

    @router.callback_query(F.data == "menu:queue")
    async def menu_queue(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_queue(message, deps.callback_user_id(cb))
        await cb.answer()

    @router.message(Command("settings"))
    async def cmd_settings(msg: Message):
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        await _show_settings(
            msg,
            deps.message_user_id(msg),
            telegram_locale=telegram_locale,
        )

    @router.callback_query(F.data == "menu:settings")
    async def menu_settings(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_settings(
            message,
            deps.callback_user_id(cb),
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(F.data == "menu:settings:locale")
    async def menu_settings_locale(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_locale_settings(
            message,
            deps.callback_user_id(cb),
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:settings:set:locale:"))
    async def menu_settings_set_locale(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return

        prefix = "menu:settings:set:locale:"
        data_value = cb.data or ""
        locale = normalize_locale_code(data_value[len(prefix) :], default="")
        available_locales = set(deps.localization.available_locales())
        if not locale or locale not in available_locales:
            await cb.answer("⚠️ Некорректный язык.", show_alert=True)
            return

        uid = deps.callback_user_id(cb)
        set_pref(deps.runtime, uid, "locale", locale)
        await _show_settings(
            message,
            uid,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer("✅ Язык обновлён")

    @router.callback_query(F.data == "menu:settings:toggle_mode")
    async def menu_settings_toggle_mode(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        current = bool(deps.runtime.user_preferences.get(uid, {}).get("pro_mode", False))
        if uid not in deps.runtime.user_preferences:
            deps.runtime.user_preferences[uid] = {}
        deps.runtime.user_preferences[uid]["pro_mode"] = not current
        await _show_settings(
            message,
            uid,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer("✅ Режим обновлён")

    @router.callback_query(F.data == "menu:settings:gen")
    async def menu_settings_generation(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        if await state.get_state() == ServiceSettingsStates.entering_generation_value.state:
            await state.clear()
        await _show_generation_settings(message, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(F.data == "menu:settings:dl")
    async def menu_settings_download(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_download_settings(message, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(F.data == "menu:settings:dl:base")
    async def menu_settings_download_base(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_download_base_settings(message, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(F.data == "menu:settings:reset:gen")
    async def menu_settings_reset_generation(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        prefs = deps.runtime.user_preferences.get(uid, {})
        for key in (
            "gen_width",
            "gen_height",
            "gen_steps",
            "gen_cfg",
            "gen_denoise",
            "gen_sampler",
            "gen_scheduler",
        ):
            prefs.pop(key, None)
        await _show_generation_settings(message, uid)
        await cb.answer("✅ Сброшено")

    @router.callback_query(F.data == "menu:settings:reset:dl")
    async def menu_settings_reset_download(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        prefs = deps.runtime.user_preferences.get(uid, {})
        for key in (
            "dl_default_source",
            "dl_default_sort",
            "dl_default_period",
            "dl_default_base",
            "dl_default_nsfw",
            "dl_default_author",
        ):
            prefs.pop(key, None)
        await _show_download_settings(message, uid)
        await cb.answer("✅ Сброшено")

    @router.callback_query(F.data.startswith("menu:settings:gen:menu:"))
    async def menu_settings_generation_menu(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        if await state.get_state() == ServiceSettingsStates.entering_generation_value.state:
            await state.clear()

        uid = deps.callback_user_id(cb)
        parts = (cb.data or "").split(":")
        if len(parts) not in {5, 6}:
            await cb.answer("⚠️ Некорректное меню.", show_alert=True)
            return
        field = parts[4]
        page = 0
        if len(parts) == 6:
            try:
                page = int(parts[5])
            except ValueError:
                page = 0
        await _show_generation_picker(message, uid, field=field, page=page)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:settings:set:gen:"))
    async def menu_settings_set_generation(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        parts = (cb.data or "").split(":", 6)
        if len(parts) < 6:
            await cb.answer("⚠️ Некорректный параметр.", show_alert=True)
            return
        key = parts[4]
        value = parts[5]
        try:
            updates = parse_generation_callback_value(key, value)
        except SettingsParseError as exc:
            await cb.answer(str(exc), show_alert=True)
            return
        except (TypeError, ValueError):
            if key == "size":
                await cb.answer("⚠️ Некорректный размер.", show_alert=True)
                return
            if key == "steps":
                await cb.answer("⚠️ Некорректное значение steps.", show_alert=True)
                return
            if key == "cfg":
                await cb.answer("⚠️ Некорректное значение cfg.", show_alert=True)
                return
            if key == "denoise":
                await cb.answer("⚠️ Некорректное значение denoise.", show_alert=True)
                return
            if key == "seed":
                await cb.answer("⚠️ Некорректный seed.", show_alert=True)
                return
            if key == "batch":
                await cb.answer("⚠️ Некорректный batch.", show_alert=True)
                return
            await cb.answer("⚠️ Неизвестный параметр.", show_alert=True)
            return

        for pref_key, pref_value in updates.items():
            set_pref(deps.runtime, uid, pref_key, pref_value)
        await _show_generation_settings(message, uid)
        await cb.answer("✅ Сохранено")

    @router.callback_query(F.data.startswith("menu:settings:set:dl:"))
    async def menu_settings_set_download(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        parts = (cb.data or "").split(":", 6)
        if len(parts) < 6:
            await cb.answer("⚠️ Некорректный параметр.", show_alert=True)
            return
        key = parts[4]
        value = parts[5]
        current_nsfw = bool(
            deps.runtime.user_preferences.get(uid, {}).get("dl_default_nsfw", False)
        )
        try:
            updates = parse_download_callback_value(
                key,
                value,
                current_nsfw=current_nsfw,
                apply_profile=lambda profile_code: _apply_download_profile(uid, profile_code),
            )
        except SettingsParseError as exc:
            await cb.answer(str(exc), show_alert=True)
            return

        for pref_key, pref_value in updates.items():
            set_pref(deps.runtime, uid, pref_key, pref_value)
        await _show_download_settings(message, uid)
        await cb.answer("✅ Сохранено")

    @router.callback_query(F.data.startswith("menu:settings:input:"))
    async def menu_settings_input_start(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return

        parts = (cb.data or "").split(":", 3)
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректный запрос.", show_alert=True)
            return
        field = parts[3]
        if field == "dl_author":
            await state.set_state(ServiceSettingsStates.entering_download_author)
            await deps.render_user_panel(
                message,
                deps.runtime,
                deps.callback_user_id(cb),
                "👤 <b>Автор по умолчанию (CivitAI)</b>\n\n"
                "Введите один или несколько ников через запятую (без @).\n"
                "Отправьте <code>-</code>, чтобы убрать фильтр.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⬅️ К настройкам поиска",
                                callback_data="menu:settings:dl",
                            )
                        ],
                        [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
                    ]
                ),
            )
            await cb.answer()
            return
        if field not in {
            "size",
            "steps",
            "cfg",
            "denoise",
            "seed",
            "batch",
            "sampler",
            "scheduler",
        }:
            await cb.answer("⚠️ Неизвестное поле.", show_alert=True)
            return

        prompt_map = {
            "size": "Введите размер в формате <code>WIDTHxHEIGHT</code> (например <code>896x1152</code>).",
            "steps": "Введите Steps (1-200).",
            "cfg": "Введите CFG (0.0-30.0).",
            "denoise": "Введите Denoise (0.0-1.0).",
            "seed": "Введите Seed (целое число, -1 = random).",
            "batch": "Введите Batch size (1-16).",
            "sampler": "Введите sampler (например <code>euler</code> или <code>dpmpp_2m</code>).",
            "scheduler": "Введите scheduler (например <code>normal</code> или <code>karras</code>).",
        }

        await state.set_state(ServiceSettingsStates.entering_generation_value)
        await state.update_data(settings_input_field=field)
        await deps.render_user_panel(
            message,
            deps.runtime,
            deps.callback_user_id(cb),
            "⚙️ <b>Ввод значения</b>\n\n"
            f"{prompt_map[field]}\n"
            "\nНапишите значение одним сообщением.\n"
            "Для отмены отправьте <code>cancel</code>.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="⬅️ К настройкам генерации", callback_data="menu:settings:gen"
                        )
                    ],
                    [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
                ]
            ),
        )
        await cb.answer()

    @router.message(ServiceSettingsStates.entering_generation_value, F.text)
    async def menu_settings_input_value(msg: Message, state: FSMContext):
        uid = deps.message_user_id(msg)
        data = await state.get_data()
        field = str(data.get("settings_input_field") or "")
        raw = (msg.text or "").strip()

        try:
            await msg.delete()
        except TelegramBadRequest:
            pass

        if raw.lower() in {"cancel", "отмена", "/cancel"}:
            await state.clear()
            await _show_generation_settings(msg, uid)
            return

        try:
            updates = parse_generation_manual_value(field, raw)
        except (SettingsParseError, TypeError, ValueError):
            await deps.render_user_panel(
                msg,
                deps.runtime,
                uid,
                "⚠️ Некорректное значение. Попробуйте ещё раз или отправьте <code>cancel</code>.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⬅️ К настройкам генерации",
                                callback_data="menu:settings:gen",
                            )
                        ],
                        [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
                    ]
                ),
            )
            return

        for pref_key, pref_value in updates.items():
            set_pref(deps.runtime, uid, pref_key, pref_value)

        await state.clear()
        await _show_generation_settings(msg, uid)

    @router.message(ServiceSettingsStates.entering_download_author, F.text)
    async def menu_settings_input_download_author(msg: Message, state: FSMContext):
        uid = deps.message_user_id(msg)
        raw = (msg.text or "").strip()
        try:
            await msg.delete()
        except TelegramBadRequest:
            pass

        if raw.lower() in {"cancel", "отмена", "/cancel"}:
            await state.clear()
            await _show_download_settings(msg, uid)
            return

        author = parse_download_author(raw)
        set_pref(deps.runtime, uid, "dl_default_author", author)
        await state.clear()
        await _show_download_settings(msg, uid)

    @router.callback_query(F.data == "menu:training")
    async def menu_training(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_training(message, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:training:page:"))
    async def menu_training_page(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректная страница.", show_alert=True)
            return
        try:
            page = int(parts[3])
        except ValueError:
            await cb.answer("⚠️ Некорректная страница.", show_alert=True)
            return
        await _show_training(message, deps.callback_user_id(cb), page=page)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:training:mode:"))
    async def menu_training_mode(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer("⚠️ Некорректный режим.", show_alert=True)
            return
        mode = parts[3]
        if mode not in {"simple", "advanced"}:
            await cb.answer("⚠️ Некорректный режим.", show_alert=True)
            return
        uid = deps.callback_user_id(cb)
        set_training_mode(deps.runtime, uid, mode)
        await _show_training(message, uid)
        await cb.answer("✅ Режим обучения обновлён")

    @router.callback_query(F.data == "noop")
    async def noop(cb: CallbackQuery):
        await cb.answer()

    @router.message(F.text, ~F.text.startswith("/"), StateFilter(None))
    async def fallback_text(msg: Message):
        uid = deps.message_user_id(msg)
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        await deps.render_user_panel(
            msg,
            deps.runtime,
            uid,
            _fallback_text_for_uid(uid, telegram_locale=telegram_locale),
            reply_markup=deps.main_menu_keyboard(locale),
        )
