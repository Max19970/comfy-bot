from __future__ import annotations

from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient
from core.formatting import human_size, short_number
from core.html_utils import h
from core.panels import render_user_panel
from core.runtime import RuntimeStore
from core.states import DownloadStates
from core.telegram import callback_user_id, message_user_id
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, noop_button
from model_downloader import (
    ModelDownloader,
    SearchResult,
    SearchVersionOption,
    apply_version_option,
)

from .download_flow_handlers import DownloadFlowDeps, register_download_flow_handlers

MODEL_TYPE_LABELS = {
    "checkpoint": "Checkpoint",
    "lora": "LoRA",
    "upscaler": "Upscaler",
    "embedding": "Embedding",
    "controlnet": "ControlNet",
    "vae": "VAE",
}

SOURCE_LABELS = {
    "civitai": "CivitAI",
    "huggingface": "HuggingFace",
    "all": "Оба источника",
}

SORT_CODE_TO_LABEL = {
    "downloads": "По скачиваниям",
    "rating": "По рейтингу",
    "newest": "Новые",
}

SORT_CODE_TO_API = {
    "downloads": "Most Downloaded",
    "rating": "Highest Rated",
    "newest": "Newest",
}

PERIOD_CODE_TO_LABEL = {
    "all": "Всё время",
    "month": "Месяц",
    "week": "Неделя",
}

PERIOD_CODE_TO_API = {
    "all": "AllTime",
    "month": "Month",
    "week": "Week",
}

FILTER_PROFILES: dict[str, dict[str, Any]] = {
    "popular": {
        "label": "🔥 Популярные",
        "source": "all",
        "sort": "downloads",
        "period": "all",
        "base": "all",
        "nsfw": False,
    },
    "fresh": {
        "label": "🆕 Новые",
        "source": "all",
        "sort": "newest",
        "period": "week",
        "base": "all",
        "nsfw": False,
    },
    "quality": {
        "label": "⭐ Рейтинг",
        "source": "all",
        "sort": "rating",
        "period": "month",
        "base": "all",
        "nsfw": False,
    },
    "anime": {
        "label": "🎎 Anime",
        "source": "civitai",
        "sort": "downloads",
        "period": "month",
        "base": "pony",
        "nsfw": False,
    },
}

BASE_CODE_TO_LABEL = {
    "all": "Все",
    "sd15": "SD 1.5",
    "sdxl09": "SDXL 0.9",
    "sdxl": "SDXL",
    "sd2": "SD 2.x",
    "sd3": "SD 3",
    "sd35": "SD 3.5",
    "pony": "Pony",
    "illustrious": "Illustrious",
    "noobai": "NoobAI",
    "flux": "Flux",
}

BASE_CODE_TO_API = {
    "all": [],
    "sd15": ["SD 1.5"],
    "sdxl09": ["SDXL 0.9"],
    "sdxl": ["SDXL 1.0"],
    "sd2": ["SD 2.0", "SD 2.1"],
    "sd3": ["SD 3"],
    "sd35": ["SD 3.5"],
    "pony": ["Pony"],
    "illustrious": ["Illustrious"],
    "noobai": ["NoobAI"],
    "flux": ["Flux.1 D"],
}


def _base_code_from_base_model(base_model: str) -> str:
    value = base_model.strip().lower()
    if not value:
        return "all"
    if "illustrious" in value:
        return "illustrious"
    if "noob" in value:
        return "noobai"
    if "pony" in value:
        return "pony"
    if "flux" in value:
        return "flux"
    if "3.5" in value:
        return "sd35"
    if "sd3" in value or "stable diffusion 3" in value:
        return "sd3"
    if "2.1" in value or "2.0" in value or "sd2" in value:
        return "sd2"
    if "0.9" in value:
        return "sdxl09"
    if "sdxl" in value or "stable diffusion xl" in value:
        return "sdxl"
    if "1.5" in value or "sd15" in value:
        return "sd15"
    return "all"


def _mark(active: bool, label: str) -> str:
    return f"✅ {label}" if active else label


def _type_title(model_type: str) -> str:
    return MODEL_TYPE_LABELS.get(model_type, model_type)


def _source_title(source: str) -> str:
    return SOURCE_LABELS.get(source, source)


def _hydrate_result(data: dict[str, Any]) -> SearchResult:
    result = SearchResult(**data)
    hydrated: list[SearchVersionOption] = []
    for item in result.available_versions:
        if isinstance(item, SearchVersionOption):
            hydrated.append(item)
        elif isinstance(item, dict):
            hydrated.append(SearchVersionOption(**item))
    result.available_versions = hydrated
    return result


def _search_filters_summary(
    *,
    model_type: str,
    sort_code: str,
    period_code: str,
    base_code: str,
    include_nsfw: bool,
    source: str,
    author_nick: str,
    page_size: int,
) -> str:
    parts = [f"Сортировка: {SORT_CODE_TO_LABEL.get(sort_code, 'По скачиваниям')}"]
    if _supports_period_filter(source):
        parts.append(f"Период: {PERIOD_CODE_TO_LABEL.get(period_code, 'Всё время')}")
    if _supports_base_filter(model_type=model_type, source=source):
        parts.append(f"Base: {BASE_CODE_TO_LABEL.get(base_code, 'Все')}")
    if _supports_nsfw_filter(source):
        parts.append(f"NSFW: {'вкл' if include_nsfw else 'выкл'}")
    if source in {"civitai", "all"} and author_nick.strip():
        parts.append(f"Автор(ы): {author_nick.strip().lstrip('@')}")
    parts.append(f"На страницу: {max(1, min(page_size, 10))}")
    return " | ".join(parts)


def _supports_period_filter(source: str) -> bool:
    return source in {"civitai", "all"}


def _supports_nsfw_filter(source: str) -> bool:
    return source in {"civitai", "all"}


def _supports_base_filter(*, model_type: str, source: str) -> bool:
    if source not in {"civitai", "all"}:
        return False
    return model_type in {"checkpoint", "lora", "controlnet"}


def _result_meta_line(result: SearchResult) -> str:
    parts: list[str] = []
    if result.creator:
        parts.append(f"by {result.creator}")
    if result.base_model:
        parts.append(result.base_model)
    if result.version_name:
        parts.append(result.version_name)
    if result.size_bytes > 0:
        parts.append(human_size(result.size_bytes))
    if result.rating > 0:
        parts.append(f"⭐ {result.rating:.2f}")
    if result.download_count > 0:
        parts.append(f"📥 {short_number(result.download_count)}")
    if not parts and result.description:
        return result.description
    return " | ".join(parts)


def _result_button_label(result: SearchResult) -> str:
    src_tag = "C" if result.source == "civitai" else "HF"
    label = result.name if len(result.name) <= 28 else result.name[:25] + "..."
    return f"[{src_tag}] {label}"


def _build_type_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [button("Checkpoint", "dltype:checkpoint")],
        [button("LoRA", "dltype:lora")],
        [button("Upscaler", "dltype:upscaler")],
        [button("Embedding", "dltype:embedding")],
        [button("ControlNet", "dltype:controlnet")],
        [button("VAE", "dltype:vae")],
        [cancel_button("dltype:cancel")],
    ]
    return build_keyboard(rows)


def _build_source_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("CivitAI", "dlsrc:civitai")],
            [button("HuggingFace", "dlsrc:huggingface")],
            [button("Оба источника", "dlsrc:all")],
            [back_button("dlsrc:back", text="⬅️ Назад к типу")],
        ]
    )


def _build_filter_keyboard(
    *,
    model_type: str,
    source: str,
    sort_code: str,
    period_code: str,
    base_code: str,
    include_nsfw: bool,
    author_nick: str,
    page_size: int,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            button(_mark(sort_code == "downloads", "📥 Скачивания"), "dlflt:sort:downloads"),
            button(_mark(sort_code == "rating", "⭐ Рейтинг"), "dlflt:sort:rating"),
            button(_mark(sort_code == "newest", "🆕 Новые"), "dlflt:sort:newest"),
        ]
    ]

    if _supports_period_filter(source):
        rows.append(
            [
                button(_mark(period_code == "all", "🕒 Всё время"), "dlflt:period:all"),
                button(_mark(period_code == "month", "📆 Месяц"), "dlflt:period:month"),
                button(_mark(period_code == "week", "🗓 Неделя"), "dlflt:period:week"),
            ]
        )

    show_base_filters = _supports_base_filter(model_type=model_type, source=source)
    if show_base_filters:
        rows.append(
            [
                button(
                    f"🧬 Базовые модели: {BASE_CODE_TO_LABEL.get(base_code, 'Все')}",
                    "dlflt:base_menu",
                )
            ]
        )

    if _supports_nsfw_filter(source):
        rows.append(
            [button(("🔞 NSFW: вкл" if include_nsfw else "🛡 NSFW: выкл"), "dlflt:nsfw:toggle")]
        )

    if source in {"civitai", "all"}:
        author_label = author_nick.strip().lstrip("@") or "любой"
        rows.append([button(f"👤 Автор: {author_label}", "dlflt:author")])

    rows.append(
        [
            button("🔥 Популярные", "dlflt:profile:popular"),
            button("🆕 Новые", "dlflt:profile:fresh"),
            button("⭐ Рейтинг", "dlflt:profile:quality"),
        ]
    )
    rows.append([button("🎎 Anime", "dlflt:profile:anime")])
    rows.append(
        [
            button(_mark(page_size == 5, "5/стр"), "dlflt:pagesize:5"),
            button(_mark(page_size == 8, "8/стр"), "dlflt:pagesize:8"),
            button(_mark(page_size == 10, "10/стр"), "dlflt:pagesize:10"),
        ]
    )

    rows.extend(
        [
            [button("➡️ Ввести запрос", "dlflt:go")],
            [back_button("dlflt:back", text="⬅️ Источники")],
        ]
    )
    return build_keyboard(rows)


def _build_base_filter_keyboard(*, base_code: str) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [button(_mark(base_code == "all", "Все"), "dlflt:base:all")],
        [
            button(_mark(base_code == "sd15", "SD 1.5"), "dlflt:base:sd15"),
            button(_mark(base_code == "sd2", "SD 2.x"), "dlflt:base:sd2"),
        ],
        [
            button(_mark(base_code == "sdxl09", "SDXL 0.9"), "dlflt:base:sdxl09"),
            button(_mark(base_code == "sdxl", "SDXL 1.0"), "dlflt:base:sdxl"),
        ],
        [
            button(_mark(base_code == "sd3", "SD 3"), "dlflt:base:sd3"),
            button(_mark(base_code == "sd35", "SD 3.5"), "dlflt:base:sd35"),
        ],
        [
            button(_mark(base_code == "pony", "Pony"), "dlflt:base:pony"),
            button(_mark(base_code == "flux", "Flux"), "dlflt:base:flux"),
        ],
        [
            button(_mark(base_code == "illustrious", "Illustrious"), "dlflt:base:illustrious"),
            button(_mark(base_code == "noobai", "NoobAI"), "dlflt:base:noobai"),
        ],
        [back_button("dlbase:back", text="⬅️ К фильтрам")],
    ]
    return build_keyboard(rows)


def _confirmation_text(result: SearchResult) -> str:
    size_info = human_size(result.size_bytes)
    extras = [item for item in (result.file_fp, result.file_format) if item]
    if extras:
        size_info = f"{size_info} ({', '.join(extras)})"

    lines = [
        "📦 <b>Скачать модель?</b>",
        "",
        f"<b>Название:</b> {h(result.name)}",
    ]
    if result.version_name:
        lines.append(f"<b>Версия:</b> {h(result.version_name)}")
    if result.creator:
        lines.append(f"<b>Автор:</b> {h(result.creator)}")
    if result.base_model:
        lines.append(f"<b>Базовая модель:</b> {h(result.base_model)}")

    if result.rating > 0 or result.download_count > 0:
        stat_parts = []
        if result.rating > 0:
            stat_parts.append(f"⭐ {result.rating:.2f}")
        if result.download_count > 0:
            stat_parts.append(f"📥 {short_number(result.download_count)}")
        lines.append(f"<b>Популярность:</b> {' | '.join(stat_parts)}")

    lines.extend(
        [
            f"<b>Файл:</b> <code>{h(result.filename)}</code>",
            f"<b>Размер:</b> {h(size_info)}",
            f"<b>Источник:</b> {h(_source_title(result.source))}",
            f"<b>Тип:</b> {h(_type_title(result.model_type))}",
        ]
    )

    if result.trained_words:
        words = ", ".join(result.trained_words[:8])
        lines.append(f"<b>Триггер-слова:</b> <code>{h(words)}</code>")

    if result.tags:
        lines.append(f"<b>Теги:</b> {h(', '.join(result.tags[:8]))}")

    return "\n".join(lines)


def register_download_handlers(
    router: Router,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
) -> None:
    async def _render_download_panel(
        message: Message,
        state: FSMContext,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        prefer_edit: bool,
    ) -> Message:
        data = await state.get_data()
        uid = data.get("dl_uid")
        if isinstance(uid, int):
            return await render_user_panel(
                message,
                runtime,
                uid,
                text,
                reply_markup=reply_markup,
                prefer_edit=prefer_edit,
            )

        if prefer_edit:
            try:
                edited = await message.edit_text(text, reply_markup=reply_markup)
                if isinstance(edited, Message):
                    return edited
            except TelegramBadRequest:
                pass

        return await message.answer(text, reply_markup=reply_markup)

    async def show_filter_menu(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        data = await state.get_data()
        model_type = data.get("dl_type", "checkpoint")
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()
        page_size = int(data.get("dl_page_size", 8) or 8)

        text = (
            "🧩 <b>Фильтры поиска</b>\n"
            f"<b>Тип:</b> {h(_type_title(model_type))}\n"
            f"<b>Источник:</b> {h(_source_title(source))}\n"
            f"<i>{h(_search_filters_summary(model_type=model_type, sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source, author_nick=author_nick, page_size=page_size))}</i>"
        )
        kb = _build_filter_keyboard(
            model_type=model_type,
            source=source,
            sort_code=sort_code,
            period_code=period_code,
            base_code=base_code,
            include_nsfw=include_nsfw,
            author_nick=author_nick,
            page_size=page_size,
        )
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_query_prompt(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        kb = build_keyboard([[back_button("dlqry:back", text="⬅️ Назад к фильтрам")]])
        text = "✏️ Введите поисковый запрос или прямую ссылку на модель CivitAI/HuggingFace:"
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_author_prompt(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        data = await state.get_data()
        current = str(data.get("dl_author", "")).strip().lstrip("@")
        kb = build_keyboard([[back_button("dlauth:back", text="⬅️ Назад к фильтрам")]])
        text = (
            "👤 <b>Фильтр по автору CivitAI</b>\n"
            "Введите один или несколько ников через запятую (без @).\n"
            "Отправьте <code>-</code>, чтобы убрать фильтр.\n"
            f"<b>Текущее значение:</b> <code>{h(current or 'любой')}</code>"
        )
        await _render_download_panel(message, state, text, kb, prefer_edit=edit)

    async def show_base_filter_menu(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        data = await state.get_data()
        base_code = str(data.get("dl_base", "all"))
        kb = _build_base_filter_keyboard(base_code=base_code)
        text = (
            "🧬 <b>Базовые модели</b>\n"
            "Выберите семейство моделей для фильтрации поиска.\n"
            f"<b>Текущий выбор:</b> {h(BASE_CODE_TO_LABEL.get(base_code, 'Все'))}"
        )
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_results_menu(
        message: Message,
        state: FSMContext,
        results: list[SearchResult],
        *,
        edit: bool,
        page: int = 0,
        page_size: int = 8,
        can_continue: bool = False,
    ) -> None:
        data = await state.get_data()
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()

        total = len(results)
        page_size = max(1, min(page_size, 10))
        total_pages = max(1, (total + page_size - 1) // page_size)
        current_page = max(0, min(page, total_pages - 1))
        start = current_page * page_size
        page_items = results[start : start + page_size]

        rows: list[list[InlineKeyboardButton]] = []
        for offset, result in enumerate(page_items):
            absolute_index = start + offset
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_result_button_label(result),
                        callback_data=f"dlpick:{absolute_index}",
                    )
                ]
            )

        nav: list[InlineKeyboardButton] = []
        if current_page > 0:
            nav.append(button("⏮ В начало", "dlpick:first"))
        if current_page > 0:
            nav.append(button("◀️ Назад", "dlpick:prev"))
        nav.append(noop_button(f"· {current_page + 1}/{total_pages} ·"))
        if current_page < total_pages - 1:
            nav.append(button("▶️ Далее", "dlpick:next"))
        if current_page < total_pages - 1:
            nav.append(button("⏭ В конец", "dlpick:last"))
        if nav:
            rows.append(nav)

        if can_continue:
            rows.append([button("🔎 Продолжить поиск", "dlpick:more")])
        rows.append([back_button("dlpick:new", text="⬅️ Новый поиск")])
        rows.append([cancel_button("dlpick:cancel")])

        lines = [
            "📦 <b>Результаты поиска</b>",
            f"<i>{h(_search_filters_summary(model_type=data.get('dl_type', 'checkpoint'), sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source, author_nick=author_nick, page_size=page_size))}</i>",
            "",
        ]
        max_text_len = 3800
        for offset, result in enumerate(page_items):
            index = start + offset
            meta = _result_meta_line(result)
            name_value = result.name if len(result.name) <= 140 else result.name[:137] + "..."
            meta_value = meta if len(meta) <= 220 else meta[:217] + "..."
            item_lines = [
                f"{index + 1}. <b>{h(name_value)}</b>",
                f"   <i>{h(meta_value)}</i>",
            ]
            if result.model_type == "lora" and result.trained_words:
                words = ", ".join(result.trained_words[:4])
                words_value = words if len(words) <= 180 else words[:177] + "..."
                item_lines.append(f"   🔑 <code>{h(words_value)}</code>")

            projected = "\n".join(lines + item_lines + ["", "Выберите модель:"])
            if len(projected) > max_text_len:
                lines.append("… список сокращён, уточните запрос для более точного результата.")
                break
            lines.extend(item_lines)
        lines.append("\nВыберите модель:")

        text = "\n".join(lines)
        kb = build_keyboard(rows)
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_download_confirmation(
        message: Message,
        state: FSMContext,
        result: SearchResult,
    ) -> None:
        rows: list[list[InlineKeyboardButton]] = []
        if result.preview_url:
            rows.append([InlineKeyboardButton(text="👀 Превью", url=result.preview_url)])
        if result.model_url:
            rows.append([InlineKeyboardButton(text="🔗 Страница модели", url=result.model_url)])
        rows.append(
            [
                button("⬇️ Скачать", "dlconfirm:yes"),
                cancel_button("dlconfirm:no"),
            ]
        )
        rows.append([back_button("dlconfirm:back", text="⬅️ К результатам")])
        kb = build_keyboard(rows)
        await _render_download_panel(
            message,
            state,
            _confirmation_text(result),
            kb,
            prefer_edit=True,
        )

    async def _start_download(
        message: Message,
        state: FSMContext,
        uid: int,
    ) -> None:
        await state.update_data(dl_uid=uid)
        await state.set_state(DownloadStates.choosing_type)
        await _render_download_panel(
            message,
            state,
            "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
            _build_type_keyboard(),
            prefer_edit=True,
        )

    register_download_flow_handlers(
        DownloadFlowDeps(
            router=router,
            runtime=runtime,
            downloader=downloader,
            client=client,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            start_download=_start_download,
            render_download_panel=_render_download_panel,
            build_type_keyboard=_build_type_keyboard,
            build_source_keyboard=_build_source_keyboard,
            type_title=_type_title,
            show_filter_menu=show_filter_menu,
            show_author_prompt=show_author_prompt,
            show_base_filter_menu=show_base_filter_menu,
            show_query_prompt=show_query_prompt,
            supports_base_filter=_supports_base_filter,
            supports_nsfw_filter=_supports_nsfw_filter,
            base_code_to_api=BASE_CODE_TO_API,
            sort_code_to_api=SORT_CODE_TO_API,
            period_code_to_api=PERIOD_CODE_TO_API,
            show_results_menu=show_results_menu,
            hydrate_result=_hydrate_result,
            human_size=human_size,
            short_number=short_number,
            apply_version_option=apply_version_option,
            show_download_confirmation=show_download_confirmation,
            filter_profiles=FILTER_PROFILES,
        )
    )
