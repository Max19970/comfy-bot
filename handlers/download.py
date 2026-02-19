from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from aiogram import F, Bot, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient
from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import DownloadStates
from core.telegram import callback_user_id, message_user_id
from model_downloader import (
    ModelDownloader,
    SearchResult,
    SearchVersionOption,
    _human_size,
    apply_version_option,
)


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

BASE_CODE_TO_LABEL = {
    "all": "Все",
    "sd15": "SD 1.5",
    "sdxl": "SDXL",
    "pony": "Pony",
    "flux": "Flux",
}

BASE_CODE_TO_API = {
    "all": [],
    "sd15": ["SD 1.5"],
    "sdxl": ["SDXL 1.0"],
    "pony": ["Pony"],
    "flux": ["Flux.1 D"],
}


def _base_code_from_base_model(base_model: str) -> str:
    value = base_model.strip().lower()
    if not value:
        return "all"
    if "pony" in value:
        return "pony"
    if "flux" in value:
        return "flux"
    if "sdxl" in value or "stable diffusion xl" in value:
        return "sdxl"
    if "1.5" in value or "sd15" in value:
        return "sd15"
    return "all"


def _short_number(value: int) -> str:
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}K"
    return f"{value / 1_000_000:.1f}M"


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
    sort_code: str,
    period_code: str,
    base_code: str,
    include_nsfw: bool,
    source: str,
) -> str:
    base_part = BASE_CODE_TO_LABEL.get(base_code, "Все")
    if source == "huggingface":
        base_part = "—"
    return (
        f"Сортировка: {SORT_CODE_TO_LABEL.get(sort_code, 'По скачиваниям')} | "
        f"Период: {PERIOD_CODE_TO_LABEL.get(period_code, 'Всё время')} | "
        f"Base: {base_part} | "
        f"NSFW: {'вкл' if include_nsfw else 'выкл'}"
    )


def _result_meta_line(result: SearchResult) -> str:
    parts: list[str] = []
    if result.creator:
        parts.append(f"by {result.creator}")
    if result.base_model:
        parts.append(result.base_model)
    if result.version_name:
        parts.append(result.version_name)
    if result.size_bytes > 0:
        parts.append(_human_size(result.size_bytes))
    if result.rating > 0:
        parts.append(f"⭐ {result.rating:.2f}")
    if result.download_count > 0:
        parts.append(f"📥 {_short_number(result.download_count)}")
    if not parts and result.description:
        return result.description
    return " | ".join(parts)


def _result_button_label(result: SearchResult) -> str:
    src_tag = "C" if result.source == "civitai" else "HF"
    label = result.name if len(result.name) <= 28 else result.name[:25] + "..."
    return f"[{src_tag}] {label}"


def _build_type_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="Checkpoint", callback_data="dltype:checkpoint")],
        [InlineKeyboardButton(text="LoRA", callback_data="dltype:lora")],
        [InlineKeyboardButton(text="Upscaler", callback_data="dltype:upscaler")],
        [InlineKeyboardButton(text="Embedding", callback_data="dltype:embedding")],
        [InlineKeyboardButton(text="ControlNet", callback_data="dltype:controlnet")],
        [InlineKeyboardButton(text="VAE", callback_data="dltype:vae")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="dltype:cancel")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_source_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="CivitAI", callback_data="dlsrc:civitai")],
            [
                InlineKeyboardButton(
                    text="HuggingFace",
                    callback_data="dlsrc:huggingface",
                )
            ],
            [
                InlineKeyboardButton(
                    text="Оба источника",
                    callback_data="dlsrc:all",
                )
            ],
            [InlineKeyboardButton(text="⬅️ Назад к типу", callback_data="dlsrc:back")],
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
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=_mark(sort_code == "downloads", "📥 Скачивания"),
                callback_data="dlflt:sort:downloads",
            ),
            InlineKeyboardButton(
                text=_mark(sort_code == "rating", "⭐ Рейтинг"),
                callback_data="dlflt:sort:rating",
            ),
            InlineKeyboardButton(
                text=_mark(sort_code == "newest", "🆕 Новые"),
                callback_data="dlflt:sort:newest",
            ),
        ]
    ]

    if source in ("civitai", "all"):
        rows.append(
            [
                InlineKeyboardButton(
                    text=_mark(period_code == "all", "🕒 Всё время"),
                    callback_data="dlflt:period:all",
                ),
                InlineKeyboardButton(
                    text=_mark(period_code == "month", "📆 Месяц"),
                    callback_data="dlflt:period:month",
                ),
                InlineKeyboardButton(
                    text=_mark(period_code == "week", "🗓 Неделя"),
                    callback_data="dlflt:period:week",
                ),
            ]
        )

    show_base_filters = source in ("civitai", "all") and model_type in {
        "checkpoint",
        "lora",
        "controlnet",
        "vae",
    }
    if show_base_filters:
        rows.extend(
            [
                [
                    InlineKeyboardButton(
                        text=_mark(base_code == "all", "Base: Все"),
                        callback_data="dlflt:base:all",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_mark(base_code == "sd15", "SD 1.5"),
                        callback_data="dlflt:base:sd15",
                    ),
                    InlineKeyboardButton(
                        text=_mark(base_code == "sdxl", "SDXL"),
                        callback_data="dlflt:base:sdxl",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_mark(base_code == "pony", "Pony"),
                        callback_data="dlflt:base:pony",
                    ),
                    InlineKeyboardButton(
                        text=_mark(base_code == "flux", "Flux"),
                        callback_data="dlflt:base:flux",
                    ),
                ],
            ]
        )

    if source in ("civitai", "all"):
        rows.append(
            [
                InlineKeyboardButton(
                    text=("🔞 NSFW: вкл" if include_nsfw else "🛡 NSFW: выкл"),
                    callback_data="dlflt:nsfw:toggle",
                )
            ]
        )

    rows.extend(
        [
            [
                InlineKeyboardButton(
                    text="➡️ Ввести запрос",
                    callback_data="dlflt:go",
                )
            ],
            [InlineKeyboardButton(text="⬅️ Источники", callback_data="dlflt:back")],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _confirmation_text(result: SearchResult) -> str:
    size_info = _human_size(result.size_bytes)
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
            stat_parts.append(f"📥 {_short_number(result.download_count)}")
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
    bot: Bot,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
) -> None:
    async def _remember_download_panel(state: FSMContext, panel: Message) -> None:
        if panel.chat is None:
            return
        await state.update_data(
            dl_panel_chat_id=panel.chat.id,
            dl_panel_message_id=panel.message_id,
        )
        data = await state.get_data()
        uid = data.get("dl_uid")
        if isinstance(uid, int):
            runtime.user_ui_panels[uid] = {
                "chat_id": panel.chat.id,
                "message_id": panel.message_id,
            }

    async def _edit_download_panel_by_anchor(
        state: FSMContext,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message | None:
        data = await state.get_data()
        chat_id = data.get("dl_panel_chat_id")
        message_id = data.get("dl_panel_message_id")
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            uid = data.get("dl_uid")
            if isinstance(uid, int):
                anchor = runtime.user_ui_panels.get(uid, {})
                chat_id = anchor.get("chat_id")
                message_id = anchor.get("message_id")
        if not isinstance(chat_id, int) or not isinstance(message_id, int):
            return None

        try:
            edited = await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup,
            )
            if isinstance(edited, Message):
                await _remember_download_panel(state, edited)
                return edited
        except Exception:
            return None
        return None

    async def _render_download_panel(
        message: Message,
        state: FSMContext,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        prefer_edit: bool,
    ) -> Message:
        anchored = await _edit_download_panel_by_anchor(state, text, reply_markup)
        if anchored is not None:
            return anchored

        if prefer_edit:
            try:
                edited = await message.edit_text(text, reply_markup=reply_markup)
                if isinstance(edited, Message):
                    await _remember_download_panel(state, edited)
                    return edited
            except Exception:
                pass

        sent = await message.answer(text, reply_markup=reply_markup)
        await _remember_download_panel(state, sent)
        return sent

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

        text = (
            "🧩 <b>Фильтры поиска</b>\n"
            f"<b>Тип:</b> {h(_type_title(model_type))}\n"
            f"<b>Источник:</b> {h(_source_title(source))}\n"
            f"<i>{h(_search_filters_summary(sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source))}</i>"
        )
        kb = _build_filter_keyboard(
            model_type=model_type,
            source=source,
            sort_code=sort_code,
            period_code=period_code,
            base_code=base_code,
            include_nsfw=include_nsfw,
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
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад к фильтрам",
                        callback_data="dlqry:back",
                    )
                ]
            ]
        )
        text = (
            "✏️ Введите поисковый запрос или прямую ссылку "
            "на модель CivitAI/HuggingFace:"
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
    ) -> None:
        data = await state.get_data()
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))

        rows: list[list[InlineKeyboardButton]] = []
        for index, result in enumerate(results[:10]):
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_result_button_label(result),
                        callback_data=f"dlpick:{index}",
                    )
                ]
            )
        rows.append(
            [InlineKeyboardButton(text="⬅️ Новый поиск", callback_data="dlpick:new")]
        )
        rows.append(
            [InlineKeyboardButton(text="❌ Отмена", callback_data="dlpick:cancel")]
        )

        lines = [
            "📦 <b>Результаты поиска</b>",
            f"<i>{h(_search_filters_summary(sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source))}</i>",
            "",
        ]
        for index, result in enumerate(results[:10]):
            meta = _result_meta_line(result)
            lines.append(f"{index + 1}. <b>{h(result.name)}</b>")
            lines.append(f"   <i>{h(meta)}</i>")
            if result.model_type == "lora" and result.trained_words:
                lines.append(
                    f"   🔑 <code>{h(', '.join(result.trained_words[:4]))}</code>"
                )
        lines.append("\nВыберите модель:")

        text = "\n".join(lines)
        kb = InlineKeyboardMarkup(inline_keyboard=rows)
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
            rows.append(
                [InlineKeyboardButton(text="👀 Превью", url=result.preview_url)]
            )
        if result.model_url:
            rows.append(
                [InlineKeyboardButton(text="🔗 Страница модели", url=result.model_url)]
            )
        rows.append(
            [
                InlineKeyboardButton(text="⬇️ Скачать", callback_data="dlconfirm:yes"),
                InlineKeyboardButton(text="❌ Отмена", callback_data="dlconfirm:no"),
            ]
        )
        kb = InlineKeyboardMarkup(inline_keyboard=rows)
        await _render_download_panel(
            message,
            state,
            _confirmation_text(result),
            kb,
            prefer_edit=True,
        )

    @router.message(Command("download"))
    async def cmd_download(msg: Message, state: FSMContext):
        await state.update_data(dl_uid=message_user_id(msg))
        await state.set_state(DownloadStates.choosing_type)
        await _render_download_panel(
            msg,
            state,
            "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
            _build_type_keyboard(),
            prefer_edit=True,
        )

    @router.callback_query(DownloadStates.choosing_type, F.data.startswith("dltype:"))
    async def dl_type_chosen(cb: CallbackQuery, state: FSMContext):
        model_type = cb.data.split(":", 1)[1]
        if model_type == "cancel":
            await state.clear()
            await cb.message.edit_text("❌ Скачивание отменено.")
            await cb.answer()
            return

        inferred_base = "all"
        uid = callback_user_id(cb)
        req = runtime.active_prompt_requests.get(uid)
        if req and req.params.checkpoint:
            meta = downloader.get_model_metadata(
                req.params.checkpoint,
                model_type="checkpoint",
            )
            base_model = str(meta.get("base_model") if meta else "").strip()
            if not base_model:
                base_model = downloader.infer_base_model(req.params.checkpoint)
            inferred_base = _base_code_from_base_model(base_model)

        await state.update_data(
            dl_type=model_type,
            dl_source="all",
            dl_sort="downloads",
            dl_period="all",
            dl_base=inferred_base,
            dl_nsfw=False,
        )
        await state.set_state(DownloadStates.choosing_source)
        await cb.message.edit_text(
            f"🔍 <b>Где искать {h(_type_title(model_type))}?</b>",
            reply_markup=_build_source_keyboard(),
        )
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_source, F.data.startswith("dlsrc:"))
    async def dl_src_chosen(cb: CallbackQuery, state: FSMContext):
        source = cb.data.split(":", 1)[1]
        if source == "back":
            await state.set_state(DownloadStates.choosing_type)
            await cb.message.edit_text(
                "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
                reply_markup=_build_type_keyboard(),
            )
            await cb.answer()
            return

        await state.update_data(dl_source=source)
        await state.set_state(DownloadStates.choosing_filters)
        await show_filter_menu(cb.message, state, edit=True)
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_filters, F.data.startswith("dlflt:"))
    async def dl_filter_chosen(cb: CallbackQuery, state: FSMContext):
        payload = cb.data.split(":")
        action = payload[1] if len(payload) > 1 else ""

        if action == "go":
            await state.set_state(DownloadStates.entering_query)
            await show_query_prompt(cb.message, state, edit=True)
            await cb.answer()
            return

        if action == "back":
            data = await state.get_data()
            model_type = data.get("dl_type", "checkpoint")
            await state.set_state(DownloadStates.choosing_source)
            await cb.message.edit_text(
                f"🔍 <b>Где искать {h(_type_title(model_type))}?</b>",
                reply_markup=_build_source_keyboard(),
            )
            await cb.answer()
            return

        if action == "sort" and len(payload) >= 3:
            await state.update_data(dl_sort=payload[2])
        elif action == "period" and len(payload) >= 3:
            await state.update_data(dl_period=payload[2])
        elif action == "base" and len(payload) >= 3:
            await state.update_data(dl_base=payload[2])
        elif action == "nsfw":
            data = await state.get_data()
            await state.update_data(dl_nsfw=not bool(data.get("dl_nsfw", False)))

        await show_filter_menu(cb.message, state, edit=True)
        await cb.answer()

    @router.callback_query(DownloadStates.entering_query, F.data == "dlqry:back")
    async def dl_query_back(cb: CallbackQuery, state: FSMContext):
        await state.set_state(DownloadStates.choosing_filters)
        await show_filter_menu(cb.message, state, edit=True)
        await cb.answer()

    @router.message(DownloadStates.entering_query, F.text)
    async def dl_query(msg: Message, state: FSMContext):
        query = (msg.text or "").strip()
        if not query:
            await msg.answer("Введите запрос:")
            return

        try:
            await msg.delete()
        except Exception:
            pass

        data = await state.get_data()
        model_type = data.get("dl_type", "checkpoint")
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))

        status = await _render_download_panel(
            msg,
            state,
            f"🔍 Ищу <code>{h(query)}</code>…",
            None,
            prefer_edit=False,
        )
        try:
            results = await downloader.search(
                query,
                model_type,
                source,
                limit=8,
                sort=SORT_CODE_TO_API.get(sort_code, "Most Downloaded"),
                base_models=BASE_CODE_TO_API.get(base_code, []),
                include_nsfw=include_nsfw,
                period=PERIOD_CODE_TO_API.get(period_code, "AllTime"),
            )
        except Exception as exc:
            await status.edit_text(
                f"❌ <b>Ошибка поиска:</b> <code>{h(exc)}</code>",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⬅️ Назад к фильтрам",
                                callback_data="dlqry:back",
                            )
                        ]
                    ]
                ),
            )
            return

        if not results:
            await status.edit_text(
                "❌ Ничего не найдено. Попробуйте другой запрос или измените фильтры.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="⬅️ Назад к фильтрам",
                                callback_data="dlqry:back",
                            )
                        ]
                    ]
                ),
            )
            return

        await state.update_data(dl_results=[asdict(item) for item in results])
        await state.set_state(DownloadStates.choosing_result)
        await show_results_menu(status, state, results, edit=True)

    @router.callback_query(DownloadStates.choosing_result, F.data.startswith("dlpick:"))
    async def dl_pick(cb: CallbackQuery, state: FSMContext):
        value = cb.data.split(":", 1)[1]
        if value == "cancel":
            await state.clear()
            await cb.message.edit_text("❌ Скачивание отменено.")
            await cb.answer()
            return

        if value == "new":
            await state.set_state(DownloadStates.entering_query)
            await show_query_prompt(cb.message, state, edit=True)
            await cb.answer()
            return

        try:
            index = int(value)
        except ValueError:
            await cb.answer("❌ Неверный выбор.", show_alert=True)
            return

        data = await state.get_data()
        results_data = data.get("dl_results", [])
        if index < 0 or index >= len(results_data):
            await cb.answer("❌ Неверный выбор.", show_alert=True)
            return

        result = _hydrate_result(results_data[index])

        if result.source == "civitai" and len(result.available_versions) > 1:
            await state.update_data(dl_chosen_base=asdict(result))
            await state.set_state(DownloadStates.choosing_version)

            rows: list[list[InlineKeyboardButton]] = []
            versions = result.available_versions[:12]
            for idx, option in enumerate(versions):
                label_parts = []
                if option.base_model:
                    label_parts.append(option.base_model)
                if option.version_name:
                    label_parts.append(option.version_name)
                if option.size_bytes > 0:
                    label_parts.append(_human_size(option.size_bytes))
                short = " | ".join(label_parts) or f"Версия {idx + 1}"
                short = short if len(short) <= 58 else short[:55] + "..."
                rows.append(
                    [
                        InlineKeyboardButton(
                            text=f"{idx + 1}. {short}",
                            callback_data=f"dlver:{idx}",
                        )
                    ]
                )
            rows.append(
                [InlineKeyboardButton(text="❌ Отмена", callback_data="dlver:cancel")]
            )
            rows.append(
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад к результатам",
                        callback_data="dlver:back",
                    )
                ]
            )

            lines = [
                f"🧬 <b>Выберите версию</b> для <b>{h(result.name)}</b>",
                f"Найдено версий: <b>{len(result.available_versions)}</b>",
                "",
            ]
            for idx, option in enumerate(versions):
                meta = []
                if option.base_model:
                    meta.append(option.base_model)
                if option.size_bytes > 0:
                    meta.append(_human_size(option.size_bytes))
                if option.download_count > 0:
                    meta.append(f"📥 {_short_number(option.download_count)}")
                if option.version_name:
                    lines.append(f"{idx + 1}. <b>{h(option.version_name)}</b>")
                else:
                    lines.append(f"{idx + 1}. <b>Версия {idx + 1}</b>")
                if meta:
                    lines.append(f"   <i>{h(' | '.join(meta))}</i>")

            await cb.message.edit_text(
                "\n".join(lines),
                reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
            )
            await cb.answer()
            return

        await state.update_data(dl_chosen=asdict(result))
        await state.set_state(DownloadStates.confirming_download)
        await show_download_confirmation(cb.message, state, result)
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_version, F.data.startswith("dlver:"))
    async def dl_version_pick(cb: CallbackQuery, state: FSMContext):
        value = cb.data.split(":", 1)[1]
        if value == "cancel":
            await state.clear()
            await cb.message.edit_text("❌ Скачивание отменено.")
            await cb.answer()
            return

        if value == "back":
            data = await state.get_data()
            results_data = data.get("dl_results", [])
            results: list[SearchResult] = []
            for item in results_data:
                if isinstance(item, dict):
                    results.append(_hydrate_result(item))

            if not results:
                await state.clear()
                await cb.message.edit_text(
                    "❌ Сессия истекла. Запустите /download заново."
                )
                await cb.answer()
                return

            await state.set_state(DownloadStates.choosing_result)
            await show_results_menu(cb.message, state, results, edit=True)
            await cb.answer()
            return

        try:
            index = int(value)
        except ValueError:
            await cb.answer("❌ Неверный выбор версии.", show_alert=True)
            return

        data = await state.get_data()
        chosen_data = data.get("dl_chosen_base")
        if not chosen_data:
            await state.clear()
            await cb.message.edit_text("❌ Сессия истекла. Запустите /download заново.")
            await cb.answer()
            return

        result = _hydrate_result(chosen_data)
        if index < 0 or index >= len(result.available_versions):
            await cb.answer("❌ Неверный выбор версии.", show_alert=True)
            return

        selected = result.available_versions[index]
        result = apply_version_option(result, selected)

        await state.update_data(dl_chosen=asdict(result), dl_chosen_base=None)
        await state.set_state(DownloadStates.confirming_download)
        await show_download_confirmation(cb.message, state, result)
        await cb.answer()

    @router.callback_query(
        DownloadStates.confirming_download,
        F.data.startswith("dlconfirm:"),
    )
    async def dl_confirm(cb: CallbackQuery, state: FSMContext):
        decision = cb.data.split(":", 1)[1]
        if decision == "no":
            await state.clear()
            await cb.message.edit_text("❌ Скачивание отменено.")
            await cb.answer()
            return

        data = await state.get_data()
        chosen_data = data.get("dl_chosen")
        if not chosen_data:
            await state.clear()
            await cb.message.edit_text("❌ Ошибка: выбранная модель не найдена.")
            await cb.answer()
            return

        uid = callback_user_id(cb)
        existing = runtime.active_downloads.get(uid)
        if existing and not existing.done():
            await cb.answer(
                "⏳ Уже идёт скачивание. Дождитесь завершения.",
                show_alert=True,
            )
            return

        result = _hydrate_result(chosen_data)
        download_cancel_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="❌ Отменить скачивание",
                        callback_data="dldl:cancel",
                    )
                ]
            ]
        )
        status_msg = await cb.message.edit_text(
            f"⬇️ <b>Скачивание:</b> {h(result.name)}…",
            reply_markup=download_cancel_kb,
        )
        await cb.answer()
        await state.clear()

        import time as _time

        last_edit_time = 0.0
        last_text = ""
        edit_interval = 2.0

        async def _progress(downloaded: int, total: int, text: str):
            nonlocal last_edit_time, last_text
            now = _time.monotonic()
            if text == last_text:
                return

            is_final = total > 0 and downloaded >= total
            if not is_final and now - last_edit_time < edit_interval:
                return

            last_text = text
            last_edit_time = now
            try:
                await status_msg.edit_text(text, reply_markup=download_cancel_kb)
            except Exception:
                pass

        async def _do_download():
            try:
                path = await downloader.download_model(result, progress_cb=_progress)

                details = [
                    "✅ <b>Скачано!</b>",
                    "",
                    f"<b>Файл:</b> <code>{h(result.filename)}</code>",
                    f"<b>Путь:</b> <code>{h(path)}</code>",
                ]
                if result.base_model:
                    details.append(f"<b>Base:</b> {h(result.base_model)}")
                if result.model_type == "lora" and result.trained_words:
                    details.append(
                        f"<b>Триггер-слова:</b> <code>{h(', '.join(result.trained_words[:8]))}</code>"
                    )
                details.append("")
                details.append("⏳ Обновляю список моделей…")
                try:
                    await client.refresh_info()
                    details.append("✅ Список моделей обновлён.")
                except Exception:
                    details.append(
                        "⚠️ Модель скачана, но не удалось обновить список. Попробуйте /models."
                    )

                await status_msg.edit_text("\n".join(details))
            except asyncio.CancelledError:
                await status_msg.edit_text("❌ Скачивание отменено.")
            except FileExistsError:
                await status_msg.edit_text(
                    f"⚠️ Файл <code>{h(result.filename)}</code> уже существует."
                )
            except Exception as exc:
                await status_msg.edit_text(
                    f"❌ <b>Ошибка скачивания:</b> <code>{h(exc)}</code>"
                )
            finally:
                runtime.active_downloads.pop(uid, None)

        task = asyncio.create_task(_do_download())
        runtime.active_downloads[uid] = task

    @router.callback_query(F.data == "dldl:cancel")
    async def dl_cancel_running(cb: CallbackQuery):
        uid = callback_user_id(cb)
        task = runtime.active_downloads.get(uid)
        if task and not task.done():
            task.cancel()
            await cb.answer("❌ Отменяю скачивание…")
            return
        await cb.answer("Нечего отменять.", show_alert=True)
