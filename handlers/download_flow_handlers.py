from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from application.download_search_use_case import DownloadSearchUseCase
from core.callbacks import ValueSelectionCallback
from core.html_utils import h
from core.interaction import require_callback_message
from core.states import DownloadStates
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, menu_root_button

from .download_flow_state import (
    clamp_page,
    next_search_limit,
    read_results_view_state,
    serialize_results_payload,
)
from .download_flow_utils import (
    apply_download_profile,
    download_defaults_for_user,
)
from .download_flow_version_view import build_version_selection_view


@dataclass
class DownloadFlowDeps:
    router: Router
    runtime: Any
    downloader: Any
    client: Any
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    start_download: Callable[[Message, FSMContext, int], Awaitable[None]]
    render_download_panel: Callable[..., Awaitable[Message]]
    build_type_keyboard: Callable[[str | None], InlineKeyboardMarkup]
    build_source_keyboard: Callable[[str | None], InlineKeyboardMarkup]
    type_title: Callable[[str, str | None], str]
    show_filter_menu: Callable[..., Awaitable[None]]
    show_author_prompt: Callable[..., Awaitable[None]]
    show_base_filter_menu: Callable[..., Awaitable[None]]
    show_query_prompt: Callable[..., Awaitable[None]]
    supports_base_filter: Callable[..., bool]
    supports_nsfw_filter: Callable[[str], bool]
    base_code_to_api: dict[str, list[str]]
    sort_code_to_api: dict[str, str]
    period_code_to_api: dict[str, str]
    show_results_menu: Callable[..., Awaitable[None]]
    hydrate_result: Callable[[dict[str, Any]], Any]
    human_size: Callable[[int], str]
    short_number: Callable[[int], str]
    apply_version_option: Callable[[Any, Any], Any]
    show_download_confirmation: Callable[..., Awaitable[None]]
    filter_profiles: dict[str, dict[str, Any]]
    translate: Callable[[str, str | None, str, Mapping[str, object] | None], str]
    resolve_locale: Callable[..., str]


def register_download_flow_handlers(deps: DownloadFlowDeps) -> None:
    router = deps.router
    default_page_size = 8
    search_use_case = DownloadSearchUseCase(
        base_code_to_api=deps.base_code_to_api,
        sort_code_to_api=deps.sort_code_to_api,
        period_code_to_api=deps.period_code_to_api,
        supports_base_filter=deps.supports_base_filter,
        supports_nsfw_filter=deps.supports_nsfw_filter,
    )

    def _resolved_locale(uid: int, *, telegram_locale: str | None) -> str:
        return deps.resolve_locale(uid, telegram_locale=telegram_locale)

    def _t(
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: Mapping[str, object] | None = None,
    ) -> str:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        return deps.translate(key, locale, default, params)

    def _t_cb(
        cb: CallbackQuery,
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        uid = deps.callback_user_id(cb)
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        return _t(uid, key, default, telegram_locale=telegram_locale, params=params)

    def _t_msg(
        msg: Message,
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        uid = deps.message_user_id(msg)
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        return _t(uid, key, default, telegram_locale=telegram_locale, params=params)

    def _models_back_keyboard(*, uid: int, telegram_locale: str | None) -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [
                    back_button(
                        "menu:models",
                        text=_t(
                            uid,
                            "common.menu.back_to_models",
                            "⬅️ Модели",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
                [
                    menu_root_button(
                        text=_t(
                            uid,
                            "common.menu.root",
                            "🏠 В меню",
                            telegram_locale=telegram_locale,
                        )
                    )
                ],
            ]
        )

    def _download_done_keyboard(*, uid: int, telegram_locale: str | None) -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [
                    back_button(
                        "dldone:results",
                        text=_t(
                            uid,
                            "download.done.back_to_results",
                            "⬅️ К результатам",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
                [
                    button(
                        _t(
                            uid,
                            "download.done.new_search",
                            "🔎 Новый поиск",
                            telegram_locale=telegram_locale,
                        ),
                        "dldone:new",
                    )
                ],
                [
                    back_button(
                        "menu:models",
                        text=_t(
                            uid,
                            "common.menu.back_to_models",
                            "⬅️ Модели",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
                [
                    menu_root_button(
                        text=_t(
                            uid,
                            "common.menu.root",
                            "🏠 В меню",
                            telegram_locale=telegram_locale,
                        )
                    )
                ],
            ]
        )

    async def _run_search(state: FSMContext, *, limit: int) -> list[Any]:
        data = await state.get_data()
        criteria = search_use_case.build_search_criteria(
            data,
            requested_limit=limit,
            default_page_size=default_page_size,
        )

        return await deps.downloader.search(
            criteria.query,
            criteria.model_type,
            criteria.source,
            limit=criteria.limit,
            sort=criteria.sort,
            base_models=list(criteria.base_models),
            include_nsfw=criteria.include_nsfw,
            period=criteria.period,
            civitai_authors=list(criteria.civitai_authors),
        )

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

    async def _callback_value(
        cb: CallbackQuery,
        *,
        prefix: str,
        invalid_text: str,
    ) -> str | None:
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
        if parsed is None:
            await cb.answer(invalid_text, show_alert=True)
            return None
        return parsed.value

    async def _show_results_from_state(
        message: Message,
        state: FSMContext,
        *,
        notice: str = "",
    ) -> bool:
        data = await state.get_data()
        view = read_results_view_state(
            data,
            default_page_size=default_page_size,
            hydrate_result=deps.hydrate_result,
        )
        if not view.results:
            return False

        page = clamp_page(view.page, total_items=len(view.results), page_size=view.page_size)
        await state.update_data(dl_results_page=page)
        await state.set_state(DownloadStates.choosing_result)
        await deps.show_results_menu(
            message,
            state,
            view.results,
            edit=True,
            page=page,
            page_size=view.page_size,
            can_continue=view.can_continue,
            notice=notice,
        )
        return True

    @router.message(Command("download"))
    async def cmd_download(msg: Message, state: FSMContext):
        await deps.start_download(msg, state, deps.message_user_id(msg))

    @router.callback_query(F.data == "menu:download")
    async def menu_download(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        await deps.start_download(message, state, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_type, F.data.startswith("dltype:"))
    async def dl_type_chosen(cb: CallbackQuery, state: FSMContext):
        model_type = await _callback_value(
            cb,
            prefix="dltype",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.invalid_type",
                "❌ Неверный выбор типа.",
            ),
        )
        if model_type is None:
            return
        if model_type == "cancel":
            await state.clear()
            message = await _callback_message(cb)
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            if message is not None:
                await message.edit_text(
                    _t(
                        uid,
                        "download.flow.cancelled",
                        "❌ Скачивание отменено.",
                        telegram_locale=telegram_locale,
                    ),
                    reply_markup=_models_back_keyboard(uid=uid, telegram_locale=telegram_locale),
                )
            await cb.answer()
            return

        inferred_base = "all"
        uid = deps.callback_user_id(cb)
        req = deps.runtime.active_prompt_requests.get(uid)
        if req and req.params.checkpoint:
            inferred_base = search_use_case.infer_base_code_for_checkpoint(
                req.params.checkpoint,
                get_model_metadata=deps.downloader.get_model_metadata,
                infer_base_model=deps.downloader.infer_base_model,
                default="all",
            )

        defaults = download_defaults_for_user(
            deps.runtime.user_preferences,
            uid,
            inferred_base=inferred_base,
        )
        await state.update_data(dl_type=model_type, **defaults)
        await state.set_state(DownloadStates.choosing_source)
        message = await _callback_message(cb)
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        if message is not None:
            await message.edit_text(
                _t(
                    uid,
                    "download.flow.choose_source",
                    "🔍 <b>Где искать {type_title}?</b>",
                    telegram_locale=telegram_locale,
                    params={"type_title": h(deps.type_title(model_type, locale))},
                ),
                reply_markup=deps.build_source_keyboard(locale),
            )
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_source, F.data.startswith("dlsrc:"))
    async def dl_src_chosen(cb: CallbackQuery, state: FSMContext):
        source = await _callback_value(
            cb,
            prefix="dlsrc",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.invalid_source",
                "❌ Неверный источник.",
            ),
        )
        if source is None:
            return
        if source == "back":
            await state.set_state(DownloadStates.choosing_type)
            message = await _callback_message(cb)
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            if message is not None:
                await message.edit_text(
                    _t(
                        uid,
                        "download.start.choose_type",
                        "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
                        telegram_locale=telegram_locale,
                    ),
                    reply_markup=deps.build_type_keyboard(locale),
                )
            await cb.answer()
            return

        await state.update_data(dl_source=source)
        await state.set_state(DownloadStates.choosing_filters)
        message = await _callback_message(cb)
        if message is not None:
            await deps.show_filter_menu(message, state, edit=True)
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_filters, F.data.startswith("dlflt:"))
    async def dl_filter_chosen(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        payload = (cb.data or "").split(":")
        action = payload[1] if len(payload) > 1 else ""

        if action == "go":
            await state.set_state(DownloadStates.entering_query)
            await deps.show_query_prompt(message, state, edit=True)
            await cb.answer()
            return

        if action == "author":
            await state.set_state(DownloadStates.entering_author)
            await deps.show_author_prompt(message, state, edit=True)
            await cb.answer()
            return

        if action == "profile" and len(payload) >= 3:
            data = await state.get_data()
            updated = apply_download_profile(data, deps.filter_profiles.get(payload[2]))
            await state.update_data(**updated)
            await deps.show_filter_menu(message, state, edit=True)
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.filter.profile_applied",
                    "✅ Профиль применён",
                )
            )
            return

        if action == "pagesize" and len(payload) >= 3:
            try:
                page_size = int(payload[2])
            except ValueError:
                await cb.answer(
                    _t_cb(
                        cb,
                        "download.flow.error.invalid_page_size",
                        "Некорректный размер страницы.",
                    ),
                    show_alert=True,
                )
                return
            if page_size not in {5, 8, 10}:
                await cb.answer(
                    _t_cb(
                        cb,
                        "download.flow.error.invalid_page_size",
                        "Некорректный размер страницы.",
                    ),
                    show_alert=True,
                )
                return
            await state.update_data(dl_page_size=page_size)
            await deps.show_filter_menu(message, state, edit=True)
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.filter.page_size_updated",
                    "✅ Размер страницы обновлён",
                )
            )
            return

        if action == "base_menu":
            await deps.show_base_filter_menu(message, state, edit=True)
            await cb.answer()
            return

        if action == "back":
            data = await state.get_data()
            model_type = data.get("dl_type", "checkpoint")
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            await state.set_state(DownloadStates.choosing_source)
            await message.edit_text(
                _t(
                    uid,
                    "download.flow.choose_source",
                    "🔍 <b>Где искать {type_title}?</b>",
                    telegram_locale=telegram_locale,
                    params={"type_title": h(deps.type_title(model_type, locale))},
                ),
                reply_markup=deps.build_source_keyboard(locale),
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

        await deps.show_filter_menu(message, state, edit=True)
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_filters, F.data == "dlbase:back")
    async def dl_base_back(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        await deps.show_filter_menu(message, state, edit=True)
        await cb.answer()

    @router.callback_query(DownloadStates.entering_author, F.data == "dlauth:back")
    async def dl_author_back(cb: CallbackQuery, state: FSMContext):
        await state.set_state(DownloadStates.choosing_filters)
        message = await _callback_message(cb)
        if message is None:
            return
        await deps.show_filter_menu(message, state, edit=True)
        await cb.answer()

    @router.message(DownloadStates.entering_author, F.text)
    async def dl_author_input(msg: Message, state: FSMContext):
        raw = (msg.text or "").strip()
        try:
            await msg.delete()
        except TelegramBadRequest:
            pass

        if raw.lower() in {"cancel", "отмена", "/cancel"}:
            await state.set_state(DownloadStates.choosing_filters)
            await deps.show_filter_menu(msg, state, edit=False)
            return

        author = "" if raw in {"-", "*"} else ",".join(search_use_case.parse_author_filters(raw))
        await state.update_data(dl_author=author)
        await state.set_state(DownloadStates.choosing_filters)
        await deps.show_filter_menu(msg, state, edit=False)

    @router.callback_query(DownloadStates.entering_query, F.data == "dlqry:back")
    async def dl_query_back(cb: CallbackQuery, state: FSMContext):
        await state.set_state(DownloadStates.choosing_filters)
        message = await _callback_message(cb)
        if message is None:
            return
        await deps.show_filter_menu(message, state, edit=True)
        await cb.answer()

    @router.message(DownloadStates.entering_query, F.text)
    async def dl_query(msg: Message, state: FSMContext):
        query = (msg.text or "").strip()
        uid = deps.message_user_id(msg)
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        if not query:
            await msg.answer(
                _t(
                    uid,
                    "download.flow.query.enter_query",
                    "Введите запрос:",
                    telegram_locale=telegram_locale,
                )
            )
            return

        try:
            await msg.delete()
        except TelegramBadRequest:
            pass

        await state.update_data(dl_query=query)

        status = await deps.render_download_panel(
            msg,
            state,
            _t(
                uid,
                "download.flow.query.searching",
                "🔍 Ищу <code>{query}</code>…",
                telegram_locale=telegram_locale,
                params={"query": h(query)},
            ),
            None,
            prefer_edit=False,
        )
        try:
            data = await state.get_data()
            page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)
            initial_limit = page_size * 2
            results = await _run_search(state, limit=initial_limit)
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
            await status.edit_text(
                _t(
                    uid,
                    "download.flow.query.search_error",
                    "❌ <b>Ошибка поиска:</b> <code>{error}</code>",
                    telegram_locale=telegram_locale,
                    params={"error": h(exc)},
                ),
                reply_markup=build_keyboard(
                    [
                        [
                            back_button(
                                "dlqry:back",
                                text=_t(
                                    uid,
                                    "download.query.back_to_filters",
                                    "⬅️ Назад к фильтрам",
                                    telegram_locale=telegram_locale,
                                ),
                            )
                        ]
                    ]
                ),
            )
            return

        if not results:
            await status.edit_text(
                _t(
                    uid,
                    "download.flow.query.not_found",
                    "❌ Ничего не найдено. Попробуйте другой запрос или измените фильтры.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=build_keyboard(
                    [
                        [
                            back_button(
                                "dlqry:back",
                                text=_t(
                                    uid,
                                    "download.query.back_to_filters",
                                    "⬅️ Назад к фильтрам",
                                    telegram_locale=telegram_locale,
                                ),
                            )
                        ]
                    ]
                ),
            )
            return

        await state.update_data(
            dl_results=serialize_results_payload(results),
            dl_results_page=0,
            dl_loaded_limit=initial_limit,
            dl_more_exhausted=False,
        )
        await state.set_state(DownloadStates.choosing_result)
        can_continue = len(results) >= initial_limit
        await deps.show_results_menu(
            status,
            state,
            results,
            edit=True,
            page=0,
            page_size=page_size,
            can_continue=can_continue,
        )

    @router.callback_query(DownloadStates.choosing_result, F.data.startswith("dlpick:"))
    async def dl_pick(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        value = await _callback_value(
            cb,
            prefix="dlpick",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.invalid_pick",
                "❌ Неверный выбор.",
            ),
        )
        if value is None:
            return
        if value == "cancel":
            await state.clear()
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            await message.edit_text(
                _t(
                    uid,
                    "download.flow.cancelled",
                    "❌ Скачивание отменено.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=_models_back_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            await cb.answer()
            return

        if value == "new":
            await state.set_state(DownloadStates.entering_query)
            await deps.show_query_prompt(message, state, edit=True)
            await cb.answer()
            return

        if value in {"first", "prev", "next", "last", "more"}:
            data = await state.get_data()
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            view = read_results_view_state(
                data,
                default_page_size=default_page_size,
                hydrate_result=deps.hydrate_result,
            )

            results = view.results
            page_size = view.page_size
            page = view.page
            loaded_limit = view.loaded_limit
            exhausted = view.exhausted
            notice = ""
            callback_answered = False

            if value == "first":
                page = 0
            elif value == "prev":
                page = max(0, page - 1)
            elif value == "next":
                page += 1
            elif value == "last":
                page = max(0, (len(results) - 1) // page_size)
            else:
                await cb.answer()
                callback_answered = True
                prev_count = len(results)
                next_limit = next_search_limit(loaded_limit=loaded_limit, page_size=page_size)
                if next_limit == loaded_limit:
                    exhausted = True
                    notice = _t(
                        uid,
                        "download.flow.results.no_more",
                        "Больше результатов не найдено.",
                        telegram_locale=telegram_locale,
                    )
                    await state.update_data(dl_more_exhausted=True)
                    max_page = max(0, (len(results) - 1) // page_size)
                    page = max(0, min(page, max_page))
                    await deps.show_results_menu(
                        message,
                        state,
                        results,
                        edit=True,
                        page=page,
                        page_size=page_size,
                        can_continue=False,
                        notice=notice,
                    )
                    return
                try:
                    results = await _run_search(state, limit=next_limit)
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, ValueError) as exc:
                    await message.answer(
                        _t(
                            uid,
                            "download.flow.results.search_error",
                            "⚠️ Ошибка поиска: {error}",
                            telegram_locale=telegram_locale,
                            params={"error": h(exc)},
                        )
                    )
                    return
                loaded_limit = next_limit
                exhausted = len(results) <= prev_count
                await state.update_data(
                    dl_results=serialize_results_payload(results),
                    dl_loaded_limit=loaded_limit,
                    dl_more_exhausted=exhausted,
                )
                max_page = max(0, (len(results) - 1) // page_size)
                page = min(page, max_page)
                if exhausted:
                    notice = _t(
                        uid,
                        "download.flow.results.no_more",
                        "Больше результатов не найдено.",
                        telegram_locale=telegram_locale,
                    )
                else:
                    notice = _t(
                        uid,
                        "download.flow.results.loaded_more",
                        "Подгрузил ещё результаты.",
                        telegram_locale=telegram_locale,
                    )

            max_page = max(0, (len(results) - 1) // page_size)
            page = max(0, min(page, max_page))
            await state.update_data(dl_results_page=page)

            can_continue = (not exhausted) and len(results) >= loaded_limit and loaded_limit < 200
            await deps.show_results_menu(
                message,
                state,
                results,
                edit=True,
                page=page,
                page_size=page_size,
                can_continue=can_continue,
                notice=notice,
            )
            if not callback_answered:
                await cb.answer()
            return

        try:
            index = int(value)
        except ValueError:
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.error.invalid_pick",
                    "❌ Неверный выбор.",
                ),
                show_alert=True,
            )
            return

        data = await state.get_data()
        results_data = data.get("dl_results", [])
        if index < 0 or index >= len(results_data):
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.error.invalid_pick",
                    "❌ Неверный выбор.",
                ),
                show_alert=True,
            )
            return

        result = deps.hydrate_result(results_data[index])

        if result.source == "civitai" and len(result.available_versions) > 1:
            await state.update_data(dl_chosen_base=asdict(result))
            await state.set_state(DownloadStates.choosing_version)

            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            text, kb = build_version_selection_view(
                result,
                human_size=deps.human_size,
                short_number=deps.short_number,
                escape_html=h,
                translate=deps.translate,
                locale=locale,
            )
            await message.edit_text(text, reply_markup=kb)
            await cb.answer()
            return

        await state.update_data(dl_chosen=asdict(result))
        await state.set_state(DownloadStates.confirming_download)
        await deps.show_download_confirmation(message, state, result)
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_version, F.data.startswith("dlver:"))
    async def dl_version_pick(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        value = await _callback_value(
            cb,
            prefix="dlver",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.invalid_version",
                "❌ Неверный выбор версии.",
            ),
        )
        if value is None:
            return
        if value == "cancel":
            await state.clear()
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            await message.edit_text(
                _t(
                    uid,
                    "download.flow.cancelled",
                    "❌ Скачивание отменено.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=_models_back_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            await cb.answer()
            return

        if value == "back":
            shown = await _show_results_from_state(message, state)
            if not shown:
                await state.clear()
                await message.edit_text(
                    _t_cb(
                        cb,
                        "download.flow.error.session_expired",
                        "❌ Сессия истекла. Запустите /download заново.",
                    )
                )
                await cb.answer()
                return
            await cb.answer()
            return

        try:
            index = int(value)
        except ValueError:
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.error.invalid_version",
                    "❌ Неверный выбор версии.",
                ),
                show_alert=True,
            )
            return

        data = await state.get_data()
        chosen_data = data.get("dl_chosen_base")
        if not chosen_data:
            await state.clear()
            await message.edit_text(
                _t_cb(
                    cb,
                    "download.flow.error.session_expired",
                    "❌ Сессия истекла. Запустите /download заново.",
                )
            )
            await cb.answer()
            return

        result = deps.hydrate_result(chosen_data)
        if index < 0 or index >= len(result.available_versions):
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.error.invalid_version",
                    "❌ Неверный выбор версии.",
                ),
                show_alert=True,
            )
            return

        selected = result.available_versions[index]
        result = deps.apply_version_option(result, selected)

        await state.update_data(dl_chosen=asdict(result), dl_chosen_base=None)
        await state.set_state(DownloadStates.confirming_download)
        await deps.show_download_confirmation(message, state, result)
        await cb.answer()

    @router.callback_query(
        DownloadStates.confirming_download,
        F.data.startswith("dlconfirm:"),
    )
    async def dl_confirm(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        decision = await _callback_value(
            cb,
            prefix="dlconfirm",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.invalid_action",
                "❌ Неверное действие.",
            ),
        )
        if decision is None:
            return
        if decision == "back":
            shown = await _show_results_from_state(message, state)
            if not shown:
                await state.clear()
                await message.edit_text(
                    _t_cb(
                        cb,
                        "download.flow.error.session_expired",
                        "❌ Сессия истекла. Запустите /download заново.",
                    )
                )
                await cb.answer()
                return
            await cb.answer()
            return

        if decision == "no":
            await state.clear()
            uid = deps.callback_user_id(cb)
            telegram_locale = cb.from_user.language_code if cb.from_user else None
            await message.edit_text(
                _t(
                    uid,
                    "download.flow.cancelled",
                    "❌ Скачивание отменено.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=_models_back_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            await cb.answer()
            return

        if decision != "yes":
            await cb.answer(
                _t_cb(
                    cb,
                    "download.flow.error.invalid_action",
                    "❌ Неверное действие.",
                ),
                show_alert=True,
            )
            return

        data = await state.get_data()
        chosen_data = data.get("dl_chosen")
        if not chosen_data:
            await state.clear()
            await message.edit_text(
                _t_cb(
                    cb,
                    "download.flow.error.model_not_found",
                    "❌ Ошибка: выбранная модель не найдена.",
                )
            )
            await cb.answer()
            return

        uid = deps.callback_user_id(cb)
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        existing = deps.runtime.active_downloads.get(uid)
        if existing and not existing.done():
            await cb.answer(
                _t(
                    uid,
                    "download.flow.download.already_running",
                    "⏳ Уже идёт скачивание. Дождитесь завершения.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        result = deps.hydrate_result(chosen_data)
        download_cancel_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "download.flow.download.cancel_button",
                            "❌ Отменить скачивание",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data="dldl:cancel",
                    )
                ]
            ]
        )
        edited = await message.edit_text(
            _t(
                uid,
                "download.flow.download.started",
                "⬇️ <b>Скачивание:</b> {name}…",
                telegram_locale=telegram_locale,
                params={"name": h(result.name)},
            ),
            reply_markup=download_cancel_kb,
        )
        status_msg: Message = edited if isinstance(edited, Message) else message
        await cb.answer()

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
            except TelegramBadRequest:
                pass

        async def _do_download():
            try:
                path = await deps.downloader.download_model(result, progress_cb=_progress)

                details = [
                    _t(
                        uid,
                        "download.flow.download.done_title",
                        "✅ <b>Скачано!</b>",
                        telegram_locale=telegram_locale,
                    ),
                    "",
                    _t(
                        uid,
                        "download.flow.download.field.file",
                        "<b>Файл:</b> <code>{value}</code>",
                        telegram_locale=telegram_locale,
                        params={"value": h(result.filename)},
                    ),
                    _t(
                        uid,
                        "download.flow.download.field.path",
                        "<b>Путь:</b> <code>{value}</code>",
                        telegram_locale=telegram_locale,
                        params={"value": h(path)},
                    ),
                ]
                if result.base_model:
                    details.append(
                        _t(
                            uid,
                            "download.flow.download.field.base",
                            "<b>Base:</b> {value}",
                            telegram_locale=telegram_locale,
                            params={"value": h(result.base_model)},
                        )
                    )
                if result.model_type == "lora" and result.trained_words:
                    details.append(
                        _t(
                            uid,
                            "download.flow.download.field.trigger_words",
                            "<b>Триггер-слова:</b> <code>{value}</code>",
                            telegram_locale=telegram_locale,
                            params={"value": h(", ".join(result.trained_words[:8]))},
                        )
                    )
                details.append("")
                details.append(
                    _t(
                        uid,
                        "common.models.updating",
                        "⏳ Обновляю список моделей…",
                        telegram_locale=telegram_locale,
                    )
                )
                try:
                    await deps.client.refresh_info()
                    details.append(
                        _t(
                            uid,
                            "download.flow.download.models_updated",
                            "✅ Список моделей обновлён.",
                            telegram_locale=telegram_locale,
                        )
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                    details.append(
                        _t(
                            uid,
                            "download.flow.download.models_update_failed",
                            "⚠️ Модель скачана, но не удалось обновить список. Попробуйте /models.",
                            telegram_locale=telegram_locale,
                        )
                    )

                await status_msg.edit_text(
                    "\n".join(details),
                    reply_markup=_download_done_keyboard(
                        uid=uid,
                        telegram_locale=telegram_locale,
                    ),
                )
                await state.set_state(DownloadStates.choosing_result)
            except asyncio.CancelledError:
                await status_msg.edit_text(
                    _t(
                        uid,
                        "download.flow.cancelled",
                        "❌ Скачивание отменено.",
                        telegram_locale=telegram_locale,
                    ),
                    reply_markup=_models_back_keyboard(
                        uid=uid,
                        telegram_locale=telegram_locale,
                    ),
                )
            except FileExistsError:
                await status_msg.edit_text(
                    _t(
                        uid,
                        "download.flow.download.file_exists",
                        "⚠️ Файл <code>{filename}</code> уже существует.",
                        telegram_locale=telegram_locale,
                        params={"filename": h(result.filename)},
                    ),
                    reply_markup=_models_back_keyboard(
                        uid=uid,
                        telegram_locale=telegram_locale,
                    ),
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                await status_msg.edit_text(
                    _t(
                        uid,
                        "download.flow.download.failed",
                        "❌ <b>Ошибка скачивания:</b> <code>{error}</code>",
                        telegram_locale=telegram_locale,
                        params={"error": h(exc)},
                    ),
                    reply_markup=_models_back_keyboard(
                        uid=uid,
                        telegram_locale=telegram_locale,
                    ),
                )
            finally:
                deps.runtime.active_downloads.pop(uid, None)

        task = asyncio.create_task(_do_download())
        deps.runtime.active_downloads[uid] = task

    @router.callback_query(F.data.startswith("dldone:"))
    async def dl_done_actions(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        action = await _callback_value(
            cb,
            prefix="dldone",
            invalid_text=_t_cb(
                cb,
                "download.flow.error.unknown_done_action",
                "❌ Неизвестное действие.",
            ),
        )
        if action is None:
            return

        if action == "new":
            await state.set_state(DownloadStates.entering_query)
            await deps.show_query_prompt(message, state, edit=True)
            await cb.answer()
            return

        if action == "results":
            shown = await _show_results_from_state(message, state)
            if not shown:
                await cb.answer(
                    _t_cb(
                        cb,
                        "download.flow.done.results_unavailable",
                        "История результатов недоступна. Начните новый поиск.",
                    ),
                    show_alert=True,
                )
                return
            await cb.answer()
            return

        await cb.answer(
            _t_cb(
                cb,
                "download.flow.error.unknown_done_action",
                "❌ Неизвестное действие.",
            ),
            show_alert=True,
        )

    @router.callback_query(F.data == "dldl:cancel")
    async def dl_cancel_running(cb: CallbackQuery):
        uid = deps.callback_user_id(cb)
        task = deps.runtime.active_downloads.get(uid)
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        if task and not task.done():
            task.cancel()
            await cb.answer(
                _t(
                    uid,
                    "download.flow.download.cancelling",
                    "❌ Отменяю скачивание…",
                    telegram_locale=telegram_locale,
                )
            )
            return
        await cb.answer(
            _t(
                uid,
                "download.flow.download.nothing_to_cancel",
                "Нечего отменять.",
                telegram_locale=telegram_locale,
            ),
            show_alert=True,
        )
