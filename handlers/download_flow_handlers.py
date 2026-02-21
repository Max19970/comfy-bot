from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from core.callbacks import ValueSelectionCallback
from core.download_filters import base_code_from_base_model
from core.html_utils import h
from core.interaction import require_callback_message
from core.states import DownloadStates
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, menu_root_button
from core.user_preferences import read_download_defaults


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
    build_type_keyboard: Callable[[], InlineKeyboardMarkup]
    build_source_keyboard: Callable[[], InlineKeyboardMarkup]
    type_title: Callable[[str], str]
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


def register_download_flow_handlers(deps: DownloadFlowDeps) -> None:
    router = deps.router
    default_page_size = 8

    def _models_back_keyboard() -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [back_button("menu:models", text="⬅️ Модели")],
                [menu_root_button()],
            ]
        )

    def _download_done_keyboard() -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [back_button("dldone:results", text="⬅️ К результатам")],
                [button("🔎 Новый поиск", "dldone:new")],
                [back_button("menu:models", text="⬅️ Модели")],
                [menu_root_button()],
            ]
        )

    def _download_defaults_for_user(uid: int, *, inferred_base: str) -> dict[str, Any]:
        prefs = deps.runtime.user_preferences.get(uid, {})
        defaults = read_download_defaults(prefs, inferred_base=inferred_base)
        return {
            "dl_source": defaults["source"],
            "dl_sort": defaults["sort"],
            "dl_period": defaults["period"],
            "dl_base": defaults["base"],
            "dl_nsfw": defaults["nsfw"],
            "dl_author": defaults["author"],
        }

    def _apply_profile(data: dict[str, Any], profile_code: str) -> dict[str, Any]:
        profile = deps.filter_profiles.get(profile_code)
        if not profile:
            return data
        updated = dict(data)
        updated["dl_source"] = profile.get("source", updated.get("dl_source", "all"))
        updated["dl_sort"] = profile.get("sort", updated.get("dl_sort", "downloads"))
        updated["dl_period"] = profile.get("period", updated.get("dl_period", "all"))
        updated["dl_base"] = profile.get("base", updated.get("dl_base", "all"))
        updated["dl_nsfw"] = bool(profile.get("nsfw", updated.get("dl_nsfw", False)))
        return updated

    def _parse_author_filters(raw: str) -> list[str]:
        items = [item.strip().lstrip("@") for item in raw.split(",")]
        cleaned = [item for item in items if item]
        unique: list[str] = []
        seen: set[str] = set()
        for item in cleaned:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:6]

    async def _run_search(state: FSMContext, *, limit: int) -> list[Any]:
        data = await state.get_data()
        query = str(data.get("dl_query", "")).strip()
        model_type = data.get("dl_type", "checkpoint")
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()
        author_filters = _parse_author_filters(author_nick)
        page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)

        base_models = deps.base_code_to_api.get(base_code, [])
        if not deps.supports_base_filter(model_type=model_type, source=source):
            base_models = []

        return await deps.downloader.search(
            query,
            model_type,
            source,
            limit=max(page_size, limit),
            sort=deps.sort_code_to_api.get(sort_code, "Most Downloaded"),
            base_models=base_models,
            include_nsfw=include_nsfw and deps.supports_nsfw_filter(source),
            period=deps.period_code_to_api.get(period_code, "AllTime"),
            civitai_authors=(author_filters if source in {"civitai", "all"} else []),
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
            cb, prefix="dltype", invalid_text="❌ Неверный выбор типа."
        )
        if model_type is None:
            return
        if model_type == "cancel":
            await state.clear()
            message = await _callback_message(cb)
            if message is not None:
                await message.edit_text(
                    "❌ Скачивание отменено.", reply_markup=_models_back_keyboard()
                )
            await cb.answer()
            return

        inferred_base = "all"
        uid = deps.callback_user_id(cb)
        req = deps.runtime.active_prompt_requests.get(uid)
        if req and req.params.checkpoint:
            meta = deps.downloader.get_model_metadata(
                req.params.checkpoint,
                model_type="checkpoint",
            )
            base_model = str(meta.get("base_model") if meta else "").strip()
            if not base_model:
                base_model = deps.downloader.infer_base_model(req.params.checkpoint)
            inferred_base = base_code_from_base_model(base_model)

        defaults = _download_defaults_for_user(uid, inferred_base=inferred_base)
        await state.update_data(dl_type=model_type, **defaults)
        await state.set_state(DownloadStates.choosing_source)
        message = await _callback_message(cb)
        if message is not None:
            await message.edit_text(
                f"🔍 <b>Где искать {h(deps.type_title(model_type))}?</b>",
                reply_markup=deps.build_source_keyboard(),
            )
        await cb.answer()

    @router.callback_query(DownloadStates.choosing_source, F.data.startswith("dlsrc:"))
    async def dl_src_chosen(cb: CallbackQuery, state: FSMContext):
        source = await _callback_value(cb, prefix="dlsrc", invalid_text="❌ Неверный источник.")
        if source is None:
            return
        if source == "back":
            await state.set_state(DownloadStates.choosing_type)
            message = await _callback_message(cb)
            if message is not None:
                await message.edit_text(
                    "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
                    reply_markup=deps.build_type_keyboard(),
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
            updated = _apply_profile(data, payload[2])
            await state.update_data(**updated)
            await deps.show_filter_menu(message, state, edit=True)
            await cb.answer("✅ Профиль применён")
            return

        if action == "pagesize" and len(payload) >= 3:
            try:
                page_size = int(payload[2])
            except ValueError:
                await cb.answer("Некорректный размер страницы.", show_alert=True)
                return
            if page_size not in {5, 8, 10}:
                await cb.answer("Некорректный размер страницы.", show_alert=True)
                return
            await state.update_data(dl_page_size=page_size)
            await deps.show_filter_menu(message, state, edit=True)
            await cb.answer("✅ Размер страницы обновлён")
            return

        if action == "base_menu":
            await deps.show_base_filter_menu(message, state, edit=True)
            await cb.answer()
            return

        if action == "back":
            data = await state.get_data()
            model_type = data.get("dl_type", "checkpoint")
            await state.set_state(DownloadStates.choosing_source)
            await message.edit_text(
                f"🔍 <b>Где искать {h(deps.type_title(model_type))}?</b>",
                reply_markup=deps.build_source_keyboard(),
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

        author = "" if raw in {"-", "*"} else ",".join(_parse_author_filters(raw))
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
        if not query:
            await msg.answer("Введите запрос:")
            return

        try:
            await msg.delete()
        except TelegramBadRequest:
            pass

        await state.update_data(dl_query=query)

        status = await deps.render_download_panel(
            msg,
            state,
            f"🔍 Ищу <code>{h(query)}</code>…",
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
                f"❌ <b>Ошибка поиска:</b> <code>{h(exc)}</code>",
                reply_markup=build_keyboard(
                    [[back_button("dlqry:back", text="⬅️ Назад к фильтрам")]]
                ),
            )
            return

        if not results:
            await status.edit_text(
                "❌ Ничего не найдено. Попробуйте другой запрос или измените фильтры.",
                reply_markup=build_keyboard(
                    [[back_button("dlqry:back", text="⬅️ Назад к фильтрам")]]
                ),
            )
            return

        await state.update_data(
            dl_results=[asdict(item) for item in results],
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
        value = await _callback_value(cb, prefix="dlpick", invalid_text="❌ Неверный выбор.")
        if value is None:
            return
        if value == "cancel":
            await state.clear()
            await message.edit_text(
                "❌ Скачивание отменено.",
                reply_markup=_models_back_keyboard(),
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
            results_data = data.get("dl_results", [])
            results: list[Any] = []
            for item in results_data:
                if isinstance(item, dict):
                    results.append(deps.hydrate_result(item))

            page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)
            page = int(data.get("dl_results_page", 0) or 0)
            loaded_limit = int(data.get("dl_loaded_limit", page_size * 2) or (page_size * 2))
            exhausted = bool(data.get("dl_more_exhausted", False))
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
                next_limit = min(200, loaded_limit + page_size * 2)
                if next_limit == loaded_limit:
                    exhausted = True
                    notice = "Больше результатов не найдено."
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
                    await message.answer(f"⚠️ Ошибка поиска: {h(exc)}")
                    return
                loaded_limit = next_limit
                exhausted = len(results) <= prev_count
                await state.update_data(
                    dl_results=[asdict(item) for item in results],
                    dl_loaded_limit=loaded_limit,
                    dl_more_exhausted=exhausted,
                )
                max_page = max(0, (len(results) - 1) // page_size)
                page = min(page, max_page)
                if exhausted:
                    notice = "Больше результатов не найдено."
                else:
                    notice = "Подгрузил ещё результаты."

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
            await cb.answer("❌ Неверный выбор.", show_alert=True)
            return

        data = await state.get_data()
        results_data = data.get("dl_results", [])
        if index < 0 or index >= len(results_data):
            await cb.answer("❌ Неверный выбор.", show_alert=True)
            return

        result = deps.hydrate_result(results_data[index])

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
                    label_parts.append(deps.human_size(option.size_bytes))
                short = " | ".join(label_parts) or f"Версия {idx + 1}"
                short = short if len(short) <= 58 else short[:55] + "..."
                rows.append([button(f"{idx + 1}. {short}", f"dlver:{idx}")])
            rows.append([cancel_button("dlver:cancel")])
            rows.append([back_button("dlver:back", text="⬅️ Назад к результатам")])

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
                    meta.append(deps.human_size(option.size_bytes))
                if option.download_count > 0:
                    meta.append(f"📥 {deps.short_number(option.download_count)}")
                if option.version_name:
                    lines.append(f"{idx + 1}. <b>{h(option.version_name)}</b>")
                else:
                    lines.append(f"{idx + 1}. <b>Версия {idx + 1}</b>")
                if meta:
                    lines.append(f"   <i>{h(' | '.join(meta))}</i>")

            await message.edit_text("\n".join(lines), reply_markup=build_keyboard(rows))
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
        value = await _callback_value(cb, prefix="dlver", invalid_text="❌ Неверный выбор версии.")
        if value is None:
            return
        if value == "cancel":
            await state.clear()
            await message.edit_text(
                "❌ Скачивание отменено.",
                reply_markup=_models_back_keyboard(),
            )
            await cb.answer()
            return

        if value == "back":
            data = await state.get_data()
            results_data = data.get("dl_results", [])
            results: list[Any] = []
            for item in results_data:
                if isinstance(item, dict):
                    results.append(deps.hydrate_result(item))

            if not results:
                await state.clear()
                await message.edit_text("❌ Сессия истекла. Запустите /download заново.")
                await cb.answer()
                return

            await state.set_state(DownloadStates.choosing_result)
            page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)
            page = int(data.get("dl_results_page", 0) or 0)
            loaded_limit = int(data.get("dl_loaded_limit", page_size * 2) or (page_size * 2))
            exhausted = bool(data.get("dl_more_exhausted", False))
            can_continue = (not exhausted) and len(results) >= loaded_limit and loaded_limit < 200
            await deps.show_results_menu(
                message,
                state,
                results,
                edit=True,
                page=page,
                page_size=page_size,
                can_continue=can_continue,
            )
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
            await message.edit_text("❌ Сессия истекла. Запустите /download заново.")
            await cb.answer()
            return

        result = deps.hydrate_result(chosen_data)
        if index < 0 or index >= len(result.available_versions):
            await cb.answer("❌ Неверный выбор версии.", show_alert=True)
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
            cb, prefix="dlconfirm", invalid_text="❌ Неверное действие."
        )
        if decision is None:
            return
        if decision == "back":
            data = await state.get_data()
            results_data = data.get("dl_results", [])
            results: list[Any] = []
            for item in results_data:
                if isinstance(item, dict):
                    results.append(deps.hydrate_result(item))

            if not results:
                await state.clear()
                await message.edit_text("❌ Сессия истекла. Запустите /download заново.")
                await cb.answer()
                return

            await state.set_state(DownloadStates.choosing_result)
            page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)
            page = int(data.get("dl_results_page", 0) or 0)
            loaded_limit = int(data.get("dl_loaded_limit", page_size * 2) or (page_size * 2))
            exhausted = bool(data.get("dl_more_exhausted", False))
            can_continue = (not exhausted) and len(results) >= loaded_limit and loaded_limit < 200
            await deps.show_results_menu(
                message,
                state,
                results,
                edit=True,
                page=page,
                page_size=page_size,
                can_continue=can_continue,
            )
            await cb.answer()
            return

        if decision == "no":
            await state.clear()
            await message.edit_text(
                "❌ Скачивание отменено.",
                reply_markup=_models_back_keyboard(),
            )
            await cb.answer()
            return

        if decision != "yes":
            await cb.answer("❌ Неверное действие.", show_alert=True)
            return

        data = await state.get_data()
        chosen_data = data.get("dl_chosen")
        if not chosen_data:
            await state.clear()
            await message.edit_text("❌ Ошибка: выбранная модель не найдена.")
            await cb.answer()
            return

        uid = deps.callback_user_id(cb)
        existing = deps.runtime.active_downloads.get(uid)
        if existing and not existing.done():
            await cb.answer(
                "⏳ Уже идёт скачивание. Дождитесь завершения.",
                show_alert=True,
            )
            return

        result = deps.hydrate_result(chosen_data)
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
        edited = await message.edit_text(
            f"⬇️ <b>Скачивание:</b> {h(result.name)}…",
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
                    await deps.client.refresh_info()
                    details.append("✅ Список моделей обновлён.")
                except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                    details.append(
                        "⚠️ Модель скачана, но не удалось обновить список. Попробуйте /models."
                    )

                await status_msg.edit_text(
                    "\n".join(details), reply_markup=_download_done_keyboard()
                )
                await state.set_state(DownloadStates.choosing_result)
            except asyncio.CancelledError:
                await status_msg.edit_text(
                    "❌ Скачивание отменено.",
                    reply_markup=_models_back_keyboard(),
                )
            except FileExistsError:
                await status_msg.edit_text(
                    f"⚠️ Файл <code>{h(result.filename)}</code> уже существует.",
                    reply_markup=_models_back_keyboard(),
                )
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                await status_msg.edit_text(
                    f"❌ <b>Ошибка скачивания:</b> <code>{h(exc)}</code>",
                    reply_markup=_models_back_keyboard(),
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
        action = await _callback_value(cb, prefix="dldone", invalid_text="❌ Неизвестное действие.")
        if action is None:
            return

        if action == "new":
            await state.set_state(DownloadStates.entering_query)
            await deps.show_query_prompt(message, state, edit=True)
            await cb.answer()
            return

        if action == "results":
            data = await state.get_data()
            results_data = data.get("dl_results", [])
            results: list[Any] = []
            for item in results_data:
                if isinstance(item, dict):
                    results.append(deps.hydrate_result(item))

            if not results:
                await cb.answer(
                    "История результатов недоступна. Начните новый поиск.", show_alert=True
                )
                return

            page_size = int(data.get("dl_page_size", default_page_size) or default_page_size)
            page = int(data.get("dl_results_page", 0) or 0)
            loaded_limit = int(data.get("dl_loaded_limit", page_size * 2) or (page_size * 2))
            exhausted = bool(data.get("dl_more_exhausted", False))
            can_continue = (not exhausted) and len(results) >= loaded_limit and loaded_limit < 200
            await state.set_state(DownloadStates.choosing_result)
            await deps.show_results_menu(
                message,
                state,
                results,
                edit=True,
                page=page,
                page_size=page_size,
                can_continue=can_continue,
            )
            await cb.answer()
            return

        await cb.answer("❌ Неизвестное действие.", show_alert=True)

    @router.callback_query(F.data == "dldl:cancel")
    async def dl_cancel_running(cb: CallbackQuery):
        uid = deps.callback_user_id(cb)
        task = deps.runtime.active_downloads.get(uid)
        if task and not task.done():
            task.cancel()
            await cb.answer("❌ Отменяю скачивание…")
            return
        await cb.answer("Нечего отменять.", show_alert=True)
