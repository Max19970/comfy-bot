from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from core.interaction import require_callback_message
from core.states import DeleteModelStates
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import menu_root_button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService


@dataclass
class CommonDeleteDeps:
    router: Router
    runtime: Any
    client: Any
    downloader: Any
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    delete_model_types: dict[str, str]
    delete_types_keyboard: Callable[[str | None], Any]
    delete_files_keyboard: Callable[..., Any]
    delete_confirm_keyboard: Callable[[str | None], Any]
    h: Callable[[Any], str]


def register_common_delete_handlers(deps: CommonDeleteDeps) -> None:
    router = deps.router

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

    def _t(
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: Mapping[str, object] | None = None,
    ) -> str:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        return deps.localization.t(
            key,
            locale=locale,
            params=params,
            default=default,
        )

    def _model_type_label(model_type: str, *, locale: str) -> str:
        default_label = deps.delete_model_types.get(model_type, model_type)
        return deps.localization.t(
            f"common.delete.model_type.{model_type}",
            locale=locale,
            default=default_label,
        )

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

    def _models_back_keyboard(*, locale: str | None) -> Any:
        back_label = deps.localization.t(
            "common.menu.back_to_models",
            locale=locale,
            default="⬅️ Модели",
        )
        root_label = deps.localization.t(
            "common.menu.root",
            locale=locale,
            default="🏠 В меню",
        )
        return build_keyboard(
            [
                [back_button("menu:models", text=back_label)],
                [menu_root_button(text=root_label)],
            ]
        )

    async def _delete_execute(
        message: Message,
        *,
        uid: int,
        model_type: str,
        filename: str,
        telegram_locale: str | None,
        state: FSMContext | None = None,
    ) -> None:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        localized_type = _model_type_label(model_type, locale=locale)
        try:
            path = await deps.downloader.delete_local_model(model_type, filename)
            try:
                await deps.client.refresh_info()
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                pass
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                _t(
                    uid,
                    "common.delete.success_title",
                    "✅ <b>Модель удалена</b>",
                    telegram_locale=telegram_locale,
                )
                + "\n"
                + _t(
                    uid,
                    "common.delete.field.type",
                    "<b>Тип:</b>",
                    telegram_locale=telegram_locale,
                )
                + f" <code>{deps.h(localized_type)}</code>\n"
                + _t(
                    uid,
                    "common.delete.field.file",
                    "<b>Файл:</b>",
                    telegram_locale=telegram_locale,
                )
                + f" <code>{deps.h(filename)}</code>\n"
                + _t(
                    uid,
                    "common.delete.field.path",
                    "<b>Путь:</b>",
                    telegram_locale=telegram_locale,
                )
                + f" <code>{deps.h(path)}</code>",
                reply_markup=_models_back_keyboard(locale=locale),
            )
            if state is not None:
                await state.clear()
        except FileNotFoundError:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                _t(
                    uid,
                    "common.delete.error.file_not_found",
                    "❌ Файл не найден: <code>{filename}</code>",
                    telegram_locale=telegram_locale,
                    params={"filename": deps.h(filename)},
                ),
                reply_markup=_models_back_keyboard(locale=locale),
            )
            if state is not None:
                await state.clear()
        except (OSError, RuntimeError, ValueError) as exc:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                _t(
                    uid,
                    "common.delete.error.failed",
                    "❌ Не удалось удалить модель: <code>{error}</code>",
                    telegram_locale=telegram_locale,
                    params={"error": deps.h(exc)},
                ),
                reply_markup=_models_back_keyboard(locale=locale),
            )
            if state is not None:
                await state.clear()

    async def _show_delete_file_list(
        message: Message,
        *,
        uid: int,
        state: FSMContext,
        model_type: str,
        page: int = 0,
        telegram_locale: str | None,
    ) -> None:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        localized_type = _model_type_label(model_type, locale=locale)
        files = deps.downloader.list_local_models(model_type)
        if not files:
            await state.set_state(DeleteModelStates.choosing_type)
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                _t(
                    uid,
                    "common.delete.title",
                    "🗑 <b>Удаление модели</b>",
                    telegram_locale=telegram_locale,
                )
                + "\n"
                + _t(
                    uid,
                    "common.delete.panel.no_files_for_type",
                    "Для типа <code>{model_type}</code> файлов не найдено.",
                    telegram_locale=telegram_locale,
                    params={"model_type": deps.h(localized_type)},
                ),
                reply_markup=deps.delete_types_keyboard(locale),
            )
            return

        await state.update_data(dmdl_type=model_type, dmdl_files=files, dmdl_page=page)
        await state.set_state(DeleteModelStates.choosing_file)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.panel.choose_file",
                "🗑 <b>Выберите файл для удаления</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.field.type",
                "<b>Тип:</b>",
                telegram_locale=telegram_locale,
            )
            + f" {deps.h(localized_type)}\n"
            + _t(
                uid,
                "common.delete.panel.found_count",
                "<b>Найдено:</b> {count}",
                telegram_locale=telegram_locale,
                params={"count": len(files)},
            ),
            reply_markup=deps.delete_files_keyboard(
                model_type=model_type,
                files=files,
                page=page,
                locale=locale,
            ),
        )

    @router.message(Command("delete_model"))
    async def cmd_delete_model(message: Message, state: FSMContext):
        uid = deps.message_user_id(message)
        telegram_locale = message.from_user.language_code if message.from_user else None
        raw = (message.text or "").strip()
        parts = raw.split(maxsplit=2)

        if len(parts) >= 3:
            model_type = parts[1].strip().lower()
            filename = parts[2].strip()
            await _delete_execute(
                message,
                uid=uid,
                model_type=model_type,
                filename=filename,
                telegram_locale=telegram_locale,
                state=state,
            )
            return

        await state.clear()
        await state.set_state(DeleteModelStates.choosing_type)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.title",
                "🗑 <b>Удаление модели</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.panel.choose_type",
                "Выберите тип модели.",
                telegram_locale=telegram_locale,
            ),
            reply_markup=deps.delete_types_keyboard(
                _resolved_locale(uid, telegram_locale=telegram_locale)
            ),
        )

    @router.callback_query(F.data == "menu:delete_model")
    async def menu_delete_model(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        await state.clear()
        await state.set_state(DeleteModelStates.choosing_type)
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.title",
                "🗑 <b>Удаление модели</b>",
                telegram_locale=cb.from_user.language_code,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.panel.choose_type",
                "Выберите тип модели.",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=deps.delete_types_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_type, F.data.startswith("dmdl:type:"))
    async def dmdl_type_chosen(cb: CallbackQuery, state: FSMContext):
        model_type = (cb.data or "").split(":", 2)[2]
        uid = deps.callback_user_id(cb)
        if model_type not in deps.delete_model_types:
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.unknown_type_alert",
                    "❌ Неизвестный тип",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        await _show_delete_file_list(
            cast(Message, cb.message),
            uid=uid,
            state=state,
            model_type=model_type,
            page=0,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_file, F.data.startswith("dmdl:page:"))
    async def dmdl_page(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        files = data.get("dmdl_files")
        uid = deps.callback_user_id(cb)
        telegram_locale = cb.from_user.language_code
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        if model_type not in deps.delete_model_types or not isinstance(files, list):
            await state.clear()
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.session_expired_alert",
                    "Сессия истекла",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        try:
            page = int((cb.data or "").split(":", 2)[2])
        except ValueError:
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.invalid_page_alert",
                    "❌ Неверная страница",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        await state.update_data(dmdl_page=page)
        localized_type = _model_type_label(model_type, locale=locale)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.panel.choose_file",
                "🗑 <b>Выберите файл для удаления</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.field.type",
                "<b>Тип:</b>",
                telegram_locale=telegram_locale,
            )
            + f" {deps.h(localized_type)}\n"
            + _t(
                uid,
                "common.delete.panel.found_count",
                "<b>Найдено:</b> {count}",
                telegram_locale=telegram_locale,
                params={"count": len(files)},
            ),
            reply_markup=deps.delete_files_keyboard(
                model_type=model_type,
                files=files,
                page=page,
                locale=locale,
            ),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_file, F.data.startswith("dmdl:file:"))
    async def dmdl_file_chosen(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        files = data.get("dmdl_files")
        uid = deps.callback_user_id(cb)
        telegram_locale = cb.from_user.language_code
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        if model_type not in deps.delete_model_types or not isinstance(files, list):
            await state.clear()
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.session_expired_alert",
                    "Сессия истекла",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        try:
            index = int((cb.data or "").split(":", 2)[2])
        except ValueError:
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.invalid_selection_alert",
                    "❌ Неверный выбор",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return
        if index < 0 or index >= len(files):
            await cb.answer(
                _t(
                    uid,
                    "common.delete.error.invalid_selection_alert",
                    "❌ Неверный выбор",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        filename = str(files[index])
        await state.update_data(dmdl_file=filename)
        await state.set_state(DeleteModelStates.confirming_delete)
        localized_type = _model_type_label(model_type, locale=locale)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.panel.confirm",
                "⚠️ <b>Подтвердите удаление</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.field.type",
                "<b>Тип:</b>",
                telegram_locale=telegram_locale,
            )
            + f" <code>{deps.h(localized_type)}</code>\n"
            + _t(
                uid,
                "common.delete.field.file",
                "<b>Файл:</b>",
                telegram_locale=telegram_locale,
            )
            + f" <code>{deps.h(filename)}</code>",
            reply_markup=deps.delete_confirm_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.confirming_delete, F.data == "dmdl:confirm:no")
    async def dmdl_confirm_no(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        page = int(data.get("dmdl_page") or 0)
        uid = deps.callback_user_id(cb)
        await _show_delete_file_list(
            cast(Message, cb.message),
            uid=uid,
            state=state,
            model_type=model_type,
            page=page,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.confirming_delete, F.data == "dmdl:confirm:yes")
    async def dmdl_confirm_yes(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        filename = str(data.get("dmdl_file") or "")
        uid = deps.callback_user_id(cb)
        await _delete_execute(
            cast(Message, cb.message),
            uid=uid,
            model_type=model_type,
            filename=filename,
            telegram_locale=cb.from_user.language_code,
            state=state,
        )
        await cb.answer(
            _t(
                uid,
                "common.delete.alert.deleted",
                "Удалено",
                telegram_locale=cb.from_user.language_code,
            )
        )

    @router.callback_query(DeleteModelStates.choosing_file, F.data == "dmdl:back:type")
    async def dmdl_back_type(cb: CallbackQuery, state: FSMContext):
        await state.set_state(DeleteModelStates.choosing_type)
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.title",
                "🗑 <b>Удаление модели</b>",
                telegram_locale=cb.from_user.language_code,
            )
            + "\n"
            + _t(
                uid,
                "common.delete.panel.choose_type",
                "Выберите тип модели.",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=deps.delete_types_keyboard(locale),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_type, F.data == "dmdl:cancel")
    @router.callback_query(DeleteModelStates.choosing_file, F.data == "dmdl:cancel")
    @router.callback_query(DeleteModelStates.confirming_delete, F.data == "dmdl:cancel")
    async def dmdl_cancel(cb: CallbackQuery, state: FSMContext):
        await state.clear()
        uid = deps.callback_user_id(cb)
        locale = _resolved_locale(uid, telegram_locale=cb.from_user.language_code)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            _t(
                uid,
                "common.delete.panel.cancelled",
                "❌ Удаление модели отменено.",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=_models_back_keyboard(locale=locale),
        )
        await cb.answer()
