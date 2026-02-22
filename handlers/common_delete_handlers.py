from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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


@dataclass
class CommonDeleteDeps:
    router: Router
    runtime: Any
    client: Any
    downloader: Any
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    delete_model_types: dict[str, str]
    delete_types_keyboard: Callable[[], Any]
    delete_files_keyboard: Callable[..., Any]
    delete_confirm_keyboard: Callable[[], Any]
    h: Callable[[Any], str]


def register_common_delete_handlers(deps: CommonDeleteDeps) -> None:
    router = deps.router

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

    def _models_back_keyboard() -> Any:
        return build_keyboard(
            [
                [back_button("menu:models", text="⬅️ Модели")],
                [menu_root_button()],
            ]
        )

    async def _delete_execute(
        message: Message,
        *,
        uid: int,
        model_type: str,
        filename: str,
        state: FSMContext | None = None,
    ) -> None:
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
                "✅ <b>Модель удалена</b>\n"
                f"<b>Тип:</b> <code>{deps.h(model_type)}</code>\n"
                f"<b>Файл:</b> <code>{deps.h(filename)}</code>\n"
                f"<b>Путь:</b> <code>{deps.h(path)}</code>",
                reply_markup=_models_back_keyboard(),
            )
            if state is not None:
                await state.clear()
        except FileNotFoundError:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                f"❌ Файл не найден: <code>{deps.h(filename)}</code>",
                reply_markup=_models_back_keyboard(),
            )
            if state is not None:
                await state.clear()
        except (OSError, RuntimeError, ValueError) as exc:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                f"❌ Не удалось удалить модель: <code>{deps.h(exc)}</code>",
                reply_markup=_models_back_keyboard(),
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
    ) -> None:
        files = deps.downloader.list_local_models(model_type)
        if not files:
            await state.set_state(DeleteModelStates.choosing_type)
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                "🗑 <b>Удаление модели</b>\n"
                f"Для типа <code>{deps.h(model_type)}</code> файлов не найдено.",
                reply_markup=deps.delete_types_keyboard(),
            )
            return

        await state.update_data(dmdl_type=model_type, dmdl_files=files, dmdl_page=page)
        await state.set_state(DeleteModelStates.choosing_file)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🗑 <b>Выберите файл для удаления</b>\n"
            f"<b>Тип:</b> {deps.h(deps.delete_model_types.get(model_type, model_type))}\n"
            f"<b>Найдено:</b> {len(files)}",
            reply_markup=deps.delete_files_keyboard(model_type=model_type, files=files, page=page),
        )

    @router.message(Command("delete_model"))
    async def cmd_delete_model(message: Message, state: FSMContext):
        uid = deps.message_user_id(message)
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
                state=state,
            )
            return

        await state.clear()
        await state.set_state(DeleteModelStates.choosing_type)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🗑 <b>Удаление модели</b>\nВыберите тип модели.",
            reply_markup=deps.delete_types_keyboard(),
        )

    @router.callback_query(F.data == "menu:delete_model")
    async def menu_delete_model(cb: CallbackQuery, state: FSMContext):
        message = await _callback_message(cb)
        if message is None:
            return
        await state.clear()
        await state.set_state(DeleteModelStates.choosing_type)
        uid = deps.callback_user_id(cb)
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "🗑 <b>Удаление модели</b>\nВыберите тип модели.",
            reply_markup=deps.delete_types_keyboard(),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_type, F.data.startswith("dmdl:type:"))
    async def dmdl_type_chosen(cb: CallbackQuery, state: FSMContext):
        model_type = (cb.data or "").split(":", 2)[2]
        uid = deps.callback_user_id(cb)
        if model_type not in deps.delete_model_types:
            await cb.answer("❌ Неизвестный тип", show_alert=True)
            return
        await _show_delete_file_list(
            cast(Message, cb.message),
            uid=uid,
            state=state,
            model_type=model_type,
            page=0,
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_file, F.data.startswith("dmdl:page:"))
    async def dmdl_page(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        files = data.get("dmdl_files")
        if model_type not in deps.delete_model_types or not isinstance(files, list):
            await state.clear()
            await cb.answer("Сессия истекла", show_alert=True)
            return

        try:
            page = int((cb.data or "").split(":", 2)[2])
        except ValueError:
            await cb.answer("❌ Неверная страница", show_alert=True)
            return

        uid = deps.callback_user_id(cb)
        await state.update_data(dmdl_page=page)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            "🗑 <b>Выберите файл для удаления</b>\n"
            f"<b>Тип:</b> {deps.h(deps.delete_model_types.get(model_type, model_type))}\n"
            f"<b>Найдено:</b> {len(files)}",
            reply_markup=deps.delete_files_keyboard(model_type=model_type, files=files, page=page),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_file, F.data.startswith("dmdl:file:"))
    async def dmdl_file_chosen(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        model_type = str(data.get("dmdl_type") or "")
        files = data.get("dmdl_files")
        if model_type not in deps.delete_model_types or not isinstance(files, list):
            await state.clear()
            await cb.answer("Сессия истекла", show_alert=True)
            return

        try:
            index = int((cb.data or "").split(":", 2)[2])
        except ValueError:
            await cb.answer("❌ Неверный выбор", show_alert=True)
            return
        if index < 0 or index >= len(files):
            await cb.answer("❌ Неверный выбор", show_alert=True)
            return

        filename = str(files[index])
        await state.update_data(dmdl_file=filename)
        await state.set_state(DeleteModelStates.confirming_delete)
        uid = deps.callback_user_id(cb)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            "⚠️ <b>Подтвердите удаление</b>\n"
            f"<b>Тип:</b> <code>{deps.h(model_type)}</code>\n"
            f"<b>Файл:</b> <code>{deps.h(filename)}</code>",
            reply_markup=deps.delete_confirm_keyboard(),
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
            state=state,
        )
        await cb.answer("Удалено")

    @router.callback_query(DeleteModelStates.choosing_file, F.data == "dmdl:back:type")
    async def dmdl_back_type(cb: CallbackQuery, state: FSMContext):
        await state.set_state(DeleteModelStates.choosing_type)
        uid = deps.callback_user_id(cb)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            "🗑 <b>Удаление модели</b>\nВыберите тип модели.",
            reply_markup=deps.delete_types_keyboard(),
        )
        await cb.answer()

    @router.callback_query(DeleteModelStates.choosing_type, F.data == "dmdl:cancel")
    @router.callback_query(DeleteModelStates.choosing_file, F.data == "dmdl:cancel")
    @router.callback_query(DeleteModelStates.confirming_delete, F.data == "dmdl:cancel")
    async def dmdl_cancel(cb: CallbackQuery, state: FSMContext):
        await state.clear()
        uid = deps.callback_user_id(cb)
        await deps.render_user_panel(
            cast(Message, cb.message),
            deps.runtime,
            uid,
            "❌ Удаление модели отменено.",
            reply_markup=_models_back_keyboard(),
        )
        await cb.answer()
