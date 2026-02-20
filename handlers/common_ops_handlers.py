from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from core.states import DeleteModelStates


@dataclass
class CommonDeleteDeps:
    router: Router
    runtime: Any
    client: Any
    downloader: Any
    callback_message: Callable[[CallbackQuery], Message | None]
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    delete_model_types: dict[str, str]
    delete_types_keyboard: Callable[[], Any]
    delete_files_keyboard: Callable[..., Any]
    delete_confirm_keyboard: Callable[[], Any]
    h: Callable[[Any], str]


@dataclass
class CommonJobsDeps:
    router: Router
    runtime: Any
    client: Any
    callback_message: Callable[[CallbackQuery], Message | None]
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    user_generations: Callable[[Any, int], list[Any]]
    queue_prompt_state: Callable[[dict[str, Any], str], str]
    jobs_keyboard: Callable[..., Any]
    job_detail_keyboard: Callable[[str], Any]
    truncate: Callable[[str, int], str]
    h: Callable[[Any], str]


def register_common_delete_handlers(deps: CommonDeleteDeps) -> None:
    router = deps.router

    def _models_back_keyboard() -> Any:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Модели", callback_data="menu:models")],
                [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
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
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
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


def register_common_jobs_handlers(deps: CommonJobsDeps) -> None:
    router = deps.router

    async def _show_job_detail(message: Message, uid: int, generation_id: str) -> None:
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != uid:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                "❌ Задача не найдена или уже завершена.",
                reply_markup=deps.jobs_keyboard(deps.user_generations(deps.runtime, uid), page=0),
            )
            return

        status_text = "в очереди"
        if item.status_msg:
            raw = (item.status_msg.text or item.status_msg.caption or "").strip()
            if raw:
                status_text = raw
        queue_snapshot: dict[str, Any] = {}
        try:
            queue_snapshot = await deps.client.get_queue_status()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            queue_snapshot = {}
        queue_state = deps.queue_prompt_state(queue_snapshot, item.prompt_id or "")
        age = max(0, int(time.time() - item.created_at))
        mins, secs = divmod(age, 60)
        title = item.title or "Задача"
        lines = [
            f"🧾 <b>{deps.h(title)}</b>",
            f"<b>ID:</b> <code>{deps.h(item.generation_id)}</code>",
            f"<b>В очереди:</b> {mins:02d}:{secs:02d}",
            f"<b>Состояние очереди:</b> {deps.h(queue_state)}",
        ]
        if item.restored:
            lines.append("<b>Режим:</b> восстановлена после перезапуска")
        if item.prompt_id:
            lines.append(f"<b>ComfyUI prompt_id:</b> <code>{deps.h(item.prompt_id)}</code>")
        lines.extend(["", "<b>Текущий статус:</b>", deps.h(deps.truncate(status_text, 3000))])
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "\n".join(lines),
            reply_markup=deps.job_detail_keyboard(item.generation_id),
        )

    async def _show_jobs(msg: Message, uid: int, *, page: int = 0) -> None:
        items = deps.user_generations(deps.runtime, uid)
        queue_snapshot: dict[str, Any] = {}
        try:
            queue_snapshot = await deps.client.get_queue_status()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            queue_snapshot = {}

        stale_ids: list[str] = []
        now = time.time()
        for item in items:
            if item.task is not None:
                continue
            if not item.prompt_id:
                continue
            state = deps.queue_prompt_state(queue_snapshot, item.prompt_id)
            if state == "не найдена в очереди" and now - item.created_at > 300:
                stale_ids.append(item.generation_id)
        for generation_id in stale_ids:
            deps.runtime.active_generations.pop(generation_id, None)
        if stale_ids:
            items = deps.user_generations(deps.runtime, uid)

        if not items:
            await deps.render_user_panel(
                msg,
                deps.runtime,
                uid,
                "🧵 <b>Мои задачи</b>\nСейчас активных генераций нет.",
                reply_markup=deps.jobs_keyboard([], page=0),
            )
            return

        lines = [f"🧵 <b>Мои задачи</b>\nАктивно: <b>{len(items)}</b>", ""]
        for idx, item in enumerate(items, start=1):
            status_text = "в очереди"
            if item.status_msg:
                raw = (item.status_msg.text or item.status_msg.caption or "").strip()
                if raw:
                    status_text = raw.split("\n", 1)[0]
            queue_state = deps.queue_prompt_state(queue_snapshot, item.prompt_id or "")
            status_text = deps.truncate(status_text, 90)
            kind_label = "🎨 Генерация"
            if item.kind == "enhancement":
                kind_label = "✨ Улучшение"
            restored = " (восстановлена)" if item.restored else ""
            lines.append(
                f"{idx}. {kind_label}{restored} — {deps.h(queue_state)} | {deps.h(status_text)}"
            )

        await deps.render_user_panel(
            msg,
            deps.runtime,
            uid,
            "\n".join(lines),
            reply_markup=deps.jobs_keyboard(items, page=page),
        )

    @router.message(Command("jobs"))
    async def cmd_jobs(msg: Message, page: int = 0):
        await _show_jobs(msg, deps.message_user_id(msg), page=page)

    @router.callback_query(F.data == "menu:jobs")
    async def menu_jobs(cb: CallbackQuery):
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        await _show_jobs(message, deps.callback_user_id(cb), page=0)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:page:"))
    async def menu_jobs_page(cb: CallbackQuery):
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        try:
            page = int((cb.data or "").split(":", 3)[3])
        except ValueError:
            page = 0
        await _show_jobs(message, deps.callback_user_id(cb), page=page)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:open:"))
    async def menu_jobs_open(cb: CallbackQuery):
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        await _show_job_detail(message, deps.callback_user_id(cb), generation_id)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:cancel:"))
    async def menu_jobs_cancel_one(cb: CallbackQuery):
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != deps.callback_user_id(cb):
            await cb.answer("Задача не найдена", show_alert=True)
            return
        if item.task is not None and not item.task.done():
            item.task.cancel()
        if item.prompt_id:
            asyncio.create_task(deps.client.cancel_prompt(item.prompt_id))
        await cb.answer("🛑 Задача отменяется")
        await _show_job_detail(message, deps.callback_user_id(cb), generation_id)

    @router.callback_query(F.data.startswith("menu:jobs:goto:"))
    async def menu_jobs_goto_source(cb: CallbackQuery):
        message = deps.callback_message(cb)
        if message is None:
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != deps.callback_user_id(cb):
            await cb.answer("Задача не найдена", show_alert=True)
            return
        status_chat_id = item.status_msg.chat.id if item.status_msg else item.status_chat_id
        status_message_id = (
            item.status_msg.message_id if item.status_msg else item.status_message_id
        )
        if status_chat_id is None or status_message_id is None:
            await cb.answer("Источник статуса недоступен", show_alert=True)
            return
        if status_chat_id != message.chat.id:
            await cb.answer("Источник в другом чате", show_alert=True)
            return
        await message.answer("📌 Исходное сообщение задачи", reply_to_message_id=status_message_id)
        await cb.answer("Готово")
