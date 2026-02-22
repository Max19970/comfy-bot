from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from core.interaction import require_callback_message


@dataclass
class CommonJobsDeps:
    router: Router
    runtime: Any
    client: Any
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    user_generations: Callable[[Any, int], list[Any]]
    queue_prompt_state: Callable[[dict[str, Any], str], str]
    jobs_keyboard: Callable[..., Any]
    job_detail_keyboard: Callable[[str], Any]
    truncate: Callable[[str, int], str]
    h: Callable[[Any], str]


def register_common_jobs_handlers(deps: CommonJobsDeps) -> None:
    router = deps.router

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

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
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_jobs(message, deps.callback_user_id(cb), page=0)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:page:"))
    async def menu_jobs_page(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        try:
            page = int((cb.data or "").split(":", 3)[3])
        except ValueError:
            page = 0
        await _show_jobs(message, deps.callback_user_id(cb), page=page)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:open:"))
    async def menu_jobs_open(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        await _show_job_detail(message, deps.callback_user_id(cb), generation_id)
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:cancel:"))
    async def menu_jobs_cancel_one(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
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
        message = await _callback_message(cb)
        if message is None:
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

    @router.callback_query(F.data == "menu:jobs:cancel_all")
    async def menu_jobs_cancel_all(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return

        uid = deps.callback_user_id(cb)
        items = deps.user_generations(deps.runtime, uid)
        if not items:
            await cb.answer("Активных задач нет.", show_alert=True)
            await _show_jobs(message, uid, page=0)
            return

        for item in items:
            if item.task is not None and not item.task.done():
                item.task.cancel()
            if item.prompt_id:
                asyncio.create_task(deps.client.cancel_prompt(item.prompt_id))

        await cb.answer(f"🛑 Отмена запрошена для {len(items)} задач")
        await _show_jobs(message, uid, page=0)
