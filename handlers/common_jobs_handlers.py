from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from core.interaction import require_callback_message
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService


@dataclass
class CommonJobsDeps:
    router: Router
    runtime: Any
    client: Any
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    render_user_panel: Callable[..., Awaitable[Message]]
    user_generations: Callable[[Any, int], list[Any]]
    queue_prompt_state: Callable[[dict[str, Any], str, str | None], str]
    jobs_keyboard: Callable[..., Any]
    job_detail_keyboard: Callable[[str, str | None], Any]
    truncate: Callable[[str, int], str]
    h: Callable[[Any], str]


def register_common_jobs_handlers(deps: CommonJobsDeps) -> None:
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

    async def _callback_message(cb: CallbackQuery) -> Message | None:
        return await require_callback_message(cb)

    async def _show_job_detail(
        message: Message,
        uid: int,
        generation_id: str,
        *,
        telegram_locale: str | None,
    ) -> None:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != uid:
            await deps.render_user_panel(
                message,
                deps.runtime,
                uid,
                _t(
                    uid,
                    "common.jobs.detail.not_found_or_done",
                    "❌ Задача не найдена или уже завершена.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=deps.jobs_keyboard(
                    deps.user_generations(deps.runtime, uid),
                    page=0,
                    locale=locale,
                ),
            )
            return

        status_text = _t(
            uid,
            "common.jobs.status.default_pending",
            "в очереди",
            telegram_locale=telegram_locale,
        )
        if item.status_msg:
            raw = (item.status_msg.text or item.status_msg.caption or "").strip()
            if raw:
                status_text = raw
        queue_snapshot: dict[str, Any] = {}
        try:
            queue_snapshot = await deps.client.get_queue_status()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            queue_snapshot = {}
        queue_state = deps.queue_prompt_state(queue_snapshot, item.prompt_id or "", locale)
        age = max(0, int(time.time() - item.created_at))
        mins, secs = divmod(age, 60)
        title = item.title or _t(
            uid,
            "common.jobs.detail.title_fallback",
            "Задача",
            telegram_locale=telegram_locale,
        )
        lines = [
            f"🧾 <b>{deps.h(title)}</b>",
            f"{_t(uid, 'common.jobs.detail.id_label', '<b>ID:</b>', telegram_locale=telegram_locale)} <code>{deps.h(item.generation_id)}</code>",
            f"{_t(uid, 'common.jobs.detail.in_queue_for_label', '<b>В очереди:</b>', telegram_locale=telegram_locale)} {mins:02d}:{secs:02d}",
            f"{_t(uid, 'common.jobs.detail.queue_state_label', '<b>Состояние очереди:</b>', telegram_locale=telegram_locale)} {deps.h(queue_state)}",
        ]
        if item.restored:
            lines.append(
                _t(
                    uid,
                    "common.jobs.detail.mode_restored",
                    "<b>Режим:</b> восстановлена после перезапуска",
                    telegram_locale=telegram_locale,
                )
            )
        if item.prompt_id:
            lines.append(
                f"{_t(uid, 'common.jobs.detail.prompt_id_label', '<b>ComfyUI prompt_id:</b>', telegram_locale=telegram_locale)} <code>{deps.h(item.prompt_id)}</code>"
            )
        lines.extend(
            [
                "",
                _t(
                    uid,
                    "common.jobs.detail.current_status_label",
                    "<b>Текущий статус:</b>",
                    telegram_locale=telegram_locale,
                ),
                deps.h(deps.truncate(status_text, 3000)),
            ]
        )
        await deps.render_user_panel(
            message,
            deps.runtime,
            uid,
            "\n".join(lines),
            reply_markup=deps.job_detail_keyboard(item.generation_id, locale),
        )

    async def _show_jobs(
        msg: Message,
        uid: int,
        *,
        page: int = 0,
        telegram_locale: str | None,
    ) -> None:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        items = deps.user_generations(deps.runtime, uid)
        queue_snapshot: dict[str, Any] = {}
        try:
            queue_snapshot = await deps.client.get_queue_status()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
            queue_snapshot = {}

        not_found_state = _t(
            uid,
            "common.jobs.queue_state.not_found",
            "не найдена в очереди",
            telegram_locale=telegram_locale,
        )
        stale_ids: list[str] = []
        now = time.time()
        for item in items:
            if item.task is not None:
                continue
            if not item.prompt_id:
                continue
            state = deps.queue_prompt_state(queue_snapshot, item.prompt_id, locale)
            if state == not_found_state and now - item.created_at > 300:
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
                _t(
                    uid,
                    "common.jobs.list.empty",
                    "🧵 <b>Мои задачи</b>\nСейчас активных генераций нет.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=deps.jobs_keyboard([], page=0, locale=locale),
            )
            return

        lines = [
            _t(
                uid,
                "common.jobs.list.title_active",
                "🧵 <b>Мои задачи</b>\nАктивно: <b>{count}</b>",
                telegram_locale=telegram_locale,
                params={"count": len(items)},
            ),
            "",
        ]
        for idx, item in enumerate(items, start=1):
            status_text = _t(
                uid,
                "common.jobs.status.default_pending",
                "в очереди",
                telegram_locale=telegram_locale,
            )
            if item.status_msg:
                raw = (item.status_msg.text or item.status_msg.caption or "").strip()
                if raw:
                    status_text = raw.split("\n", 1)[0]
            queue_state = deps.queue_prompt_state(queue_snapshot, item.prompt_id or "", locale)
            status_text = deps.truncate(status_text, 90)
            kind_label = _t(
                uid,
                "common.jobs.kind.generation",
                "🎨 Генерация",
                telegram_locale=telegram_locale,
            )
            if item.kind == "enhancement":
                kind_label = _t(
                    uid,
                    "common.jobs.kind.enhancement",
                    "✨ Улучшение",
                    telegram_locale=telegram_locale,
                )
            restored = (
                _t(
                    uid,
                    "common.jobs.list.restored_suffix",
                    " (восстановлена)",
                    telegram_locale=telegram_locale,
                )
                if item.restored
                else ""
            )
            lines.append(
                f"{idx}. {kind_label}{restored} — {deps.h(queue_state)} | {deps.h(status_text)}"
            )

        await deps.render_user_panel(
            msg,
            deps.runtime,
            uid,
            "\n".join(lines),
            reply_markup=deps.jobs_keyboard(items, page=page, locale=locale),
        )

    @router.message(Command("jobs"))
    async def cmd_jobs(msg: Message, page: int = 0):
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        await _show_jobs(
            msg,
            deps.message_user_id(msg),
            page=page,
            telegram_locale=telegram_locale,
        )

    @router.callback_query(F.data == "menu:jobs")
    async def menu_jobs(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        await _show_jobs(
            message,
            deps.callback_user_id(cb),
            page=0,
            telegram_locale=cb.from_user.language_code,
        )
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
        await _show_jobs(
            message,
            deps.callback_user_id(cb),
            page=page,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:open:"))
    async def menu_jobs_open(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        await _show_job_detail(
            message,
            deps.callback_user_id(cb),
            generation_id,
            telegram_locale=cb.from_user.language_code,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("menu:jobs:cancel:"))
    async def menu_jobs_cancel_one(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        uid = deps.callback_user_id(cb)
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != uid:
            await cb.answer(
                _t(
                    uid,
                    "common.jobs.alert.not_found",
                    "Задача не найдена",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        if item.task is not None and not item.task.done():
            item.task.cancel()
        if item.prompt_id:
            asyncio.create_task(deps.client.cancel_prompt(item.prompt_id))
        await cb.answer(
            _t(
                uid,
                "common.jobs.alert.cancelling",
                "🛑 Задача отменяется",
                telegram_locale=cb.from_user.language_code,
            )
        )
        await _show_job_detail(
            message,
            uid,
            generation_id,
            telegram_locale=cb.from_user.language_code,
        )

    @router.callback_query(F.data.startswith("menu:jobs:goto:"))
    async def menu_jobs_goto_source(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return
        generation_id = (cb.data or "").split(":", 3)[3] if cb.data else ""
        uid = deps.callback_user_id(cb)
        item = deps.runtime.active_generations.get(generation_id)
        if item is None or item.owner_uid != uid:
            await cb.answer(
                _t(
                    uid,
                    "common.jobs.alert.not_found",
                    "Задача не найдена",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        status_chat_id = item.status_msg.chat.id if item.status_msg else item.status_chat_id
        status_message_id = (
            item.status_msg.message_id if item.status_msg else item.status_message_id
        )
        if status_chat_id is None or status_message_id is None:
            await cb.answer(
                _t(
                    uid,
                    "common.jobs.alert.source_unavailable",
                    "Источник статуса недоступен",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        if status_chat_id != message.chat.id:
            await cb.answer(
                _t(
                    uid,
                    "common.jobs.alert.source_other_chat",
                    "Источник в другом чате",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        await message.answer(
            _t(
                uid,
                "common.jobs.alert.source_message",
                "📌 Исходное сообщение задачи",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_to_message_id=status_message_id,
        )
        await cb.answer(
            _t(
                uid,
                "common.jobs.alert.done",
                "Готово",
                telegram_locale=cb.from_user.language_code,
            )
        )

    @router.callback_query(F.data == "menu:jobs:cancel_all")
    async def menu_jobs_cancel_all(cb: CallbackQuery):
        message = await _callback_message(cb)
        if message is None:
            return

        uid = deps.callback_user_id(cb)
        items = deps.user_generations(deps.runtime, uid)
        if not items:
            await cb.answer(
                _t(
                    uid,
                    "common.jobs.alert.no_active",
                    "Активных задач нет.",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            await _show_jobs(
                message,
                uid,
                page=0,
                telegram_locale=cb.from_user.language_code,
            )
            return

        for item in items:
            if item.task is not None and not item.task.done():
                item.task.cancel()
            if item.prompt_id:
                asyncio.create_task(deps.client.cancel_prompt(item.prompt_id))

        await cb.answer(
            _t(
                uid,
                "common.jobs.alert.cancel_all_requested",
                "🛑 Отмена запрошена для {count} задач",
                telegram_locale=cb.from_user.language_code,
                params={"count": len(items)},
            )
        )
        await _show_jobs(
            message,
            uid,
            page=0,
            telegram_locale=cb.from_user.language_code,
        )
