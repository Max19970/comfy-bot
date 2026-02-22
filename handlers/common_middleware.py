from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message
from aiogram.types.base import TelegramObject


def register_access_middlewares(
    router: Router,
    *,
    allowed_users: Collection[int],
    runtime_persist: Callable[[], None],
) -> None:
    async def wl_msg(
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)
        command_text = (event.text or "").strip()
        should_delete_command = command_text.startswith("/")
        if allowed_users and event.from_user and event.from_user.id not in allowed_users:
            await event.answer("⛔ <b>Доступ запрещён</b>\nВаш ID не в списке разрешённых.")
            return
        try:
            return await handler(event, data)
        finally:
            if should_delete_command:
                try:
                    await event.delete()
                except TelegramBadRequest:
                    pass
            runtime_persist()

    async def wl_cb(
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, CallbackQuery):
            return await handler(event, data)
        if allowed_users and event.from_user and event.from_user.id not in allowed_users:
            await event.answer(
                "⛔ Доступ запрещён.",
                show_alert=True,
            )
            return
        try:
            return await handler(event, data)
        finally:
            runtime_persist()

    router.message.outer_middleware(wl_msg)
    router.callback_query.outer_middleware(wl_cb)
