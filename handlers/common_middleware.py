from __future__ import annotations

from collections.abc import Awaitable, Callable, Collection
from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Message
from aiogram.types.base import TelegramObject
from domain.localization import LocalizationService


def register_access_middlewares(
    router: Router,
    *,
    allowed_users: Collection[int],
    runtime_persist: Callable[[], None],
    localization: LocalizationService | None = None,
    resolve_locale_for_user: Callable[[int | None, str | None], str] | None = None,
) -> None:
    def _locale_for_event(*, user_id: int | None, telegram_locale: str | None) -> str | None:
        if localization is None:
            return None
        if resolve_locale_for_user is None:
            return localization.default_locale()
        return resolve_locale_for_user(user_id, telegram_locale)

    def _t(
        key: str,
        default: str,
        *,
        user_id: int | None,
        telegram_locale: str | None,
    ) -> str:
        if localization is None:
            return default
        return localization.t(
            key,
            locale=_locale_for_event(user_id=user_id, telegram_locale=telegram_locale),
            default=default,
        )

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
            await event.answer(
                _t(
                    "common.access.denied.message",
                    "⛔ <b>Доступ запрещён</b>\nВаш ID не в списке разрешённых.",
                    user_id=event.from_user.id,
                    telegram_locale=event.from_user.language_code,
                )
            )
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
                _t(
                    "common.access.denied.alert",
                    "⛔ Доступ запрещён.",
                    user_id=event.from_user.id,
                    telegram_locale=event.from_user.language_code,
                ),
                show_alert=True,
            )
            return
        try:
            return await handler(event, data)
        finally:
            runtime_persist()

    router.message.outer_middleware(wl_msg)
    router.callback_query.outer_middleware(wl_cb)
