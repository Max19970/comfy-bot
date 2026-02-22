from __future__ import annotations

from collections.abc import Awaitable, Callable

from aiogram.types import CallbackQuery, Message

from core.interaction import require_callback_message
from core.runtime import PromptRequest

RequirePromptRequestForCallback = Callable[
    [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
]


async def require_message_and_request(
    cb: CallbackQuery,
    *,
    require_prompt_request_for_callback: RequirePromptRequestForCallback,
) -> tuple[Message, int, PromptRequest] | None:
    payload = await require_prompt_request_for_callback(cb)
    if payload is None:
        return None

    message = await require_callback_message(cb)
    if message is None:
        return None

    uid, req = payload
    return message, uid, req
