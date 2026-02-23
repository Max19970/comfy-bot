from __future__ import annotations

from collections.abc import Callable
from typing import cast

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

TranslateText = Callable[[str, str | None, str], str]

CALLBACK_MESSAGE_UNAVAILABLE_TEXT = "⚠️ Сообщение недоступно."
CALLBACK_MESSAGE_UNAVAILABLE_KEY = "core.interaction.callback_message_unavailable"


def _tx(translate: TranslateText | None, key: str, locale: str | None, default: str) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


def callback_message(cb: CallbackQuery) -> Message | None:
    message = cb.message
    if message is None or not hasattr(message, "edit_text"):
        return None
    return cast(Message, message)


async def require_callback_message(
    cb: CallbackQuery,
    *,
    alert_text: str | None = None,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> Message | None:
    message = callback_message(cb)
    if message is None:
        resolved_alert_text = (
            alert_text
            if alert_text is not None
            else _tx(
                translate,
                CALLBACK_MESSAGE_UNAVAILABLE_KEY,
                locale,
                CALLBACK_MESSAGE_UNAVAILABLE_TEXT,
            )
        )
        await cb.answer(resolved_alert_text, show_alert=True)
        return None
    return message


def is_message_not_modified_error(exc: TelegramBadRequest) -> bool:
    return "message is not modified" in str(exc).lower()


async def edit_message_by_anchor(
    source_message: Message,
    *,
    chat_id: int | None,
    message_id: int | None,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
    not_modified_fallback: Message | None = None,
) -> Message | None:
    if chat_id is None or message_id is None:
        return None

    bot = source_message.bot
    if bot is None:
        return None

    try:
        edited = await bot.edit_message_text(
            text=text,
            chat_id=chat_id,
            message_id=message_id,
            reply_markup=reply_markup,
        )
    except TelegramBadRequest as exc:
        if is_message_not_modified_error(exc):
            return not_modified_fallback
        return None
    if isinstance(edited, Message):
        return edited
    return None


async def edit_or_answer(
    message: Message,
    *,
    text: str,
    reply_markup: InlineKeyboardMarkup | None,
    prefer_edit: bool,
) -> Message:
    if prefer_edit:
        try:
            edited = await message.edit_text(text, reply_markup=reply_markup)
            if isinstance(edited, Message):
                return edited
        except TelegramBadRequest:
            pass
    return await message.answer(text, reply_markup=reply_markup)
