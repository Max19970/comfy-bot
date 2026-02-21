from __future__ import annotations

import logging

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, Message

from core.interaction import edit_message_by_anchor, is_message_not_modified_error
from core.runtime import RuntimeStore

logger = logging.getLogger(__name__)


def remember_user_panel(runtime: RuntimeStore, uid: int, panel: Message) -> None:
    if panel.chat is None:
        return
    runtime.user_ui_panels[uid] = {
        "chat_id": panel.chat.id,
        "message_id": panel.message_id,
    }


async def render_user_panel(
    message: Message,
    runtime: RuntimeStore,
    uid: int,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
    prefer_edit: bool = True,
) -> Message:
    anchor = runtime.user_ui_panels.get(uid, {})
    chat_id = anchor.get("chat_id")
    message_id = anchor.get("message_id")

    if isinstance(chat_id, int) and isinstance(message_id, int):
        edited_by_anchor = await edit_message_by_anchor(
            message,
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=reply_markup,
            not_modified_fallback=message,
        )
        if edited_by_anchor is not None:
            remember_user_panel(runtime, uid, edited_by_anchor)
            return edited_by_anchor

    if prefer_edit:
        try:
            edited = await message.edit_text(text, reply_markup=reply_markup)
            if isinstance(edited, Message):
                remember_user_panel(runtime, uid, edited)
                return edited
        except TelegramBadRequest as exc:
            if is_message_not_modified_error(exc):
                remember_user_panel(runtime, uid, message)
                return message
            logger.debug("Failed to edit current panel", exc_info=True)

    sent = await message.answer(text, reply_markup=reply_markup)
    remember_user_panel(runtime, uid, sent)
    return sent
