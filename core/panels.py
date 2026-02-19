from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup, Message

from core.runtime import RuntimeStore


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

    if isinstance(chat_id, int) and isinstance(message_id, int) and message.bot:
        try:
            edited = await message.bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=reply_markup,
            )
            if isinstance(edited, Message):
                remember_user_panel(runtime, uid, edited)
                return edited
        except Exception:
            pass

    if prefer_edit:
        try:
            edited = await message.edit_text(text, reply_markup=reply_markup)
            if isinstance(edited, Message):
                remember_user_panel(runtime, uid, edited)
                return edited
        except Exception:
            pass

    sent = await message.answer(text, reply_markup=reply_markup)
    remember_user_panel(runtime, uid, sent)
    return sent
