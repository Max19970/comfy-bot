from __future__ import annotations

from aiogram.types import CallbackQuery, Message


def message_user_id(msg: Message) -> int:
    return msg.from_user.id if msg.from_user else 0


def callback_user_id(cb: CallbackQuery) -> int:
    return cb.from_user.id if cb.from_user else 0
