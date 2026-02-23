from __future__ import annotations

from typing import Any

from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from core.callbacks import ValueSelectionCallback


def parse_callback_data(data: str | None, *, prefix: str) -> str | None:
    parsed = ValueSelectionCallback.parse(data or "", prefix=prefix)
    if parsed is None:
        return None
    return parsed.value


def overwrite_state_reset(*, save_source: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "preset_overwrite_name": None,
        "preset_overwrite_params": None,
        "preset_overwrite_source": None,
        "preset_overwrite_input_message_id": None,
        "preset_overwrite_prompt_message_id": None,
        "preset_overwrite_chat_id": None,
    }
    if save_source is not None:
        payload["preset_save_source"] = save_source
    return payload


def overwrite_state_prepare(
    *,
    save_source: str,
    overwrite_source: str,
    preset_name: str,
    preset_params: dict[str, Any],
) -> dict[str, Any]:
    payload = overwrite_state_reset(save_source=save_source)
    payload.update(
        {
            "preset_overwrite_name": preset_name,
            "preset_overwrite_params": preset_params,
            "preset_overwrite_source": overwrite_source,
        }
    )
    return payload


async def remember_overwrite_messages(
    state: FSMContext,
    *,
    input_message: Message,
    confirm_message: Message,
) -> None:
    await state.update_data(
        preset_overwrite_input_message_id=input_message.message_id,
        preset_overwrite_prompt_message_id=confirm_message.message_id,
        preset_overwrite_chat_id=input_message.chat.id,
    )


async def cleanup_overwrite_messages(
    source_message: Message,
    state_data: dict[str, object],
) -> None:
    bot = source_message.bot
    if bot is None:
        return

    chat_id = state_data.get("preset_overwrite_chat_id")
    if not isinstance(chat_id, int):
        chat_id = source_message.chat.id

    ids_to_delete: list[int] = []
    for key in (
        "preset_overwrite_input_message_id",
        "preset_overwrite_prompt_message_id",
    ):
        value = state_data.get(key)
        if isinstance(value, int):
            ids_to_delete.append(value)

    for message_id in set(ids_to_delete):
        try:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
        except TelegramBadRequest:
            pass
