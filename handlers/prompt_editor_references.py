from __future__ import annotations

import logging
import uuid
from io import BytesIO

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from comfyui_client import ComfyUIClient
from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates
from core.ui import MAX_REFERENCE_IMAGES

logger = logging.getLogger(__name__)


def make_reference_image(file_id: str) -> dict[str, str]:
    return {"id": uuid.uuid4().hex, "file_id": file_id}


async def download_reference_image(message: Message, file_id: str) -> bytes:
    bot = message.bot
    if bot is None:
        raise RuntimeError("Bot недоступен в текущем сообщении")

    buffer = BytesIO()
    stream = await bot.download(file_id, destination=buffer)
    if stream is None:
        raise RuntimeError("Не удалось загрузить файл из Telegram")
    return buffer.getvalue()


async def collect_reference_images(
    message: Message,
    refs: list[dict[str, str]],
) -> tuple[list[bytes], int]:
    images: list[bytes] = []
    failed = 0

    for ref in refs:
        file_id = str(ref.get("file_id", "")).strip()
        if not file_id:
            failed += 1
            continue
        try:
            images.append(await download_reference_image(message, file_id))
        except Exception:
            logger.exception("Failed to download reference image")
            failed += 1

    return images, failed


def denoise_from_reference_strength(strength: float) -> float:
    clamped = max(0.0, min(2.0, strength))
    return max(0.35, min(0.9, 0.9 - 0.25 * clamped))


def add_reference_from_message(
    message: Message,
    refs: list[dict[str, str]],
) -> tuple[int, bool]:
    added = 0
    hit_limit = False

    def add_file_id(file_id: str) -> None:
        nonlocal added, hit_limit
        if len(refs) >= MAX_REFERENCE_IMAGES:
            hit_limit = True
            return
        refs.append(make_reference_image(file_id))
        added += 1

    if message.photo:
        add_file_id(message.photo[-1].file_id)

    document = message.document
    if document and (document.mime_type or "").startswith("image/"):
        add_file_id(document.file_id)

    return added, hit_limit


async def show_reference_menu(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    runtime: RuntimeStore,
    client: ComfyUIClient,
    edit: bool = True,
    notice: str = "",
) -> None:
    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer("Активный запрос не найден. Используйте /generate.")
        return

    refs = req.params.reference_images
    ipadapter_icon = "✅" if client.supports_ipadapter() else "❌"
    lines: list[str] = []
    if notice:
        lines.append(f"💬 <i>{h(notice)}</i>")
        lines.append("")
    lines.extend(
        [
            f"🖼 <b>Референс-картинки</b>  {len(refs)}/{MAX_REFERENCE_IMAGES}",
            "",
            "Отправьте фото в чат для добавления. Можно альбомом.",
            f"IP-Adapter: {ipadapter_icon} {'доступен' if client.supports_ipadapter() else 'недоступен'}",
        ]
    )

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👁 Превью",
                    callback_data="pe:refs:view",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➖ Удалить последнюю",
                    callback_data="pe:refs:remove_last",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Очистить все",
                    callback_data="pe:refs:clear",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data="pe:back",
                )
            ],
        ]
    )

    text = "\n".join(lines)
    if req.ui_chat_id is not None and req.ui_message_id is not None and message.bot:
        try:
            edited = await message.bot.edit_message_text(
                text=text,
                chat_id=req.ui_chat_id,
                message_id=req.ui_message_id,
                reply_markup=kb,
            )
            if isinstance(edited, Message):
                req.ui_chat_id = edited.chat.id
                req.ui_message_id = edited.message_id
                runtime.user_ui_panels[uid] = {
                    "chat_id": edited.chat.id,
                    "message_id": edited.message_id,
                }
                await state.set_state(PromptEditorStates.editing)
                return
        except Exception:
            pass

    if edit:
        try:
            edited = await message.edit_text(text, reply_markup=kb)
            if isinstance(edited, Message):
                req.ui_chat_id = edited.chat.id
                req.ui_message_id = edited.message_id
                runtime.user_ui_panels[uid] = {
                    "chat_id": edited.chat.id,
                    "message_id": edited.message_id,
                }
                await state.set_state(PromptEditorStates.editing)
                return
        except Exception:
            pass

    sent = await message.answer(text, reply_markup=kb)
    req.ui_chat_id = sent.chat.id
    req.ui_message_id = sent.message_id
    runtime.user_ui_panels[uid] = {
        "chat_id": sent.chat.id,
        "message_id": sent.message_id,
    }
    await state.set_state(PromptEditorStates.editing)
