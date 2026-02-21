from __future__ import annotations

import logging
import uuid
from io import BytesIO

from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from comfyui_client import ComfyUIClient
from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates
from core.ui import MAX_REFERENCE_IMAGES
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button

from .prompt_editor_session import show_prompt_panel

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
        except (TelegramBadRequest, RuntimeError, OSError):
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

    kb = build_keyboard(
        [
            [button("👁 Превью", "pe:refs:view")],
            [button("➖ Удалить последнюю", "pe:refs:remove_last")],
            [button("🗑 Очистить все", "pe:refs:clear")],
            [back_button("pe:back")],
        ]
    )

    text = "\n".join(lines)
    await show_prompt_panel(runtime, message, req, text, kb, prefer_edit=edit)
    await state.set_state(PromptEditorStates.editing)
