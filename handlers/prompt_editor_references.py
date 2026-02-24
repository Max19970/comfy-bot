from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from io import BytesIO

from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates
from core.ui import MAX_REFERENCE_IMAGES
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button
from infrastructure.comfyui_client import ComfyUIClient

from .prompt_editor_session import show_prompt_panel

logger = logging.getLogger(__name__)

TranslateText = Callable[[str, str | None, str], str]


def _tx(translate: TranslateText | None, key: str, locale: str | None, default: str) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


def make_reference_image(file_id: str) -> dict[str, str]:
    return {"id": uuid.uuid4().hex, "file_id": file_id}


async def download_reference_image(
    message: Message,
    file_id: str,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> bytes:
    bot = message.bot
    if bot is None:
        raise RuntimeError(
            _tx(
                translate,
                "prompt_editor.references.error.bot_unavailable",
                locale,
                "Bot недоступен в текущем сообщении",
            )
        )

    buffer = BytesIO()
    stream = await bot.download(file_id, destination=buffer)
    if stream is None:
        raise RuntimeError(
            _tx(
                translate,
                "prompt_editor.references.error.telegram_download_failed",
                locale,
                "Не удалось загрузить файл из Telegram",
            )
        )
    return buffer.getvalue()


async def collect_reference_images(
    message: Message,
    refs: list[dict[str, str]],
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> tuple[list[bytes], int]:
    images: list[bytes] = []
    failed = 0

    for ref in refs:
        file_id = str(ref.get("file_id", "")).strip()
        if not file_id:
            failed += 1
            continue
        try:
            images.append(
                await download_reference_image(
                    message,
                    file_id,
                    translate=translate,
                    locale=locale,
                )
            )
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
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> None:
    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer(
            _tx(
                translate,
                "prompt_editor.references.error.active_request_not_found",
                locale,
                "Активный запрос не найден. Используйте /generate.",
            )
        )
        return

    refs = req.params.reference_images
    ipadapter_icon = "✅" if client.supports_ipadapter() else "❌"
    lines: list[str] = []
    if notice:
        lines.append(f"💬 <i>{h(notice)}</i>")
        lines.append("")
    lines.extend(
        [
            (
                _tx(
                    translate,
                    "prompt_editor.references.menu.title",
                    locale,
                    "🖼 <b>Референс-картинки</b>",
                )
                + f"  {len(refs)}/{MAX_REFERENCE_IMAGES}"
            ),
            "",
            _tx(
                translate,
                "prompt_editor.references.menu.send_photo_hint",
                locale,
                "Отправьте фото в чат для добавления. Можно альбомом.",
            ),
            (
                _tx(
                    translate,
                    "prompt_editor.references.menu.ipadapter_prefix",
                    locale,
                    "IP-Adapter:",
                )
                + f" {ipadapter_icon} "
                + _tx(
                    translate,
                    (
                        "prompt_editor.references.menu.ipadapter_available"
                        if client.supports_ipadapter()
                        else "prompt_editor.references.menu.ipadapter_unavailable"
                    ),
                    locale,
                    "доступен" if client.supports_ipadapter() else "недоступен",
                )
            ),
        ]
    )

    kb = build_keyboard(
        [
            [
                button(
                    _tx(
                        translate,
                        "prompt_editor.references.button.preview",
                        locale,
                        "👁 Превью",
                    ),
                    "pe:refs:view",
                )
            ],
            [
                button(
                    _tx(
                        translate,
                        "prompt_editor.references.button.remove_last",
                        locale,
                        "➖ Удалить последнюю",
                    ),
                    "pe:refs:remove_last",
                )
            ],
            [
                button(
                    _tx(
                        translate,
                        "prompt_editor.references.button.clear_all",
                        locale,
                        "🗑 Очистить все",
                    ),
                    "pe:refs:clear",
                )
            ],
            [
                back_button(
                    "pe:back",
                    text=_tx(translate, "common.action.back", locale, "⬅️ Назад"),
                )
            ],
        ]
    )

    text = "\n".join(lines)
    await show_prompt_panel(runtime, message, req, text, kb, prefer_edit=edit)
    await state.set_state(PromptEditorStates.editing)
