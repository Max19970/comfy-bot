from __future__ import annotations

import asyncio
from collections.abc import Mapping

from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardMarkup,
    Message,
)

from core.image_utils import compress_for_photo
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button
from domain.localization import LocalizationService


def _t(
    localization: LocalizationService | None,
    key: str,
    *,
    locale: str | None,
    default: str,
    params: Mapping[str, object] | None = None,
) -> str:
    if localization is None:
        return default
    return localization.t(key, locale=locale, params=params, default=default)


def generation_result_keyboard(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [
                button(
                    _t(
                        localization,
                        "presets.save_button",
                        locale=locale,
                        default="💾 Сохранить как пресет",
                    ),
                    "save_preset",
                )
            ],
            [
                button(
                    _t(
                        localization,
                        "common.menu.generation.new",
                        locale=locale,
                        default="🔄 Новая генерация",
                    ),
                    "send:new",
                ),
                cancel_button("send:cancel", localization=localization, locale=locale),
            ],
            [
                back_button(
                    "menu:root",
                    text=_t(
                        localization,
                        "common.menu.back_to_menu",
                        locale=locale,
                        default="⬅️ В меню",
                    ),
                )
            ],
        ]
    )


def preview_image_keyboard(
    artifact_id: str,
    parent_artifact_id: str | None = None,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    rows = [
        [
            button(
                _t(
                    localization,
                    "prompt_editor.send.button.png",
                    locale=locale,
                    default="📄 PNG без сжатия",
                ),
                f"img:png:{artifact_id}",
            ),
            button(
                _t(
                    localization,
                    "prompt_editor.send.button.enhance",
                    locale=locale,
                    default="✨ Улучшить",
                ),
                f"img:open:{artifact_id}",
            ),
        ],
    ]
    if parent_artifact_id:
        rows.append(
            [
                button(
                    _t(
                        localization,
                        "prompt_editor.send.button.back_to_source",
                        locale=locale,
                        default="↩️ К исходнику",
                    ),
                    f"img:goto_parent:{artifact_id}",
                )
            ]
        )
    rows.append(
        [
            back_button(
                "menu:root",
                text=_t(
                    localization,
                    "common.menu.back_to_menu",
                    locale=locale,
                    default="⬅️ В меню",
                ),
            )
        ]
    )
    return build_keyboard(rows)


async def deliver_generated_images(
    message: Message,
    images: list[bytes],
    *,
    used_seed: int | str,
    mode: str,
    preview_keyboards: list[InlineKeyboardMarkup] | None = None,
    index_offset: int = 0,
    total_count: int | None = None,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[Message]:
    preview_messages: list[Message] = []
    total = total_count if total_count and total_count > 0 else len(images)

    async def _send_photo_with_retry(
        *,
        msg: Message,
        image_bytes: bytes,
        filename: str,
        caption: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message:
        for attempt in range(2):
            try:
                return await msg.answer_photo(
                    photo=BufferedInputFile(image_bytes, filename),
                    caption=caption,
                    reply_markup=reply_markup,
                )
            except TelegramNetworkError:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.35)
        raise RuntimeError("unreachable")

    async def _send_document_with_retry(
        *,
        msg: Message,
        image_bytes: bytes,
        filename: str,
        caption: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message:
        for attempt in range(2):
            try:
                return await msg.answer_document(
                    document=BufferedInputFile(image_bytes, filename),
                    caption=caption,
                    reply_markup=reply_markup,
                )
            except TelegramNetworkError:
                if attempt == 1:
                    raise
                await asyncio.sleep(0.35)
        raise RuntimeError("unreachable")

    for index, img_bytes in enumerate(images):
        caption = _t(
            localization,
            "prompt_editor.send.caption.image_index_seed",
            locale=locale,
            default="🖼 {index}/{total} | Seed: {seed}",
            params={"index": index_offset + index + 1, "total": total, "seed": used_seed},
        )
        preview_kb = None
        if preview_keyboards and index < len(preview_keyboards):
            preview_kb = preview_keyboards[index]
        if mode in ("photo", "both"):
            try:
                compressed = compress_for_photo(img_bytes)
                sent = await _send_photo_with_retry(
                    msg=message,
                    image_bytes=compressed,
                    filename=f"comfy_{index + 1}.jpg",
                    caption=caption,
                    reply_markup=preview_kb,
                )
                preview_messages.append(sent)
            except TelegramBadRequest:
                sent = await _send_document_with_retry(
                    msg=message,
                    image_bytes=img_bytes,
                    filename=f"comfy_{index + 1}.png",
                    caption=_t(
                        localization,
                        "prompt_editor.send.caption.fallback",
                        locale=locale,
                        default="{caption} (fallback)",
                        params={"caption": caption},
                    ),
                    reply_markup=preview_kb,
                )
                preview_messages.append(sent)

        if mode in ("file", "both"):
            await _send_document_with_retry(
                msg=message,
                image_bytes=img_bytes,
                filename=f"comfy_{index + 1}.png",
                caption=_t(
                    localization,
                    "prompt_editor.send.caption.png",
                    locale=locale,
                    default="{caption} (PNG)",
                    params={"caption": caption},
                ),
                reply_markup=None,
            )
    return preview_messages
