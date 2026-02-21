from __future__ import annotations

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardMarkup,
    Message,
)

from core.image_utils import compress_for_photo
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button


def generation_result_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [button("💾 Сохранить как пресет", "save_preset")],
            [button("🔄 Новая генерация", "send:new"), cancel_button("send:cancel")],
            [back_button("menu:root", text="⬅️ В меню")],
        ]
    )


def preview_image_keyboard(
    artifact_id: str,
    parent_artifact_id: str | None = None,
) -> InlineKeyboardMarkup:
    rows = [
        [
            button("📄 PNG без сжатия", f"img:png:{artifact_id}"),
            button("✨ Улучшить", f"img:open:{artifact_id}"),
        ],
    ]
    if parent_artifact_id:
        rows.append([button("↩️ К исходнику", f"img:goto_parent:{artifact_id}")])
    rows.append([back_button("menu:root", text="⬅️ В меню")])
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
) -> list[Message]:
    preview_messages: list[Message] = []
    total = total_count if total_count and total_count > 0 else len(images)
    for index, img_bytes in enumerate(images):
        caption = f"🖼 {index_offset + index + 1}/{total} | Seed: {used_seed}"
        preview_kb = None
        if preview_keyboards and index < len(preview_keyboards):
            preview_kb = preview_keyboards[index]
        if mode in ("photo", "both"):
            try:
                compressed = compress_for_photo(img_bytes)
                sent = await message.answer_photo(
                    photo=BufferedInputFile(compressed, f"comfy_{index + 1}.jpg"),
                    caption=caption,
                    reply_markup=preview_kb,
                )
                preview_messages.append(sent)
            except TelegramBadRequest:
                sent = await message.answer_document(
                    document=BufferedInputFile(img_bytes, f"comfy_{index + 1}.png"),
                    caption=f"{caption} (fallback)",
                    reply_markup=preview_kb,
                )
                preview_messages.append(sent)

        if mode in ("file", "both"):
            await message.answer_document(
                document=BufferedInputFile(img_bytes, f"comfy_{index + 1}.png"),
                caption=f"{caption} (PNG)",
            )
    return preview_messages
