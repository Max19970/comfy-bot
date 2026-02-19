from __future__ import annotations

from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from core.image_utils import compress_for_photo


def generation_result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="💾 Сохранить как пресет",
                    callback_data="save_preset",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🔄 Новая генерация",
                    callback_data="send:new",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="send:cancel",
                ),
            ],
        ]
    )


def preview_image_keyboard(artifact_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="📄 Отправить PNG",
                    callback_data=f"img:png:{artifact_id}",
                ),
                InlineKeyboardButton(
                    text="✨ Улучшить",
                    callback_data=f"img:open:{artifact_id}",
                ),
            ]
        ]
    )


async def deliver_generated_images(
    message: Message,
    images: list[bytes],
    *,
    used_seed: int | str,
    mode: str,
    preview_keyboards: list[InlineKeyboardMarkup] | None = None,
) -> None:
    for index, img_bytes in enumerate(images):
        caption = f"🖼 {index + 1}/{len(images)} | Seed: {used_seed}"
        preview_kb = None
        if preview_keyboards and index < len(preview_keyboards):
            preview_kb = preview_keyboards[index]
        if mode in ("photo", "both"):
            try:
                compressed = compress_for_photo(img_bytes)
                await message.answer_photo(
                    photo=BufferedInputFile(compressed, f"comfy_{index + 1}.jpg"),
                    caption=caption,
                    reply_markup=preview_kb,
                )
            except TelegramBadRequest:
                await message.answer_document(
                    document=BufferedInputFile(img_bytes, f"comfy_{index + 1}.png"),
                    caption=f"{caption} (fallback)",
                    reply_markup=preview_kb,
                )

        if mode in ("file", "both"):
            await message.answer_document(
                document=BufferedInputFile(img_bytes, f"comfy_{index + 1}.png"),
                caption=f"{caption} (PNG)",
            )
