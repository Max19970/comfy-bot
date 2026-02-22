from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast

from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from PIL import Image

from handlers.prompt_editor_send import deliver_generated_images


def _png_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


@dataclass
class _FakeChat:
    id: int


@dataclass
class _FakeSentMessage:
    chat: _FakeChat
    message_id: int


class _FakeMessage:
    def __init__(self) -> None:
        self.chat = _FakeChat(id=100)
        self._next_message_id = 1
        self.photo_calls = 0
        self.document_calls = 0
        self.fail_photo_first_with_network = False
        self.force_photo_bad_request = False
        self.fail_document_first_with_network = False

    async def answer_photo(self, *, photo, caption, reply_markup=None):
        del photo, caption, reply_markup
        self.photo_calls += 1
        if self.force_photo_bad_request:
            raise TelegramBadRequest(method=cast(Any, None), message="bad photo")
        if self.fail_photo_first_with_network and self.photo_calls == 1:
            raise TelegramNetworkError(method=cast(Any, None), message="flaky network")
        msg = _FakeSentMessage(chat=self.chat, message_id=self._next_message_id)
        self._next_message_id += 1
        return msg

    async def answer_document(self, *, document, caption, reply_markup=None):
        del document, caption, reply_markup
        self.document_calls += 1
        if self.fail_document_first_with_network and self.document_calls == 1:
            raise TelegramNetworkError(method=cast(Any, None), message="flaky network")
        msg = _FakeSentMessage(chat=self.chat, message_id=self._next_message_id)
        self._next_message_id += 1
        return msg


def test_deliver_generated_images_retries_photo_on_network_error() -> None:
    message = _FakeMessage()
    message.fail_photo_first_with_network = True

    sent = asyncio.run(
        deliver_generated_images(
            cast(Any, message),
            [_png_bytes()],
            used_seed=123,
            mode="photo",
        )
    )

    assert message.photo_calls == 2
    assert len(sent) == 1
    assert sent[0].chat.id == 100


def test_deliver_generated_images_retries_document_fallback_on_network_error() -> None:
    message = _FakeMessage()
    message.force_photo_bad_request = True
    message.fail_document_first_with_network = True

    sent = asyncio.run(
        deliver_generated_images(
            cast(Any, message),
            [_png_bytes()],
            used_seed=321,
            mode="photo",
        )
    )

    assert message.photo_calls == 1
    assert message.document_calls == 2
    assert len(sent) == 1
    assert sent[0].chat.id == 100
