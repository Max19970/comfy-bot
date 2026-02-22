from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

from core.models import GenerationParams
from core.runtime import PromptRequest
from handlers.prompt_editor_handler_guards import require_message_and_request


class _FakeMessage:
    async def edit_text(self, *args, **kwargs):
        del args, kwargs
        return self


@dataclass
class _FakeCallback:
    message: Any
    answered: int = 0

    async def answer(self, *args, **kwargs):
        del args, kwargs
        self.answered += 1


def _request() -> PromptRequest:
    return PromptRequest(params=GenerationParams())


def test_require_message_and_request_returns_context_when_available() -> None:
    cb = _FakeCallback(message=_FakeMessage())

    async def _require(_cb):
        del _cb
        return (42, _request())

    result = asyncio.run(
        require_message_and_request(
            cast(Any, cb),
            require_prompt_request_for_callback=_require,
        )
    )

    assert result is not None
    message, uid, req = result
    assert isinstance(message, _FakeMessage)
    assert uid == 42
    assert isinstance(req, PromptRequest)


def test_require_message_and_request_stops_when_request_missing() -> None:
    cb = _FakeCallback(message=_FakeMessage())

    async def _require(_cb):
        del _cb
        return None

    result = asyncio.run(
        require_message_and_request(
            cast(Any, cb),
            require_prompt_request_for_callback=_require,
        )
    )

    assert result is None
    assert cb.answered == 0


def test_require_message_and_request_stops_when_message_missing() -> None:
    cb = _FakeCallback(message=None)

    async def _require(_cb):
        del _cb
        return (7, _request())

    result = asyncio.run(
        require_message_and_request(
            cast(Any, cb),
            require_prompt_request_for_callback=_require,
        )
    )

    assert result is None
    assert cb.answered == 1
