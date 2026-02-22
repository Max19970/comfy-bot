from __future__ import annotations

import asyncio
from typing import Any

from config import Config
from smart_prompt import SmartPromptService


class _FakeTipoBackend:
    def __init__(
        self,
        *,
        missing: list[str] | None = None,
        backend_error: str = "",
        supports_min_p: bool = True,
        format_prompt: str = "cat, watermark, <|meta|>fox",
    ) -> None:
        self._missing = list(missing or [])
        self._backend_error = backend_error
        self._supports_min_p = supports_min_p
        self._ready = False
        self._format_prompt = format_prompt

        self.closed = False
        self.seed_tags: list[str] = []
        self.last_ban_tags: list[str] = []
        self.last_runner_kwargs: dict[str, Any] = {}

    def missing_dependencies(self) -> list[str]:
        return list(self._missing)

    def is_ready(self) -> bool:
        return self._ready

    @property
    def backend_error(self) -> str:
        return self._backend_error

    @property
    def supports_min_p(self) -> bool:
        return self._supports_min_p

    async def ensure_loaded(self) -> None:
        self._ready = True

    def ensure_loaded_sync(self) -> None:
        self._ready = True

    async def close(self) -> None:
        self.closed = True
        self._ready = False

    def separate_tags(self, tags: list[str]) -> Any:
        self.seed_tags = list(tags)
        return {"general": tags}

    def parse_request(
        self,
        *,
        tag_map: Any,
        nl_prompt: str,
        want_nl: bool,
        tag_length: str,
        nl_length: str,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        del tag_map, want_nl, tag_length, nl_length
        return {}, ["op"], ["general"], nl_prompt

    def set_ban_tags(self, tags: list[str]) -> None:
        self.last_ban_tags = list(tags)

    def run(
        self,
        meta: dict[str, Any],
        operations: Any,
        general: Any,
        parsed_nl: Any,
        *,
        runner_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        del meta, operations, general, parsed_nl
        self.last_runner_kwargs = dict(runner_kwargs)
        return {"quality": ["masterpiece"]}, None

    def apply_format(self, result_map: dict[str, Any], format_key: str) -> str:
        del result_map, format_key
        return self._format_prompt


def _cfg(**kwargs: Any) -> Config:
    base = Config(telegram_token="token")
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def test_generate_prompts_uses_backend_and_preserves_prompt_hygiene() -> None:
    backend = _FakeTipoBackend(format_prompt="cat, watermark, <|meta|>fox")
    service = SmartPromptService(
        _cfg(
            smart_prompt_provider="tipo",
            smart_prompt_ban_tags="watermark, signature",
            smart_prompt_negative_base="bad quality, blurry",
        ),
        backend=backend,
    )

    result = asyncio.run(
        service.generate_prompts(
            description="кот в лесу",
            checkpoint="pony-model.safetensors",
        )
    )

    assert "watermark" not in result.positive.casefold()
    assert "fox" in result.positive.casefold()
    assert "bad quality" in result.negative.casefold()
    assert "signature" in result.negative.casefold()
    assert "min_p" in backend.last_runner_kwargs
    assert backend.last_ban_tags == ["watermark", "signature"]
    assert backend.seed_tags


def test_generate_prompts_skips_min_p_when_backend_does_not_support_it() -> None:
    backend = _FakeTipoBackend(supports_min_p=False, format_prompt="cat, fox")
    service = SmartPromptService(_cfg(smart_prompt_provider="tipo"), backend=backend)

    result = asyncio.run(
        service.generate_prompts(
            description="cat",
            checkpoint="sdxl-model.safetensors",
        )
    )

    assert result.positive == "cat, fox"
    assert "min_p" not in backend.last_runner_kwargs


def test_configuration_hint_reflects_missing_dependencies_and_backend_errors() -> None:
    missing_backend = _FakeTipoBackend(missing=["torch", "tipo-kgen"])
    service_with_missing = SmartPromptService(
        _cfg(smart_prompt_provider="tipo"),
        backend=missing_backend,
    )
    assert "pip install torch tipo-kgen" in service_with_missing.configuration_hint()

    broken_backend = _FakeTipoBackend(backend_error="backend-failed")
    service_with_error = SmartPromptService(
        _cfg(smart_prompt_provider="tipo"),
        backend=broken_backend,
    )
    assert service_with_error.configuration_hint() == "backend-failed"


def test_close_delegates_to_backend() -> None:
    backend = _FakeTipoBackend()
    service = SmartPromptService(_cfg(smart_prompt_provider="tipo"), backend=backend)

    asyncio.run(service.close())

    assert backend.closed is True
