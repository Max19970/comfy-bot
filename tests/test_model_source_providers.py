from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from application.model_source_providers import (
    SOURCE_ALL,
    SOURCE_CIVITAI,
    SOURCE_HUGGINGFACE,
    CivitaiModelSourceProvider,
    HuggingFaceModelSourceProvider,
    ModelSourceProviderRegistry,
    ModelSourceSearchRequest,
    build_default_model_source_registry,
)


@dataclass
class _ProviderStub:
    source: str

    async def search(self, request: ModelSourceSearchRequest) -> list[Any]:
        return [self.source, request.query]

    async def resolve_direct(self, request: ModelSourceSearchRequest) -> list[Any]:
        return [self.source, request.model_type]

    def download_headers(self) -> dict[str, str]:
        return {"X-Source": self.source}


def test_provider_registry_routes_all_sources_in_registration_order() -> None:
    registry = ModelSourceProviderRegistry()
    civitai = registry.register(_ProviderStub(source=SOURCE_CIVITAI))
    huggingface = registry.register(_ProviderStub(source=SOURCE_HUGGINGFACE))

    assert registry.sources() == (SOURCE_CIVITAI, SOURCE_HUGGINGFACE)
    assert registry.resolve(SOURCE_ALL) == (civitai, huggingface)
    assert registry.resolve("civitai") == (civitai,)
    assert registry.resolve("unknown") == ()


def test_provider_registry_rejects_duplicate_and_reserved_sources() -> None:
    registry = ModelSourceProviderRegistry()
    registry.register(_ProviderStub(source=SOURCE_CIVITAI))

    with pytest.raises(ValueError, match="Duplicate model source provider"):
        registry.register(_ProviderStub(source=SOURCE_CIVITAI))

    with pytest.raises(ValueError, match="reserved"):
        registry.register(_ProviderStub(source=SOURCE_ALL))


def test_civitai_provider_maps_search_and_direct_contracts() -> None:
    captured: dict[str, Any] = {}

    async def _search(*args: Any, **kwargs: Any) -> list[str]:
        captured["search_args"] = args
        captured["search_kwargs"] = kwargs
        return ["search-ok"]

    async def _fetch(*args: Any, **kwargs: Any) -> list[str]:
        captured["fetch_args"] = args
        captured["fetch_kwargs"] = kwargs
        return ["direct-ok"]

    provider = CivitaiModelSourceProvider(
        search_fn=_search,
        fetch_model_fn=_fetch,
        extract_model_id=lambda query: "123" if "civitai.com/models/" in query else None,
        api_key="token-civitai",
    )

    request = ModelSourceSearchRequest(
        query="https://civitai.com/models/123",
        model_type="lora",
        limit=12,
        sort="Most Downloaded",
        period="AllTime",
        base_models=("SDXL",),
        include_nsfw=True,
        civitai_authors=("alice", "bob"),
    )

    assert asyncio.run(provider.search(request)) == ["search-ok"]
    assert captured["search_args"] == ("https://civitai.com/models/123", "lora", 12)
    assert captured["search_kwargs"] == {
        "sort": "Most Downloaded",
        "period": "AllTime",
        "base_models": ["SDXL"],
        "include_nsfw": True,
        "authors": ["alice", "bob"],
        "strict_type": True,
    }

    assert asyncio.run(provider.resolve_direct(request)) == ["direct-ok"]
    assert captured["fetch_args"] == ("123",)
    assert captured["fetch_kwargs"] == {"model_type": "lora"}
    assert provider.download_headers() == {"Authorization": "Bearer token-civitai"}


def test_huggingface_provider_maps_search_and_direct_contracts() -> None:
    captured: dict[str, Any] = {}

    async def _search(*args: Any, **kwargs: Any) -> list[str]:
        captured["search_args"] = args
        captured["search_kwargs"] = kwargs
        return ["search-ok"]

    async def _fetch(*args: Any, **kwargs: Any) -> list[str]:
        captured["fetch_args"] = args
        captured["fetch_kwargs"] = kwargs
        return ["direct-ok"]

    provider = HuggingFaceModelSourceProvider(
        search_fn=_search,
        fetch_repo_fn=_fetch,
        extract_repo_id=lambda query: "org/repo" if "huggingface.co/" in query else None,
        token="token-hf",
    )

    request = ModelSourceSearchRequest(
        query="https://huggingface.co/org/repo",
        model_type="checkpoint",
        limit=5,
    )

    assert asyncio.run(provider.search(request)) == ["search-ok"]
    assert captured["search_args"] == ("https://huggingface.co/org/repo", "checkpoint", 5)
    assert captured["search_kwargs"] == {}

    assert asyncio.run(provider.resolve_direct(request)) == ["direct-ok"]
    assert captured["fetch_args"] == ("org/repo",)
    assert captured["fetch_kwargs"] == {"model_type": "checkpoint"}
    assert provider.download_headers() == {"Authorization": "Bearer token-hf"}


def test_build_default_provider_registry_exposes_builtin_sources() -> None:
    async def _search(*_: Any, **__: Any) -> list[Any]:
        return []

    async def _fetch(*_: Any, **__: Any) -> list[Any]:
        return []

    registry = build_default_model_source_registry(
        civitai_search=_search,
        civitai_fetch_model=_fetch,
        civitai_extract_model_id=lambda _: None,
        civitai_api_key="",
        huggingface_search=_search,
        huggingface_fetch_repo=_fetch,
        huggingface_extract_repo_id=lambda _: None,
        huggingface_token="",
    )

    assert registry.sources() == (SOURCE_CIVITAI, SOURCE_HUGGINGFACE)
