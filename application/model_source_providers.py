from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

SOURCE_ALL = "all"
SOURCE_CIVITAI = "civitai"
SOURCE_HUGGINGFACE = "huggingface"

BUILTIN_MODEL_SOURCES = (
    SOURCE_CIVITAI,
    SOURCE_HUGGINGFACE,
)


def normalize_model_source(value: str) -> str:
    return str(value or "").strip().lower()


@dataclass(frozen=True, slots=True)
class ModelSourceSearchRequest:
    query: str
    model_type: str
    limit: int = 8
    sort: str = "Most Downloaded"
    period: str = "AllTime"
    base_models: tuple[str, ...] = ()
    include_nsfw: bool = False
    civitai_authors: tuple[str, ...] = ()
    strict_type: bool = True


@runtime_checkable
class ModelSourceProvider(Protocol):
    source: str

    async def search(self, request: ModelSourceSearchRequest) -> list[Any]: ...

    async def resolve_direct(self, request: ModelSourceSearchRequest) -> list[Any]: ...

    def download_headers(self) -> dict[str, str]: ...


@dataclass(slots=True)
class ModelSourceProviderRegistry:
    _providers: dict[str, ModelSourceProvider] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def register(self, provider: ModelSourceProvider) -> ModelSourceProvider:
        source = normalize_model_source(getattr(provider, "source", ""))
        if not source:
            raise ValueError("Model source provider must define non-empty source")
        if source == SOURCE_ALL:
            raise ValueError(f"'{SOURCE_ALL}' is reserved and cannot be used as provider source")
        if source in self._providers:
            raise ValueError(f"Duplicate model source provider: {source}")
        self._providers[source] = provider
        self._order.append(source)
        return provider

    def get(self, source: str) -> ModelSourceProvider | None:
        return self._providers.get(normalize_model_source(source))

    def sources(self) -> tuple[str, ...]:
        return tuple(self._order)

    def resolve(self, source: str) -> tuple[ModelSourceProvider, ...]:
        selected_source = normalize_model_source(source)
        if selected_source in {"", SOURCE_ALL}:
            return tuple(self._providers[item] for item in self._order)
        provider = self.get(selected_source)
        if provider is None:
            return ()
        return (provider,)


CivitaiSearchCallable = Callable[..., Awaitable[list[Any]]]
CivitaiFetchCallable = Callable[..., Awaitable[list[Any]]]
ExtractCivitaiModelId = Callable[[str], str | None]

HuggingFaceSearchCallable = Callable[..., Awaitable[list[Any]]]
HuggingFaceFetchCallable = Callable[..., Awaitable[list[Any]]]
ExtractHuggingFaceRepoId = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class ModelSourceProviderBuildContext:
    civitai_search: CivitaiSearchCallable
    civitai_fetch_model: CivitaiFetchCallable
    civitai_extract_model_id: ExtractCivitaiModelId
    civitai_api_key: str
    huggingface_search: HuggingFaceSearchCallable
    huggingface_fetch_repo: HuggingFaceFetchCallable
    huggingface_extract_repo_id: ExtractHuggingFaceRepoId
    huggingface_token: str


@dataclass(slots=True)
class CivitaiModelSourceProvider:
    search_fn: CivitaiSearchCallable
    fetch_model_fn: CivitaiFetchCallable
    extract_model_id: ExtractCivitaiModelId
    api_key: str = ""
    source: str = SOURCE_CIVITAI

    async def search(self, request: ModelSourceSearchRequest) -> list[Any]:
        return await self.search_fn(
            request.query,
            request.model_type,
            request.limit,
            sort=request.sort,
            period=request.period,
            base_models=list(request.base_models),
            include_nsfw=request.include_nsfw,
            authors=list(request.civitai_authors),
            strict_type=request.strict_type,
        )

    async def resolve_direct(self, request: ModelSourceSearchRequest) -> list[Any]:
        model_id = self.extract_model_id(request.query)
        if not model_id:
            return []
        return await self.fetch_model_fn(model_id, model_type=request.model_type)

    def download_headers(self) -> dict[str, str]:
        key = str(self.api_key or "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}


@dataclass(slots=True)
class HuggingFaceModelSourceProvider:
    search_fn: HuggingFaceSearchCallable
    fetch_repo_fn: HuggingFaceFetchCallable
    extract_repo_id: ExtractHuggingFaceRepoId
    token: str = ""
    source: str = SOURCE_HUGGINGFACE

    async def search(self, request: ModelSourceSearchRequest) -> list[Any]:
        return await self.search_fn(
            request.query,
            request.model_type,
            request.limit,
        )

    async def resolve_direct(self, request: ModelSourceSearchRequest) -> list[Any]:
        repo_id = self.extract_repo_id(request.query)
        if not repo_id:
            return []
        return await self.fetch_repo_fn(repo_id, model_type=request.model_type)

    def download_headers(self) -> dict[str, str]:
        token = str(self.token or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}


def build_default_model_source_registry(
    *,
    civitai_search: CivitaiSearchCallable,
    civitai_fetch_model: CivitaiFetchCallable,
    civitai_extract_model_id: ExtractCivitaiModelId,
    civitai_api_key: str,
    huggingface_search: HuggingFaceSearchCallable,
    huggingface_fetch_repo: HuggingFaceFetchCallable,
    huggingface_extract_repo_id: ExtractHuggingFaceRepoId,
    huggingface_token: str,
) -> ModelSourceProviderRegistry:
    context = ModelSourceProviderBuildContext(
        civitai_search=civitai_search,
        civitai_fetch_model=civitai_fetch_model,
        civitai_extract_model_id=civitai_extract_model_id,
        civitai_api_key=civitai_api_key,
        huggingface_search=huggingface_search,
        huggingface_fetch_repo=huggingface_fetch_repo,
        huggingface_extract_repo_id=huggingface_extract_repo_id,
        huggingface_token=huggingface_token,
    )
    registry = ModelSourceProviderRegistry()
    register_default_model_source_providers(registry, context)
    return registry


def register_default_model_source_providers(
    registry: ModelSourceProviderRegistry,
    context: ModelSourceProviderBuildContext,
) -> None:
    registry.register(
        CivitaiModelSourceProvider(
            search_fn=context.civitai_search,
            fetch_model_fn=context.civitai_fetch_model,
            extract_model_id=context.civitai_extract_model_id,
            api_key=context.civitai_api_key,
        )
    )
    registry.register(
        HuggingFaceModelSourceProvider(
            search_fn=context.huggingface_search,
            fetch_repo_fn=context.huggingface_fetch_repo,
            extract_repo_id=context.huggingface_extract_repo_id,
            token=context.huggingface_token,
        )
    )
