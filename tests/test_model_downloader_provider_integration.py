from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any

from application.model_downloader import ModelDownloader, SearchResult
from application.model_source_providers import ModelSourceProviderRegistry, ModelSourceSearchRequest
from core.config import Config
from infrastructure.model_source_clients import FileDownloadResult


def _write_provider_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


class _ProviderStub:
    def __init__(self, source: str) -> None:
        self.source = source
        self.search_calls: list[ModelSourceSearchRequest] = []
        self.direct_calls: list[ModelSourceSearchRequest] = []
        self.search_results: list[list[SearchResult]] = []
        self.direct_results: list[list[SearchResult]] = []
        self.headers: dict[str, str] = {}

    async def search(self, request: ModelSourceSearchRequest) -> list[Any]:
        self.search_calls.append(request)
        if self.search_results:
            return self.search_results.pop(0)
        return []

    async def resolve_direct(self, request: ModelSourceSearchRequest) -> list[Any]:
        self.direct_calls.append(request)
        if self.direct_results:
            return self.direct_results.pop(0)
        return []

    def download_headers(self) -> dict[str, str]:
        return dict(self.headers)


def _downloader(tmp_path: Path) -> ModelDownloader:
    cfg = Config(
        telegram_token="test-token",
        comfyui_models_path=str(tmp_path / "models"),
    )
    return ModelDownloader(cfg)


def _search_result(
    *, source: str, model_id: str, downloads: int = 0, rating: float = 0.0
) -> SearchResult:
    return SearchResult(
        name=f"{source}-{model_id}",
        source=source,
        model_id=model_id,
        version_id="v1",
        filename=f"{model_id}.safetensors",
        download_url=f"https://example.com/{model_id}",
        model_type="checkpoint",
        download_count=downloads,
        rating=rating,
    )


def test_search_uses_provider_routing_for_selected_source(tmp_path: Path) -> None:
    downloader = _downloader(tmp_path)

    civitai = _ProviderStub("civitai")
    huggingface = _ProviderStub("huggingface")
    civitai.search_results = [[_search_result(source="civitai", model_id="a", downloads=10)]]

    registry = ModelSourceProviderRegistry()
    registry.register(civitai)
    registry.register(huggingface)
    downloader._source_providers = registry

    results = asyncio.run(
        downloader.search(
            "anime",
            "checkpoint",
            source="civitai",
            limit=8,
            sort="Most Downloaded",
            period="AllTime",
            base_models=["SDXL"],
            include_nsfw=True,
            civitai_authors=["alice"],
        )
    )

    assert [item.source for item in results] == ["civitai"]
    assert len(civitai.search_calls) == 1
    assert len(huggingface.search_calls) == 0
    request = civitai.search_calls[0]
    assert request.base_models == ("SDXL",)
    assert request.civitai_authors == ("alice",)
    assert request.strict_type is True


def test_search_runs_relaxed_pass_when_primary_results_empty(tmp_path: Path) -> None:
    downloader = _downloader(tmp_path)

    civitai = _ProviderStub("civitai")
    civitai.search_results = [
        [],
        [_search_result(source="civitai", model_id="relaxed", downloads=1)],
    ]

    registry = ModelSourceProviderRegistry()
    registry.register(civitai)
    downloader._source_providers = registry

    results = asyncio.run(
        downloader.search(
            "query",
            "lora",
            source="civitai",
            limit=4,
            sort="Highest Rated",
            period="Week",
            base_models=["SD1.5"],
            include_nsfw=True,
            civitai_authors=["bob"],
        )
    )

    assert len(results) == 1
    assert len(civitai.search_calls) == 2
    primary, relaxed = civitai.search_calls
    assert primary.strict_type is True
    assert primary.base_models == ("SD1.5",)
    assert primary.civitai_authors == ("bob",)
    assert relaxed.strict_type is False
    assert relaxed.sort == "Most Downloaded"
    assert relaxed.period == "AllTime"
    assert relaxed.base_models == ()
    assert relaxed.civitai_authors == ()


def test_search_uses_provider_direct_resolution_before_source_search(tmp_path: Path) -> None:
    downloader = _downloader(tmp_path)

    civitai = _ProviderStub("civitai")
    civitai.direct_results = [[_search_result(source="civitai", model_id="direct")]]

    registry = ModelSourceProviderRegistry()
    registry.register(civitai)
    downloader._source_providers = registry

    results = asyncio.run(
        downloader.search(
            "https://civitai.com/models/123",
            "checkpoint",
            source="all",
            limit=3,
        )
    )

    assert len(results) == 1
    assert results[0].model_id == "direct"
    assert len(civitai.direct_calls) == 1
    assert len(civitai.search_calls) == 0


def test_download_model_uses_provider_download_headers(tmp_path: Path) -> None:
    downloader = _downloader(tmp_path)

    provider = _ProviderStub("custom")
    provider.headers = {"Authorization": "Bearer custom-token"}

    registry = ModelSourceProviderRegistry()
    registry.register(provider)
    downloader._source_providers = registry

    captured: dict[str, Any] = {}

    class _RemoteDownloaderStub:
        async def download_to_file(self, **kwargs: Any) -> FileDownloadResult:
            captured.update(kwargs)
            Path(kwargs["tmp_path"]).write_bytes(b"abc")
            return FileDownloadResult(downloaded_bytes=3, total_bytes=3)

    async def _record_metadata(*_: Any, **__: Any) -> None:
        return None

    setattr(downloader, "_remote_file_downloader", _RemoteDownloaderStub())
    setattr(downloader, "_record_download_metadata", _record_metadata)

    result = SearchResult(
        name="custom-model",
        source="custom",
        model_id="m1",
        version_id="v1",
        filename="custom-model.safetensors",
        download_url="https://example.com/custom-model",
        size_bytes=3,
        model_type="checkpoint",
    )

    target_path = asyncio.run(downloader.download_model(result))

    assert captured["headers"] == {"Authorization": "Bearer custom-token"}
    assert Path(target_path).exists()


def test_model_downloader_loads_dynamic_provider_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_name = "fixture_model_source_dynamic"
    _write_provider_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "main.py": """
from application.model_downloader import SearchResult


class _DynamicProvider:
    source = "dynamic"

    async def search(self, request):
        return [
            SearchResult(
                name="Dynamic provider model",
                source=self.source,
                model_id="dynamic/model",
                version_id="v1",
                filename="dynamic.safetensors",
                download_url="https://example.com/dynamic",
                model_type=request.model_type,
                download_count=7,
                rating=1.0,
            )
        ]

    async def resolve_direct(self, request):
        return []

    def download_headers(self):
        return {"X-Provider": "dynamic"}


def register_providers(registry, context):
    registry.register(_DynamicProvider())
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = Config(
        telegram_token="test-token",
        comfyui_models_path=str(tmp_path / "models"),
        model_source_provider_packages=package_name,
    )
    downloader = ModelDownloader(cfg)

    try:
        assert downloader.source_providers.sources() == ("dynamic",)

        results = asyncio.run(
            downloader.search(
                "anything",
                "checkpoint",
                source="dynamic",
                limit=3,
            )
        )
        assert len(results) == 1
        assert results[0].source == "dynamic"
        assert results[0].model_id == "dynamic/model"
    finally:
        asyncio.run(downloader.close())
