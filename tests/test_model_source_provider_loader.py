from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from application.model_source_provider_loader import (
    ModelSourceProviderLoaderError,
    load_model_source_provider_registry,
    parse_model_source_provider_packages,
)
from application.model_source_providers import ModelSourceProviderBuildContext


def _write_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _provider_module(*, source: str) -> str:
    return f"""
class _Provider:
    source = "{source}"

    async def search(self, request):
        return [self.source, request.query]

    async def resolve_direct(self, request):
        return [self.source, request.model_type]

    def download_headers(self):
        return {{"X-Source": self.source}}


def register_providers(registry, context):
    registry.register(_Provider())
"""


def _context() -> ModelSourceProviderBuildContext:
    async def _search(*_: Any, **__: Any) -> list[Any]:
        return []

    async def _fetch(*_: Any, **__: Any) -> list[Any]:
        return []

    return ModelSourceProviderBuildContext(
        civitai_search=_search,
        civitai_fetch_model=_fetch,
        civitai_extract_model_id=lambda _: None,
        civitai_api_key="",
        huggingface_search=_search,
        huggingface_fetch_repo=_fetch,
        huggingface_extract_repo_id=lambda _: None,
        huggingface_token="",
    )


def test_parse_model_source_provider_packages_deduplicates_and_trims() -> None:
    assert parse_model_source_provider_packages(" app.providers, app.extra , app.providers, ") == (
        "app.providers",
        "app.extra",
    )


def test_load_model_source_provider_registry_discovers_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_model_source_plugins"
    _write_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "a_civitai.py": _provider_module(source="civitai"),
            "b_huggingface.py": _provider_module(source="huggingface"),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = load_model_source_provider_registry(
        package_name,
        context=_context(),
    )

    assert registry.sources() == ("civitai", "huggingface")


def test_load_model_source_provider_registry_raises_on_duplicate_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_model_source_duplicate"
    _write_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "first.py": _provider_module(source="dup"),
            "second.py": _provider_module(source="dup"),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModelSourceProviderLoaderError, match="Duplicate model source provider"):
        load_model_source_provider_registry(package_name, context=_context())


def test_load_model_source_provider_registry_uses_default_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_package = "fixture_model_source_fallback"
    _write_package(
        tmp_path,
        fallback_package,
        {
            "__init__.py": "",
            "main.py": _provider_module(source="fallback-source"),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = load_model_source_provider_registry(
        "",
        context=_context(),
        default_package=fallback_package,
    )

    assert registry.sources() == ("fallback-source",)
