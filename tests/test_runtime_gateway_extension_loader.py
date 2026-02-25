from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from core.runtime import RuntimeStore
from core.runtime_gateway_extension_loader import (
    RuntimeGatewayExtensionLoaderError,
    load_runtime_gateway_extensions,
    parse_runtime_gateway_extension_packages,
)
from core.runtime_gateways import create_runtime_gateways


def _write_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_parse_runtime_gateway_extension_packages_deduplicates_and_trims() -> None:
    assert parse_runtime_gateway_extension_packages(
        " app.runtime_ext, app.extra , app.runtime_ext, "
    ) == ("app.runtime_ext", "app.extra")


def test_load_runtime_gateway_extensions_registers_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_runtime_gateway_plugins"
    _write_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "main.py": """
def register_runtime_gateway_extensions(gateways, runtime):
    gateways.register_extension("fixture.metrics", {"runtime_id": id(runtime)})
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    runtime = RuntimeStore()
    gateways = create_runtime_gateways(runtime)
    loaded_packages = load_runtime_gateway_extensions(package_name, gateways=gateways)

    assert loaded_packages == (package_name,)
    extension = gateways.get_extension("fixture.metrics")
    assert extension == {"runtime_id": id(runtime)}


def test_load_runtime_gateway_extensions_uses_default_fallback_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_package = "fixture_runtime_gateway_fallback"
    _write_package(
        tmp_path,
        fallback_package,
        {
            "__init__.py": "",
            "fallback.py": """
def register_runtime_gateway_extensions(gateways, runtime):
    gateways.register_extension("fixture.fallback", {"ok": True})
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    gateways = create_runtime_gateways(RuntimeStore())
    loaded_packages = load_runtime_gateway_extensions(
        "",
        gateways=gateways,
        default_package=fallback_package,
    )

    assert loaded_packages == (fallback_package,)
    assert gateways.get_extension("fixture.fallback") == {"ok": True}


def test_load_runtime_gateway_extensions_raises_for_non_callable_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_runtime_gateway_invalid"
    _write_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "invalid.py": "register_runtime_gateway_extensions = 42",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    gateways = create_runtime_gateways(RuntimeStore())
    with pytest.raises(RuntimeGatewayExtensionLoaderError, match="non-callable"):
        load_runtime_gateway_extensions(package_name, gateways=gateways)
