from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from plugins.loader import (
    HandlerPluginLoaderError,
    load_handler_plugins_from_packages,
    parse_plugin_packages,
)


def _write_plugin_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = textwrap.dedent(content).strip() + "\n"
        file_path.write_text(payload, encoding="utf-8")


def _plugin_module(
    *, plugin_id: str, display_name: str, order: int, api_major: int, api_minor: int
) -> str:
    return f"""
from plugins.contracts import HandlerPluginContext, PluginApiVersion, PluginDescriptor


class _Plugin:
    descriptor = PluginDescriptor(
        plugin_id="{plugin_id}",
        display_name="{display_name}",
        api_version=PluginApiVersion({api_major}, {api_minor}),
        order={order},
    )

    def register(self, context: HandlerPluginContext) -> None:
        context.shared[self.descriptor.plugin_id] = True


def register_plugins(registry):
    registry.register(_Plugin())
"""


def test_parse_plugin_packages_deduplicates_and_trims() -> None:
    assert parse_plugin_packages(
        " handlers.plugins.builtin,  foo.bar, handlers.plugins.builtin , "
    ) == (
        "handlers.plugins.builtin",
        "foo.bar",
    )


def test_load_handler_plugins_from_packages_orders_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_plugins_ordered"
    _write_plugin_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "b_second.py": _plugin_module(
                plugin_id="plugin.z",
                display_name="Second plugin",
                order=200,
                api_major=1,
                api_minor=0,
            ),
            "a_first.py": _plugin_module(
                plugin_id="plugin.a",
                display_name="First plugin",
                order=100,
                api_major=1,
                api_minor=0,
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    plugins = load_handler_plugins_from_packages(f"{package_name}, {package_name}")
    assert [plugin.descriptor.plugin_id for plugin in plugins] == ["plugin.a", "plugin.z"]


def test_load_handler_plugins_rejects_duplicate_plugin_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_plugins_duplicate"
    _write_plugin_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "first.py": _plugin_module(
                plugin_id="plugin.duplicate",
                display_name="Duplicate plugin A",
                order=100,
                api_major=1,
                api_minor=0,
            ),
            "second.py": _plugin_module(
                plugin_id="plugin.duplicate",
                display_name="Duplicate plugin B",
                order=200,
                api_major=1,
                api_minor=0,
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(HandlerPluginLoaderError, match="Duplicate plugin id"):
        load_handler_plugins_from_packages(package_name)


def test_load_handler_plugins_rejects_incompatible_api_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_name = "fixture_plugins_api"
    _write_plugin_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "main.py": _plugin_module(
                plugin_id="plugin.future",
                display_name="Future plugin",
                order=100,
                api_major=1,
                api_minor=1,
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(HandlerPluginLoaderError, match="requires API"):
        load_handler_plugins_from_packages(package_name)
