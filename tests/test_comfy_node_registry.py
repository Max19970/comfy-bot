from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from infrastructure.comfy_nodes.registry import (
    load_discovered_nodes_from_packages,
    parse_node_packages,
)


def _write_node_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def _node_module(
    *,
    node_id: str,
    phase: int,
    order: int,
    class_type: str,
    localization_key: str,
    default_text: str,
) -> str:
    return f"""
from infrastructure.comfy_nodes.contracts import WorkflowStageLabel


class _Node:
    node_id = "{node_id}"
    phase = {phase}
    order = {order}

    def apply(self, state):
        return None

    def stage_labels(self):
        return {{
            "{class_type}": WorkflowStageLabel(
                localization_key="{localization_key}",
                default_text="{default_text}",
            )
        }}


def register_nodes(registry):
    registry.register(_Node())
"""


def test_parse_node_packages_deduplicates_and_applies_default() -> None:
    assert parse_node_packages("pkg.a, pkg.b, pkg.a") == ("pkg.a", "pkg.b")
    assert parse_node_packages("") == ("infrastructure.comfy_nodes.nodes",)


def test_load_discovered_nodes_from_packages_merges_packages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_node_package(
        tmp_path,
        "fixture_nodes_a",
        {
            "__init__.py": "",
            "a_first.py": _node_module(
                node_id="node.a",
                phase=10,
                order=20,
                class_type="NodeA",
                localization_key="stage.node_a",
                default_text="Node A",
            ),
        },
    )
    _write_node_package(
        tmp_path,
        "fixture_nodes_b",
        {
            "__init__.py": "",
            "main.py": _node_module(
                node_id="node.b",
                phase=20,
                order=10,
                class_type="NodeB",
                localization_key="stage.node_b",
                default_text="Node B",
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    nodes = load_discovered_nodes_from_packages(("fixture_nodes_a", "fixture_nodes_b"))
    assert [node.node_id for node in nodes] == ["node.a", "node.b"]


def test_load_discovered_nodes_from_packages_rejects_duplicate_node_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_node_package(
        tmp_path,
        "fixture_nodes_duplicate",
        {
            "__init__.py": "",
            "first.py": _node_module(
                node_id="node.dup",
                phase=10,
                order=10,
                class_type="NodeA",
                localization_key="stage.node_a",
                default_text="Node A",
            ),
            "second.py": _node_module(
                node_id="node.dup",
                phase=10,
                order=20,
                class_type="NodeB",
                localization_key="stage.node_b",
                default_text="Node B",
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="Duplicate comfy node id"):
        load_discovered_nodes_from_packages(("fixture_nodes_duplicate",))


def test_load_discovered_nodes_from_packages_rejects_conflicting_stage_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_node_package(
        tmp_path,
        "fixture_nodes_stage_conflict",
        {
            "__init__.py": "",
            "first.py": _node_module(
                node_id="node.alpha",
                phase=10,
                order=10,
                class_type="SharedClass",
                localization_key="stage.shared",
                default_text="Shared",
            ),
            "second.py": _node_module(
                node_id="node.beta",
                phase=20,
                order=10,
                class_type="SharedClass",
                localization_key="stage.shared",
                default_text="Shared changed",
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="Conflicting stage label"):
        load_discovered_nodes_from_packages(("fixture_nodes_stage_conflict",))
