from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models import GenerationParams
from infrastructure.comfy_workflow_builder import (
    build_comfy_workflow,
    generation_node_packages,
    set_generation_node_packages,
)


def _write_node_package(tmp_path: Path, package_name: str, files: dict[str, str]) -> None:
    package_dir = tmp_path / package_name
    package_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in files.items():
        file_path = package_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


@dataclass
class _InfoStub:
    ipadapter_apply_node: str = ""
    clip_vision_models: list[str] = field(default_factory=list)
    ipadapter_models: list[str] = field(default_factory=list)


def _defaults(_: str) -> dict[str, Any]:
    return {}


def _select(field_specs: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in field_specs:
            return alias
    return None


def test_build_workflow_loads_dynamic_node_package_without_core_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_name = "fixture_dynamic_comfy_nodes"
    _write_node_package(
        tmp_path,
        package_name,
        {
            "__init__.py": "",
            "dynamic_node.py": """
from infrastructure.comfy_nodes.contracts import WorkflowStageLabel


class _DynamicNode:
    node_id = "dynamic_debug"
    phase = 95
    order = 10

    def apply(self, state):
        state.add_node("DynamicDebug", {"message": "ok"})

    def stage_labels(self):
        return {
            "DynamicDebug": WorkflowStageLabel(
                localization_key="workflow.stage.dynamic_debug",
                default_text="Dynamic debug",
            )
        }


def register_nodes(registry):
    registry.register(_DynamicNode())
""",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    previous = generation_node_packages()
    try:
        set_generation_node_packages(f"infrastructure.comfy_nodes.nodes,{package_name}")
        workflow = build_comfy_workflow(
            GenerationParams(
                positive="portrait",
                negative="blurry",
                checkpoint="sdxl.safetensors",
                seed=123,
            ),
            reference_image_name=None,
            reference_mode="none",
            object_info={},
            info=_InfoStub(),
            required_input_defaults=_defaults,
            select_field_name=_select,
        )

        class_types = {
            node.get("class_type") for node in workflow.values() if isinstance(node, dict)
        }
        assert "DynamicDebug" in class_types
    finally:
        set_generation_node_packages(",".join(previous))
