from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import ModuleType

from .contracts import ComfyWorkflowNode, WorkflowStageLabel

DEFAULT_NODES_PACKAGE = "infrastructure.comfy_nodes.nodes"


@dataclass(slots=True)
class ComfyNodeRegistry:
    _nodes: list[ComfyWorkflowNode] = field(default_factory=list)
    _stage_labels: dict[str, WorkflowStageLabel] = field(default_factory=dict)

    def register(self, node: ComfyWorkflowNode) -> None:
        node_id = str(getattr(node, "node_id", "")).strip()
        if not node_id:
            raise ValueError("Comfy node id must not be empty")
        if any(existing.node_id == node_id for existing in self._nodes):
            raise ValueError(f"Duplicate comfy node id: {node_id}")

        labels = node.stage_labels()
        if not isinstance(labels, dict):
            raise ValueError(f"Node '{node_id}' stage_labels() must return dict")
        for class_type, label in labels.items():
            class_name = str(class_type).strip()
            if not class_name:
                raise ValueError(f"Node '{node_id}' returned empty class_type in stage_labels")
            if not isinstance(label, WorkflowStageLabel):
                raise ValueError(
                    f"Node '{node_id}' returned invalid WorkflowStageLabel for '{class_name}'"
                )
            if not label.localization_key.strip() or not label.default_text.strip():
                raise ValueError(
                    f"Node '{node_id}' returned empty stage label payload for '{class_name}'"
                )
            existing = self._stage_labels.get(class_name)
            if existing is not None and existing != label:
                raise ValueError(f"Conflicting stage label for class type '{class_name}'")
            self._stage_labels[class_name] = label

        self._nodes.append(node)

    def ordered(self) -> list[ComfyWorkflowNode]:
        return sorted(
            self._nodes,
            key=lambda node: (int(node.phase), int(node.order), node.node_id),
        )

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return dict(self._stage_labels)


def parse_node_packages(
    value: str | Iterable[str] | None,
    *,
    default_packages: tuple[str, ...] = (DEFAULT_NODES_PACKAGE,),
) -> tuple[str, ...]:
    if value is None:
        items = list(default_packages)
    elif isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        if not any(items):
            items = list(default_packages)
    else:
        items = [str(item).strip() for item in value]
        if not any(items):
            items = list(default_packages)

    parsed: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        parsed.append(item)
    return tuple(parsed)


def discover_node_modules(package_name: str = DEFAULT_NODES_PACKAGE) -> list[ModuleType]:
    normalized = package_name.strip()
    if not normalized:
        raise ValueError("Comfy node package name must not be empty")

    package = importlib.import_module(normalized)
    modules = [package]
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return modules

    module_names = [
        module_info.name
        for module_info in pkgutil.iter_modules(package_paths, prefix=f"{normalized}.")
    ]
    modules.extend(importlib.import_module(module_name) for module_name in sorted(module_names))
    return modules


def load_discovered_nodes_from_packages(
    package_names: str | Iterable[str] | None,
) -> list[ComfyWorkflowNode]:
    registry = ComfyNodeRegistry()
    for package_name in parse_node_packages(package_names):
        for module in discover_node_modules(package_name):
            register_nodes = getattr(module, "register_nodes", None)
            if callable(register_nodes):
                register_nodes(registry)
    return registry.ordered()


def load_discovered_nodes(package_name: str = DEFAULT_NODES_PACKAGE) -> list[ComfyWorkflowNode]:
    return load_discovered_nodes_from_packages((package_name,))
