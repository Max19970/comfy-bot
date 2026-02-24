from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from types import ModuleType

from .contracts import ComfyWorkflowNode

DEFAULT_NODES_PACKAGE = "infrastructure.comfy_nodes.nodes"


@dataclass(slots=True)
class ComfyNodeRegistry:
    _nodes: list[ComfyWorkflowNode] = field(default_factory=list)

    def register(self, node: ComfyWorkflowNode) -> None:
        if any(existing.node_id == node.node_id for existing in self._nodes):
            raise ValueError(f"Duplicate comfy node id: {node.node_id}")
        self._nodes.append(node)

    def ordered(self) -> list[ComfyWorkflowNode]:
        return sorted(
            self._nodes,
            key=lambda node: (int(node.phase), int(node.order), node.node_id),
        )


def discover_node_modules(package_name: str = DEFAULT_NODES_PACKAGE) -> list[ModuleType]:
    package = importlib.import_module(package_name)
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return []

    module_names = [
        module_info.name
        for module_info in pkgutil.iter_modules(package_paths, prefix=f"{package_name}.")
    ]
    return [importlib.import_module(module_name) for module_name in sorted(module_names)]


def load_discovered_nodes(package_name: str = DEFAULT_NODES_PACKAGE) -> list[ComfyWorkflowNode]:
    registry = ComfyNodeRegistry()
    for module in discover_node_modules(package_name):
        register_nodes = getattr(module, "register_nodes", None)
        if callable(register_nodes):
            register_nodes(registry)
    return registry.ordered()
