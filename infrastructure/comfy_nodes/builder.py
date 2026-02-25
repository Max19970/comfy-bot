from __future__ import annotations

from typing import Any

from .context import ComfyWorkflowBuildContext
from .contracts import WorkflowStageLabel
from .registry import load_discovered_nodes, load_discovered_nodes_from_packages
from .state import ComfyWorkflowState


def build_workflow_from_nodes(
    context: ComfyWorkflowBuildContext,
    *,
    package_name: str | None = None,
    package_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    state = ComfyWorkflowState(context=context)
    if package_names is not None:
        nodes = load_discovered_nodes_from_packages(package_names)
    elif package_name:
        nodes = load_discovered_nodes(package_name)
    else:
        nodes = load_discovered_nodes()

    for node in nodes:
        labels = node.stage_labels()
        if labels:
            state.add_stage_labels(labels)
        node.apply(state)

    if not state.workflow:
        raise RuntimeError("Comfy workflow node pipeline produced empty workflow")
    return state.workflow


def collect_workflow_stage_labels(*package_names: str) -> dict[str, WorkflowStageLabel]:
    nodes = (
        load_discovered_nodes_from_packages(package_names)
        if package_names
        else load_discovered_nodes()
    )
    labels: dict[str, WorkflowStageLabel] = {}
    for node in nodes:
        labels.update(node.stage_labels())
    return labels
