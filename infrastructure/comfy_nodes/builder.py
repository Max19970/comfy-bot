from __future__ import annotations

from typing import Any

from .context import ComfyWorkflowBuildContext
from .contracts import WorkflowStageLabel
from .registry import load_discovered_nodes
from .state import ComfyWorkflowState


def build_workflow_from_nodes(context: ComfyWorkflowBuildContext) -> dict[str, Any]:
    state = ComfyWorkflowState(context=context)
    for node in load_discovered_nodes():
        labels = node.stage_labels()
        if labels:
            state.add_stage_labels(labels)
        node.apply(state)

    if not state.workflow:
        raise RuntimeError("Comfy workflow node pipeline produced empty workflow")
    return state.workflow


def collect_workflow_stage_labels() -> dict[str, WorkflowStageLabel]:
    labels: dict[str, WorkflowStageLabel] = {}
    for node in load_discovered_nodes():
        labels.update(node.stage_labels())
    return labels
