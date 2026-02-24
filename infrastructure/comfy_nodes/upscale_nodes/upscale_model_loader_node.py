from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..nodes._stage_labels import stage_labels_for
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState


class UpscaleModelLoaderNode:
    node_id = "upscale_model_loader"
    phase = 10
    order = 20

    def apply(self, state: ComfyWorkflowState) -> None:
        model_name = state.models.upscale_model
        if not model_name:
            raise ValueError("upscale_model is required for upscale workflow")
        model_loader_id = state.add_node("UpscaleModelLoader", {"model_name": model_name})
        state.upscale_model_out = [model_loader_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("UpscaleModelLoader")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(UpscaleModelLoaderNode())
