from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class UpscaleNode:
    node_id = "upscale"
    phase = 50
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.models.upscale_model:
            return
        if state.image_out is None:
            raise RuntimeError("Upscale node requires image output")

        upscale_loader_id = state.add_node(
            "UpscaleModelLoader",
            {
                "model_name": state.models.upscale_model,
            },
        )
        upscale_id = state.add_node(
            "ImageUpscaleWithModel",
            {
                "upscale_model": [upscale_loader_id, 0],
                "image": state.image_out,
            },
        )
        state.image_out = [upscale_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("UpscaleModelLoader", "ImageUpscaleWithModel")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(UpscaleNode())
