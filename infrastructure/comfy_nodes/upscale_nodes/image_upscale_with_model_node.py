from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..nodes._stage_labels import stage_labels_for
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState


class ImageUpscaleWithModelNode:
    node_id = "image_upscale_with_model"
    phase = 10
    order = 30

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.image_out is None:
            raise RuntimeError("ImageUpscaleWithModel node requires image output")
        if state.upscale_model_out is None:
            raise RuntimeError("ImageUpscaleWithModel node requires upscale model output")
        upscale_id = state.add_node(
            "ImageUpscaleWithModel",
            {
                "upscale_model": state.upscale_model_out,
                "image": state.image_out,
            },
        )
        state.image_out = [upscale_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("ImageUpscaleWithModel")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(ImageUpscaleWithModelNode())
