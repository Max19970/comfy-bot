from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..nodes._stage_labels import stage_labels_for
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState


class UpscaleLoadImageNode:
    node_id = "upscale_load_image"
    phase = 10
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        reference_image_name = state.context.reference_image_name
        if not reference_image_name:
            raise ValueError("reference_image_name is required for upscale workflow")
        image_node_id = state.add_node("LoadImage", {"image": reference_image_name})
        state.image_out = [image_node_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("LoadImage")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(UpscaleLoadImageNode())
