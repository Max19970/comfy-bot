from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..nodes._stage_labels import stage_labels_for
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState


class UpscaleSaveImageNode:
    node_id = "upscale_save_image"
    phase = 10
    order = 40

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.image_out is None:
            raise RuntimeError("SaveImage node requires image output")
        state.add_node(
            "SaveImage",
            {
                "images": state.image_out,
                "filename_prefix": "ComfyBot",
            },
        )

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("SaveImage")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(UpscaleSaveImageNode())
