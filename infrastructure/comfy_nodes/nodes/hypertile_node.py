from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class HyperTileNode:
    node_id = "hypertile"
    phase = 30
    order = 30

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.enhancements.enable_tiled_diffusion:
            return
        if state.model_out is None:
            raise RuntimeError("HyperTile node requires model output")

        hypertile_id = state.add_node(
            "HyperTile",
            {
                "model": state.model_out,
                "tile_size": state.enhancements.tile_size,
                "swap_size": 2,
                "max_depth": 0,
                "scale_depth": False,
            },
        )
        state.model_out = [hypertile_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("HyperTile")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(HyperTileNode())
