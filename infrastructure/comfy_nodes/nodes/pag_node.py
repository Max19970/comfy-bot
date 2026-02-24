from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class PagNode:
    node_id = "pag"
    phase = 30
    order = 20

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.enhancements.enable_pag:
            return
        if state.model_out is None:
            raise RuntimeError("PAG node requires model output")

        pag_id = state.add_node(
            "PerturbedAttentionGuidance",
            {
                "model": state.model_out,
                "scale": state.enhancements.pag_scale,
            },
        )
        state.model_out = [pag_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("PerturbedAttentionGuidance")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(PagNode())
