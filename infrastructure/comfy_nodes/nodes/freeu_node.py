from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class FreeuNode:
    node_id = "freeu"
    phase = 30
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.enhancements.enable_freeu:
            return
        if state.model_out is None:
            raise RuntimeError("FreeU node requires model output")

        freeu_id = state.add_node(
            "FreeU_V2",
            {
                "model": state.model_out,
                "b1": 1.3,
                "b2": 1.4,
                "s1": 0.9,
                "s2": 0.2,
            },
        )
        state.model_out = [freeu_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("FreeU_V2")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(FreeuNode())
