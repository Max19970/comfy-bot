from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class CheckpointLoaderNode:
    node_id = "checkpoint_loader"
    phase = 10
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        ckpt_id = state.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": state.models.checkpoint},
        )
        state.model_out = [ckpt_id, 0]
        state.clip_out = [ckpt_id, 1]
        state.vae_out = [ckpt_id, 2]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("CheckpointLoaderSimple")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(CheckpointLoaderNode())
