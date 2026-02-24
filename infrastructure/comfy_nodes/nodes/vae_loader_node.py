from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class VaeLoaderNode:
    node_id = "vae_loader"
    phase = 10
    order = 20

    def apply(self, state: ComfyWorkflowState) -> None:
        if not state.models.vae_name:
            return
        vae_loader_id = state.add_node("VAELoader", {"vae_name": state.models.vae_name})
        state.vae_out = [vae_loader_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("VAELoader")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(VaeLoaderNode())
