from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class LoraLoaderNode:
    node_id = "lora_loader"
    phase = 10
    order = 30

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.model_out is None or state.clip_out is None:
            raise RuntimeError("LoraLoader node requires model and clip outputs")

        for attachment in state.models.loras:
            lora_id = state.add_node(
                "LoraLoader",
                {
                    "lora_name": attachment.lora_name,
                    "strength_model": attachment.strength_model,
                    "strength_clip": attachment.strength_clip,
                    "model": state.model_out,
                    "clip": state.clip_out,
                },
            )
            state.model_out = [lora_id, 0]
            state.clip_out = [lora_id, 1]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("LoraLoader")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(LoraLoaderNode())
