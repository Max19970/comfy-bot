from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class ClipTextEncodeNode:
    node_id = "clip_text_encode"
    phase = 20
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.clip_out is None:
            raise RuntimeError("CLIPTextEncode node requires clip output")

        positive_text = state.prompt.positive
        negative_text = state.prompt.negative
        if state.models.embedding_name:
            token = f"embedding:{state.models.embedding_name}"
            if negative_text.strip():
                negative_text = f"{negative_text}, {token}"
            else:
                negative_text = token

        pos_id = state.add_node(
            "CLIPTextEncode",
            {
                "text": positive_text,
                "clip": state.clip_out,
            },
        )
        neg_id = state.add_node(
            "CLIPTextEncode",
            {
                "text": negative_text,
                "clip": state.clip_out,
            },
        )
        state.positive_ref = [pos_id, 0]
        state.negative_ref = [neg_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("CLIPTextEncode")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(ClipTextEncodeNode())
