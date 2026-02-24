from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class DecodeNode:
    node_id = "decode"
    phase = 40
    order = 30

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.sampler_id is None:
            raise RuntimeError("Decode node requires sampler output")
        if state.vae_out is None:
            raise RuntimeError("Decode node requires VAE output")

        if state.enhancements.enable_tiled_diffusion:
            decode_id = state.add_node(
                "VAEDecodeTiled",
                {
                    "samples": [state.sampler_id, 0],
                    "vae": state.vae_out,
                    "tile_size": state.enhancements.vae_tile_size,
                    "overlap": state.enhancements.tile_overlap,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            )
        else:
            decode_id = state.add_node(
                "VAEDecode",
                {
                    "samples": [state.sampler_id, 0],
                    "vae": state.vae_out,
                },
            )
        state.image_out = [decode_id, 0]

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("VAEDecode", "VAEDecodeTiled")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(DecodeNode())
