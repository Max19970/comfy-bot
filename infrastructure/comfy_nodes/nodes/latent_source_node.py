from __future__ import annotations

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class LatentSourceNode:
    node_id = "latent_source"
    phase = 40
    order = 10

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.vae_out is None:
            raise RuntimeError("LatentSource node requires VAE output")

        width = (state.image.width // 8) * 8 or 8
        height = (state.image.height // 8) * 8 or 8
        state.base_width = width
        state.base_height = height

        if state.context.reference_mode == "img2img" and state.context.reference_image_name:
            load_image_id = state.add_node(
                "LoadImage",
                {"image": state.context.reference_image_name},
            )
            if state.enhancements.enable_tiled_diffusion:
                latent_id = state.add_node(
                    "VAEEncodeTiled",
                    {
                        "pixels": [load_image_id, 0],
                        "vae": state.vae_out,
                        "tile_size": state.enhancements.vae_tile_size,
                        "overlap": state.enhancements.tile_overlap,
                        "temporal_size": 64,
                        "temporal_overlap": 8,
                    },
                )
            else:
                latent_id = state.add_node(
                    "VAEEncode",
                    {
                        "pixels": [load_image_id, 0],
                        "vae": state.vae_out,
                    },
                )
            if state.sampling.batch_size > 1:
                latent_id = state.add_node(
                    "RepeatLatentBatch",
                    {
                        "samples": [latent_id, 0],
                        "amount": state.sampling.batch_size,
                    },
                )
            state.latent_id = latent_id
            return

        state.latent_id = state.add_node(
            "EmptyLatentImage",
            {
                "width": width,
                "height": height,
                "batch_size": state.sampling.batch_size,
            },
        )

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for(
            "LoadImage",
            "VAEEncode",
            "VAEEncodeTiled",
            "RepeatLatentBatch",
            "EmptyLatentImage",
        )


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(LatentSourceNode())
