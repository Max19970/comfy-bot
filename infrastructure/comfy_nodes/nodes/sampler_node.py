from __future__ import annotations

from typing import Any

from ..contracts import WorkflowStageLabel
from ..registry import ComfyNodeRegistry
from ..state import ComfyWorkflowState
from ._stage_labels import stage_labels_for


class SamplerNode:
    node_id = "sampler"
    phase = 40
    order = 20

    def apply(self, state: ComfyWorkflowState) -> None:
        if state.model_out is None:
            raise RuntimeError("Sampler node requires model output")
        if state.positive_ref is None or state.negative_ref is None:
            raise RuntimeError("Sampler node requires conditioning references")
        if state.latent_id is None:
            raise RuntimeError("Sampler node requires latent source")

        def _ksampler_inputs(*, latent_image: list[Any], denoise: float) -> dict[str, Any]:
            return {
                "model": state.model_out,
                "positive": state.positive_ref,
                "negative": state.negative_ref,
                "latent_image": latent_image,
                "seed": state.seed,
                "steps": state.sampling.steps,
                "cfg": state.sampling.cfg,
                "sampler_name": state.sampling.sampler,
                "scheduler": state.sampling.scheduler,
                "denoise": denoise,
            }

        skip_base_sampling = (
            state.context.skip_base_sampler_pass and state.enhancements.enable_hires_fix
        )
        base_sampler_id: str | None = None
        if not skip_base_sampling:
            base_sampler_id = state.add_node(
                "KSampler",
                _ksampler_inputs(
                    latent_image=[state.latent_id, 0],
                    denoise=state.sampling.denoise,
                ),
            )

        if state.enhancements.enable_hires_fix:
            if skip_base_sampling:
                hires_sampler_source = [state.latent_id, 0]
            elif base_sampler_id is not None:
                hires_sampler_source = [base_sampler_id, 0]
            else:
                hires_sampler_source = [state.latent_id, 0]

            hires_latent_id = state.add_node(
                "LatentUpscale",
                {
                    "samples": hires_sampler_source,
                    "upscale_method": "bislerp",
                    "width": max(
                        64,
                        ((int(state.base_width * state.enhancements.hires_scale) + 7) // 8) * 8,
                    ),
                    "height": max(
                        64,
                        ((int(state.base_height * state.enhancements.hires_scale) + 7) // 8) * 8,
                    ),
                    "crop": "disabled",
                },
            )
            state.sampler_id = state.add_node(
                "KSampler",
                _ksampler_inputs(
                    latent_image=[hires_latent_id, 0],
                    denoise=state.enhancements.hires_denoise,
                ),
            )
            return

        if base_sampler_id is None:
            base_sampler_id = state.add_node(
                "KSampler",
                _ksampler_inputs(
                    latent_image=[state.latent_id, 0],
                    denoise=state.sampling.denoise,
                ),
            )
        state.sampler_id = base_sampler_id

    def stage_labels(self) -> dict[str, WorkflowStageLabel]:
        return stage_labels_for("KSampler", "LatentUpscale")


def register_nodes(registry: ComfyNodeRegistry) -> None:
    registry.register(SamplerNode())
