from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.generation_request import (
    GenerationEnhancementConfig,
    GenerationImageConfig,
    GenerationModelStack,
    GenerationPromptText,
    GenerationRequest,
    GenerationSamplingConfig,
    ReferenceImageInput,
)
from domain.loras import (
    EditorLoraSelection,
    WorkflowLoraAttachment,
    legacy_lora_pairs,
    workflow_lora_attachments_from_legacy,
)


@dataclass
class GenerationParams:
    """All parameters needed to build a ComfyUI workflow."""

    positive: str = ""
    negative: str = ""
    checkpoint: str = ""
    loras: list[tuple[str, float]] = field(default_factory=list)
    vae_name: str = ""
    controlnet_name: str = ""
    controlnet_strength: float = 1.0
    embedding_name: str = ""
    upscale_model: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    denoise: float = 1.0
    seed: int = -1
    batch_size: int = 1
    reference_strength: float = 0.8
    reference_images: list[dict[str, str]] = field(default_factory=list)
    enable_hires_fix: bool = False
    hires_scale: float = 1.5
    hires_denoise: float = 0.5
    enable_freeu: bool = False
    enable_pag: bool = False
    pag_scale: float = 3.0
    enable_tiled_diffusion: bool = False
    tile_size: int = 256
    tile_overlap: int = 64
    vae_tile_size: int = 512

    def lora_selections(self) -> list[EditorLoraSelection]:
        return [EditorLoraSelection.create(name, strength) for name, strength in self.loras]

    def set_lora_selections(self, selections: list[EditorLoraSelection]) -> None:
        self.loras = legacy_lora_pairs([item.to_legacy_pair() for item in selections])

    def workflow_lora_attachments(self) -> list[WorkflowLoraAttachment]:
        return workflow_lora_attachments_from_legacy(self.loras)

    def to_generation_request(self) -> GenerationRequest:
        return GenerationRequest(
            prompt=GenerationPromptText(
                positive=self.positive,
                negative=self.negative,
            ),
            models=GenerationModelStack(
                checkpoint=self.checkpoint,
                loras=tuple(self.workflow_lora_attachments()),
                vae_name=self.vae_name,
                controlnet_name=self.controlnet_name,
                controlnet_strength=self.controlnet_strength,
                embedding_name=self.embedding_name,
                upscale_model=self.upscale_model,
            ),
            sampling=GenerationSamplingConfig(
                steps=self.steps,
                cfg=self.cfg,
                sampler=self.sampler,
                scheduler=self.scheduler,
                denoise=self.denoise,
                seed=self.seed,
                batch_size=self.batch_size,
            ),
            image=GenerationImageConfig(
                width=self.width,
                height=self.height,
                reference_strength=self.reference_strength,
                reference_images=_to_reference_image_inputs(self.reference_images),
            ),
            enhancements=GenerationEnhancementConfig(
                enable_hires_fix=self.enable_hires_fix,
                hires_scale=self.hires_scale,
                hires_denoise=self.hires_denoise,
                enable_freeu=self.enable_freeu,
                enable_pag=self.enable_pag,
                pag_scale=self.pag_scale,
                enable_tiled_diffusion=self.enable_tiled_diffusion,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                vae_tile_size=self.vae_tile_size,
            ),
        )

    @classmethod
    def from_generation_request(cls, request: GenerationRequest) -> GenerationParams:
        return cls(
            positive=request.prompt.positive,
            negative=request.prompt.negative,
            checkpoint=request.models.checkpoint,
            loras=[(item.lora_name, float(item.strength_model)) for item in request.models.loras],
            vae_name=request.models.vae_name,
            controlnet_name=request.models.controlnet_name,
            controlnet_strength=request.models.controlnet_strength,
            embedding_name=request.models.embedding_name,
            upscale_model=request.models.upscale_model,
            width=request.image.width,
            height=request.image.height,
            steps=request.sampling.steps,
            cfg=request.sampling.cfg,
            sampler=request.sampling.sampler,
            scheduler=request.sampling.scheduler,
            denoise=request.sampling.denoise,
            seed=request.sampling.seed,
            batch_size=request.sampling.batch_size,
            reference_strength=request.image.reference_strength,
            reference_images=_from_reference_image_inputs(request.image.reference_images),
            enable_hires_fix=request.enhancements.enable_hires_fix,
            hires_scale=request.enhancements.hires_scale,
            hires_denoise=request.enhancements.hires_denoise,
            enable_freeu=request.enhancements.enable_freeu,
            enable_pag=request.enhancements.enable_pag,
            pag_scale=request.enhancements.pag_scale,
            enable_tiled_diffusion=request.enhancements.enable_tiled_diffusion,
            tile_size=request.enhancements.tile_size,
            tile_overlap=request.enhancements.tile_overlap,
            vae_tile_size=request.enhancements.vae_tile_size,
        )


def _to_reference_image_inputs(raw: Any) -> tuple[ReferenceImageInput, ...]:
    if not isinstance(raw, list):
        return ()

    normalized: list[ReferenceImageInput] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("file_id") or "").strip()
        if not file_id:
            continue
        ref_id = str(item.get("id") or f"ref_{index + 1}").strip()
        normalized.append(ReferenceImageInput(id=ref_id, file_id=file_id))

    return tuple(normalized)


def _from_reference_image_inputs(raw: tuple[ReferenceImageInput, ...]) -> list[dict[str, str]]:
    return [
        {
            "id": item.id,
            "file_id": item.file_id,
        }
        for item in raw
        if item.file_id
    ]
