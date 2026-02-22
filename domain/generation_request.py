from __future__ import annotations

from dataclasses import dataclass

from .loras import WorkflowLoraAttachment


@dataclass(frozen=True, slots=True)
class GenerationPromptText:
    positive: str
    negative: str


@dataclass(frozen=True, slots=True)
class ReferenceImageInput:
    id: str
    file_id: str


@dataclass(frozen=True, slots=True)
class GenerationModelStack:
    checkpoint: str
    loras: tuple[WorkflowLoraAttachment, ...] = ()
    vae_name: str = ""
    controlnet_name: str = ""
    controlnet_strength: float = 1.0
    embedding_name: str = ""
    upscale_model: str = ""


@dataclass(frozen=True, slots=True)
class GenerationSamplingConfig:
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    denoise: float = 1.0
    seed: int = -1
    batch_size: int = 1


@dataclass(frozen=True, slots=True)
class GenerationImageConfig:
    width: int = 512
    height: int = 512
    reference_strength: float = 0.8
    reference_images: tuple[ReferenceImageInput, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationEnhancementConfig:
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


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: GenerationPromptText
    models: GenerationModelStack
    sampling: GenerationSamplingConfig
    image: GenerationImageConfig
    enhancements: GenerationEnhancementConfig
