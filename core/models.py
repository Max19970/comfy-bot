from __future__ import annotations

from dataclasses import dataclass, field


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
