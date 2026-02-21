from __future__ import annotations

from core.models import GenerationParams


def enhancements_count(params: GenerationParams) -> int:
    return sum(
        [
            params.enable_hires_fix,
            params.enable_freeu,
            params.enable_pag,
            bool(params.upscale_model),
            params.enable_tiled_diffusion,
        ]
    )


def enhancements_menu_label(params: GenerationParams) -> str:
    count = enhancements_count(params)
    return f"✨ Улучшения ({count})" if count else "✨ Улучшения"
