from __future__ import annotations

from collections.abc import Callable

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


def enhancements_menu_label(
    params: GenerationParams,
    *,
    translate: Callable[[str, str | None, str], str] | None = None,
    locale: str | None = None,
) -> str:
    count = enhancements_count(params)
    if translate is None:
        return f"✨ Enhancements ({count})" if count else "✨ Enhancements"
    if count:
        return translate(
            "prompt_editor.ui.button.enhancements.count",
            locale,
            "✨ Enhancements ({count})",
        ).format(count=count)
    return translate(
        "prompt_editor.ui.button.enhancements",
        locale,
        "✨ Enhancements",
    )
