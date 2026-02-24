from __future__ import annotations

from core.models import GenerationParams
from handlers.prompt_editor_enhancements import enhancements_count, enhancements_menu_label


def test_enhancements_count_and_label_zero() -> None:
    params = GenerationParams()
    assert enhancements_count(params) == 0
    assert enhancements_menu_label(params) == "✨ Улучшения"


def test_enhancements_count_and_label_non_zero() -> None:
    params = GenerationParams(
        enable_hires_fix=True,
        enable_freeu=True,
        enable_pag=True,
        upscale_model="RealESRGAN_x4plus",
        enable_tiled_diffusion=True,
    )
    assert enhancements_count(params) == 5
    assert enhancements_menu_label(params) == "✨ Улучшения (5)"


def test_enhancements_menu_label_uses_translate_callback() -> None:
    params = GenerationParams(enable_hires_fix=True)

    def _translate(key: str, locale: str | None, default: str) -> str:
        mapping = {
            ("prompt_editor.ui.button.enhancements", "ru"): "✨ Улучшения",
            ("prompt_editor.ui.button.enhancements.count", "ru"): "✨ Улучшения ({count})",
            ("prompt_editor.ui.button.enhancements", "en"): "✨ Enhancements",
            ("prompt_editor.ui.button.enhancements.count", "en"): "✨ Enhancements ({count})",
        }
        return mapping.get((key, locale or ""), default)

    assert enhancements_menu_label(params, translate=_translate, locale="ru") == "✨ Улучшения (1)"
    assert (
        enhancements_menu_label(params, translate=_translate, locale="en") == "✨ Enhancements (1)"
    )
