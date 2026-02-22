from __future__ import annotations

from typing import Any

from application.lora_catalog_service import LoraCatalogService


def _build_catalog(metadata: dict[tuple[str, str], dict[str, Any]]) -> LoraCatalogService:
    def get_model_metadata(
        filename: str, *, model_type: str | None = None
    ) -> dict[str, Any] | None:
        if model_type is None:
            return None
        return metadata.get((filename, model_type))

    def infer_base_model(text: str) -> str:
        low = text.lower()
        if "flux" in low:
            return "Flux"
        if "sdxl" in low:
            return "SDXL"
        return ""

    def base_models_compatible(checkpoint_base: str, lora_base: str) -> bool:
        return checkpoint_base == lora_base

    return LoraCatalogService(
        get_model_metadata=get_model_metadata,
        infer_base_model=infer_base_model,
        base_models_compatible=base_models_compatible,
    )


def test_lora_catalog_service_returns_typed_entry_and_trigger_words() -> None:
    catalog = _build_catalog(
        {
            ("anime.safetensors", "lora"): {
                "base_model": "SDXL",
                "trained_words": ["anime_style", "ANIME_STYLE", "best quality", ""],
                "path": "models/loras/anime.safetensors",
            }
        }
    )

    entry = catalog.lora_entry("anime.safetensors")
    assert entry is not None
    assert entry.base_model == "SDXL"
    assert entry.file_path == "models/loras/anime.safetensors"
    assert catalog.lora_trigger_words("anime.safetensors", limit=2) == [
        "anime_style",
        "best quality",
    ]


def test_lora_catalog_service_compatibility_resolves_statuses() -> None:
    catalog = _build_catalog(
        {
            ("ckpt.safetensors", "checkpoint"): {"base_model": "SDXL"},
            ("anime.safetensors", "lora"): {"base_model": "SDXL"},
            ("flux_style.safetensors", "lora"): {"base_model": "Flux"},
        }
    )

    assert catalog.lora_compatibility("ckpt.safetensors", "anime.safetensors") == (
        "compatible",
        "SDXL",
        "SDXL",
    )
    assert catalog.lora_compatibility("ckpt.safetensors", "flux_style.safetensors") == (
        "incompatible",
        "SDXL",
        "Flux",
    )


def test_lora_catalog_service_falls_back_to_base_inference() -> None:
    catalog = _build_catalog({})
    assert catalog.checkpoint_base_model("custom_sdxl_checkpoint.safetensors") == "SDXL"
    assert catalog.lora_base_model("flux_style.safetensors") == "Flux"
