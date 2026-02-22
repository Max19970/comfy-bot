from __future__ import annotations

from application.lora_catalog_service import LoraCatalogService
from handlers.prompt_editor_lora import (
    checkpoint_base_model,
    lora_compatibility,
    lora_trained_words,
    merge_prompt_with_words,
)


class _DownloaderStub:
    def __init__(self) -> None:
        self._meta = {
            ("ckpt.safetensors", "checkpoint"): {"base_model": "SDXL"},
            ("anime.safetensors", "lora"): {
                "base_model": "SDXL",
                "trained_words": ["anime_style", "best quality", "masterpiece"],
            },
            ("flux_style.safetensors", "lora"): {"base_model": "Flux"},
        }

    def get_model_metadata(self, filename: str, *, model_type: str | None = None) -> dict | None:
        if model_type is None:
            return None
        return self._meta.get((filename, model_type))

    def infer_base_model(self, text: str) -> str:
        if "flux" in text.lower():
            return "Flux"
        return ""

    def base_models_compatible(self, checkpoint_base: str, lora_base: str) -> bool:
        return checkpoint_base == lora_base


def _catalog() -> LoraCatalogService:
    stub = _DownloaderStub()
    return LoraCatalogService(
        get_model_metadata=stub.get_model_metadata,
        infer_base_model=stub.infer_base_model,
        base_models_compatible=stub.base_models_compatible,
    )


def test_checkpoint_base_model_prefers_metadata() -> None:
    assert checkpoint_base_model("ckpt.safetensors", _catalog()) == "SDXL"


def test_lora_compatibility_reports_compatible_and_incompatible() -> None:
    catalog = _catalog()

    status_ok, ckpt_ok, lora_ok = lora_compatibility(
        "ckpt.safetensors", "anime.safetensors", catalog
    )
    assert status_ok == "compatible"
    assert ckpt_ok == "SDXL"
    assert lora_ok == "SDXL"

    status_bad, ckpt_bad, lora_bad = lora_compatibility(
        "ckpt.safetensors",
        "flux_style.safetensors",
        catalog,
    )
    assert status_bad == "incompatible"
    assert ckpt_bad == "SDXL"
    assert lora_bad == "Flux"


def test_lora_trained_words_limited_to_twelve_items() -> None:
    assert lora_trained_words("anime.safetensors", _catalog()) == [
        "anime_style",
        "best quality",
        "masterpiece",
    ]


def test_merge_prompt_with_words_appends_only_missing_tokens() -> None:
    prompt = "portrait, best quality"
    merged = merge_prompt_with_words(prompt, ["best quality", "anime_style", "masterpiece"])
    assert merged == "portrait, best quality, anime_style, masterpiece"
