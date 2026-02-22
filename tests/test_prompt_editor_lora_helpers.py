from __future__ import annotations

from typing import Any, cast

from domain.loras import LoraCatalogEntry
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
            ("anime.safetensors", "lora"): {"base_model": "SDXL"},
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

    def get_lora_trained_words(self, lora_name: str) -> list[str]:
        if lora_name == "anime.safetensors":
            return ["anime_style", "best quality", "masterpiece"]
        return []

    def get_lora_entry(self, lora_name: str) -> LoraCatalogEntry | None:
        meta = self.get_model_metadata(lora_name, model_type="lora")
        if not meta:
            return None
        return LoraCatalogEntry.from_metadata(lora_name, meta)

    def base_models_compatible(self, checkpoint_base: str, lora_base: str) -> bool:
        return checkpoint_base == lora_base


def test_checkpoint_base_model_prefers_metadata() -> None:
    downloader = cast(Any, _DownloaderStub())
    assert checkpoint_base_model("ckpt.safetensors", downloader) == "SDXL"


def test_lora_compatibility_reports_compatible_and_incompatible() -> None:
    downloader = cast(Any, _DownloaderStub())

    status_ok, ckpt_ok, lora_ok = lora_compatibility(
        "ckpt.safetensors", "anime.safetensors", downloader
    )
    assert status_ok == "compatible"
    assert ckpt_ok == "SDXL"
    assert lora_ok == "SDXL"

    status_bad, ckpt_bad, lora_bad = lora_compatibility(
        "ckpt.safetensors",
        "flux_style.safetensors",
        downloader,
    )
    assert status_bad == "incompatible"
    assert ckpt_bad == "SDXL"
    assert lora_bad == "Flux"


def test_lora_trained_words_limited_to_twelve_items() -> None:
    downloader = cast(Any, _DownloaderStub())
    assert lora_trained_words("anime.safetensors", downloader) == [
        "anime_style",
        "best quality",
        "masterpiece",
    ]


def test_merge_prompt_with_words_appends_only_missing_tokens() -> None:
    prompt = "portrait, best quality"
    merged = merge_prompt_with_words(prompt, ["best quality", "anime_style", "masterpiece"])
    assert merged == "portrait, best quality, anime_style, masterpiece"
