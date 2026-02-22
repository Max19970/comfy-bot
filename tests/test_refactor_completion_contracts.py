from __future__ import annotations

from core.download_filters import base_code_from_base_model
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from domain.base_model_policy import BaseModelPolicy


def test_runtime_persisted_payload_uses_schema_v2_and_editor_loras() -> None:
    runtime = RuntimeStore(
        active_prompt_requests={
            1001: PromptRequest(
                params=GenerationParams(
                    checkpoint="sdxl.safetensors",
                    loras=[("anime_style.safetensors", 0.8)],
                ),
                operation="generate",
            )
        }
    )

    payload = runtime.to_persisted_dict()

    assert payload["schema_version"] == 2
    entry = payload["active_prompt_requests"]["1001"]
    assert isinstance(entry.get("editor_loras"), list)
    assert entry["editor_loras"][0]["name"] == "anime_style.safetensors"


def test_download_base_code_wrapper_matches_domain_policy() -> None:
    policy = BaseModelPolicy()
    samples = [
        "Illustrious XL",
        "NoobAI checkpoint",
        "Flux.1 D",
        "SDXL 1.0",
        "SDXL 0.9",
        "SD 3.5",
        "SD 3",
        "SD 2.1",
        "SD 1.5",
        "",
    ]

    for sample in samples:
        assert base_code_from_base_model(sample) == policy.download_base_code(sample)
