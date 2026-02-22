from __future__ import annotations

from domain.base_model_policy import BaseModelPolicy


def test_infer_base_model_detects_known_families() -> None:
    policy = BaseModelPolicy()

    assert policy.infer_base_model("ponyDiffusionXL_v6.safetensors") == "Pony"
    assert policy.infer_base_model("flux-dev-checkpoint") == "Flux"
    assert policy.infer_base_model("illustrious-v2") == "Illustrious"
    assert policy.infer_base_model("sdxl_base_1.0") == "SDXL"
    assert policy.infer_base_model("custom_sd1.5_model") == "SD 1.5"


def test_base_models_compatible_handles_related_families() -> None:
    policy = BaseModelPolicy()

    assert policy.base_models_compatible("SDXL", "Illustrious") is True
    assert policy.base_models_compatible("Flux", "Flux") is True
    assert policy.base_models_compatible("SDXL", "Flux") is False
