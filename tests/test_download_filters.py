from __future__ import annotations

from core.download_filters import base_code_from_base_model


def test_base_code_from_base_model_known_values() -> None:
    assert base_code_from_base_model("SDXL 1.0") == "sdxl"
    assert base_code_from_base_model("Stable Diffusion XL") == "sdxl"
    assert base_code_from_base_model("SDXL 0.9") == "sdxl09"
    assert base_code_from_base_model("Stable Diffusion 3.5") == "sd35"
    assert base_code_from_base_model("Stable Diffusion 3") == "sd3"
    assert base_code_from_base_model("SD 2.1") == "sd2"
    assert base_code_from_base_model("SD 1.5") == "sd15"
    assert base_code_from_base_model("Pony Realism") == "pony"
    assert base_code_from_base_model("Flux Dev") == "flux"
    assert base_code_from_base_model("Illustrious XL") == "illustrious"
    assert base_code_from_base_model("NoobAI XL") == "noobai"


def test_base_code_from_base_model_unknown_defaults_to_all() -> None:
    assert base_code_from_base_model("") == "all"
    assert base_code_from_base_model("mystery-model") == "all"
