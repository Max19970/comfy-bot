from __future__ import annotations

from core.models import GenerationParams
from core.storage import dict_to_params, params_to_dict


def test_params_serialization_roundtrip() -> None:
    params = GenerationParams(
        positive="a cat",
        negative="blurry",
        loras=[("cat.safetensors", 0.7)],
        reference_images=[{"id": "ref_1", "file_id": "file_1"}],
    )

    payload = params_to_dict(params)
    restored = dict_to_params(payload)

    assert restored.positive == params.positive
    assert restored.negative == params.negative
    assert restored.loras == params.loras
    assert restored.reference_images == params.reference_images


def test_dict_to_params_normalizes_loras_from_mixed_payload() -> None:
    payload = {
        "positive": "a cat",
        "loras": [
            ["style.safetensors", "0.9"],
            {"name": "detail.safetensors", "strength_model": 1.5},
            ["", 0.5],
        ],
    }

    restored = dict_to_params(payload)

    assert restored.loras == [
        ("style.safetensors", 0.9),
        ("detail.safetensors", 1.5),
    ]
