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
