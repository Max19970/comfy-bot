from __future__ import annotations

from core.models import GenerationParams
from core.prompt_exchange import export_prompt_token, import_prompt_token


def test_prompt_exchange_roundtrip() -> None:
    src = GenerationParams(
        positive="masterpiece, portrait",
        negative="blurry, text",
        checkpoint="model.safetensors",
        loras=[("style.safetensors", 0.8)],
        width=1024,
        height=768,
        steps=30,
        cfg=6.5,
        seed=123,
        reference_images=[{"id": "r1", "file_id": "abc"}],
    )

    token = export_prompt_token(src)
    restored = import_prompt_token(token)

    assert restored.positive == src.positive
    assert restored.negative == src.negative
    assert restored.checkpoint == src.checkpoint
    assert restored.loras == src.loras
    assert restored.width == src.width
    assert restored.height == src.height
    assert restored.steps == src.steps
    assert restored.cfg == src.cfg
    assert restored.seed == src.seed
    assert restored.reference_images == src.reference_images
