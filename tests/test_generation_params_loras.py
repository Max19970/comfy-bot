from __future__ import annotations

from core.models import GenerationParams
from domain.loras import EditorLoraSelection


def test_generation_params_exposes_typed_lora_selections() -> None:
    params = GenerationParams(loras=[("anime.safetensors", 0.8)])
    assert params.lora_selections() == [
        EditorLoraSelection(name="anime.safetensors", strength=0.8),
    ]


def test_generation_params_set_lora_selections_updates_legacy_chain() -> None:
    params = GenerationParams()
    params.set_lora_selections(
        [
            EditorLoraSelection.create("anime.safetensors", "0.7"),
            EditorLoraSelection.create("detail.safetensors", 5),
        ]
    )

    assert params.loras == [
        ("anime.safetensors", 0.7),
        ("detail.safetensors", 2.0),
    ]


def test_generation_params_builds_workflow_lora_attachments() -> None:
    params = GenerationParams(loras=[("anime.safetensors", 1.1)])
    attachments = params.workflow_lora_attachments()

    assert [(item.lora_name, item.strength_model, item.strength_clip) for item in attachments] == [
        ("anime.safetensors", 1.1, 1.1),
    ]
