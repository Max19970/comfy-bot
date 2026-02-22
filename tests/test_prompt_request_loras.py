from __future__ import annotations

from core.models import GenerationParams
from core.runtime import PromptRequest
from domain.loras import EditorLoraSelection


def test_prompt_request_initializes_editor_loras_from_params() -> None:
    req = PromptRequest(params=GenerationParams(loras=[("anime.safetensors", 0.8)]))

    assert [(item.name, item.strength, item.file_path) for item in req.editor_loras] == [
        ("anime.safetensors", 0.8, ""),
    ]


def test_prompt_request_add_editor_lora_keeps_editor_context() -> None:
    req = PromptRequest(params=GenerationParams())
    req.add_editor_lora(
        EditorLoraSelection.create(
            "anime.safetensors",
            0.9,
            file_path="models/loras/anime.safetensors",
        )
    )

    assert req.params.loras == [("anime.safetensors", 0.9)]
    assert req.editor_loras[0].file_path == "models/loras/anime.safetensors"


def test_prompt_request_remove_and_clear_editor_loras() -> None:
    req = PromptRequest(params=GenerationParams())
    req.add_editor_lora(EditorLoraSelection.create("anime.safetensors", 0.8))
    req.add_editor_lora(EditorLoraSelection.create("detail.safetensors", 1.1))

    assert req.remove_last_editor_lora() is True
    assert req.params.loras == [("anime.safetensors", 0.8)]
    assert req.clear_editor_loras() == 1
    assert req.params.loras == []
    assert req.remove_last_editor_lora() is False


def test_prompt_request_sync_editor_loras_from_params_preserves_known_paths() -> None:
    req = PromptRequest(params=GenerationParams())
    req.add_editor_lora(
        EditorLoraSelection.create(
            "anime.safetensors",
            0.8,
            file_path="models/loras/anime.safetensors",
        )
    )

    req.params.loras = [("anime.safetensors", 1.2)]
    req.sync_editor_loras_from_params()

    assert req.editor_loras == [
        EditorLoraSelection(
            name="anime.safetensors",
            strength=1.2,
            file_path="models/loras/anime.safetensors",
        )
    ]
