from __future__ import annotations

from domain.loras import (
    EditorLoraSelection,
    clamp_editor_lora_strength,
    editor_lora_selections_from_legacy,
    legacy_lora_pairs,
    lora_catalog_entry_from_metadata,
    workflow_lora_attachments_from_legacy,
)


def test_editor_lora_selection_from_legacy_clamps_strength() -> None:
    raw = [
        ["anime.safetensors", "1.2"],
        ["portrait.safetensors", 9],
        ["", 0.7],
    ]

    selections = editor_lora_selections_from_legacy(raw)

    assert selections == [
        EditorLoraSelection(name="anime.safetensors", strength=1.2),
        EditorLoraSelection(name="portrait.safetensors", strength=2.0),
    ]


def test_legacy_lora_pairs_and_workflow_attachments_from_raw_payload() -> None:
    raw = [
        ["anime.safetensors", "0.8"],
        {"name": "detail.safetensors", "strength_model": 1.1},
    ]

    assert legacy_lora_pairs(raw) == [
        ("anime.safetensors", 0.8),
        ("detail.safetensors", 1.1),
    ]

    attachments = workflow_lora_attachments_from_legacy(raw)
    assert [(item.lora_name, item.strength_model, item.strength_clip) for item in attachments] == [
        ("anime.safetensors", 0.8, 0.8),
        ("detail.safetensors", 1.1, 1.1),
    ]


def test_lora_catalog_entry_from_metadata_normalizes_words() -> None:
    entry = lora_catalog_entry_from_metadata(
        "anime.safetensors",
        {
            "base_model": "SDXL",
            "trained_words": ["anime_style", "ANIME_STYLE", "best quality", ""],
            "path": "models/loras/anime.safetensors",
            "source": "civitai",
        },
    )

    assert entry.name == "anime.safetensors"
    assert entry.base_model == "SDXL"
    assert entry.file_path == "models/loras/anime.safetensors"
    assert entry.trained_words == ("anime_style", "best quality")
    assert entry.trigger_words(limit=1) == ["anime_style"]


def test_clamp_editor_lora_strength_uses_default_for_invalid_values() -> None:
    assert clamp_editor_lora_strength("invalid") == 1.0
