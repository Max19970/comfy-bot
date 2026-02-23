from __future__ import annotations

from core.models import GenerationParams
from handlers.presets_storage import (
    is_valid_preset_name,
    normalize_preset_name,
    params_from_payload,
    params_payload_from,
    resolve_preset_name,
)


def test_normalize_and_validate_preset_name() -> None:
    assert normalize_preset_name("  my preset  ") == "my preset"
    assert is_valid_preset_name("ok")
    assert not is_valid_preset_name("")
    assert not is_valid_preset_name("x" * 51)


def test_resolve_preset_name_prefers_snapshot() -> None:
    presets: dict[str, dict[str, object]] = {"a": {}, "b": {}}
    snapshot = ["b", "a"]

    assert resolve_preset_name(0, presets, snapshot) == "b"
    assert resolve_preset_name(1, presets, snapshot) == "a"


def test_resolve_preset_name_falls_back_to_sorted_names() -> None:
    presets: dict[str, dict[str, object]] = {"z": {}, "a": {}}

    assert resolve_preset_name(0, presets, object()) == "a"
    assert resolve_preset_name(1, presets, object()) == "z"
    assert resolve_preset_name(2, presets, object()) is None


def test_params_payload_roundtrip() -> None:
    params = GenerationParams(positive="test", width=768, height=1024)

    payload = params_payload_from(params)
    restored = params_from_payload(payload)

    assert restored.positive == "test"
    assert restored.width == 768
    assert restored.height == 1024
