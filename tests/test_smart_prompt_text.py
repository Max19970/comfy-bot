from __future__ import annotations

from application.smart_prompt_text import (
    dedupe_tags,
    detect_quality_preset,
    extract_anchor_tags,
    quality_tags_for_checkpoint,
    sanitize_final_prompt,
    split_tags,
)


def test_detect_quality_preset_and_quality_tags() -> None:
    assert detect_quality_preset("pony-model.safetensors") == "pony"
    assert detect_quality_preset("illustrious-xl") == "illustrious"
    assert detect_quality_preset("unknown-model") == "default"

    assert quality_tags_for_checkpoint("pony-model")[:2] == ["score_9", "score_8_up"]


def test_extract_anchor_tags_translates_russian_words() -> None:
    tags = extract_anchor_tags("Девушка в лес, красный волосы")

    assert "girl" in tags
    assert "forest" in tags
    assert "red" in tags


def test_split_dedupe_and_sanitize_helpers() -> None:
    assert split_tags(" cat, dog; cat\n") == ["cat", "dog", "cat"]
    assert dedupe_tags(["cat", "Cat", "dog"]) == ["cat", "dog"]

    sanitized = sanitize_final_prompt("<|meta|>cat, , dog\n target:bad\nfox")
    assert sanitized == "cat, dog\nfox"
