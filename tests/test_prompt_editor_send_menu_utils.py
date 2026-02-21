from __future__ import annotations

from core.models import GenerationParams
from core.runtime import PreviewArtifact
from handlers.prompt_editor_send_menu_utils import (
    apply_field_value,
    parse_shrink_size,
)


def _artifact() -> PreviewArtifact:
    return PreviewArtifact(
        artifact_id="a1",
        owner_uid=1,
        image_bytes=b"img",
        params=GenerationParams(),
        used_seed=1,
    )


def test_parse_shrink_size_accepts_cyrillic_x() -> None:
    assert parse_shrink_size("1280х720") == (1280, 720)
    assert parse_shrink_size("0x100") is None
    assert parse_shrink_size("abc") is None


def test_apply_field_value_respects_enhancement_ranges() -> None:
    artifact = _artifact()
    assert apply_field_value(artifact, field="hires_scale", value=2.0) is True
    assert artifact.params.hires_scale == 2.0
    assert apply_field_value(artifact, field="hires_scale", value=9.0) is False


def test_apply_field_value_enables_sampler_for_steps() -> None:
    artifact = _artifact()
    artifact.enable_sampler_pass = False
    assert apply_field_value(artifact, field="steps", value=30) is True
    assert artifact.params.steps == 30
    assert artifact.enable_sampler_pass is True
