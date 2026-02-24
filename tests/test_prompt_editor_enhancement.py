from __future__ import annotations

from core.models import GenerationParams
from core.runtime import PreviewArtifact
from handlers.prompt_editor_enhancement import _requires_image_generation


def _artifact(
    *,
    params: GenerationParams | None = None,
    enable_sampler_pass: bool = False,
) -> PreviewArtifact:
    return PreviewArtifact(
        artifact_id="a1",
        owner_uid=1,
        image_bytes=b"img",
        params=params or GenerationParams(),
        used_seed=42,
        enable_sampler_pass=enable_sampler_pass,
    )


def test_requires_image_generation_for_sampler_pass() -> None:
    artifact = _artifact(enable_sampler_pass=True)
    assert _requires_image_generation(artifact, artifact.params) is True


def test_requires_image_generation_for_hires_without_sampler_pass() -> None:
    params = GenerationParams(enable_hires_fix=True)
    artifact = _artifact(params=params, enable_sampler_pass=False)
    assert _requires_image_generation(artifact, artifact.params) is True


def test_requires_image_generation_for_other_sampling_enhancements() -> None:
    freeu_artifact = _artifact(params=GenerationParams(enable_freeu=True))
    pag_artifact = _artifact(params=GenerationParams(enable_pag=True))
    tiled_artifact = _artifact(params=GenerationParams(enable_tiled_diffusion=True))

    assert _requires_image_generation(freeu_artifact, freeu_artifact.params) is True
    assert _requires_image_generation(pag_artifact, pag_artifact.params) is True
    assert _requires_image_generation(tiled_artifact, tiled_artifact.params) is True


def test_requires_image_generation_false_for_upscaler_only() -> None:
    params = GenerationParams(upscale_model="RealESRGAN_x4plus")
    artifact = _artifact(params=params, enable_sampler_pass=False)
    assert _requires_image_generation(artifact, artifact.params) is False
