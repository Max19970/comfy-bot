from __future__ import annotations

from application.prompt_generation_use_case import PromptGenerationUseCase
from core.models import GenerationParams


def _clone(params: GenerationParams) -> GenerationParams:
    return GenerationParams.from_generation_request(params.to_generation_request())


def test_prepare_builds_seeded_generation_params() -> None:
    use_case = PromptGenerationUseCase(
        normalize_params=_clone,
        incompatible_loras=lambda _: [("style.safetensors", "SDXL", "Flux")],
        denoise_from_reference_strength=lambda strength: max(0.1, min(1.0, strength)),
        supports_ipadapter=lambda: True,
        resolve_reference_mode=lambda has_refs: "img2img" if has_refs else "none",
        random_seed=lambda low, high: 123,
    )

    source = GenerationParams(checkpoint="model.safetensors", seed=-1, denoise=0.9)
    prepared = use_case.prepare(source)

    assert prepared.params.checkpoint == "model.safetensors"
    assert prepared.used_seed == 123
    assert prepared.generation_params.seed == 123
    assert prepared.incompatible_loras == (("style.safetensors", "SDXL", "Flux"),)
    assert prepared.generation_params is not prepared.params


def test_reference_validation_notice_matches_expected_rules() -> None:
    use_case = PromptGenerationUseCase(
        normalize_params=_clone,
        incompatible_loras=lambda _: [],
        denoise_from_reference_strength=lambda strength: strength,
        supports_ipadapter=lambda: False,
        resolve_reference_mode=lambda has_refs: "img2img" if has_refs else "none",
    )

    assert (
        use_case.reference_validation_notice(
            GenerationParams(controlnet_name="controlnet.safetensors"),
            reference_image_count=0,
        )
        == "❌ Для ControlNet нужно добавить хотя бы один референс."
    )
    assert (
        use_case.reference_validation_notice(
            GenerationParams(reference_images=[{"id": "ref_1", "file_id": "x"}]),
            reference_image_count=0,
        )
        == "❌ Не удалось загрузить референс-картинки. Загрузите их заново в редакторе."
    )
    assert (
        use_case.reference_validation_notice(
            GenerationParams(),
            reference_image_count=0,
        )
        is None
    )


def test_resolve_mode_and_apply_reference_adjustments() -> None:
    use_case = PromptGenerationUseCase(
        normalize_params=_clone,
        incompatible_loras=lambda _: [],
        denoise_from_reference_strength=lambda strength: 0.25 if strength > 0 else 1.0,
        supports_ipadapter=lambda: True,
        resolve_reference_mode=lambda has_refs: "img2img" if has_refs else "none",
    )

    control_params = GenerationParams(controlnet_name="controlnet.safetensors")
    assert use_case.resolve_mode(control_params, has_reference_images=True) == "ipadapter"

    params = GenerationParams(denoise=0.9, reference_strength=0.8)
    assert use_case.resolve_mode(params, has_reference_images=True) == "img2img"
    use_case.apply_reference_adjustments(
        params,
        reference_mode="img2img",
        has_reference_images=True,
    )
    assert params.denoise == 0.25

    use_case.apply_reference_adjustments(
        params,
        reference_mode="none",
        has_reference_images=True,
    )
    assert params.denoise == 0.25


def test_lora_warning_payload_handles_suffix() -> None:
    payload = PromptGenerationUseCase.lora_warning_payload(
        (
            ("a.safetensors", "", ""),
            ("b.safetensors", "", ""),
            ("c.safetensors", "", ""),
            ("d.safetensors", "", ""),
        )
    )

    assert payload == ("a.safetensors, b.safetensors, c.safetensors", " и ещё 1")
