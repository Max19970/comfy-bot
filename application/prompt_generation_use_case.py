from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from core.models import GenerationParams

TranslateText = Callable[[str, str | None, str], str]


def _tx(translate: TranslateText | None, key: str, locale: str | None, default: str) -> str:
    if translate is None:
        return default
    return translate(key, locale, default)


@dataclass(frozen=True, slots=True)
class GenerationPreparation:
    params: GenerationParams
    generation_params: GenerationParams
    used_seed: int
    incompatible_loras: tuple[tuple[str, str, str], ...]


@dataclass(slots=True)
class PromptGenerationUseCase:
    normalize_params: Callable[[GenerationParams], GenerationParams]
    incompatible_loras: Callable[[GenerationParams], list[tuple[str, str, str]]]
    denoise_from_reference_strength: Callable[[float], float]
    supports_ipadapter: Callable[[], bool]
    resolve_reference_mode: Callable[[bool], str]
    random_seed: Callable[[int, int], int] = random.randint

    def prepare(self, source_params: GenerationParams) -> GenerationPreparation:
        params = self.normalize_params(source_params)
        used_seed = params.seed if params.seed >= 0 else self.random_seed(0, 2**63 - 1)
        generation_params = GenerationParams.from_generation_request(params.to_generation_request())
        generation_params.seed = used_seed
        incompatible = tuple(self.incompatible_loras(params))
        return GenerationPreparation(
            params=params,
            generation_params=generation_params,
            used_seed=used_seed,
            incompatible_loras=incompatible,
        )

    def reference_validation_notice(
        self,
        params: GenerationParams,
        *,
        reference_image_count: int,
        translate: TranslateText | None = None,
        locale: str | None = None,
    ) -> str | None:
        if params.controlnet_name and reference_image_count <= 0:
            return _tx(
                translate,
                "application.prompt_generation.reference.controlnet_required",
                locale,
                "❌ Для ControlNet нужно добавить хотя бы один референс.",
            )
        if params.reference_images and reference_image_count <= 0:
            return _tx(
                translate,
                "application.prompt_generation.reference.reload_failed",
                locale,
                "❌ Не удалось загрузить референс-картинки. Загрузите их заново в редакторе.",
            )
        return None

    def resolve_mode(self, params: GenerationParams, *, has_reference_images: bool) -> str:
        if params.controlnet_name and has_reference_images:
            return "ipadapter" if self.supports_ipadapter() else "none"
        return self.resolve_reference_mode(has_reference_images)

    def apply_reference_adjustments(
        self,
        generation_params: GenerationParams,
        *,
        reference_mode: str,
        has_reference_images: bool,
    ) -> None:
        if reference_mode == "img2img" and has_reference_images:
            generation_params.denoise = min(
                generation_params.denoise,
                self.denoise_from_reference_strength(generation_params.reference_strength),
            )

    @staticmethod
    def lora_warning_payload(
        incompatible_loras: tuple[tuple[str, str, str], ...],
        *,
        max_items: int = 3,
        translate: TranslateText | None = None,
        locale: str | None = None,
    ) -> tuple[str, str] | None:
        if not incompatible_loras:
            return None
        listed = ", ".join(name for name, _, _ in incompatible_loras[:max_items])
        suffix = (
            ""
            if len(incompatible_loras) <= max_items
            else _tx(
                translate,
                "application.prompt_generation.lora.more_suffix",
                locale,
                " и ещё {count}",
            ).format(count=len(incompatible_loras) - max_items)
        )
        return listed, suffix
