from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.models import GenerationParams
from domain.generation_request import GenerationRequest


class ComfyInfoLike(Protocol):
    ipadapter_apply_node: str
    clip_vision_models: list[str]
    ipadapter_models: list[str]


@dataclass(slots=True)
class ComfyWorkflowBuildContext:
    params: GenerationParams
    reference_image_name: str | None
    reference_mode: str
    object_info: dict[str, Any]
    info: ComfyInfoLike
    required_input_defaults: Callable[[str], dict[str, Any]]
    select_field_name: Callable[[dict[str, Any], tuple[str, ...]], str | None]
    skip_base_sampler_pass: bool = False

    def generation_request(self) -> GenerationRequest:
        return self.params.to_generation_request()
