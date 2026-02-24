from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.models import GenerationParams

from .comfy_nodes.builder import build_workflow_from_nodes, collect_workflow_stage_labels
from .comfy_nodes.context import ComfyInfoLike, ComfyWorkflowBuildContext
from .comfy_nodes.contracts import WorkflowStageLabel

UPSCALE_NODES_PACKAGE = "infrastructure.comfy_nodes.upscale_nodes"
GENERATION_NODES_PACKAGE = "infrastructure.comfy_nodes.nodes"


@dataclass(frozen=True, slots=True)
class _EmptyComfyInfo:
    ipadapter_apply_node: str = ""
    clip_vision_models: list[str] | None = None
    ipadapter_models: list[str] | None = None

    def __post_init__(self) -> None:
        if self.clip_vision_models is None:
            object.__setattr__(self, "clip_vision_models", [])
        if self.ipadapter_models is None:
            object.__setattr__(self, "ipadapter_models", [])


def _noop_required_defaults(_: str) -> dict[str, Any]:
    return {}


def _noop_select_field_name(_: dict[str, Any], __: tuple[str, ...]) -> str | None:
    return None


def build_comfy_workflow(
    params: GenerationParams,
    *,
    reference_image_name: str | None,
    reference_mode: str,
    object_info: dict[str, Any],
    info: ComfyInfoLike,
    required_input_defaults: Callable[[str], dict[str, Any]],
    select_field_name: Callable[[dict[str, Any], tuple[str, ...]], str | None],
    skip_base_sampler_pass: bool = False,
) -> dict[str, Any]:
    params_skip_flag = bool(getattr(params, "_skip_base_sampler_pass", False))
    context = ComfyWorkflowBuildContext(
        params=params,
        reference_image_name=reference_image_name,
        reference_mode=reference_mode,
        object_info=object_info,
        info=info,
        required_input_defaults=required_input_defaults,
        select_field_name=select_field_name,
        skip_base_sampler_pass=skip_base_sampler_pass or params_skip_flag,
    )
    return build_workflow_from_nodes(context)


def build_comfy_upscale_workflow(
    *,
    reference_image_name: str,
    upscale_model: str,
) -> dict[str, Any]:
    if not reference_image_name:
        raise ValueError("reference_image_name is required")
    if not upscale_model:
        raise ValueError("upscale_model is required")

    params = GenerationParams(upscale_model=upscale_model)
    context = ComfyWorkflowBuildContext(
        params=params,
        reference_image_name=reference_image_name,
        reference_mode="upscale_only",
        object_info={},
        info=_EmptyComfyInfo(),
        required_input_defaults=_noop_required_defaults,
        select_field_name=_noop_select_field_name,
    )
    return build_workflow_from_nodes(context, package_name=UPSCALE_NODES_PACKAGE)


def comfy_workflow_stage_labels() -> dict[str, WorkflowStageLabel]:
    return collect_workflow_stage_labels(
        GENERATION_NODES_PACKAGE,
        UPSCALE_NODES_PACKAGE,
    )
