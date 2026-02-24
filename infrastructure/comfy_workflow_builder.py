from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.models import GenerationParams

from .comfy_nodes.builder import build_workflow_from_nodes, collect_workflow_stage_labels
from .comfy_nodes.context import ComfyInfoLike, ComfyWorkflowBuildContext
from .comfy_nodes.contracts import WorkflowStageLabel


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
    context = ComfyWorkflowBuildContext(
        params=params,
        reference_image_name=reference_image_name,
        reference_mode=reference_mode,
        object_info=object_info,
        info=info,
        required_input_defaults=required_input_defaults,
        select_field_name=select_field_name,
        skip_base_sampler_pass=skip_base_sampler_pass,
    )
    return build_workflow_from_nodes(context)


def comfy_workflow_stage_labels() -> dict[str, WorkflowStageLabel]:
    return collect_workflow_stage_labels()
