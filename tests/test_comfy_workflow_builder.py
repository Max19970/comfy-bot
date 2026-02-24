from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.models import GenerationParams
from infrastructure.comfy_workflow_builder import build_comfy_workflow


@dataclass
class _InfoStub:
    ipadapter_apply_node: str = ""
    clip_vision_models: list[str] = field(default_factory=list)
    ipadapter_models: list[str] = field(default_factory=list)


def _defaults(_: str) -> dict[str, Any]:
    return {}


def _select(field_specs: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in field_specs:
            return alias
    return None


def _class_types(workflow: dict[str, Any]) -> list[str]:
    return [
        node["class_type"] for _, node in sorted(workflow.items(), key=lambda item: int(item[0]))
    ]


def test_build_comfy_workflow_minimal_graph() -> None:
    workflow = build_comfy_workflow(
        GenerationParams(
            positive="portrait",
            negative="blurry",
            checkpoint="sdxl.safetensors",
            seed=42,
        ),
        reference_image_name=None,
        reference_mode="none",
        object_info={},
        info=_InfoStub(),
        required_input_defaults=_defaults,
        select_field_name=_select,
    )

    class_types = _class_types(workflow)
    assert "CheckpointLoaderSimple" in class_types
    assert class_types.count("CLIPTextEncode") == 2
    assert "KSampler" in class_types
    assert "SaveImage" in class_types


def test_build_comfy_workflow_includes_lora_and_upscale_nodes() -> None:
    workflow = build_comfy_workflow(
        GenerationParams(
            positive="portrait",
            negative="blurry",
            checkpoint="sdxl.safetensors",
            loras=[("style.safetensors", 0.8)],
            upscale_model="4x-ultrasharp.pth",
            seed=99,
        ),
        reference_image_name=None,
        reference_mode="none",
        object_info={},
        info=_InfoStub(),
        required_input_defaults=_defaults,
        select_field_name=_select,
    )

    class_types = _class_types(workflow)
    assert "LoraLoader" in class_types
    assert "UpscaleModelLoader" in class_types
    assert "ImageUpscaleWithModel" in class_types


def test_build_comfy_workflow_hires_img2img_uses_two_samplers_by_default() -> None:
    workflow = build_comfy_workflow(
        GenerationParams(
            positive="portrait",
            negative="blurry",
            checkpoint="sdxl.safetensors",
            enable_hires_fix=True,
            seed=7,
        ),
        reference_image_name="source.png",
        reference_mode="img2img",
        object_info={},
        info=_InfoStub(),
        required_input_defaults=_defaults,
        select_field_name=_select,
    )

    class_types = _class_types(workflow)
    assert class_types.count("KSampler") == 2


def test_build_comfy_workflow_hires_img2img_skips_base_sampler_when_requested() -> None:
    workflow = build_comfy_workflow(
        GenerationParams(
            positive="portrait",
            negative="blurry",
            checkpoint="sdxl.safetensors",
            enable_hires_fix=True,
            seed=7,
        ),
        reference_image_name="source.png",
        reference_mode="img2img",
        object_info={},
        info=_InfoStub(),
        required_input_defaults=_defaults,
        select_field_name=_select,
        skip_base_sampler_pass=True,
    )

    class_types = _class_types(workflow)
    assert class_types.count("KSampler") == 1
    assert "LatentUpscale" in class_types
