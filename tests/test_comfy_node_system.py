from __future__ import annotations

import inspect

from infrastructure.comfy_nodes.registry import load_discovered_nodes
from infrastructure.comfy_workflow_builder import build_comfy_workflow


def _fingerprint() -> list[tuple[int, int, str]]:
    return [(int(node.phase), int(node.order), node.node_id) for node in load_discovered_nodes()]


def test_load_discovered_nodes_detects_nodes_from_directory() -> None:
    node_ids = [node.node_id for node in load_discovered_nodes()]
    assert "checkpoint_loader" in node_ids
    assert "clip_text_encode" in node_ids
    assert "controlnet" in node_ids
    assert "ipadapter" in node_ids
    assert "sampler" in node_ids
    assert "save_image" in node_ids
    assert len(node_ids) >= 10


def test_discovered_nodes_order_is_deterministic() -> None:
    first = _fingerprint()
    second = _fingerprint()
    assert first == second
    assert first == sorted(first)


def test_load_discovered_nodes_supports_upscale_package() -> None:
    node_ids = [
        node.node_id for node in load_discovered_nodes("infrastructure.comfy_nodes.upscale_nodes")
    ]
    assert node_ids == [
        "upscale_load_image",
        "upscale_model_loader",
        "image_upscale_with_model",
        "upscale_save_image",
    ]


def test_build_comfy_workflow_signature_is_compatible() -> None:
    signature = inspect.signature(build_comfy_workflow)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == [
        "params",
        "reference_image_name",
        "reference_mode",
        "object_info",
        "info",
        "required_input_defaults",
        "select_field_name",
        "skip_base_sampler_pass",
    ]

    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for parameter in parameters[1:]:
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["skip_base_sampler_pass"].default is False
