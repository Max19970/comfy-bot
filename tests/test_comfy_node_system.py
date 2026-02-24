from __future__ import annotations

import inspect

from infrastructure.comfy_nodes.registry import load_discovered_nodes
from infrastructure.comfy_workflow_builder import build_comfy_workflow


def _fingerprint() -> list[tuple[int, int, str]]:
    return [(int(node.phase), int(node.order), node.node_id) for node in load_discovered_nodes()]


def test_load_discovered_nodes_detects_nodes_from_directory() -> None:
    node_ids = [node.node_id for node in load_discovered_nodes()]
    assert "legacy_workflow" in node_ids


def test_discovered_nodes_order_is_deterministic() -> None:
    first = _fingerprint()
    second = _fingerprint()
    assert first == second
    assert first == sorted(first)


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
