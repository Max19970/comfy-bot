from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from core.models import GenerationParams
from handlers.prompt_editor_session import normalize_generation_params


@dataclass
class _ClientInfo:
    checkpoints: list[str] = field(default_factory=lambda: ["ckpt-a"])
    samplers: list[str] = field(default_factory=lambda: ["sampler-a"])
    schedulers: list[str] = field(default_factory=lambda: ["scheduler-a"])
    upscale_models: list[str] = field(default_factory=lambda: ["upscaler-a"])
    vaes: list[str] = field(default_factory=lambda: ["vae-a"])
    controlnets: list[str] = field(default_factory=lambda: ["cn-a"])
    embeddings: list[str] = field(default_factory=lambda: ["emb-a"])
    loras: list[str] = field(default_factory=lambda: ["lora-a"])


@dataclass
class _FakeClient:
    info: _ClientInfo = field(default_factory=_ClientInfo)


def test_normalize_generation_params_clamps_and_filters_values() -> None:
    params = GenerationParams(
        sampler="unknown",
        scheduler="unknown",
        checkpoint="",
        upscale_model="missing",
        vae_name="missing",
        controlnet_name="missing",
        embedding_name="missing",
        controlnet_strength=cast(Any, "bad"),
        reference_strength=cast(Any, "3.7"),
        loras=[("lora-a", 0.8), ("missing", 1.0)],
    )

    normalized = normalize_generation_params(
        params, cast(Any, _FakeClient()), max_reference_images=4
    )

    assert normalized.checkpoint == "ckpt-a"
    assert normalized.sampler == "sampler-a"
    assert normalized.scheduler == "scheduler-a"
    assert normalized.upscale_model == ""
    assert normalized.vae_name == ""
    assert normalized.controlnet_name == ""
    assert normalized.embedding_name == ""
    assert normalized.controlnet_strength == 1.0
    assert normalized.reference_strength == 2.0
    assert normalized.loras == [("lora-a", 0.8)]


def test_normalize_generation_params_normalizes_reference_items() -> None:
    params = GenerationParams()
    params.reference_images = cast(
        Any,
        [
            {"file_id": "img-1", "id": "ref-1"},
            {"file_id": "img-2"},
            {"file_id": "   "},
            "img-3",
            "   ",
        ],
    )

    normalized = normalize_generation_params(
        params, cast(Any, _FakeClient()), max_reference_images=3
    )

    assert len(normalized.reference_images) == 3
    assert normalized.reference_images[0] == {"id": "ref-1", "file_id": "img-1"}
    assert normalized.reference_images[1]["file_id"] == "img-2"
    assert normalized.reference_images[1]["id"]
    assert normalized.reference_images[2]["file_id"] == "img-3"
    assert normalized.reference_images[2]["id"]
