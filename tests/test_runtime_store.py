from __future__ import annotations

from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore


def test_runtime_store_roundtrip_preserves_persisted_fields() -> None:
    uid = 1001
    params = GenerationParams(
        positive="cat portrait",
        negative="blurry",
        checkpoint="sdxl.safetensors",
        loras=[("anime_style.safetensors", 0.8)],
        seed=777,
    )

    runtime = RuntimeStore(
        last_params={uid: params},
        last_seeds={uid: 777},
        active_prompt_requests={uid: PromptRequest(params=params, operation="generate")},
        user_preferences={uid: {"pro_mode": True, "download_source": "all"}},
        user_ui_panels={uid: {"chat_id": 10, "message_id": 20}},
    )
    runtime.preview_artifacts["art_1"] = PreviewArtifact(
        artifact_id="art_1",
        owner_uid=uid,
        image_bytes=b"",
        params=params,
        used_seed=777,
        preview_chat_id=10,
        preview_message_id=20,
    )
    runtime.active_generations["gen_1"] = ActiveGeneration(
        owner_uid=uid,
        generation_id="gen_1",
        kind="generate",
        title="Генерация",
        prompt_id="prompt_1",
        status_chat_id=10,
        status_message_id=20,
    )

    payload = runtime.to_persisted_dict()
    restored = RuntimeStore.from_persisted_dict(payload)

    assert restored.last_seeds[uid] == 777
    assert restored.last_params[uid].checkpoint == "sdxl.safetensors"
    assert restored.active_prompt_requests[uid].params.loras == [("anime_style.safetensors", 0.8)]
    assert restored.user_preferences[uid]["pro_mode"] is True
    assert restored.user_ui_panels[uid] == {"chat_id": 10, "message_id": 20}

    restored_artifact = restored.preview_artifacts["art_1"]
    assert restored_artifact.owner_uid == uid
    assert restored_artifact.preview_chat_id == 10
    assert restored_artifact.preview_message_id == 20

    restored_generation = restored.active_generations["gen_1"]
    assert restored_generation.owner_uid == uid
    assert restored_generation.prompt_id == "prompt_1"
    assert restored_generation.restored is True


def test_runtime_store_from_persisted_ignores_invalid_user_payloads() -> None:
    restored = RuntimeStore.from_persisted_dict(
        {
            "last_seeds": {
                "bad": 1,
                "-10": 2,
            },
            "user_ui_panels": {
                "1001": {"chat_id": "nope", "message_id": 42},
            },
            "active_prompt_requests": {
                "1002": {"operation": "generate", "params": "not_a_dict"},
            },
        }
    )

    assert restored.last_seeds == {}
    assert restored.user_ui_panels == {}
    assert restored.active_prompt_requests == {}
