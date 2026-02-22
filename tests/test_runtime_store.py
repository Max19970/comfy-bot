from __future__ import annotations

from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore
from domain.loras import EditorLoraSelection


def test_runtime_store_roundtrip_preserves_persisted_fields() -> None:
    uid = 1001
    params = GenerationParams(
        positive="cat portrait",
        negative="blurry",
        checkpoint="sdxl.safetensors",
        loras=[("anime_style.safetensors", 0.8)],
        seed=777,
    )

    req = PromptRequest(params=params, operation="generate")
    req.set_editor_loras(
        [
            EditorLoraSelection.create(
                "anime_style.safetensors",
                0.8,
                file_path="models/loras/anime_style.safetensors",
            )
        ]
    )

    runtime = RuntimeStore(
        last_params={uid: params},
        last_seeds={uid: 777},
        active_prompt_requests={uid: req},
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
    assert restored.active_prompt_requests[uid].editor_loras[0].file_path == (
        "models/loras/anime_style.safetensors"
    )
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


def test_runtime_store_migrates_legacy_prompt_request_payload_to_schema_v2() -> None:
    restored = RuntimeStore.from_persisted_dict(
        {
            "schema_version": 1,
            "active_prompt_requests": {
                "1001": {
                    "operation": "generate",
                    "params": {
                        "checkpoint": "sdxl.safetensors",
                        "loras": [["anime_style.safetensors", 0.9]],
                    },
                }
            },
        }
    )

    req = restored.active_prompt_requests[1001]
    assert req.params.loras == [("anime_style.safetensors", 0.9)]
    assert req.editor_loras == [EditorLoraSelection(name="anime_style.safetensors", strength=0.9)]

    persisted = restored.to_persisted_dict()
    assert persisted["schema_version"] == 2
    entry = persisted["active_prompt_requests"]["1001"]
    assert entry["editor_loras"] == [
        {
            "name": "anime_style.safetensors",
            "strength": 0.9,
            "file_path": "",
        }
    ]
