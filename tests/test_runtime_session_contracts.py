from __future__ import annotations

from core.models import GenerationParams
from core.runtime import RUNTIME_SCHEMA_VERSION, PromptRequest, RuntimeStore


def _expected_runtime_payload_keys() -> set[str]:
    return {
        "schema_version",
        "last_params",
        "last_seeds",
        "active_prompt_requests",
        "user_preferences",
        "user_ui_panels",
        "preview_artifacts",
        "active_generations",
    }


def test_runtime_persisted_payload_keys_are_backward_compatible() -> None:
    uid = 1001
    params = GenerationParams(checkpoint="sdxl.safetensors", seed=123)
    request = PromptRequest(params=params, operation="generate")
    request.ui_chat_id = 10
    request.ui_message_id = 20

    runtime = RuntimeStore(
        last_params={uid: params},
        last_seeds={uid: 123},
        active_prompt_requests={uid: request},
        user_preferences={uid: {"pro_mode": True, "download_source": "all"}},
        user_ui_panels={uid: {"chat_id": 10, "message_id": 20}},
    )

    payload = runtime.to_persisted_dict()

    assert set(payload.keys()) == _expected_runtime_payload_keys()
    assert payload["schema_version"] == RUNTIME_SCHEMA_VERSION
    assert set(payload["last_params"].keys()) == {"1001"}
    assert set(payload["last_seeds"].keys()) == {"1001"}
    assert set(payload["active_prompt_requests"].keys()) == {"1001"}
    assert set(payload["user_preferences"].keys()) == {"1001"}
    assert set(payload["user_ui_panels"].keys()) == {"1001"}

    request_payload = payload["active_prompt_requests"]["1001"]
    assert set(request_payload.keys()) == {
        "operation",
        "params",
        "editor_loras",
        "ui_chat_id",
        "ui_message_id",
    }
    assert request_payload["operation"] == "generate"
    assert request_payload["ui_chat_id"] == 10
    assert request_payload["ui_message_id"] == 20


def test_runtime_schema_v1_migration_preserves_expected_contract_keys() -> None:
    legacy_payload = {
        "schema_version": 1,
        "last_seeds": {"1001": 123},
        "active_prompt_requests": {
            "1001": {
                "operation": "generate",
                "params": {
                    "checkpoint": "sdxl.safetensors",
                    "loras": [["anime_style.safetensors", 0.9]],
                },
            }
        },
        "user_preferences": {"1001": {"pro_mode": True}},
        "user_ui_panels": {"1001": {"chat_id": 10, "message_id": 20}},
    }

    restored = RuntimeStore.from_persisted_dict(legacy_payload)
    migrated_payload = restored.to_persisted_dict()

    assert set(migrated_payload.keys()) == _expected_runtime_payload_keys()
    assert migrated_payload["schema_version"] == RUNTIME_SCHEMA_VERSION

    request_payload = migrated_payload["active_prompt_requests"]["1001"]
    assert "editor_loras" in request_payload
    assert request_payload["editor_loras"] == [
        {
            "name": "anime_style.safetensors",
            "strength": 0.9,
            "file_path": "",
        }
    ]
