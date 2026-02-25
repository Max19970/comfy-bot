from __future__ import annotations

import asyncio

from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore
from core.runtime_gateways import create_runtime_gateways


def test_prompt_session_gateway_tracks_requests_and_pending_inputs() -> None:
    runtime = RuntimeStore()
    gateways = create_runtime_gateways(runtime)

    uid = 1001
    request = PromptRequest(
        params=GenerationParams(checkpoint="sdxl.safetensors", seed=42),
        operation="generate",
    )

    gateways.prompt_sessions.set(uid, request)
    assert gateways.prompt_sessions.has(uid) is True
    assert gateways.prompt_sessions.get(uid) is request

    pending = gateways.prompt_sessions.set_pending_input(
        uid,
        {
            "image_name": "source.png",
            "reference_mode": "img2img",
        },
    )
    assert pending == {
        "image_name": "source.png",
        "reference_mode": "img2img",
    }
    assert gateways.prompt_sessions.get_pending_input(uid) == pending

    removed_pending = gateways.prompt_sessions.pop_pending_input(uid)
    assert removed_pending == pending
    assert gateways.prompt_sessions.get_pending_input(uid) is None

    removed_request = gateways.prompt_sessions.pop(uid)
    assert removed_request is request
    assert gateways.prompt_sessions.has(uid) is False


def test_generation_gateway_records_last_params_and_generations() -> None:
    runtime = RuntimeStore()
    gateways = create_runtime_gateways(runtime)

    uid = 1002
    params = GenerationParams(checkpoint="sdxl.safetensors", seed=777)
    copied = gateways.generations.set_last_params(uid, params)
    gateways.generations.set_last_seed(uid, 777)

    params.checkpoint = "changed-after-set"
    assert copied.checkpoint == "sdxl.safetensors"
    assert gateways.generations.get_last_params(uid).checkpoint == "sdxl.safetensors"
    assert gateways.generations.get_last_seed(uid) == 777

    generation = ActiveGeneration(owner_uid=uid, generation_id="gen_1")
    gateways.generations.set_active(generation)
    assert gateways.generations.get_active("gen_1") is generation
    assert gateways.generations.list_for_user(uid) == [generation]

    artifact = PreviewArtifact(
        artifact_id="art_1",
        owner_uid=uid,
        image_bytes=b"",
        params=GenerationParams(checkpoint="sdxl.safetensors"),
        used_seed=777,
    )
    gateways.generations.register_preview_artifact(artifact)
    assert gateways.generations.get_preview_artifact("art_1") is artifact

    gateways.generations.drop_preview_artifact("art_1")
    assert gateways.generations.get_preview_artifact("art_1") is None

    popped_generation = gateways.generations.pop_active("gen_1")
    assert popped_generation is generation
    assert gateways.generations.get_active("gen_1") is None


def test_download_gateway_cancel_clears_active_task() -> None:
    async def _scenario() -> tuple[bool, bool, object]:
        runtime = RuntimeStore()
        gateways = create_runtime_gateways(runtime)

        uid = 1003
        task = asyncio.create_task(asyncio.sleep(5))
        gateways.downloads.set(uid, task)

        cancelled = gateways.downloads.cancel(uid)
        await asyncio.sleep(0)
        return cancelled, task.cancelled(), gateways.downloads.get(uid)

    cancelled, task_cancelled, remaining = asyncio.run(_scenario())
    assert cancelled is True
    assert task_cancelled is True
    assert remaining is None


def test_user_preferences_gateway_manages_prefs_and_panels() -> None:
    runtime = RuntimeStore()
    gateways = create_runtime_gateways(runtime)

    uid = 1004
    gateways.user_preferences.set_pref(uid, "pro_mode", True)
    gateways.user_preferences.update(uid, {"download_source": "all", "locale": "ru"})

    assert gateways.user_preferences.get_pref(uid, "pro_mode", False) is True
    assert gateways.user_preferences.get(uid)["download_source"] == "all"

    panel = gateways.user_preferences.set_ui_panel(uid, chat_id=10, message_id=20)
    assert panel == {"chat_id": 10, "message_id": 20}
    assert gateways.user_preferences.get_ui_panel(uid) == panel
    assert gateways.user_preferences.clear_ui_panel(uid) is True
    assert gateways.user_preferences.get_ui_panel(uid) is None


def test_runtime_gateways_extension_registry_validates_keys_and_duplicates() -> None:
    runtime = RuntimeStore()
    gateways = create_runtime_gateways(runtime)

    extension = {"kind": "custom"}
    assert gateways.register_extension("custom.gateway", extension) is extension
    assert gateways.get_extension("custom.gateway") is extension

    try:
        gateways.register_extension(" ", object())
        raise AssertionError("Expected ValueError for empty extension name")
    except ValueError as exc:
        assert "must not be empty" in str(exc)

    try:
        gateways.register_extension("custom.gateway", object())
        raise AssertionError("Expected ValueError for duplicate extension")
    except ValueError as exc:
        assert "Duplicate runtime gateway extension" in str(exc)
