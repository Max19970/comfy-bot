from __future__ import annotations

import core.runtime_persistence as runtime_persistence
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.runtime_persistence import load_runtime_store, persist_runtime_store


def test_persist_runtime_store_writes_runtime_snapshot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_save(payload: dict[str, object]) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(runtime_persistence, "save_runtime_session", _fake_save)

    uid = 1001
    params = GenerationParams(checkpoint="sdxl.safetensors", seed=777)
    runtime = RuntimeStore(
        last_params={uid: params},
        last_seeds={uid: 777},
        active_prompt_requests={uid: PromptRequest(params=params, operation="generate")},
    )

    persist_runtime_store(runtime)

    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("schema_version") == 2
    assert payload.get("last_params", {}).get("1001", {}).get("checkpoint") == "sdxl.safetensors"


def test_load_runtime_store_returns_empty_when_storage_raises(monkeypatch) -> None:
    def _raise_load() -> dict[str, object]:
        raise OSError("storage down")

    monkeypatch.setattr(runtime_persistence, "load_runtime_session", _raise_load)

    runtime = load_runtime_store()

    assert isinstance(runtime, RuntimeStore)
    assert runtime.last_params == {}
    assert runtime.last_seeds == {}
