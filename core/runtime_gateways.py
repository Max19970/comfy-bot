from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore


def _clone_params(params: GenerationParams) -> GenerationParams:
    return GenerationParams.from_generation_request(params.to_generation_request())


@dataclass(slots=True)
class PromptSessionGateway:
    runtime: RuntimeStore

    def get(self, uid: int) -> PromptRequest | None:
        return self.runtime.active_prompt_requests.get(uid)

    def set(self, uid: int, request: PromptRequest) -> PromptRequest:
        self.runtime.active_prompt_requests[uid] = request
        return request

    def pop(self, uid: int) -> PromptRequest | None:
        return self.runtime.active_prompt_requests.pop(uid, None)

    def has(self, uid: int) -> bool:
        return uid in self.runtime.active_prompt_requests

    def set_pending_input(self, uid: int, payload: Mapping[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for key, value in payload.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            normalized[key_name] = str(value)
        self.runtime.pending_image_inputs[uid] = normalized
        return normalized

    def get_pending_input(self, uid: int) -> dict[str, str] | None:
        payload = self.runtime.pending_image_inputs.get(uid)
        if payload is None:
            return None
        return dict(payload)

    def pop_pending_input(self, uid: int) -> dict[str, str] | None:
        payload = self.runtime.pending_image_inputs.pop(uid, None)
        if payload is None:
            return None
        return dict(payload)


@dataclass(slots=True)
class GenerationGateway:
    runtime: RuntimeStore

    def get_active(self, generation_id: str) -> ActiveGeneration | None:
        return self.runtime.active_generations.get(generation_id)

    def set_active(self, generation: ActiveGeneration) -> ActiveGeneration:
        self.runtime.active_generations[generation.generation_id] = generation
        return generation

    def pop_active(self, generation_id: str) -> ActiveGeneration | None:
        return self.runtime.active_generations.pop(generation_id, None)

    def list_for_user(self, uid: int) -> list[ActiveGeneration]:
        return [item for item in self.runtime.active_generations.values() if item.owner_uid == uid]

    def set_last_params(self, uid: int, params: GenerationParams) -> GenerationParams:
        copied = _clone_params(params)
        self.runtime.last_params[uid] = copied
        return copied

    def get_last_params(self, uid: int, *, clone: bool = False) -> GenerationParams | None:
        params = self.runtime.last_params.get(uid)
        if params is None:
            return None
        return _clone_params(params) if clone else params

    def set_last_seed(self, uid: int, seed: int) -> int:
        normalized = int(seed)
        self.runtime.last_seeds[uid] = normalized
        return normalized

    def get_last_seed(self, uid: int, default: int | None = None) -> int | None:
        return self.runtime.last_seeds.get(uid, default)

    def register_preview_artifact(self, artifact: PreviewArtifact) -> None:
        self.runtime.register_preview_artifact(artifact)

    def get_preview_artifact(self, artifact_id: str) -> PreviewArtifact | None:
        return self.runtime.preview_artifacts.get(artifact_id)

    def drop_preview_artifact(self, artifact_id: str) -> None:
        self.runtime.drop_preview_artifact(artifact_id)

    def artifact_bytes(self, artifact: PreviewArtifact) -> bytes | None:
        return self.runtime.artifact_bytes(artifact)

    def prune_preview_artifacts(self, uid: int, *, max_items: int = 40) -> None:
        self.runtime.prune_preview_artifacts(uid, max_items=max_items)


@dataclass(slots=True)
class DownloadSessionGateway:
    runtime: RuntimeStore

    def get(self, uid: int) -> asyncio.Task[Any] | None:
        return self.runtime.active_downloads.get(uid)

    def set(self, uid: int, task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        self.runtime.active_downloads[uid] = task
        return task

    def pop(self, uid: int) -> asyncio.Task[Any] | None:
        return self.runtime.active_downloads.pop(uid, None)

    def cancel(self, uid: int) -> bool:
        task = self.runtime.active_downloads.get(uid)
        if task is None:
            return False
        if not task.done():
            task.cancel()
        self.runtime.active_downloads.pop(uid, None)
        return True


@dataclass(slots=True)
class UserPreferencesGateway:
    runtime: RuntimeStore

    def ensure(self, uid: int) -> dict[str, Any]:
        if uid not in self.runtime.user_preferences:
            self.runtime.user_preferences[uid] = {}
        return self.runtime.user_preferences[uid]

    def get(self, uid: int) -> dict[str, Any]:
        return self.runtime.user_preferences.get(uid, {})

    def get_pref(self, uid: int, key: str, default: Any = None) -> Any:
        return self.runtime.user_preferences.get(uid, {}).get(key, default)

    def set_pref(self, uid: int, key: str, value: Any) -> Any:
        prefs = self.ensure(uid)
        prefs[key] = value
        return value

    def update(self, uid: int, values: Mapping[str, Any]) -> dict[str, Any]:
        prefs = self.ensure(uid)
        prefs.update(values)
        return prefs

    def get_ui_panel(self, uid: int) -> dict[str, int] | None:
        panel = self.runtime.user_ui_panels.get(uid)
        if panel is None:
            return None
        return {"chat_id": int(panel["chat_id"]), "message_id": int(panel["message_id"])}

    def set_ui_panel(self, uid: int, *, chat_id: int, message_id: int) -> dict[str, int]:
        panel = {"chat_id": int(chat_id), "message_id": int(message_id)}
        self.runtime.user_ui_panels[uid] = panel
        return panel

    def clear_ui_panel(self, uid: int) -> bool:
        return self.runtime.user_ui_panels.pop(uid, None) is not None


@dataclass(slots=True)
class RuntimeGateways:
    runtime: RuntimeStore
    prompt_sessions: PromptSessionGateway
    generations: GenerationGateway
    downloads: DownloadSessionGateway
    user_preferences: UserPreferencesGateway
    extensions: dict[str, object] = field(default_factory=dict)

    def register_extension(self, name: str, extension: object) -> object:
        key = str(name).strip()
        if not key:
            raise ValueError("Runtime gateway extension name must not be empty")
        if key in self.extensions:
            raise ValueError(f"Duplicate runtime gateway extension: {key}")
        self.extensions[key] = extension
        return extension

    def get_extension(self, name: str) -> object | None:
        return self.extensions.get(str(name).strip())


def create_runtime_gateways(runtime: RuntimeStore) -> RuntimeGateways:
    return RuntimeGateways(
        runtime=runtime,
        prompt_sessions=PromptSessionGateway(runtime),
        generations=GenerationGateway(runtime),
        downloads=DownloadSessionGateway(runtime),
        user_preferences=UserPreferencesGateway(runtime),
    )
