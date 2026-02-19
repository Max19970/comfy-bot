from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aiogram.types import Message

from comfyui_client import GenerationParams
from core.storage import (
    dict_to_params,
    load_runtime_session,
    params_to_dict,
    save_runtime_session,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptRequest:
    params: GenerationParams
    operation: str = "generate"
    ui_chat_id: int | None = None
    ui_message_id: int | None = None


@dataclass
class ActiveGeneration:
    owner_uid: int
    generation_id: str
    task: asyncio.Task[Any]
    prompt_id: str | None = None
    status_msg: Message | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class PreviewArtifact:
    artifact_id: str
    owner_uid: int
    image_bytes: bytes
    params: GenerationParams
    used_seed: int
    created_at: float = field(default_factory=time.time)
    parent_artifact_id: str | None = None
    generation_step: int = 0
    enable_sampler_pass: bool = False


@dataclass
class RuntimeStore:
    last_params: dict[int, GenerationParams] = field(default_factory=dict)
    last_seeds: dict[int, int] = field(default_factory=dict)
    active_prompt_requests: dict[int, PromptRequest] = field(default_factory=dict)
    active_downloads: dict[int, asyncio.Task[Any]] = field(default_factory=dict)
    active_generations: dict[str, ActiveGeneration] = field(default_factory=dict)
    active_image_jobs: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    pending_image_inputs: dict[int, dict[str, str]] = field(default_factory=dict)
    user_preferences: dict[int, dict] = field(default_factory=dict)
    user_ui_panels: dict[int, dict[str, int]] = field(default_factory=dict)
    preview_artifacts: dict[str, PreviewArtifact] = field(default_factory=dict)

    def to_persisted_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "last_params": {
                str(uid): params_to_dict(params)
                for uid, params in self.last_params.items()
            },
            "last_seeds": {
                str(uid): int(seed)
                for uid, seed in self.last_seeds.items()
                if isinstance(seed, int)
            },
            "active_prompt_requests": {
                str(uid): {
                    "operation": req.operation,
                    "params": params_to_dict(req.params),
                }
                for uid, req in self.active_prompt_requests.items()
            },
            "user_preferences": {
                str(uid): _normalize_user_preferences(prefs)
                for uid, prefs in self.user_preferences.items()
            },
            "user_ui_panels": {
                str(uid): _normalize_user_ui_panel(panel)
                for uid, panel in self.user_ui_panels.items()
            },
        }

    def persist(self) -> None:
        try:
            save_runtime_session(self.to_persisted_dict())
        except Exception:
            logger.warning("Failed to persist runtime session", exc_info=True)

    @classmethod
    def from_persisted_dict(cls, raw: dict[str, Any]) -> "RuntimeStore":
        runtime = cls()

        for uid, payload in _iter_user_dict(raw.get("last_params")):
            if not isinstance(payload, dict):
                continue
            try:
                runtime.last_params[uid] = dict_to_params(payload)
            except Exception:
                continue

        for uid, payload in _iter_user_dict(raw.get("last_seeds")):
            try:
                runtime.last_seeds[uid] = int(payload)
            except Exception:
                continue

        for uid, payload in _iter_user_dict(raw.get("active_prompt_requests")):
            if not isinstance(payload, dict):
                continue
            params_raw = payload.get("params")
            if not isinstance(params_raw, dict):
                continue
            try:
                params = dict_to_params(params_raw)
            except Exception:
                continue
            operation = str(payload.get("operation") or "generate")
            runtime.active_prompt_requests[uid] = PromptRequest(
                params=params,
                operation=operation,
            )

        for uid, payload in _iter_user_dict(raw.get("user_preferences")):
            runtime.user_preferences[uid] = _normalize_user_preferences(payload)

        for uid, payload in _iter_user_dict(raw.get("user_ui_panels")):
            panel = _normalize_user_ui_panel(payload)
            if panel:
                runtime.user_ui_panels[uid] = panel

        return runtime


def load_runtime_store() -> RuntimeStore:
    raw = load_runtime_session()
    if not raw:
        return RuntimeStore()
    try:
        return RuntimeStore.from_persisted_dict(raw)
    except Exception:
        return RuntimeStore()


def _iter_user_dict(raw: Any) -> list[tuple[int, Any]]:
    if not isinstance(raw, dict):
        return []

    items: list[tuple[int, Any]] = []
    for raw_uid, value in raw.items():
        try:
            uid = int(raw_uid)
        except Exception:
            continue
        if uid <= 0:
            continue
        items.append((uid, value))
    return items


def _normalize_user_preferences(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}

    return {
        "pro_mode": bool(raw.get("pro_mode", False)),
    }


def get_user_pro_mode(runtime: RuntimeStore, uid: int) -> bool:
    return runtime.user_preferences.get(uid, {}).get("pro_mode", False)


def set_user_pro_mode(runtime: RuntimeStore, uid: int, value: bool) -> None:
    if uid not in runtime.user_preferences:
        runtime.user_preferences[uid] = {}
    runtime.user_preferences[uid]["pro_mode"] = value


def _normalize_user_ui_panel(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}

    chat_id = raw.get("chat_id")
    message_id = raw.get("message_id")
    if not isinstance(chat_id, int) or not isinstance(message_id, int):
        return {}
    return {
        "chat_id": chat_id,
        "message_id": message_id,
    }
