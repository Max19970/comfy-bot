from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiogram.types import Message

from core.models import GenerationParams
from core.storage import (
    SESSIONS_DIR,
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
    task: asyncio.Task[Any] | None = None
    kind: str = "generate"
    title: str = "Генерация"
    prompt_id: str | None = None
    status_msg: Message | None = None
    status_chat_id: int | None = None
    status_message_id: int | None = None
    restored: bool = False
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
    preview_chat_id: int | None = None
    preview_message_id: int | None = None
    image_path: str | None = None


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

    def _preview_cache_dir(self) -> Path:
        path = SESSIONS_DIR / "preview_artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_preview_artifact(self, artifact: PreviewArtifact) -> None:
        if artifact.image_bytes:
            path = self._preview_cache_dir() / f"{artifact.owner_uid}_{artifact.artifact_id}.png"
            try:
                path.write_bytes(artifact.image_bytes)
                artifact.image_path = str(path)
            except OSError:
                logger.warning("Failed to write preview artifact image", exc_info=True)
        self.preview_artifacts[artifact.artifact_id] = artifact

    def drop_preview_artifact(self, artifact_id: str) -> None:
        artifact = self.preview_artifacts.pop(artifact_id, None)
        if artifact is None or not artifact.image_path:
            return
        try:
            path = Path(artifact.image_path)
            if path.exists():
                path.unlink()
        except OSError:
            logger.debug("Failed to remove preview artifact file", exc_info=True)

    def artifact_bytes(self, artifact: PreviewArtifact) -> bytes | None:
        if artifact.image_bytes:
            return artifact.image_bytes
        if not artifact.image_path:
            return None
        try:
            data = Path(artifact.image_path).read_bytes()
            artifact.image_bytes = data
            return data
        except OSError:
            return None

    def prune_preview_artifacts(self, owner_uid: int, *, max_items: int = 40) -> None:
        owned = [item for item in self.preview_artifacts.values() if item.owner_uid == owner_uid]
        if len(owned) <= max_items:
            return
        owned.sort(key=lambda item: item.created_at)
        to_remove = owned[: len(owned) - max_items]
        for item in to_remove:
            self.drop_preview_artifact(item.artifact_id)

    def to_persisted_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "last_params": {
                str(uid): params_to_dict(params) for uid, params in self.last_params.items()
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
            "preview_artifacts": {
                aid: _preview_artifact_to_dict(item)
                for aid, item in self.preview_artifacts.items()
                if item.owner_uid > 0
            },
            "active_generations": {
                gid: _active_generation_to_dict(item)
                for gid, item in self.active_generations.items()
                if item.owner_uid > 0
            },
        }

    def persist(self) -> None:
        try:
            save_runtime_session(self.to_persisted_dict())
        except (OSError, TypeError, ValueError):
            logger.warning("Failed to persist runtime session", exc_info=True)

    @classmethod
    def from_persisted_dict(cls, raw: dict[str, Any]) -> RuntimeStore:
        runtime = cls()

        for uid, payload in _iter_user_dict(raw.get("last_params")):
            if not isinstance(payload, dict):
                continue
            try:
                runtime.last_params[uid] = dict_to_params(payload)
            except (TypeError, ValueError, KeyError):
                continue

        for uid, payload in _iter_user_dict(raw.get("last_seeds")):
            try:
                runtime.last_seeds[uid] = int(payload)
            except (TypeError, ValueError):
                continue

        for uid, payload in _iter_user_dict(raw.get("active_prompt_requests")):
            if not isinstance(payload, dict):
                continue
            params_raw = payload.get("params")
            if not isinstance(params_raw, dict):
                continue
            try:
                params = dict_to_params(params_raw)
            except (TypeError, ValueError, KeyError):
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

        artifacts_raw = raw.get("preview_artifacts")
        if isinstance(artifacts_raw, dict):
            for artifact_id, payload in artifacts_raw.items():
                if not isinstance(payload, dict):
                    continue
                artifact = _preview_artifact_from_dict(str(artifact_id), payload)
                if artifact is None:
                    continue
                runtime.preview_artifacts[artifact.artifact_id] = artifact

        active_raw = raw.get("active_generations")
        if isinstance(active_raw, dict):
            for generation_id, payload in active_raw.items():
                if not isinstance(payload, dict):
                    continue
                generation = _active_generation_from_dict(str(generation_id), payload)
                if generation is None:
                    continue
                runtime.active_generations[generation.generation_id] = generation

        return runtime


def load_runtime_store() -> RuntimeStore:
    raw = load_runtime_session()
    if not raw:
        return RuntimeStore()
    try:
        return RuntimeStore.from_persisted_dict(raw)
    except (TypeError, ValueError, OSError):
        return RuntimeStore()


def _iter_user_dict(raw: Any) -> list[tuple[int, Any]]:
    if not isinstance(raw, dict):
        return []

    items: list[tuple[int, Any]] = []
    for raw_uid, value in raw.items():
        try:
            uid = int(raw_uid)
        except (TypeError, ValueError):
            continue
        if uid <= 0:
            continue
        items.append((uid, value))
    return items


def _normalize_user_preferences(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    training_mode_raw = str(raw.get("training_mode", "simple")).strip().lower()
    training_mode = training_mode_raw if training_mode_raw in {"simple", "advanced"} else "simple"
    training_page_raw = raw.get("training_page", 0)
    training_page = training_page_raw if isinstance(training_page_raw, int) else 0

    normalized: dict[str, Any] = {
        "pro_mode": bool(raw.get("pro_mode", False)),
        "training_mode": training_mode,
        "training_page": max(0, training_page),
    }

    int_fields = {
        "gen_width": (64, 4096),
        "gen_height": (64, 4096),
        "gen_steps": (1, 200),
        "gen_seed": (-1, 2**31 - 1),
        "gen_batch": (1, 16),
    }
    for key, (min_v, max_v) in int_fields.items():
        value = raw.get(key)
        if isinstance(value, int):
            normalized[key] = max(min_v, min(max_v, value))

    float_fields = {
        "gen_cfg": (0.0, 30.0),
        "gen_denoise": (0.0, 1.0),
    }
    for key, (min_f, max_f) in float_fields.items():
        value = raw.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = max(min_f, min(max_f, float(value)))

    for key in ("gen_sampler", "gen_scheduler"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()

    source = raw.get("dl_default_source")
    if isinstance(source, str) and source in {"all", "civitai", "huggingface"}:
        normalized["dl_default_source"] = source
    sort_code = raw.get("dl_default_sort")
    if isinstance(sort_code, str) and sort_code in {"downloads", "rating", "newest"}:
        normalized["dl_default_sort"] = sort_code
    period = raw.get("dl_default_period")
    if isinstance(period, str) and period in {"all", "month", "week"}:
        normalized["dl_default_period"] = period
    base = raw.get("dl_default_base")
    if isinstance(base, str) and base in {"all", "sd15", "sdxl", "pony", "flux"}:
        normalized["dl_default_base"] = base
    if isinstance(raw.get("dl_default_nsfw"), bool):
        normalized["dl_default_nsfw"] = raw.get("dl_default_nsfw")
    author = raw.get("dl_default_author")
    if isinstance(author, str):
        normalized["dl_default_author"] = author.strip().replace("@", "")[:256]

    return normalized


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


def _preview_artifact_to_dict(item: PreviewArtifact) -> dict[str, Any]:
    return {
        "owner_uid": int(item.owner_uid),
        "params": params_to_dict(item.params),
        "used_seed": int(item.used_seed),
        "created_at": float(item.created_at),
        "parent_artifact_id": item.parent_artifact_id,
        "generation_step": int(item.generation_step),
        "enable_sampler_pass": bool(item.enable_sampler_pass),
        "preview_chat_id": item.preview_chat_id,
        "preview_message_id": item.preview_message_id,
        "image_path": item.image_path,
    }


def _preview_artifact_from_dict(
    artifact_id: str,
    payload: dict[str, Any],
) -> PreviewArtifact | None:
    try:
        owner_uid = int(payload.get("owner_uid") or 0)
        if owner_uid <= 0:
            return None
        params_raw = payload.get("params")
        if not isinstance(params_raw, dict):
            return None
        params = dict_to_params(params_raw)
        image_path_raw = payload.get("image_path")
        image_path = str(image_path_raw).strip() if image_path_raw else None
        image_bytes = b""
        if image_path:
            path = Path(image_path)
            if path.exists():
                image_bytes = path.read_bytes()
            else:
                image_path = None

        preview_chat_id = _as_int_or_none(payload.get("preview_chat_id"))
        preview_message_id = _as_int_or_none(payload.get("preview_message_id"))

        used_seed = _to_int(payload.get("used_seed"), default=-1)
        created_at = _to_float(payload.get("created_at"), default=time.time())
        generation_step = _to_int(payload.get("generation_step"), default=0)

        return PreviewArtifact(
            artifact_id=artifact_id,
            owner_uid=owner_uid,
            image_bytes=image_bytes,
            params=params,
            used_seed=used_seed,
            created_at=created_at,
            parent_artifact_id=(
                str(payload.get("parent_artifact_id")).strip()
                if payload.get("parent_artifact_id")
                else None
            ),
            generation_step=generation_step,
            enable_sampler_pass=bool(payload.get("enable_sampler_pass", False)),
            preview_chat_id=preview_chat_id,
            preview_message_id=preview_message_id,
            image_path=image_path,
        )
    except (TypeError, ValueError, KeyError, OSError):
        return None


def _active_generation_to_dict(item: ActiveGeneration) -> dict[str, Any]:
    status_chat_id = item.status_chat_id
    status_message_id = item.status_message_id
    if item.status_msg is not None:
        status_chat_id = item.status_msg.chat.id
        status_message_id = item.status_msg.message_id
    return {
        "owner_uid": int(item.owner_uid),
        "kind": str(item.kind or "generate"),
        "title": str(item.title or "Генерация"),
        "prompt_id": item.prompt_id,
        "created_at": float(item.created_at),
        "status_chat_id": status_chat_id,
        "status_message_id": status_message_id,
    }


def _active_generation_from_dict(
    generation_id: str,
    payload: dict[str, Any],
) -> ActiveGeneration | None:
    try:
        owner_uid = _to_int(payload.get("owner_uid"), default=0)
        if owner_uid <= 0:
            return None
        status_chat_id = _as_int_or_none(payload.get("status_chat_id"))
        status_message_id = _as_int_or_none(payload.get("status_message_id"))

        return ActiveGeneration(
            owner_uid=owner_uid,
            generation_id=generation_id,
            task=None,
            kind=str(payload.get("kind") or "generate"),
            title=str(payload.get("title") or "Генерация"),
            prompt_id=(str(payload.get("prompt_id")).strip() if payload.get("prompt_id") else None),
            status_chat_id=status_chat_id,
            status_message_id=status_message_id,
            restored=True,
            created_at=_to_float(payload.get("created_at"), default=time.time()),
        )
    except (TypeError, ValueError, KeyError):
        return None


def _as_int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _to_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
