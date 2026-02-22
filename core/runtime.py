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
from core.user_preferences import normalize_user_preferences
from domain.loras import EditorLoraSelection, editor_lora_selections_from_legacy

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = 2


@dataclass
class PromptRequest:
    params: GenerationParams
    operation: str = "generate"
    ui_chat_id: int | None = None
    ui_message_id: int | None = None
    editor_loras: list[EditorLoraSelection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.editor_loras:
            self.set_editor_loras(self.editor_loras)
            return
        self.editor_loras = self.params.lora_selections()

    def set_editor_loras(self, selections: list[EditorLoraSelection]) -> None:
        normalized = [
            EditorLoraSelection.create(
                item.name,
                item.strength,
                file_path=item.file_path,
            )
            for item in selections
        ]
        self.editor_loras = normalized
        self.params.set_lora_selections(normalized)

    def sync_editor_loras_from_params(self) -> None:
        by_name = {
            item.name.casefold(): item.file_path for item in self.editor_loras if item.file_path
        }
        synced = [
            EditorLoraSelection.create(
                item.name,
                item.strength,
                file_path=by_name.get(item.name.casefold(), ""),
            )
            for item in self.params.lora_selections()
        ]
        self.editor_loras = synced

    def add_editor_lora(self, selection: EditorLoraSelection) -> None:
        chain = list(self.editor_loras)
        chain.append(selection)
        self.set_editor_loras(chain)

    def remove_last_editor_lora(self) -> bool:
        if not self.editor_loras:
            return False
        chain = list(self.editor_loras)
        chain.pop()
        self.set_editor_loras(chain)
        return True

    def clear_editor_loras(self) -> int:
        count = len(self.editor_loras)
        if count > 0:
            self.set_editor_loras([])
        return count


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
    compression_percent: int = 100
    shrink_width: int | None = None
    shrink_height: int | None = None
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
        snapshot = _RuntimeSessionSnapshot.from_runtime(self)
        return snapshot.to_dict()

    def persist(self) -> None:
        try:
            save_runtime_session(self.to_persisted_dict())
        except (OSError, TypeError, ValueError):
            logger.warning("Failed to persist runtime session", exc_info=True)

    @classmethod
    def from_persisted_dict(cls, raw: dict[str, Any]) -> RuntimeStore:
        snapshot = _RuntimeSessionSnapshot.from_raw(raw)
        return snapshot.to_runtime_store()


def _clone_prompt_request(req: PromptRequest) -> PromptRequest:
    clone = PromptRequest(
        params=GenerationParams.from_generation_request(req.params.to_generation_request()),
        operation=req.operation,
        editor_loras=list(req.editor_loras),
    )
    clone.ui_chat_id = req.ui_chat_id
    clone.ui_message_id = req.ui_message_id
    return clone


def _editor_loras_to_dict(items: list[EditorLoraSelection]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
        name = str(item.name or "").strip()
        if not name:
            continue
        payload.append(
            {
                "name": name,
                "strength": float(item.strength),
                "file_path": str(item.file_path or "").strip(),
            }
        )
    return payload


def _editor_loras_from_raw(raw: Any) -> list[EditorLoraSelection]:
    if not isinstance(raw, list):
        return []

    parsed: list[EditorLoraSelection] = []
    for item in raw:
        if isinstance(item, dict):
            selection = EditorLoraSelection.from_legacy(item)
            if selection is not None:
                parsed.append(selection)
    return parsed


def _migrate_runtime_payload(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    try:
        schema_version = int(raw.get("schema_version") or 1)
    except (TypeError, ValueError):
        schema_version = 1
    if schema_version >= RUNTIME_SCHEMA_VERSION:
        return raw

    migrated: dict[str, Any] = dict(raw)
    migrated["schema_version"] = RUNTIME_SCHEMA_VERSION

    active_requests = raw.get("active_prompt_requests")
    if isinstance(active_requests, dict):
        upgraded_active: dict[str, Any] = {}
        for raw_uid, payload in active_requests.items():
            if not isinstance(payload, dict):
                continue

            upgraded_payload = dict(payload)
            if "editor_loras" not in upgraded_payload:
                params_raw = upgraded_payload.get("params")
                loras_raw = params_raw.get("loras", []) if isinstance(params_raw, dict) else []
                upgraded_payload["editor_loras"] = _editor_loras_to_dict(
                    editor_lora_selections_from_legacy(loras_raw)
                )
            upgraded_active[str(raw_uid)] = upgraded_payload
        migrated["active_prompt_requests"] = upgraded_active

    return migrated


@dataclass
class _RuntimeSessionSnapshot:
    schema_version: int = RUNTIME_SCHEMA_VERSION
    last_params: dict[int, GenerationParams] = field(default_factory=dict)
    last_seeds: dict[int, int] = field(default_factory=dict)
    active_prompt_requests: dict[int, PromptRequest] = field(default_factory=dict)
    user_preferences: dict[int, dict[str, Any]] = field(default_factory=dict)
    user_ui_panels: dict[int, dict[str, int]] = field(default_factory=dict)
    preview_artifacts: dict[str, PreviewArtifact] = field(default_factory=dict)
    active_generations: dict[str, ActiveGeneration] = field(default_factory=dict)

    @classmethod
    def from_runtime(cls, runtime: RuntimeStore) -> _RuntimeSessionSnapshot:
        snapshot = cls()
        snapshot.last_params = {
            uid: GenerationParams.from_generation_request(params.to_generation_request())
            for uid, params in runtime.last_params.items()
            if uid > 0
        }
        snapshot.last_seeds = {
            uid: int(seed)
            for uid, seed in runtime.last_seeds.items()
            if uid > 0 and isinstance(seed, int)
        }
        snapshot.active_prompt_requests = {
            uid: _clone_prompt_request(req)
            for uid, req in runtime.active_prompt_requests.items()
            if uid > 0
        }
        snapshot.user_preferences = {
            uid: _normalize_user_preferences(prefs)
            for uid, prefs in runtime.user_preferences.items()
            if uid > 0
        }
        snapshot.user_ui_panels = {
            uid: panel
            for uid, panel in (
                (uid, _normalize_user_ui_panel(payload))
                for uid, payload in runtime.user_ui_panels.items()
                if uid > 0
            )
            if panel
        }
        snapshot.preview_artifacts = {
            artifact_id: item
            for artifact_id, item in runtime.preview_artifacts.items()
            if item.owner_uid > 0
        }
        snapshot.active_generations = {
            generation_id: item
            for generation_id, item in runtime.active_generations.items()
            if item.owner_uid > 0
        }
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "last_params": {
                str(uid): params_to_dict(params) for uid, params in self.last_params.items()
            },
            "last_seeds": {str(uid): int(seed) for uid, seed in self.last_seeds.items()},
            "active_prompt_requests": {
                str(uid): {
                    "operation": req.operation,
                    "params": params_to_dict(req.params),
                    "editor_loras": _editor_loras_to_dict(req.editor_loras),
                    "ui_chat_id": req.ui_chat_id,
                    "ui_message_id": req.ui_message_id,
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

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> _RuntimeSessionSnapshot:
        migrated = _migrate_runtime_payload(raw)
        snapshot = cls(
            schema_version=_to_int(migrated.get("schema_version"), default=RUNTIME_SCHEMA_VERSION)
        )

        for uid, payload in _iter_user_dict(migrated.get("last_params")):
            if not isinstance(payload, dict):
                continue
            try:
                snapshot.last_params[uid] = dict_to_params(payload)
            except (TypeError, ValueError, KeyError):
                continue

        for uid, payload in _iter_user_dict(migrated.get("last_seeds")):
            try:
                snapshot.last_seeds[uid] = int(payload)
            except (TypeError, ValueError):
                continue

        for uid, payload in _iter_user_dict(migrated.get("active_prompt_requests")):
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
            editor_loras = _editor_loras_from_raw(payload.get("editor_loras"))
            if not editor_loras:
                editor_loras = params.lora_selections()
            req = PromptRequest(
                params=params,
                operation=operation,
                editor_loras=editor_loras,
            )
            req.ui_chat_id = _as_int_or_none(payload.get("ui_chat_id"))
            req.ui_message_id = _as_int_or_none(payload.get("ui_message_id"))
            snapshot.active_prompt_requests[uid] = req

        for uid, payload in _iter_user_dict(migrated.get("user_preferences")):
            snapshot.user_preferences[uid] = _normalize_user_preferences(payload)

        for uid, payload in _iter_user_dict(migrated.get("user_ui_panels")):
            panel = _normalize_user_ui_panel(payload)
            if panel:
                snapshot.user_ui_panels[uid] = panel

        artifacts_raw = migrated.get("preview_artifacts")
        if isinstance(artifacts_raw, dict):
            for artifact_id, payload in artifacts_raw.items():
                if not isinstance(payload, dict):
                    continue
                artifact = _preview_artifact_from_dict(str(artifact_id), payload)
                if artifact is None:
                    continue
                snapshot.preview_artifacts[artifact.artifact_id] = artifact

        active_raw = migrated.get("active_generations")
        if isinstance(active_raw, dict):
            for generation_id, payload in active_raw.items():
                if not isinstance(payload, dict):
                    continue
                generation = _active_generation_from_dict(str(generation_id), payload)
                if generation is None:
                    continue
                snapshot.active_generations[generation.generation_id] = generation

        return snapshot

    def to_runtime_store(self) -> RuntimeStore:
        runtime = RuntimeStore()
        runtime.last_params.update(self.last_params)
        runtime.last_seeds.update(self.last_seeds)
        runtime.active_prompt_requests.update(self.active_prompt_requests)
        runtime.user_preferences.update(self.user_preferences)
        runtime.user_ui_panels.update(self.user_ui_panels)
        runtime.preview_artifacts.update(self.preview_artifacts)
        runtime.active_generations.update(self.active_generations)
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
    return normalize_user_preferences(raw)


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
        "compression_percent": int(item.compression_percent),
        "shrink_width": item.shrink_width,
        "shrink_height": item.shrink_height,
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
        compression_percent = _to_int(payload.get("compression_percent"), default=100)
        compression_percent = max(1, min(100, compression_percent))
        shrink_width = _as_int_or_none(payload.get("shrink_width"))
        shrink_height = _as_int_or_none(payload.get("shrink_height"))
        if (
            shrink_width is not None
            and shrink_height is not None
            and (shrink_width < 1 or shrink_height < 1)
        ):
            shrink_width = None
            shrink_height = None

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
            compression_percent=compression_percent,
            shrink_width=shrink_width,
            shrink_height=shrink_height,
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
