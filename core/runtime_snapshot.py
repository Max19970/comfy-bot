from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.models import GenerationParams
from core.storage import dict_to_params, params_to_dict
from core.user_preferences import normalize_user_preferences
from domain.loras import EditorLoraSelection, editor_lora_selections_from_legacy

if TYPE_CHECKING:
    from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore


RUNTIME_SCHEMA_VERSION = 2


def runtime_to_persisted_dict(runtime: RuntimeStore) -> dict[str, Any]:
    snapshot = _RuntimeSessionSnapshot.from_runtime(runtime)
    return snapshot.to_dict()


def runtime_from_persisted_dict(raw: dict[str, Any]) -> RuntimeStore:
    snapshot = _RuntimeSessionSnapshot.from_raw(raw)
    return snapshot.to_runtime_store()


def _clone_prompt_request(req: PromptRequest) -> PromptRequest:
    from core.runtime import PromptRequest

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
        from core.runtime import PromptRequest

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
        from core.runtime import RuntimeStore

        runtime = RuntimeStore()
        runtime.last_params.update(self.last_params)
        runtime.last_seeds.update(self.last_seeds)
        runtime.active_prompt_requests.update(self.active_prompt_requests)
        runtime.user_preferences.update(self.user_preferences)
        runtime.user_ui_panels.update(self.user_ui_panels)
        runtime.preview_artifacts.update(self.preview_artifacts)
        runtime.active_generations.update(self.active_generations)
        return runtime


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
    from core.runtime import PreviewArtifact

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
    from core.runtime import ActiveGeneration

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
