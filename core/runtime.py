from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiogram.types import Message

from core.models import GenerationParams
from core.runtime_persistence import load_runtime_store as _load_runtime_store
from core.runtime_persistence import persist_runtime_store
from core.runtime_snapshot import RUNTIME_SCHEMA_VERSION as _RUNTIME_SCHEMA_VERSION
from core.runtime_snapshot import runtime_from_persisted_dict, runtime_to_persisted_dict
from core.storage import SESSIONS_DIR
from domain.loras import EditorLoraSelection

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = _RUNTIME_SCHEMA_VERSION
GENERATION_TITLE_DEFAULT = "Генерация"
GENERATION_TITLE_KEY = "core.runtime.active_generation.title"

TranslateText = Callable[[str, str | None, str], str]


def resolve_generation_title(
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    if translate is None:
        return GENERATION_TITLE_DEFAULT
    return translate(GENERATION_TITLE_KEY, locale, GENERATION_TITLE_DEFAULT)


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
    title: str = GENERATION_TITLE_DEFAULT
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
        return runtime_to_persisted_dict(self)

    def persist(self) -> None:
        persist_runtime_store(self)

    @classmethod
    def from_persisted_dict(cls, raw: dict[str, Any]) -> RuntimeStore:
        return runtime_from_persisted_dict(raw)


def load_runtime_store() -> RuntimeStore:
    return _load_runtime_store()


def get_user_pro_mode(runtime: RuntimeStore, uid: int) -> bool:
    return runtime.user_preferences.get(uid, {}).get("pro_mode", False)


def set_user_pro_mode(runtime: RuntimeStore, uid: int, value: bool) -> None:
    if uid not in runtime.user_preferences:
        runtime.user_preferences[uid] = {}
    runtime.user_preferences[uid]["pro_mode"] = value
