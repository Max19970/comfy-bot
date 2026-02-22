from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ModelMetadataIndexRepository:
    """Persistent local index for downloaded model metadata."""

    def __init__(
        self,
        models_root_path: str,
        *,
        index_filename: str = ".comfybot_model_index.json",
    ) -> None:
        self._models_root_path = str(models_root_path)
        self._metadata_path = os.path.join(self._models_root_path, index_filename)
        self._metadata_lock = asyncio.Lock()
        self._metadata: dict[str, dict[str, Any]] = self._load_metadata()

    def get(
        self,
        filename: str,
        *,
        model_type: str | None = None,
    ) -> dict[str, Any] | None:
        key = self._metadata_key(filename)
        if not key:
            return None

        raw = self._metadata.get(key)
        if not raw:
            return None

        if model_type and str(raw.get("model_type", "")).strip() != model_type:
            return None

        return dict(raw)

    async def upsert(
        self,
        filename: str,
        entry: dict[str, Any],
        *,
        aliases: list[str] | None = None,
    ) -> None:
        key = self._metadata_key(filename)
        if not key:
            return

        payload = dict(entry)
        alias_keys = [
            alias_key
            for alias in aliases or []
            for alias_key in [self._metadata_key(alias)]
            if alias_key and alias_key != key
        ]

        async with self._metadata_lock:
            self._metadata[key] = payload
            for alias_key in alias_keys:
                self._metadata[alias_key] = payload
            await self._save_metadata()

    async def delete(self, filename: str) -> None:
        key = self._metadata_key(filename)
        if not key:
            return

        async with self._metadata_lock:
            self._metadata.pop(key, None)
            await self._save_metadata()

    def _metadata_key(self, filename: str) -> str:
        return os.path.basename(str(filename or "")).strip().casefold()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        path = self._metadata_path
        if not os.path.exists(path):
            return {}

        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
                payload = payload["models"]
            if not isinstance(payload, dict):
                return {}

            normalized: dict[str, dict[str, Any]] = {}
            for key, value in payload.items():
                if isinstance(value, dict):
                    normalized[str(key)] = value
            return normalized
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.exception("Failed to load model metadata index: %s", path)
            return {}

    async def _save_metadata(self) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": self._metadata,
        }
        os.makedirs(self._models_root_path, exist_ok=True)
        tmp_path = self._metadata_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._metadata_path)
