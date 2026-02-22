from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from domain.loras import EditorLoraSelection, LoraCatalogEntry, lora_catalog_entry_from_metadata


class ModelMetadataLookup(Protocol):
    def __call__(
        self,
        filename: str,
        *,
        model_type: str | None = None,
    ) -> dict[str, Any] | None: ...


class BaseModelInferer(Protocol):
    def __call__(self, text: str) -> str: ...


class BaseModelsCompatible(Protocol):
    def __call__(self, checkpoint_base: str, lora_base: str) -> bool: ...


@dataclass(slots=True)
class LoraCatalogService:
    get_model_metadata: ModelMetadataLookup
    infer_base_model: BaseModelInferer
    base_models_compatible: BaseModelsCompatible

    def checkpoint_base_model(self, checkpoint_name: str) -> str:
        return self._resolve_base_model(checkpoint_name, model_type="checkpoint")

    def lora_base_model(self, lora_name: str) -> str:
        entry = self.lora_entry(lora_name)
        if entry and entry.base_model:
            return entry.base_model
        return self._resolve_base_model(lora_name, model_type="lora")

    def lora_entry(self, lora_name: str) -> LoraCatalogEntry | None:
        name = str(lora_name or "").strip()
        if not name:
            return None

        metadata = self.get_model_metadata(name, model_type="lora")
        if not metadata:
            return None

        entry = lora_catalog_entry_from_metadata(name, metadata)
        if not entry.name:
            return None
        return entry

    def lora_trigger_words(self, lora_name: str, *, limit: int = 12) -> list[str]:
        entry = self.lora_entry(lora_name)
        if not entry:
            return []
        return entry.trigger_words(limit=limit)

    def editor_selection(self, lora_name: str, *, strength: Any) -> EditorLoraSelection:
        entry = self.lora_entry(lora_name)
        if entry:
            return entry.editor_selection(strength=strength)
        return EditorLoraSelection.create(lora_name, strength)

    def lora_compatibility(self, checkpoint_name: str, lora_name: str) -> tuple[str, str, str]:
        checkpoint_base = self.checkpoint_base_model(checkpoint_name)
        lora_base = self.lora_base_model(lora_name)
        if checkpoint_base and lora_base:
            if self.base_models_compatible(checkpoint_base, lora_base):
                return "compatible", checkpoint_base, lora_base
            return "incompatible", checkpoint_base, lora_base
        return "unknown", checkpoint_base, lora_base

    def _resolve_base_model(self, filename: str, *, model_type: str) -> str:
        name = str(filename or "").strip()
        if not name:
            return ""

        metadata = self.get_model_metadata(name, model_type=model_type)
        base_model = str(metadata.get("base_model") if metadata else "").strip()
        if base_model:
            return base_model
        return self.infer_base_model(name)
