from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EDITOR_LORA_STRENGTH_MIN = 0.0
EDITOR_LORA_STRENGTH_MAX = 2.0
EDITOR_LORA_DEFAULT_STRENGTH = 1.0


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, *, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _clean_words(raw: Any, *, limit: int = 24) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()

    words: list[str] = []
    seen: set[str] = set()
    for item in raw:
        word = str(item or "").strip()
        if not word:
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        words.append(word)
        if len(words) >= max(1, limit):
            break
    return tuple(words)


def clamp_editor_lora_strength(value: Any) -> float:
    return _clamp(
        _as_float(value, default=EDITOR_LORA_DEFAULT_STRENGTH),
        low=EDITOR_LORA_STRENGTH_MIN,
        high=EDITOR_LORA_STRENGTH_MAX,
    )


@dataclass(frozen=True, slots=True)
class EditorLoraSelection:
    name: str
    strength: float
    file_path: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        strength: Any = EDITOR_LORA_DEFAULT_STRENGTH,
        *,
        file_path: str = "",
    ) -> EditorLoraSelection:
        cleaned = str(name or "").strip()
        return cls(
            name=cleaned,
            strength=clamp_editor_lora_strength(strength),
            file_path=str(file_path or "").strip(),
        )

    @classmethod
    def from_legacy(cls, raw: Any) -> EditorLoraSelection | None:
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            name = str(raw[0] or "").strip()
            if not name:
                return None
            return cls.create(name, raw[1])

        if isinstance(raw, dict):
            name = str(raw.get("name") or raw.get("lora_name") or "").strip()
            if not name:
                return None
            strength = raw.get("strength", raw.get("strength_model", EDITOR_LORA_DEFAULT_STRENGTH))
            file_path = raw.get("file_path", raw.get("path", ""))
            return cls.create(name, strength, file_path=file_path)

        return None

    def to_legacy_pair(self) -> tuple[str, float]:
        return (self.name, float(self.strength))


@dataclass(frozen=True, slots=True)
class WorkflowLoraAttachment:
    lora_name: str
    strength_model: float
    strength_clip: float
    file_path: str = ""

    @classmethod
    def from_editor_selection(cls, selection: EditorLoraSelection) -> WorkflowLoraAttachment:
        return cls(
            lora_name=selection.name,
            strength_model=float(selection.strength),
            strength_clip=float(selection.strength),
            file_path=selection.file_path,
        )


@dataclass(frozen=True, slots=True)
class LoraCatalogEntry:
    name: str
    file_path: str = ""
    base_model: str = ""
    trained_words: tuple[str, ...] = ()
    source: str = ""

    @classmethod
    def from_metadata(
        cls,
        filename: str,
        metadata: dict[str, Any] | None,
    ) -> LoraCatalogEntry:
        payload = metadata if isinstance(metadata, dict) else {}
        name = str(payload.get("filename") or filename or "").strip()
        return cls(
            name=name,
            file_path=str(payload.get("path") or "").strip(),
            base_model=str(payload.get("base_model") or "").strip(),
            trained_words=_clean_words(payload.get("trained_words"), limit=24),
            source=str(payload.get("source") or "").strip(),
        )

    def trigger_words(self, *, limit: int = 12) -> list[str]:
        return list(self.trained_words[: max(1, limit)])

    def editor_selection(
        self, *, strength: Any = EDITOR_LORA_DEFAULT_STRENGTH
    ) -> EditorLoraSelection:
        return EditorLoraSelection.create(self.name, strength, file_path=self.file_path)


def editor_lora_selections_from_legacy(raw: Any) -> list[EditorLoraSelection]:
    if not isinstance(raw, list):
        return []

    selections: list[EditorLoraSelection] = []
    for item in raw:
        parsed = EditorLoraSelection.from_legacy(item)
        if parsed is None:
            continue
        selections.append(parsed)
    return selections


def legacy_lora_pairs(raw: Any) -> list[tuple[str, float]]:
    return [selection.to_legacy_pair() for selection in editor_lora_selections_from_legacy(raw)]


def workflow_lora_attachments_from_legacy(raw: Any) -> list[WorkflowLoraAttachment]:
    return [
        WorkflowLoraAttachment.from_editor_selection(selection)
        for selection in editor_lora_selections_from_legacy(raw)
    ]


def lora_catalog_entry_from_metadata(
    filename: str,
    metadata: dict[str, Any] | None,
) -> LoraCatalogEntry:
    return LoraCatalogEntry.from_metadata(filename, metadata)
