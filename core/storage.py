from __future__ import annotations

import dataclasses as _dc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.models import GenerationParams

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"
SMART_PROMPT_GUIDES_DIR = Path(__file__).resolve().parent.parent / "smart_prompt_guides"
SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"


def _presets_path(user_id: int) -> Path:
    PRESETS_DIR.mkdir(exist_ok=True)
    return PRESETS_DIR / f"{user_id}.json"


def load_presets(user_id: int) -> dict[str, dict]:
    path = _presets_path(user_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_presets(user_id: int, presets: dict[str, dict]) -> None:
    _presets_path(user_id).write_text(
        json.dumps(presets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _smart_prompt_guides_path(user_id: int) -> Path:
    SMART_PROMPT_GUIDES_DIR.mkdir(exist_ok=True)
    return SMART_PROMPT_GUIDES_DIR / f"{user_id}.json"


def load_smart_prompt_guides(user_id: int) -> dict[str, str]:
    path = _smart_prompt_guides_path(user_id)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}

    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, str] = {}
    for checkpoint, guide in raw.items():
        key = str(checkpoint).strip()
        value = str(guide).strip()
        if key and value:
            normalized[key] = value
    return normalized


def save_smart_prompt_guides(user_id: int, guides: dict[str, str]) -> None:
    payload = {
        str(checkpoint).strip(): str(guide).strip()
        for checkpoint, guide in guides.items()
        if str(checkpoint).strip() and str(guide).strip()
    }
    _smart_prompt_guides_path(user_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _runtime_session_path() -> Path:
    SESSIONS_DIR.mkdir(exist_ok=True)
    return SESSIONS_DIR / "runtime.json"


def load_runtime_session() -> dict[str, Any]:
    path = _runtime_session_path()
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}

    return raw if isinstance(raw, dict) else {}


def save_runtime_session(payload: dict[str, Any]) -> None:
    path = _runtime_session_path()
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def params_to_dict(params: GenerationParams) -> dict[str, Any]:
    data = asdict(params)
    data["loras"] = [list(item) for item in data.get("loras", [])]
    data["reference_images"] = _normalize_reference_images(data.get("reference_images", []))
    return data


def dict_to_params(data: dict[str, Any]) -> GenerationParams:
    payload = dict(data)
    payload["loras"] = [tuple(item) for item in payload.get("loras", [])]
    payload["reference_images"] = _normalize_reference_images(payload.get("reference_images", []))
    known = {f.name for f in _dc.fields(GenerationParams)}
    payload = {k: v for k, v in payload.items() if k in known}
    return GenerationParams(**payload)


def _normalize_reference_images(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            file_id = str(item.get("file_id", "")).strip()
            if not file_id:
                continue
            ref_id = str(item.get("id") or f"ref_{index + 1}")
            normalized.append({"id": ref_id, "file_id": file_id})
            continue

        if isinstance(item, str):
            file_id = item.strip()
            if file_id:
                normalized.append({"id": f"ref_{index + 1}", "file_id": file_id})

    return normalized
