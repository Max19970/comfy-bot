from __future__ import annotations

from core.models import GenerationParams
from core.storage import dict_to_params, load_presets, params_to_dict, save_presets

PresetPayload = dict[str, object]
PresetMap = dict[str, PresetPayload]


def load_user_presets(uid: int) -> PresetMap:
    return load_presets(uid)


def save_user_presets(uid: int, presets: PresetMap) -> None:
    save_presets(uid, presets)


def normalize_preset_name(raw_name: str) -> str:
    return raw_name.strip()


def is_valid_preset_name(name: str) -> bool:
    return bool(name) and len(name) <= 50


def resolve_preset_name(
    index: int,
    presets: PresetMap,
    snapshot: object,
) -> str | None:
    if isinstance(snapshot, list) and 0 <= index < len(snapshot):
        candidate = snapshot[index]
        if isinstance(candidate, str) and candidate in presets:
            return candidate

    names = sorted(presets.keys())
    if 0 <= index < len(names):
        return names[index]
    return None


def params_payload_from(params: GenerationParams | PresetPayload) -> PresetPayload:
    if isinstance(params, GenerationParams):
        return params_to_dict(params)
    return params


def params_from_payload(payload: PresetPayload) -> GenerationParams:
    return dict_to_params(payload)
