from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.download_filters import (
    normalize_download_base_code,
    normalize_download_period_code,
    normalize_download_sort_code,
    normalize_download_source,
)


class SettingsParseError(ValueError):
    pass


def parse_generation_callback_value(key: str, value: str) -> dict[str, Any]:
    if key == "size":
        width_s, height_s = value.split("x", 1)
        return {
            "gen_width": int(width_s),
            "gen_height": int(height_s),
        }
    if key == "steps":
        return {"gen_steps": int(value)}
    if key == "cfg":
        return {"gen_cfg": float(value)}
    if key == "denoise":
        return {"gen_denoise": float(value)}
    if key == "sampler":
        return {"gen_sampler": value}
    if key == "scheduler":
        return {"gen_scheduler": value}
    if key == "seed":
        return {"gen_seed": int(value)}
    if key == "batch":
        return {"gen_batch": int(value)}
    raise SettingsParseError("common.alert.unknown_param")


def parse_generation_manual_value(field: str, raw: str) -> dict[str, Any]:
    normalized = raw.strip()
    if field == "size":
        width_s, height_s = normalized.lower().replace(" ", "").split("x", 1)
        width = int(width_s)
        height = int(height_s)
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise SettingsParseError
        return {
            "gen_width": width,
            "gen_height": height,
        }
    if field == "steps":
        steps_value = int(float(normalized.replace(",", ".")))
        if not (1 <= steps_value <= 200):
            raise SettingsParseError
        return {"gen_steps": steps_value}
    if field == "cfg":
        cfg_value = float(normalized.replace(",", "."))
        if not (0.0 <= cfg_value <= 30.0):
            raise SettingsParseError
        return {"gen_cfg": cfg_value}
    if field == "denoise":
        denoise_value = float(normalized.replace(",", "."))
        if not (0.0 <= denoise_value <= 1.0):
            raise SettingsParseError
        return {"gen_denoise": denoise_value}
    if field == "seed":
        seed_value = int(normalized.replace(" ", ""))
        if seed_value < -1:
            raise SettingsParseError
        return {"gen_seed": seed_value}
    if field == "batch":
        batch_value = int(normalized.replace(" ", ""))
        if not (1 <= batch_value <= 16):
            raise SettingsParseError
        return {"gen_batch": batch_value}
    if field == "sampler":
        sampler_value = normalized
        if not sampler_value:
            raise SettingsParseError
        return {"gen_sampler": sampler_value}
    if field == "scheduler":
        scheduler_value = normalized
        if not scheduler_value:
            raise SettingsParseError
        return {"gen_scheduler": scheduler_value}
    raise SettingsParseError


def parse_download_callback_value(
    key: str,
    value: str,
    *,
    current_nsfw: bool,
    apply_profile: Callable[[str], bool],
) -> dict[str, Any]:
    normalized = value.strip().lower()
    if key == "source":
        source = normalize_download_source(normalized, default="")
        if not source:
            raise SettingsParseError("common.alert.invalid_param")
        return {"dl_default_source": source}
    if key == "sort":
        sort_code = normalize_download_sort_code(normalized, default="")
        if not sort_code:
            raise SettingsParseError("common.alert.invalid_param")
        return {"dl_default_sort": sort_code}
    if key == "period":
        period_code = normalize_download_period_code(normalized, default="")
        if not period_code:
            raise SettingsParseError("common.alert.invalid_param")
        return {"dl_default_period": period_code}
    if key == "base":
        base_code = normalize_download_base_code(normalized, default="")
        if not base_code:
            raise SettingsParseError("common.alert.invalid_param")
        return {"dl_default_base": base_code}
    if key == "profile":
        if not apply_profile(normalized):
            raise SettingsParseError("common.alert.invalid_param")
        return {}
    if key == "nsfw" and normalized == "toggle":
        return {"dl_default_nsfw": not current_nsfw}
    raise SettingsParseError("common.alert.unknown_param")


def parse_download_author(raw: str) -> str:
    value = raw.strip()
    if value in {"-", "*"}:
        return ""

    tokens = [item.strip().lstrip("@") for item in value.split(",")]
    dedup: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(token[:64])
    return ",".join(dedup[:6])
