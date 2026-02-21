from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.download_filters import (
    normalize_download_base_code,
    normalize_download_period_code,
    normalize_download_sort_code,
    normalize_download_source,
)


def read_generation_defaults(
    prefs: Mapping[str, Any],
    *,
    default_width: int,
    default_height: int,
    default_steps: int,
    default_cfg: float,
    default_denoise: float,
    default_sampler: str,
    default_scheduler: str,
) -> dict[str, Any]:
    width = _clamp_int(prefs.get("gen_width"), default_width, min_value=64, max_value=4096)
    height = _clamp_int(prefs.get("gen_height"), default_height, min_value=64, max_value=4096)
    steps = _clamp_int(prefs.get("gen_steps"), default_steps, min_value=1, max_value=200)
    cfg_value = _clamp_float(prefs.get("gen_cfg"), default_cfg, min_value=0.0, max_value=30.0)
    denoise = _clamp_float(
        prefs.get("gen_denoise"),
        default_denoise,
        min_value=0.0,
        max_value=1.0,
    )
    seed = _clamp_int(prefs.get("gen_seed"), -1, min_value=-1, max_value=2**31 - 1)
    batch = _clamp_int(prefs.get("gen_batch"), 1, min_value=1, max_value=16)

    sampler = str(prefs.get("gen_sampler", default_sampler)).strip() or default_sampler
    scheduler = str(prefs.get("gen_scheduler", default_scheduler)).strip() or default_scheduler

    return {
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg_value,
        "denoise": denoise,
        "seed": seed,
        "batch": batch,
        "sampler": sampler,
        "scheduler": scheduler,
    }


def read_download_defaults(
    prefs: Mapping[str, Any],
    *,
    inferred_base: str = "all",
) -> dict[str, Any]:
    source = normalize_download_source(str(prefs.get("dl_default_source", "all")))
    sort_code = normalize_download_sort_code(str(prefs.get("dl_default_sort", "downloads")))
    period = normalize_download_period_code(str(prefs.get("dl_default_period", "all")))
    default_base = normalize_download_base_code(inferred_base)
    base = normalize_download_base_code(
        str(prefs.get("dl_default_base", default_base)),
        default=default_base,
    )
    author = str(prefs.get("dl_default_author", "")).strip()

    return {
        "source": source,
        "sort": sort_code,
        "period": period,
        "base": base,
        "nsfw": bool(prefs.get("dl_default_nsfw", False)),
        "author": author,
    }


def normalize_user_preferences(raw: Any) -> dict[str, Any]:
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
    if isinstance(source, str):
        source_norm = normalize_download_source(source, default="")
        if source_norm:
            normalized["dl_default_source"] = source_norm

    sort_code = raw.get("dl_default_sort")
    if isinstance(sort_code, str):
        sort_norm = normalize_download_sort_code(sort_code, default="")
        if sort_norm:
            normalized["dl_default_sort"] = sort_norm

    period = raw.get("dl_default_period")
    if isinstance(period, str):
        period_norm = normalize_download_period_code(period, default="")
        if period_norm:
            normalized["dl_default_period"] = period_norm

    base = raw.get("dl_default_base")
    if isinstance(base, str):
        base_norm = normalize_download_base_code(base, default="")
        if base_norm:
            normalized["dl_default_base"] = base_norm

    if isinstance(raw.get("dl_default_nsfw"), bool):
        normalized["dl_default_nsfw"] = raw.get("dl_default_nsfw")

    author = raw.get("dl_default_author")
    if isinstance(author, str):
        normalized["dl_default_author"] = author.strip().replace("@", "")[:256]

    return normalized


def _clamp_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _clamp_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))
