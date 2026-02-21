from __future__ import annotations

from typing import Any

from core.user_preferences import read_download_defaults


def download_defaults_for_user(
    user_preferences: dict[int, dict[str, Any]],
    uid: int,
    *,
    inferred_base: str,
) -> dict[str, Any]:
    prefs = user_preferences.get(uid, {})
    defaults = read_download_defaults(prefs, inferred_base=inferred_base)
    return {
        "dl_source": defaults["source"],
        "dl_sort": defaults["sort"],
        "dl_period": defaults["period"],
        "dl_base": defaults["base"],
        "dl_nsfw": defaults["nsfw"],
        "dl_author": defaults["author"],
    }


def apply_download_profile(data: dict[str, Any], profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return data
    updated = dict(data)
    updated["dl_source"] = profile.get("source", updated.get("dl_source", "all"))
    updated["dl_sort"] = profile.get("sort", updated.get("dl_sort", "downloads"))
    updated["dl_period"] = profile.get("period", updated.get("dl_period", "all"))
    updated["dl_base"] = profile.get("base", updated.get("dl_base", "all"))
    updated["dl_nsfw"] = bool(profile.get("nsfw", updated.get("dl_nsfw", False)))
    return updated


def parse_author_filters(raw: str) -> list[str]:
    items = [item.strip().lstrip("@") for item in raw.split(",")]
    cleaned = [item for item in items if item]
    unique: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:6]
