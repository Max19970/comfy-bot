from __future__ import annotations

from typing import Any


def queue_item_prompt_id(item: Any) -> str:
    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], str):
        return item[1]
    if isinstance(item, dict):
        for key in ("prompt_id", "id"):
            value = item.get(key)
            if isinstance(value, str):
                return value
    return ""
