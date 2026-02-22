from __future__ import annotations

from collections.abc import Callable


def parse_scalar_value(
    raw_value: str,
    *,
    parse_value: Callable[[str], int | float],
    validate_value: Callable[[int | float], bool],
) -> int | float | None:
    try:
        parsed = parse_value(raw_value.strip())
    except ValueError:
        return None

    if not validate_value(parsed):
        return None
    return parsed
