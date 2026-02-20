from __future__ import annotations

from core.formatting import human_size, short_number


def test_human_size() -> None:
    assert human_size(0) == "unknown"
    assert human_size(1024) == "1.0 KB"
    assert human_size(1024 * 1024) == "1.0 MB"


def test_short_number() -> None:
    assert short_number(999) == "999"
    assert short_number(1_200) == "1.2K"
    assert short_number(2_400_000) == "2.4M"
