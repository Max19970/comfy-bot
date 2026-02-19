from __future__ import annotations

import html as _html
from typing import Any


def h(text: Any) -> str:
    """HTML-escape user-provided text for Telegram HTML mode."""
    return _html.escape(str(text))


def truncate(text: str, max_len: int = 80) -> str:
    """Trim long text and append an ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "\u2026"


def progress_bar(current: int, total: int, *, width: int = 12) -> str:
    """Render a compact unicode progress bar."""
    if total <= 0:
        return ""
    filled = max(0, min(width, (current * width) // total))
    pct = min(100, (current * 100) // total)
    return f"{'▰' * filled}{'▱' * (width - filled)} {pct}%"
