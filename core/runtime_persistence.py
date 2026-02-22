from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.runtime_snapshot import runtime_from_persisted_dict, runtime_to_persisted_dict
from core.storage import load_runtime_session, save_runtime_session

if TYPE_CHECKING:
    from core.runtime import RuntimeStore


logger = logging.getLogger(__name__)


def persist_runtime_store(runtime: RuntimeStore) -> None:
    try:
        save_runtime_session(runtime_to_persisted_dict(runtime))
    except (OSError, TypeError, ValueError):
        logger.warning("Failed to persist runtime session", exc_info=True)


def load_runtime_store() -> RuntimeStore:
    from core.runtime import RuntimeStore

    try:
        raw = load_runtime_session()
    except (OSError, TypeError, ValueError):
        return RuntimeStore()
    if not raw:
        return RuntimeStore()
    try:
        return runtime_from_persisted_dict(raw)
    except (TypeError, ValueError, OSError):
        return RuntimeStore()
