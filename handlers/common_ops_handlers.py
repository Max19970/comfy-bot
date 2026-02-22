from __future__ import annotations

from .common_delete_handlers import CommonDeleteDeps, register_common_delete_handlers
from .common_jobs_handlers import CommonJobsDeps, register_common_jobs_handlers

__all__ = [
    "CommonDeleteDeps",
    "CommonJobsDeps",
    "register_common_delete_handlers",
    "register_common_jobs_handlers",
]
