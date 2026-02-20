"""Backward-compatible facade over refactored core modules."""

from __future__ import annotations

from core.html_utils import h, progress_bar, truncate
from core.image_utils import (
    LANCZOS_RESAMPLE,
    TG_PHOTO_MAX_BYTES,
    TG_PHOTO_MAX_SIDE,
    TG_PHOTO_MAX_SUM,
    compress_for_photo,
)
from core.runtime import (
    ActiveGeneration,
    PromptRequest,
    RuntimeStore,
    get_user_pro_mode,
    set_user_pro_mode,
)
from core.states import DownloadStates, GenStates, PresetStates, PromptEditorStates
from core.storage import (
    PRESETS_DIR,
    SMART_PROMPT_GUIDES_DIR,
    dict_to_params,
    load_presets,
    load_smart_prompt_guides,
    params_to_dict,
    save_presets,
    save_smart_prompt_guides,
)
from core.telegram import callback_user_id, message_user_id
from core.ui import (
    MAX_REFERENCE_IMAGES,
    PAGE_SIZE,
    SIZES,
    custom_btn,
    loras_text,
    paginated_keyboard,
    params_summary,
    params_summary_for_mode,
    params_summary_full,
    params_summary_simple,
)

__all__ = [
    "ActiveGeneration",
    "DownloadStates",
    "GenStates",
    "LANCZOS_RESAMPLE",
    "MAX_REFERENCE_IMAGES",
    "PAGE_SIZE",
    "PRESETS_DIR",
    "PresetStates",
    "PromptEditorStates",
    "PromptRequest",
    "RuntimeStore",
    "SIZES",
    "SMART_PROMPT_GUIDES_DIR",
    "TG_PHOTO_MAX_BYTES",
    "TG_PHOTO_MAX_SIDE",
    "TG_PHOTO_MAX_SUM",
    "callback_user_id",
    "compress_for_photo",
    "custom_btn",
    "dict_to_params",
    "get_user_pro_mode",
    "h",
    "load_presets",
    "load_smart_prompt_guides",
    "loras_text",
    "message_user_id",
    "paginated_keyboard",
    "params_summary_full",
    "params_summary_simple",
    "params_summary",
    "params_summary_for_mode",
    "params_to_dict",
    "progress_bar",
    "save_presets",
    "save_smart_prompt_guides",
    "set_user_pro_mode",
    "truncate",
]
