from __future__ import annotations

from domain.base_model_policy import BaseModelPolicy

DOWNLOAD_SOURCE_LABELS = {
    "all": "Оба источника",
    "civitai": "CivitAI",
    "huggingface": "HuggingFace",
}
DOWNLOAD_SOURCE_CODES = frozenset(DOWNLOAD_SOURCE_LABELS)

DOWNLOAD_SORT_CODE_TO_LABEL = {
    "downloads": "По скачиваниям",
    "rating": "По рейтингу",
    "newest": "Новые",
}
DOWNLOAD_SORT_CODE_TO_API = {
    "downloads": "Most Downloaded",
    "rating": "Highest Rated",
    "newest": "Newest",
}
DOWNLOAD_SORT_CODES = frozenset(DOWNLOAD_SORT_CODE_TO_LABEL)

DOWNLOAD_PERIOD_CODE_TO_LABEL = {
    "all": "Всё время",
    "month": "Месяц",
    "week": "Неделя",
}
DOWNLOAD_PERIOD_CODE_TO_API = {
    "all": "AllTime",
    "month": "Month",
    "week": "Week",
}
DOWNLOAD_PERIOD_CODES = frozenset(DOWNLOAD_PERIOD_CODE_TO_LABEL)

DOWNLOAD_BASE_CODE_TO_LABEL = {
    "all": "Все",
    "sd15": "SD 1.5",
    "sd2": "SD 2.x",
    "sdxl09": "SDXL 0.9",
    "sdxl": "SDXL 1.0",
    "sd3": "SD 3",
    "sd35": "SD 3.5",
    "pony": "Pony",
    "flux": "Flux",
    "illustrious": "Illustrious",
    "noobai": "NoobAI",
}
DOWNLOAD_BASE_CODE_TO_API = {
    "all": [],
    "sd15": ["SD 1.5"],
    "sd2": ["SD 2.0", "SD 2.1"],
    "sdxl09": ["SDXL 0.9"],
    "sdxl": ["SDXL 1.0"],
    "sd3": ["SD 3"],
    "sd35": ["SD 3.5"],
    "pony": ["Pony"],
    "flux": ["Flux.1 D"],
    "illustrious": ["Illustrious"],
    "noobai": ["NoobAI"],
}
DOWNLOAD_BASE_CODES = frozenset(DOWNLOAD_BASE_CODE_TO_LABEL)

DOWNLOAD_FILTER_PROFILES = {
    "popular": {
        "label": "🔥 Популярные",
        "source": "all",
        "sort": "downloads",
        "period": "all",
        "base": "all",
        "nsfw": False,
    },
    "fresh": {
        "label": "🆕 Новые",
        "source": "all",
        "sort": "newest",
        "period": "week",
        "base": "all",
        "nsfw": False,
    },
    "quality": {
        "label": "⭐ Рейтинг",
        "source": "all",
        "sort": "rating",
        "period": "month",
        "base": "all",
        "nsfw": False,
    },
    "anime": {
        "label": "🎎 Anime",
        "source": "civitai",
        "sort": "downloads",
        "period": "month",
        "base": "pony",
        "nsfw": False,
    },
}

_base_model_policy = BaseModelPolicy()


def normalize_download_source(value: str, *, default: str = "all") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DOWNLOAD_SOURCE_CODES:
        return normalized
    return default


def normalize_download_sort_code(value: str, *, default: str = "downloads") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DOWNLOAD_SORT_CODES:
        return normalized
    return default


def normalize_download_period_code(value: str, *, default: str = "all") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DOWNLOAD_PERIOD_CODES:
        return normalized
    return default


def normalize_download_base_code(value: str, *, default: str = "all") -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DOWNLOAD_BASE_CODES:
        return normalized
    return default


def download_base_label(code: str) -> str:
    return DOWNLOAD_BASE_CODE_TO_LABEL.get(code, DOWNLOAD_BASE_CODE_TO_LABEL["all"])


def download_source_label(code: str) -> str:
    return DOWNLOAD_SOURCE_LABELS.get(code, code)


def base_code_from_base_model(base_model: str) -> str:
    return _base_model_policy.download_base_code(base_model)
