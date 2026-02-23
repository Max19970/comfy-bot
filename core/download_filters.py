from __future__ import annotations

from collections.abc import Callable

from domain.base_model_policy import BaseModelPolicy

DOWNLOAD_SOURCE_LABELS = {
    "all": "Оба источника",
    "civitai": "CivitAI",
    "huggingface": "HuggingFace",
}
DOWNLOAD_SOURCE_CODE_TO_I18N_KEY = {
    "all": "download.filter.source.all",
    "civitai": "download.filter.source.civitai",
    "huggingface": "download.filter.source.huggingface",
}
DOWNLOAD_SOURCE_CODES = frozenset(DOWNLOAD_SOURCE_LABELS)

DOWNLOAD_SORT_CODE_TO_LABEL = {
    "downloads": "По скачиваниям",
    "rating": "По рейтингу",
    "newest": "Новые",
}
DOWNLOAD_SORT_CODE_TO_I18N_KEY = {
    "downloads": "download.filter.sort.downloads",
    "rating": "download.filter.sort.rating",
    "newest": "download.filter.sort.newest",
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
DOWNLOAD_PERIOD_CODE_TO_I18N_KEY = {
    "all": "download.filter.period.all",
    "month": "download.filter.period.month",
    "week": "download.filter.period.week",
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
DOWNLOAD_BASE_CODE_TO_I18N_KEY = {
    "all": "download.filter.base.all",
    "sd15": "download.filter.base.sd15",
    "sd2": "download.filter.base.sd2",
    "sdxl09": "download.filter.base.sdxl09",
    "sdxl": "download.filter.base.sdxl",
    "sd3": "download.filter.base.sd3",
    "sd35": "download.filter.base.sd35",
    "pony": "download.filter.base.pony",
    "flux": "download.filter.base.flux",
    "illustrious": "download.filter.base.illustrious",
    "noobai": "download.filter.base.noobai",
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
DOWNLOAD_PROFILE_CODE_TO_I18N_KEY = {
    "popular": "download.filter.profile.popular",
    "fresh": "download.filter.profile.fresh",
    "quality": "download.filter.profile.quality",
    "anime": "download.filter.profile.anime",
}

_base_model_policy = BaseModelPolicy()

TranslateLabel = Callable[[str, str | None, str], str]


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


def _localized_label(
    *,
    code: str,
    defaults: dict[str, str],
    keys: dict[str, str],
    fallback: str,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    default = defaults.get(code, fallback)
    if translate is None:
        return default
    key = keys.get(code)
    if not key:
        return default
    return translate(key, locale, default)


def download_base_label(
    code: str,
    *,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    return _localized_label(
        code=code,
        defaults=DOWNLOAD_BASE_CODE_TO_LABEL,
        keys=DOWNLOAD_BASE_CODE_TO_I18N_KEY,
        fallback=DOWNLOAD_BASE_CODE_TO_LABEL["all"],
        translate=translate,
        locale=locale,
    )


def download_source_label(
    code: str,
    *,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    return _localized_label(
        code=code,
        defaults=DOWNLOAD_SOURCE_LABELS,
        keys=DOWNLOAD_SOURCE_CODE_TO_I18N_KEY,
        fallback=code,
        translate=translate,
        locale=locale,
    )


def download_sort_label(
    code: str,
    *,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    return _localized_label(
        code=code,
        defaults=DOWNLOAD_SORT_CODE_TO_LABEL,
        keys=DOWNLOAD_SORT_CODE_TO_I18N_KEY,
        fallback=DOWNLOAD_SORT_CODE_TO_LABEL["downloads"],
        translate=translate,
        locale=locale,
    )


def download_period_label(
    code: str,
    *,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    return _localized_label(
        code=code,
        defaults=DOWNLOAD_PERIOD_CODE_TO_LABEL,
        keys=DOWNLOAD_PERIOD_CODE_TO_I18N_KEY,
        fallback=DOWNLOAD_PERIOD_CODE_TO_LABEL["all"],
        translate=translate,
        locale=locale,
    )


def download_profile_label(
    code: str,
    *,
    translate: TranslateLabel | None = None,
    locale: str | None = None,
) -> str:
    defaults = {k: str(v.get("label", "")) for k, v in DOWNLOAD_FILTER_PROFILES.items()}
    return _localized_label(
        code=code,
        defaults=defaults,
        keys=DOWNLOAD_PROFILE_CODE_TO_I18N_KEY,
        fallback=code,
        translate=translate,
        locale=locale,
    )


def base_code_from_base_model(base_model: str) -> str:
    return _base_model_policy.download_base_code(base_model)
