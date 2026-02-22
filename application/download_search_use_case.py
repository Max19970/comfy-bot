from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.download_filters import base_code_from_base_model


class ModelMetadataLookup(Protocol):
    def __call__(
        self,
        filename: str,
        *,
        model_type: str | None = None,
    ) -> dict[str, Any] | None: ...


class BaseModelInferer(Protocol):
    def __call__(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class DownloadSearchCriteria:
    query: str
    model_type: str
    source: str
    limit: int
    sort: str
    base_models: tuple[str, ...]
    include_nsfw: bool
    period: str
    civitai_authors: tuple[str, ...]


@dataclass(slots=True)
class DownloadSearchUseCase:
    base_code_to_api: dict[str, list[str]]
    sort_code_to_api: dict[str, str]
    period_code_to_api: dict[str, str]
    supports_base_filter: Callable[..., bool]
    supports_nsfw_filter: Callable[[str], bool]

    def build_search_criteria(
        self,
        data: dict[str, Any],
        *,
        requested_limit: int,
        default_page_size: int,
    ) -> DownloadSearchCriteria:
        query = str(data.get("dl_query", "")).strip()
        model_type = str(data.get("dl_type", "checkpoint") or "checkpoint")
        source = str(data.get("dl_source", "all") or "all")
        sort_code = str(data.get("dl_sort", "downloads") or "downloads")
        period_code = str(data.get("dl_period", "all") or "all")
        base_code = str(data.get("dl_base", "all") or "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()

        page_size = _to_positive_int(data.get("dl_page_size"), default=default_page_size)

        base_models = self.base_code_to_api.get(base_code, [])
        if not self.supports_base_filter(model_type=model_type, source=source):
            base_models = []

        authors = self.parse_author_filters(author_nick) if source in {"civitai", "all"} else []

        return DownloadSearchCriteria(
            query=query,
            model_type=model_type,
            source=source,
            limit=max(page_size, requested_limit),
            sort=self.sort_code_to_api.get(sort_code, "Most Downloaded"),
            base_models=tuple(base_models),
            include_nsfw=include_nsfw and self.supports_nsfw_filter(source),
            period=self.period_code_to_api.get(period_code, "AllTime"),
            civitai_authors=tuple(authors),
        )

    def infer_base_code_for_checkpoint(
        self,
        checkpoint_name: str,
        *,
        get_model_metadata: ModelMetadataLookup,
        infer_base_model: BaseModelInferer,
        default: str = "all",
    ) -> str:
        name = str(checkpoint_name or "").strip()
        if not name:
            return default

        metadata = get_model_metadata(name, model_type="checkpoint")
        base_model = str(metadata.get("base_model") if metadata else "").strip()
        if not base_model:
            base_model = infer_base_model(name)
        code = base_code_from_base_model(base_model)
        return code or default

    @staticmethod
    def parse_author_filters(raw: str) -> list[str]:
        items = [item.strip().lstrip("@") for item in str(raw or "").split(",")]
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


def _to_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
