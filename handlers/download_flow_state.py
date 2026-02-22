from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MAX_RESULTS_LIMIT = 200


@dataclass(slots=True)
class DownloadResultsViewState:
    results: list[Any]
    page_size: int
    page: int
    loaded_limit: int
    exhausted: bool

    @property
    def can_continue(self) -> bool:
        return compute_can_continue(
            results_count=len(self.results),
            loaded_limit=self.loaded_limit,
            exhausted=self.exhausted,
        )


def hydrate_results_payload(
    raw_results: Any,
    *,
    hydrate_result: Any,
) -> list[Any]:
    if not isinstance(raw_results, list):
        return []

    hydrated: list[Any] = []
    for item in raw_results:
        if isinstance(item, dict):
            hydrated.append(hydrate_result(item))
    return hydrated


def serialize_results_payload(results: list[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in results]


def normalize_page_size(raw_value: Any, *, default_page_size: int) -> int:
    try:
        page_size = int(raw_value or default_page_size)
    except (TypeError, ValueError):
        page_size = default_page_size
    return max(1, min(page_size, 10))


def clamp_page(page: int, *, total_items: int, page_size: int) -> int:
    max_page = max(0, (total_items - 1) // page_size)
    return max(0, min(page, max_page))


def compute_can_continue(
    *,
    results_count: int,
    loaded_limit: int,
    exhausted: bool,
) -> bool:
    return (not exhausted) and results_count >= loaded_limit and loaded_limit < MAX_RESULTS_LIMIT


def next_search_limit(*, loaded_limit: int, page_size: int) -> int:
    return min(MAX_RESULTS_LIMIT, loaded_limit + page_size * 2)


def read_results_view_state(
    data: dict[str, Any],
    *,
    default_page_size: int,
    hydrate_result: Any,
) -> DownloadResultsViewState:
    results = hydrate_results_payload(data.get("dl_results", []), hydrate_result=hydrate_result)
    page_size = normalize_page_size(data.get("dl_page_size"), default_page_size=default_page_size)

    try:
        page = int(data.get("dl_results_page", 0) or 0)
    except (TypeError, ValueError):
        page = 0

    try:
        loaded_limit = int(data.get("dl_loaded_limit", page_size * 2) or (page_size * 2))
    except (TypeError, ValueError):
        loaded_limit = page_size * 2
    loaded_limit = max(page_size, min(loaded_limit, MAX_RESULTS_LIMIT))

    exhausted = bool(data.get("dl_more_exhausted", False))
    page = clamp_page(page, total_items=len(results), page_size=page_size)

    return DownloadResultsViewState(
        results=results,
        page_size=page_size,
        page=page,
        loaded_limit=loaded_limit,
        exhausted=exhausted,
    )
