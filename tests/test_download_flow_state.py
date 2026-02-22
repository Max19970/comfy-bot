from __future__ import annotations

from dataclasses import dataclass

from handlers.download_flow_state import (
    MAX_RESULTS_LIMIT,
    clamp_page,
    compute_can_continue,
    hydrate_results_payload,
    next_search_limit,
    read_results_view_state,
    serialize_results_payload,
)


@dataclass
class _Item:
    value: int


def test_hydrate_results_payload_skips_invalid_items() -> None:
    hydrated = hydrate_results_payload(
        [{"value": 1}, "bad", {"value": 2}],
        hydrate_result=lambda payload: _Item(value=int(payload["value"])),
    )

    assert hydrated == [_Item(1), _Item(2)]


def test_serialize_results_payload_roundtrip() -> None:
    payload = serialize_results_payload([_Item(10), _Item(20)])
    assert payload == [{"value": 10}, {"value": 20}]


def test_read_results_view_state_normalizes_and_clamps() -> None:
    data = {
        "dl_results": [{"value": 1}, {"value": 2}, {"value": 3}],
        "dl_page_size": 2,
        "dl_results_page": 10,
        "dl_loaded_limit": 4,
        "dl_more_exhausted": False,
    }

    view = read_results_view_state(
        data,
        default_page_size=8,
        hydrate_result=lambda payload: _Item(value=int(payload["value"])),
    )

    assert view.results == [_Item(1), _Item(2), _Item(3)]
    assert view.page_size == 2
    assert view.page == 1
    assert view.loaded_limit == 4
    assert view.exhausted is False
    assert view.can_continue is False


def test_next_search_limit_honors_hard_cap() -> None:
    assert next_search_limit(loaded_limit=8, page_size=8) == 24
    assert next_search_limit(loaded_limit=MAX_RESULTS_LIMIT, page_size=8) == MAX_RESULTS_LIMIT


def test_can_continue_and_clamp_helpers() -> None:
    assert compute_can_continue(results_count=16, loaded_limit=16, exhausted=False) is True
    assert compute_can_continue(results_count=15, loaded_limit=16, exhausted=False) is False
    assert compute_can_continue(results_count=16, loaded_limit=16, exhausted=True) is False

    assert clamp_page(0, total_items=0, page_size=8) == 0
    assert clamp_page(5, total_items=3, page_size=2) == 1
