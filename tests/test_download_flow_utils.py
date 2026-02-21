from __future__ import annotations

from handlers.download_flow_utils import apply_download_profile, parse_author_filters


def test_parse_author_filters_deduplicates_and_limits() -> None:
    raw = "@Alice, bob,alice,  CAROL  ,dave,erin,frank,grace"
    assert parse_author_filters(raw) == ["Alice", "bob", "CAROL", "dave", "erin", "frank"]


def test_apply_download_profile_keeps_existing_when_missing() -> None:
    state = {
        "dl_source": "all",
        "dl_sort": "downloads",
        "dl_period": "all",
        "dl_base": "all",
        "dl_nsfw": True,
    }
    updated = apply_download_profile(state, {"sort": "rating", "nsfw": False})
    assert updated["dl_source"] == "all"
    assert updated["dl_sort"] == "rating"
    assert updated["dl_period"] == "all"
    assert updated["dl_base"] == "all"
    assert updated["dl_nsfw"] is False
