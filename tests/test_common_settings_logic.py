from __future__ import annotations

import pytest

from handlers.common_settings_logic import (
    SettingsParseError,
    parse_download_author,
    parse_download_callback_value,
    parse_generation_callback_value,
    parse_generation_manual_value,
)


def test_parse_generation_callback_value_for_size_and_sampler() -> None:
    size_updates = parse_generation_callback_value("size", "896x1152")
    assert size_updates == {"gen_width": 896, "gen_height": 1152}

    sampler_updates = parse_generation_callback_value("sampler", "euler")
    assert sampler_updates == {"gen_sampler": "euler"}


def test_parse_generation_manual_value_validates_ranges() -> None:
    cfg_updates = parse_generation_manual_value("cfg", "7.5")
    assert cfg_updates == {"gen_cfg": 7.5}

    with pytest.raises(SettingsParseError):
        parse_generation_manual_value("batch", "32")


def test_parse_download_callback_value_normalizes_and_toggles() -> None:
    source_updates = parse_download_callback_value(
        "source",
        "HuGgInGfAcE",
        current_nsfw=False,
        apply_profile=lambda _value: False,
    )
    assert source_updates == {"dl_default_source": "huggingface"}

    nsfw_updates = parse_download_callback_value(
        "nsfw",
        "toggle",
        current_nsfw=True,
        apply_profile=lambda _value: False,
    )
    assert nsfw_updates == {"dl_default_nsfw": False}


def test_parse_download_callback_value_profile_and_errors() -> None:
    profile_calls: list[str] = []

    def _apply(profile_code: str) -> bool:
        profile_calls.append(profile_code)
        return profile_code == "popular"

    updates = parse_download_callback_value(
        "profile",
        "popular",
        current_nsfw=False,
        apply_profile=_apply,
    )
    assert updates == {}
    assert profile_calls == ["popular"]

    with pytest.raises(SettingsParseError):
        parse_download_callback_value(
            "period",
            "yesterday",
            current_nsfw=False,
            apply_profile=_apply,
        )


def test_parse_download_author_deduplicates_and_limits() -> None:
    author = parse_download_author(" @Alice, bob, ALICE, ,carol ")
    assert author == "Alice,bob,carol"

    assert parse_download_author("-") == ""
