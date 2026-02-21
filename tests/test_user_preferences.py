from __future__ import annotations

from core.user_preferences import (
    normalize_user_preferences,
    read_download_defaults,
    read_generation_defaults,
)


def test_read_generation_defaults_clamps_and_falls_back() -> None:
    prefs = {
        "gen_width": "9999",
        "gen_height": "32",
        "gen_steps": "abc",
        "gen_cfg": "31.5",
        "gen_denoise": "-1",
        "gen_seed": "-25",
        "gen_batch": "20",
        "gen_sampler": "  dpmpp_2m  ",
        "gen_scheduler": "",
    }

    defaults = read_generation_defaults(
        prefs,
        default_width=512,
        default_height=512,
        default_steps=20,
        default_cfg=7.0,
        default_denoise=1.0,
        default_sampler="euler",
        default_scheduler="normal",
    )

    assert defaults == {
        "width": 4096,
        "height": 64,
        "steps": 20,
        "cfg": 30.0,
        "denoise": 0.0,
        "seed": -1,
        "batch": 16,
        "sampler": "dpmpp_2m",
        "scheduler": "normal",
    }


def test_read_download_defaults_uses_inferred_base() -> None:
    prefs = {
        "dl_default_source": "invalid",
        "dl_default_sort": "rating",
        "dl_default_period": "week",
        "dl_default_base": "unknown",
        "dl_default_nsfw": 1,
        "dl_default_author": " @foo ",
    }

    defaults = read_download_defaults(prefs, inferred_base="sdxl")
    assert defaults == {
        "source": "all",
        "sort": "rating",
        "period": "week",
        "base": "sdxl",
        "nsfw": True,
        "author": "@foo",
    }


def test_normalize_user_preferences_sanitizes_known_fields() -> None:
    raw = {
        "pro_mode": 1,
        "training_mode": "ADVANCED",
        "training_page": -5,
        "gen_width": 8192,
        "gen_cfg": 99,
        "gen_sampler": "  euler_a  ",
        "dl_default_source": "huggingface",
        "dl_default_author": "@@creator@@",
        "unknown": "ignored",
    }

    normalized = normalize_user_preferences(raw)
    assert normalized["pro_mode"] is True
    assert normalized["training_mode"] == "advanced"
    assert normalized["training_page"] == 0
    assert normalized["gen_width"] == 4096
    assert normalized["gen_cfg"] == 30.0
    assert normalized["gen_sampler"] == "euler_a"
    assert normalized["dl_default_source"] == "huggingface"
    assert normalized["dl_default_author"] == "creator"
    assert "unknown" not in normalized
