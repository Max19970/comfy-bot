from __future__ import annotations

from handlers.presets_flow import (
    overwrite_state_prepare,
    overwrite_state_reset,
    parse_callback_data,
)


def test_parse_callback_data_returns_value_for_valid_payload() -> None:
    value = parse_callback_data("preset:overwrite:yes", prefix="preset:overwrite")

    assert value == "yes"


def test_parse_callback_data_returns_none_for_invalid_payload() -> None:
    value = parse_callback_data("wrong:data", prefix="preset:overwrite")

    assert value is None


def test_overwrite_state_reset_and_prepare() -> None:
    reset_payload = overwrite_state_reset(save_source="editor")
    prepared_payload = overwrite_state_prepare(
        save_source="result",
        overwrite_source="result",
        preset_name="my",
        preset_params={"width": 512},
    )

    assert reset_payload["preset_save_source"] == "editor"
    assert reset_payload["preset_overwrite_name"] is None

    assert prepared_payload["preset_save_source"] == "result"
    assert prepared_payload["preset_overwrite_source"] == "result"
    assert prepared_payload["preset_overwrite_name"] == "my"
    assert prepared_payload["preset_overwrite_params"] == {"width": 512}
