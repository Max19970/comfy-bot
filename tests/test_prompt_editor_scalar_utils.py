from __future__ import annotations

from handlers.prompt_editor_scalar_utils import parse_scalar_value


def test_parse_scalar_value_parses_valid_number() -> None:
    value = parse_scalar_value(
        " 1.25 ",
        parse_value=float,
        validate_value=lambda candidate: 0.0 <= float(candidate) <= 2.0,
    )

    assert value == 1.25


def test_parse_scalar_value_returns_none_for_invalid_value() -> None:
    value = parse_scalar_value(
        "not-a-number",
        parse_value=float,
        validate_value=lambda candidate: 0.0 <= float(candidate) <= 2.0,
    )

    assert value is None


def test_parse_scalar_value_returns_none_for_out_of_range() -> None:
    value = parse_scalar_value(
        "3.5",
        parse_value=float,
        validate_value=lambda candidate: 0.0 <= float(candidate) <= 2.0,
    )

    assert value is None
