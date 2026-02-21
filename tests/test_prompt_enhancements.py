from __future__ import annotations

from core.prompt_enhancements import (
    numeric_control_range_text,
    numeric_control_rows,
    numeric_enhancement_control,
)


def test_numeric_enhancement_controls_have_expected_ranges() -> None:
    hires = numeric_enhancement_control("hires_scale")
    denoise = numeric_enhancement_control("hires_denoise")
    pag = numeric_enhancement_control("pag_scale")

    assert hires is not None
    assert denoise is not None
    assert pag is not None

    assert (hires.min_value, hires.max_value) == (1.0, 3.0)
    assert (denoise.min_value, denoise.max_value) == (0.0, 1.0)
    assert (pag.min_value, pag.max_value) == (0.5, 10.0)


def test_numeric_control_rows_chunks_presets() -> None:
    assert numeric_control_rows("hires_scale") == [["1.25", "1.5", "1.75"], ["2.0"]]
    assert numeric_control_rows("pag_scale") == [["1.0", "2.0", "3.0"], ["4.0", "5.0"]]
    assert numeric_control_range_text("hires_denoise") == "0.0-1.0"
