from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericEnhancementControl:
    field: str
    label: str
    min_value: float
    max_value: float
    presets: tuple[str, ...]


HIRES_SCALE_CONTROL = NumericEnhancementControl(
    field="hires_scale",
    label="Hi-res scale",
    min_value=1.0,
    max_value=3.0,
    presets=("1.25", "1.5", "1.75", "2.0"),
)

HIRES_DENOISE_CONTROL = NumericEnhancementControl(
    field="hires_denoise",
    label="Hi-res denoise",
    min_value=0.0,
    max_value=1.0,
    presets=("0.3", "0.4", "0.5", "0.6", "0.7"),
)

PAG_SCALE_CONTROL = NumericEnhancementControl(
    field="pag_scale",
    label="PAG scale",
    min_value=0.5,
    max_value=10.0,
    presets=("1.0", "2.0", "3.0", "4.0", "5.0"),
)

NUMERIC_ENHANCEMENT_CONTROLS: dict[str, NumericEnhancementControl] = {
    control.field: control
    for control in (
        HIRES_SCALE_CONTROL,
        HIRES_DENOISE_CONTROL,
        PAG_SCALE_CONTROL,
    )
}


def numeric_enhancement_control(field: str) -> NumericEnhancementControl | None:
    return NUMERIC_ENHANCEMENT_CONTROLS.get(field)


def numeric_control_rows(field: str, *, row_size: int = 3) -> list[list[str]]:
    control = NUMERIC_ENHANCEMENT_CONTROLS[field]
    values = list(control.presets)
    rows: list[list[str]] = []
    for index in range(0, len(values), row_size):
        rows.append(values[index : index + row_size])
    return rows


def numeric_control_range_text(field: str) -> str:
    control = NUMERIC_ENHANCEMENT_CONTROLS[field]
    return f"{control.min_value}-{control.max_value}"
