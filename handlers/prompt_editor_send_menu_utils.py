from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.prompt_enhancements import numeric_enhancement_control
from core.runtime import PreviewArtifact


def submenu_back_callback(menu_key: str, artifact_id: str) -> str:
    if menu_key in {"steps", "cfg", "den", "sampler", "scheduler"}:
        return f"img:sub:smp:{artifact_id}"
    if menu_key in {"hrs", "hrd", "pags", "up"}:
        return f"img:sub:enh:{artifact_id}"
    if menu_key in {"cmp", "shk"}:
        return f"img:sub:size:{artifact_id}"
    return f"img:open:{artifact_id}"


def submenu_for_field(field: str) -> str:
    if field in {"steps", "cfg", "denoise"}:
        return "smp"
    if field in {"hires_scale", "hires_denoise", "pag_scale"}:
        return "enh"
    if field in {"compression_percent", "shrink_size"}:
        return "size"
    return "hub"


def simple_value_keyboard(
    *,
    artifact_id: str,
    key: str,
    values: list[str],
    back_callback: str,
) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for value in values:
        row.append(
            InlineKeyboardButton(
                text=value,
                callback_data=f"img:set:{key}:{artifact_id}:{value}",
            )
        )
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(
        [
            InlineKeyboardButton(
                text="✏️ Ввести свое",
                callback_data=f"img:custom:{key}:{artifact_id}",
            )
        ]
    )
    rows.append(
        [
            InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=back_callback,
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def paginated_pick_keyboard(
    *,
    artifact_id: str,
    menu: str,
    items: list[str],
    page: int,
    back_callback: str,
) -> InlineKeyboardMarkup:
    page_size = 8
    total_pages = max(1, (len(items) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size

    rows: list[list[InlineKeyboardButton]] = []
    for idx, label in enumerate(items[start : start + page_size], start=start):
        rows.append(
            [
                InlineKeyboardButton(
                    text=label if len(label) <= 40 else label[:37] + "...",
                    callback_data=f"img:pick:{menu}:{artifact_id}:{idx}",
                )
            ]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="◀️",
                callback_data=f"img:page:{menu}:{artifact_id}:{page - 1}",
            )
        )
    nav.append(InlineKeyboardButton(text=f"· {page + 1}/{total_pages} ·", callback_data="noop"))
    if page < total_pages - 1:
        nav.append(
            InlineKeyboardButton(
                text="▶️",
                callback_data=f"img:page:{menu}:{artifact_id}:{page + 1}",
            )
        )
    rows.append(nav)
    rows.append(
        [
            InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=back_callback,
            )
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def shrink_value_keyboard(*, artifact_id: str, back_callback: str) -> InlineKeyboardMarkup:
    rows = [
        [
            InlineKeyboardButton(
                text="1024x1024",
                callback_data=f"img:set:shrink_size:{artifact_id}:1024x1024",
            ),
            InlineKeyboardButton(
                text="1280x720",
                callback_data=f"img:set:shrink_size:{artifact_id}:1280x720",
            ),
        ],
        [
            InlineKeyboardButton(
                text="1920x1080",
                callback_data=f"img:set:shrink_size:{artifact_id}:1920x1080",
            ),
            InlineKeyboardButton(
                text="2048x2048",
                callback_data=f"img:set:shrink_size:{artifact_id}:2048x2048",
            ),
        ],
        [
            InlineKeyboardButton(
                text="❌ Выключить",
                callback_data=f"img:set:shrink_size:{artifact_id}:off",
            )
        ],
        [
            InlineKeyboardButton(
                text="✏️ Ввести XxY",
                callback_data=f"img:custom:shrink_size:{artifact_id}",
            )
        ],
        [
            InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=back_callback,
            )
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def parse_shrink_size(value: str) -> tuple[int, int] | None:
    raw = value.strip().lower().replace(" ", "").replace("х", "x")
    if "x" not in raw:
        return None
    left, right = raw.split("x", 1)
    if not left or not right:
        return None
    try:
        width = int(left)
        height = int(right)
    except ValueError:
        return None
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        return None
    return (width, height)


def enhancement_preset_values(field: str) -> list[str]:
    control = numeric_enhancement_control(field)
    if control is None:
        return []
    return list(control.presets)


def custom_field_meta(field: str) -> tuple[str, float | int, float | int]:
    control = numeric_enhancement_control(field)
    if control is not None:
        return (control.label, control.min_value, control.max_value)
    if field == "steps":
        return ("Steps", 1, 150)
    if field == "cfg":
        return ("CFG", 0.0, 30.0)
    if field == "denoise":
        return ("Denoise", 0.0, 1.0)
    if field == "compression_percent":
        return ("Сжатие (%)", 1, 100)
    raise ValueError("unknown field")


def apply_field_value(
    artifact: PreviewArtifact,
    *,
    field: str,
    value: float | int,
) -> bool:
    if field == "steps":
        artifact.params.steps = int(value)
        artifact.enable_sampler_pass = True
        return True
    if field == "cfg":
        artifact.params.cfg = float(value)
        artifact.enable_sampler_pass = True
        return True
    if field == "denoise":
        artifact.params.denoise = float(value)
        artifact.enable_sampler_pass = True
        return True
    if field in {"hires_scale", "hires_denoise", "pag_scale"}:
        control = numeric_enhancement_control(field)
        if control is None:
            return False
        numeric = float(value)
        if numeric < control.min_value or numeric > control.max_value:
            return False
        setattr(artifact.params, field, numeric)
        return True
    if field == "compression_percent":
        artifact.compression_percent = int(value)
        return True
    return False
