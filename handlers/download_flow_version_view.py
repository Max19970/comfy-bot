from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button

TranslateText = Callable[[str, str | None, str, Mapping[str, object] | None], str]


def _tx(
    translate: TranslateText | None,
    key: str,
    locale: str | None,
    default: str,
    *,
    params: Mapping[str, object] | None = None,
) -> str:
    if translate is None:
        if not params:
            return default
        try:
            return default.format_map({str(k): v for k, v in params.items()})
        except (KeyError, ValueError, TypeError):
            return default
    return translate(key, locale, default, params)


def build_version_selection_view(
    result: Any,
    *,
    human_size: Callable[[int], str],
    short_number: Callable[[int], str],
    escape_html: Callable[[Any], str],
    max_versions: int = 12,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    rows: list[list[InlineKeyboardButton]] = []
    versions = list(result.available_versions[:max_versions])

    for idx, option in enumerate(versions):
        label_parts: list[str] = []
        if option.base_model:
            label_parts.append(option.base_model)
        if option.version_name:
            label_parts.append(option.version_name)
        if option.size_bytes > 0:
            label_parts.append(human_size(option.size_bytes))
        short = " | ".join(label_parts) or _tx(
            translate,
            "download.version.option.default",
            locale,
            "Версия {index}",
            params={"index": idx + 1},
        )
        short = short if len(short) <= 58 else short[:55] + "..."
        rows.append([button(f"{idx + 1}. {short}", f"dlver:{idx}")])

    rows.append([cancel_button("dlver:cancel")])
    rows.append(
        [
            back_button(
                "dlver:back",
                text=_tx(
                    translate,
                    "download.version.back_to_results",
                    locale,
                    "⬅️ К результатам",
                ),
            )
        ]
    )

    lines = [
        _tx(
            translate,
            "download.version.title",
            locale,
            "🧬 <b>Выберите версию</b> для <b>{name}</b>",
            params={"name": escape_html(result.name)},
        ),
        _tx(
            translate,
            "download.version.found_count",
            locale,
            "Найдено версий: <b>{count}</b>",
            params={"count": len(result.available_versions)},
        ),
        "",
    ]
    for idx, option in enumerate(versions):
        meta: list[str] = []
        if option.base_model:
            meta.append(option.base_model)
        if option.size_bytes > 0:
            meta.append(human_size(option.size_bytes))
        if option.download_count > 0:
            meta.append(f"📥 {short_number(option.download_count)}")
        if option.version_name:
            lines.append(f"{idx + 1}. <b>{escape_html(option.version_name)}</b>")
        else:
            lines.append(
                _tx(
                    translate,
                    "download.version.option.indexed",
                    locale,
                    "{index}. <b>Версия {index}</b>",
                    params={"index": idx + 1},
                )
            )
        if meta:
            lines.append(f"   <i>{escape_html(' | '.join(meta))}</i>")

    return "\n".join(lines), build_keyboard(rows)
