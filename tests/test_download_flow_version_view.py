from __future__ import annotations

from handlers.download_flow_version_view import build_version_selection_view
from model_downloader import SearchResult, SearchVersionOption


def _callback_payloads(markup) -> set[str]:
    payloads: set[str] = set()
    for row in markup.inline_keyboard:
        for item in row:
            if item.callback_data is not None:
                payloads.add(item.callback_data)
    return payloads


def test_build_version_selection_view_renders_callbacks_and_text() -> None:
    result = SearchResult(
        name="My Model",
        source="civitai",
        available_versions=[
            SearchVersionOption(
                version_id="v1",
                version_name="v1",
                base_model="SDXL",
                size_bytes=1024,
                download_count=1200,
            ),
            SearchVersionOption(
                version_id="v2",
                version_name="v2",
                base_model="Pony",
                size_bytes=2048,
                download_count=6400,
            ),
        ],
    )

    text, markup = build_version_selection_view(
        result,
        human_size=lambda size: f"{size} B",
        short_number=lambda value: str(value),
        escape_html=lambda value: str(value),
    )

    assert "Выберите версию" in text
    assert "My Model" in text
    assert "Найдено версий" in text
    assert "SDXL" in text
    assert "Pony" in text

    payloads = _callback_payloads(markup)
    assert "dlver:0" in payloads
    assert "dlver:1" in payloads
    assert "dlver:cancel" in payloads
    assert "dlver:back" in payloads
