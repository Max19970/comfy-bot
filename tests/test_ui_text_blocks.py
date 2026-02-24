from __future__ import annotations

from dataclasses import dataclass

from core.ui_kit.text_blocks import resolve_ui_text, text_button, ui_text


@dataclass(slots=True)
class _StubUITextService:
    def text(
        self,
        text_id: str,
        *,
        locale: str | None = None,
        profile: str | None = None,
        params: dict[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        value = f"{text_id}|{locale or '-'}|{profile or '-'}"
        if params:
            value += f"|{params.get('n')}"
        return value


def test_resolve_ui_text_uses_value_and_service() -> None:
    service = _StubUITextService()
    value = ui_text("common.menu.back_to_menu", "Back", params={"n": 3}, profile="brand")

    assert resolve_ui_text(service, value, locale="ru") == "common.menu.back_to_menu|ru|brand|3"


def test_text_button_builds_button_from_ui_text_value() -> None:
    service = _StubUITextService()
    value = ui_text("ui.start", "Start")

    btn = text_button(service, value, "menu:start", locale="en")

    assert btn.text == "ui.start|en|-"
    assert btn.callback_data == "menu:start"
