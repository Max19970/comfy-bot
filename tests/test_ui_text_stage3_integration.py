from __future__ import annotations

from dataclasses import dataclass

from core.ui_copy import fallback_text, main_menu_keyboard, start_text
from core.ui_kit.buttons import back_button, custom_value_button
from core.ui_kit.dialogs import confirm_cancel_rows


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
        suffix = locale or "-"
        return f"{text_id}:{suffix}"


@dataclass(slots=True)
class _StubLocalization:
    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: dict[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        return f"loc:{key}:{locale or '-'}"


def test_ui_copy_prefers_text_service_when_provided() -> None:
    text_service = _StubUITextService()
    localization = _StubLocalization()

    assert start_text(localization, locale="ru", text_service=text_service) == "ui.start:ru"
    assert fallback_text(localization, locale="ru", text_service=text_service) == "ui.fallback:ru"


def test_ui_copy_main_menu_uses_text_service_blocks() -> None:
    text_service = _StubUITextService()

    keyboard = main_menu_keyboard(text_service=text_service, locale="en")

    rows = keyboard.inline_keyboard
    assert rows[0][0].text == "ui.menu.generation:en"
    assert rows[0][1].text == "ui.menu.models:en"
    assert rows[1][0].text == "ui.menu.service:en"
    assert rows[1][1].text == "ui.menu.cancel:en"


def test_ui_kit_buttons_support_text_service_resolution() -> None:
    text_service = _StubUITextService()

    back = back_button("menu:back", text_service=text_service, locale="en")
    manual = custom_value_button("pick:manual", text_service=text_service, locale="en")

    assert back.text == "common.action.back:en"
    assert manual.text == "common.action.enter_manual:en"


def test_ui_kit_dialogs_support_text_service_resolution() -> None:
    text_service = _StubUITextService()

    rows = confirm_cancel_rows(
        confirm_callback="ok",
        cancel_callback="cancel",
        text_service=text_service,
        locale="ru",
    )

    assert rows[0][0].text == "core.ui_kit.dialogs.confirm:ru"
    assert rows[1][0].text == "common.action.cancel:ru"
