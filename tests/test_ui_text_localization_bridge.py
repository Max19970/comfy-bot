from __future__ import annotations

from dataclasses import dataclass

from application.ui_text_localization_bridge import UITextLocalizationBridge


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
        if default is not None:
            return f"loc:{key}:{default}"
        return f"loc:{key}"

    def default_locale(self) -> str:
        return "ru"

    def available_locales(self) -> tuple[str, ...]:
        return ("ru", "en")


@dataclass(slots=True)
class _StubUIText:
    def text(
        self,
        text_id: str,
        *,
        locale: str | None = None,
        profile: str | None = None,
        params: dict[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        return f"ui:{text_id}:{locale or '-'}:{default or ''}"


def test_bridge_routes_valid_text_keys_to_ui_text_service() -> None:
    bridge = UITextLocalizationBridge(
        localization=_StubLocalization(),
        ui_text=_StubUIText(),
    )

    value = bridge.t("download.filter.panel.title", locale="en", default="X")

    assert value == "ui:download.filter.panel.title:en:X"


def test_bridge_keeps_non_text_keys_on_localization_service() -> None:
    bridge = UITextLocalizationBridge(
        localization=_StubLocalization(),
        ui_text=_StubUIText(),
    )

    value = bridge.t("NOT_A_TEXT_KEY", locale="en", default="fallback")

    assert value == "loc:NOT_A_TEXT_KEY:fallback"


def test_bridge_delegates_locale_metadata_access() -> None:
    bridge = UITextLocalizationBridge(
        localization=_StubLocalization(),
        ui_text=_StubUIText(),
    )

    assert bridge.default_locale() == "ru"
    assert bridge.available_locales() == ("ru", "en")
