from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from application.ui_text_service import DefaultUITextService
from domain.ui_text import UICopyProfile, UITextDefinition
from infrastructure.ui_text_modifiers import CopyProfileModifier, LocalizationModifier


@dataclass(slots=True)
class _StubRegistry:
    definitions: dict[str, UITextDefinition]

    def list_text_ids(self) -> list[str]:
        return sorted(self.definitions.keys())

    def get_definition(self, text_id: str) -> UITextDefinition | None:
        return self.definitions.get(text_id)


@dataclass(slots=True)
class _StubProfiles:
    profiles: dict[str, UICopyProfile]
    default: str

    def list_profiles(self) -> list[str]:
        return sorted(self.profiles.keys())

    def default_profile(self) -> str:
        return self.default

    def get_profile(self, profile: str) -> UICopyProfile | None:
        return self.profiles.get(profile)


@dataclass(slots=True)
class _StubLocalizer:
    values: dict[tuple[str, str | None], str]

    def default_locale(self) -> str:
        return "ru"

    def available_locales(self) -> tuple[str, ...]:
        return ("en", "ru")

    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        template = self.values.get((key, locale), default or key)
        if params:
            return template.format_map({str(k): v for k, v in params.items()})
        return template


def _build_service() -> DefaultUITextService:
    registry = _StubRegistry(
        definitions={
            "ui.start": UITextDefinition(
                text_id="ui.start",
                localization_key="ui.start",
                default_text="Start default {name}",
            )
        }
    )
    profiles = _StubProfiles(
        default="base",
        profiles={
            "base": UICopyProfile(
                profile="base",
                enabled=True,
                is_default=True,
                overrides={},
                locale_overrides={},
            ),
            "brand": UICopyProfile(
                profile="brand",
                enabled=True,
                is_default=False,
                overrides={"ui.start": "Brand default {name}"},
                locale_overrides={"ru": {"ui.start": "Бренд {name}"}},
            ),
        },
    )
    localizer = _StubLocalizer(
        values={
            ("ui.start", "en"): "Localized EN {name}",
            ("ui.start", "ru"): "Локализовано RU {name}",
        }
    )
    return DefaultUITextService(
        registry=registry,
        profiles=profiles,
        modifiers=(
            LocalizationModifier(localization=localizer),
            CopyProfileModifier(profiles=profiles),
        ),
    )


def test_ui_text_service_prefers_profile_overrides() -> None:
    service = _build_service()

    assert (
        service.text("ui.start", locale="ru", profile="brand", params={"name": "Alex"})
        == "Бренд Alex"
    )
    assert (
        service.text("ui.start", locale="en", profile="brand", params={"name": "Alex"})
        == "Brand default Alex"
    )


def test_ui_text_service_falls_back_to_localizer_and_default() -> None:
    service = _build_service()

    assert service.text("ui.start", locale="en", params={"name": "Sam"}) == "Localized EN Sam"
    assert service.text("ui.start", locale="de", params={"name": "Sam"}) == "Start default Sam"


def test_ui_text_service_supports_unknown_text_id_defaults() -> None:
    service = _build_service()

    assert service.text("custom.card.title", default="Card {n}", params={"n": 3}) == "Card 3"


def test_ui_text_service_exposes_profiles() -> None:
    service = _build_service()

    assert service.default_profile() == "base"
    assert service.available_profiles() == ("base", "brand")
