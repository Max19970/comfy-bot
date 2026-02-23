from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from application.user_locale_resolver import DefaultUserLocaleResolver


@dataclass
class _StubLocalization:
    locales: tuple[str, ...]
    default: str

    def default_locale(self) -> str:
        return self.default

    def available_locales(self) -> tuple[str, ...]:
        return self.locales

    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str:
        _ = (key, locale, params)
        return default or ""


def test_resolver_prefers_explicit_user_locale() -> None:
    resolver = DefaultUserLocaleResolver(_StubLocalization(locales=("en", "ru"), default="ru"))

    resolved = resolver.resolve(user_locale="en", telegram_locale="ru")
    assert resolved == "en"


def test_resolver_uses_telegram_locale_when_user_not_set() -> None:
    resolver = DefaultUserLocaleResolver(_StubLocalization(locales=("en", "ru"), default="ru"))

    resolved = resolver.resolve(user_locale=None, telegram_locale="en-US")
    assert resolved == "en"


def test_resolver_falls_back_to_default_locale() -> None:
    resolver = DefaultUserLocaleResolver(_StubLocalization(locales=("en", "ru"), default="ru"))

    resolved = resolver.resolve(user_locale="de", telegram_locale="pt-BR")
    assert resolved == "ru"
