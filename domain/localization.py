from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol

_LOCALE_CODE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")


def normalize_locale_code(value: str, *, default: str = "") -> str:
    normalized = str(value or "").strip().replace("_", "-").lower()
    if not normalized:
        return default
    if not _LOCALE_CODE_RE.match(normalized):
        return default
    return normalized


@dataclass(frozen=True, slots=True)
class LocaleMetadata:
    schema_version: int
    locale: str
    name: str
    native_name: str
    messages_file: str
    is_default: bool
    enabled: bool
    fallback_locale: str | None


@dataclass(frozen=True, slots=True)
class TranslationBundle:
    metadata: LocaleMetadata
    messages: Mapping[str, str]


class TranslationCatalog(Protocol):
    def list_locales(self) -> list[str]: ...

    def default_locale(self) -> str: ...

    def get_bundle(self, locale: str) -> TranslationBundle | None: ...


class LocalizationService(Protocol):
    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str: ...

    def default_locale(self) -> str: ...

    def available_locales(self) -> tuple[str, ...]: ...


class UserLocaleResolver(Protocol):
    def resolve(
        self,
        *,
        user_locale: str | None,
        telegram_locale: str | None,
    ) -> str: ...
