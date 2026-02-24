from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from .localization import normalize_locale_code

_TEXT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_PROFILE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


def normalize_text_id(value: str, *, default: str = "") -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if not _TEXT_ID_RE.match(normalized):
        return default
    return normalized


def normalize_profile_name(value: str, *, default: str = "") -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if not _PROFILE_RE.match(normalized):
        return default
    return normalized


@dataclass(frozen=True, slots=True)
class UITextDefinition:
    text_id: str
    default_text: str
    localization_key: str | None = None


@dataclass(frozen=True, slots=True)
class UIText:
    text_id: str
    default_text: str
    params: Mapping[str, object] | None = None
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class UITextRequest:
    text_id: str
    localization_key: str
    default_text: str
    locale: str | None
    profile: str
    params: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class UICopyProfile:
    profile: str
    enabled: bool
    is_default: bool
    overrides: Mapping[str, str]
    locale_overrides: Mapping[str, Mapping[str, str]]


class UITextRegistry(Protocol):
    def list_text_ids(self) -> list[str]: ...

    def get_definition(self, text_id: str) -> UITextDefinition | None: ...


class UICopyProfileCatalog(Protocol):
    def list_profiles(self) -> list[str]: ...

    def default_profile(self) -> str: ...

    def get_profile(self, profile: str) -> UICopyProfile | None: ...


class UITextLocalizer(Protocol):
    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str: ...


class UITextModifier(Protocol):
    def modify(self, *, request: UITextRequest, text: str) -> str: ...


class UITextService(Protocol):
    def text(
        self,
        text_id: str,
        *,
        locale: str | None = None,
        profile: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str: ...

    def default_profile(self) -> str: ...

    def available_profiles(self) -> tuple[str, ...]: ...


def normalize_locale_or_empty(value: str | None) -> str:
    return normalize_locale_code(str(value or ""), default="")
