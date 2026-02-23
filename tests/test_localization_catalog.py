from __future__ import annotations

import json
from pathlib import Path

import pytest

from infrastructure.localization_catalog import (
    FileSystemTranslationCatalog,
    LocalizationCatalogError,
)


def _write_locale(
    root: Path,
    locale: str,
    *,
    is_default: bool,
    fallback_locale: str | None = None,
    messages: dict[str, str] | None = None,
) -> None:
    locale_dir = root / locale
    locale_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "locale": locale,
        "name": locale.upper(),
        "native_name": locale.upper(),
        "messages_file": "messages.json",
        "is_default": is_default,
        "enabled": True,
        "fallback_locale": fallback_locale,
    }
    (locale_dir / "locale.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (locale_dir / "messages.json").write_text(
        json.dumps(messages or {"system.hello": locale}),
        encoding="utf-8",
    )


def test_filesystem_translation_catalog_loads_locales_and_default(tmp_path: Path) -> None:
    _write_locale(tmp_path, "ru", is_default=True, messages={"k": "Привет"})
    _write_locale(
        tmp_path,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={"k": "Hello"},
    )

    catalog = FileSystemTranslationCatalog(str(tmp_path))

    assert catalog.default_locale() == "ru"
    assert catalog.list_locales() == ["en", "ru"]
    bundle = catalog.get_bundle("en")
    assert bundle is not None
    assert bundle.metadata.fallback_locale == "ru"
    assert bundle.messages["k"] == "Hello"


def test_filesystem_translation_catalog_fails_when_metadata_missing(tmp_path: Path) -> None:
    (tmp_path / "ru").mkdir(parents=True)
    with pytest.raises(LocalizationCatalogError):
        FileSystemTranslationCatalog(str(tmp_path))


def test_filesystem_translation_catalog_fails_on_broken_metadata_json(tmp_path: Path) -> None:
    locale_dir = tmp_path / "ru"
    locale_dir.mkdir(parents=True)
    (locale_dir / "locale.meta.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(LocalizationCatalogError):
        FileSystemTranslationCatalog(str(tmp_path))


def test_filesystem_translation_catalog_fails_on_missing_messages_file(tmp_path: Path) -> None:
    locale_dir = tmp_path / "ru"
    locale_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "locale": "ru",
        "name": "Russian",
        "native_name": "Русский",
        "messages_file": "messages.json",
        "is_default": True,
        "enabled": True,
        "fallback_locale": None,
    }
    (locale_dir / "locale.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(LocalizationCatalogError):
        FileSystemTranslationCatalog(str(tmp_path))


def test_filesystem_translation_catalog_fails_on_broken_messages_json(tmp_path: Path) -> None:
    locale_dir = tmp_path / "ru"
    locale_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "locale": "ru",
        "name": "Russian",
        "native_name": "Русский",
        "messages_file": "messages.json",
        "is_default": True,
        "enabled": True,
        "fallback_locale": None,
    }
    (locale_dir / "locale.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (locale_dir / "messages.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(LocalizationCatalogError):
        FileSystemTranslationCatalog(str(tmp_path))
