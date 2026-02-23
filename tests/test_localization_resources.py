from __future__ import annotations

import json
from pathlib import Path

from application.localization_service import DefaultLocalizationService
from infrastructure.localization_catalog import FileSystemTranslationCatalog


def _messages_path(locale: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "locales" / locale / "messages.json"


def _write_locale(
    root: Path,
    locale: str,
    *,
    is_default: bool,
    fallback_locale: str | None,
    messages: dict[str, str],
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
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (locale_dir / "messages.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_ru_and_en_locales_have_same_message_keyset() -> None:
    ru_messages = json.loads(_messages_path("ru").read_text(encoding="utf-8"))
    en_messages = json.loads(_messages_path("en").read_text(encoding="utf-8"))

    ru_keys = set(ru_messages.keys())
    en_keys = set(en_messages.keys())

    missing_in_en = sorted(ru_keys - en_keys)
    missing_in_ru = sorted(en_keys - ru_keys)
    assert not missing_in_en, f"Missing keys in EN bundle: {missing_in_en}"
    assert not missing_in_ru, f"Missing keys in RU bundle: {missing_in_ru}"


def test_dynamic_locale_discovery_works_with_files_only(tmp_path: Path) -> None:
    _write_locale(
        tmp_path,
        "ru",
        is_default=True,
        fallback_locale=None,
        messages={"hello": "Привет"},
    )
    _write_locale(
        tmp_path,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={"hello": "Hello"},
    )

    base_catalog = FileSystemTranslationCatalog(str(tmp_path))
    assert set(base_catalog.list_locales()) == {"ru", "en"}

    _write_locale(
        tmp_path,
        "de",
        is_default=False,
        fallback_locale="en",
        messages={"hello": "Hallo"},
    )

    catalog = FileSystemTranslationCatalog(str(tmp_path))
    service = DefaultLocalizationService(catalog)

    assert set(catalog.list_locales()) == {"ru", "en", "de"}
    assert catalog.default_locale() == "ru"
    assert service.t("hello", locale="de") == "Hallo"
