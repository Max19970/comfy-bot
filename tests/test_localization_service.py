from __future__ import annotations

import json
from pathlib import Path

from application.localization_service import DefaultLocalizationService
from infrastructure.localization_catalog import FileSystemTranslationCatalog


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
        json.dumps(messages or {}),
        encoding="utf-8",
    )


def _build_service(tmp_path: Path) -> DefaultLocalizationService:
    _write_locale(
        tmp_path,
        "ru",
        is_default=True,
        messages={
            "greeting": "Привет",
            "jobs.active": "Активно: {count}, {missing}",
            "shared.only_ru": "Только RU",
        },
    )
    _write_locale(
        tmp_path,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={
            "greeting": "Hello",
        },
    )
    catalog = FileSystemTranslationCatalog(str(tmp_path))
    return DefaultLocalizationService(catalog)


def test_localization_service_resolves_locale_with_fallback_chain(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    assert service.t("greeting", locale="en") == "Hello"
    assert service.t("shared.only_ru", locale="en") == "Только RU"
    assert service.t("shared.only_ru", locale="de") == "Только RU"


def test_localization_service_formats_params_safely(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    assert service.t("jobs.active", locale="ru", params={"count": 3}) == "Активно: 3, {missing}"


def test_localization_service_returns_default_or_key_for_missing_message(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    assert service.t("unknown.key", locale="en", default="Fallback") == "Fallback"
    assert service.t("unknown.key", locale="en") == "unknown.key"
    assert service.available_locales() == ("en", "ru")
    assert service.default_locale() == "ru"
