from __future__ import annotations

import json
from pathlib import Path

from domain.localization import (
    LocaleMetadata,
    TranslationBundle,
    TranslationCatalog,
    normalize_locale_code,
)


class LocalizationCatalogError(RuntimeError):
    pass


class FileSystemTranslationCatalog(TranslationCatalog):
    def __init__(
        self,
        locales_root_path: str,
        *,
        metadata_filename: str = "locale.meta.json",
    ) -> None:
        self._locales_root = Path(locales_root_path)
        self._metadata_filename = metadata_filename
        self._bundles: dict[str, TranslationBundle] = {}
        self._default_locale = ""
        self._load()

    def list_locales(self) -> list[str]:
        return sorted(self._bundles.keys())

    def default_locale(self) -> str:
        return self._default_locale

    def get_bundle(self, locale: str) -> TranslationBundle | None:
        locale_code = normalize_locale_code(locale)
        if not locale_code:
            return None
        return self._bundles.get(locale_code)

    def _load(self) -> None:
        if not self._locales_root.exists() or not self._locales_root.is_dir():
            raise LocalizationCatalogError(f"Localization root not found: {self._locales_root}")

        bundles: dict[str, TranslationBundle] = {}
        for locale_dir in sorted(self._locales_root.iterdir()):
            if not locale_dir.is_dir():
                continue

            bundle = self._load_bundle(locale_dir)
            if bundle is None:
                continue

            locale = bundle.metadata.locale
            if locale in bundles:
                raise LocalizationCatalogError(f"Duplicate locale detected: {locale}")
            bundles[locale] = bundle

        if not bundles:
            raise LocalizationCatalogError("No enabled locales loaded from localization resources")

        default_locales = [code for code, bundle in bundles.items() if bundle.metadata.is_default]
        if len(default_locales) != 1:
            raise LocalizationCatalogError("Exactly one locale must have is_default=true")

        self._validate_fallbacks(bundles)
        self._bundles = bundles
        self._default_locale = default_locales[0]

    def _load_bundle(self, locale_dir: Path) -> TranslationBundle | None:
        metadata_path = locale_dir / self._metadata_filename
        if not metadata_path.exists():
            raise LocalizationCatalogError(f"Missing locale metadata file: {metadata_path}")

        metadata_payload = self._read_json_object(metadata_path)
        metadata = self._parse_metadata(metadata_payload, locale_dir)
        if not metadata.enabled:
            return None

        messages_path = locale_dir / metadata.messages_file
        if not messages_path.exists() or not messages_path.is_file():
            raise LocalizationCatalogError(
                f"Missing messages file for locale '{metadata.locale}': {messages_path}"
            )

        messages_payload = self._read_json_object(messages_path)
        messages = self._parse_messages(messages_payload, locale=metadata.locale)
        return TranslationBundle(metadata=metadata, messages=messages)

    def _parse_metadata(self, payload: object, locale_dir: Path) -> LocaleMetadata:
        if not isinstance(payload, dict):
            raise LocalizationCatalogError(
                f"Locale metadata must be an object: {locale_dir / self._metadata_filename}"
            )

        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise LocalizationCatalogError(
                f"Invalid schema_version in metadata: {locale_dir / self._metadata_filename}"
            )

        locale = normalize_locale_code(str(payload.get("locale") or ""))
        if not locale:
            raise LocalizationCatalogError(
                f"Invalid locale code in metadata: {locale_dir / self._metadata_filename}"
            )

        expected_locale = normalize_locale_code(locale_dir.name)
        if expected_locale and expected_locale != locale:
            raise LocalizationCatalogError(
                f"Locale directory '{locale_dir.name}' does not match metadata locale '{locale}'"
            )

        name = str(payload.get("name") or "").strip()
        if not name:
            raise LocalizationCatalogError(
                f"Missing locale name in metadata: {locale_dir / self._metadata_filename}"
            )

        native_name = str(payload.get("native_name") or "").strip() or name

        messages_file = str(payload.get("messages_file") or "").strip()
        if not messages_file:
            raise LocalizationCatalogError(
                f"Missing messages_file in metadata: {locale_dir / self._metadata_filename}"
            )

        fallback_locale_raw = str(payload.get("fallback_locale") or "").strip()
        fallback_locale = (
            normalize_locale_code(fallback_locale_raw, default="") if fallback_locale_raw else ""
        )
        if fallback_locale == locale:
            raise LocalizationCatalogError(
                f"fallback_locale cannot reference itself: {locale_dir / self._metadata_filename}"
            )

        return LocaleMetadata(
            schema_version=schema_version,
            locale=locale,
            name=name,
            native_name=native_name,
            messages_file=messages_file,
            is_default=bool(payload.get("is_default", False)),
            enabled=bool(payload.get("enabled", True)),
            fallback_locale=fallback_locale or None,
        )

    def _parse_messages(self, payload: object, *, locale: str) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise LocalizationCatalogError(
                f"Messages payload for locale '{locale}' must be an object"
            )

        messages: dict[str, str] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key or "").strip()
            if not key:
                raise LocalizationCatalogError(f"Locale '{locale}' contains an empty message key")
            if not isinstance(raw_value, str):
                raise LocalizationCatalogError(
                    f"Locale '{locale}' has non-string value for key '{key}'"
                )
            messages[key] = raw_value
        return messages

    def _read_json_object(self, path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as exc:
            raise LocalizationCatalogError(f"Localization file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise LocalizationCatalogError(f"Invalid JSON in localization file: {path}") from exc
        except OSError as exc:
            raise LocalizationCatalogError(f"Failed to read localization file: {path}") from exc

    def _validate_fallbacks(self, bundles: dict[str, TranslationBundle]) -> None:
        for locale, bundle in bundles.items():
            fallback_locale = bundle.metadata.fallback_locale
            if not fallback_locale:
                continue
            if fallback_locale not in bundles:
                raise LocalizationCatalogError(
                    f"Locale '{locale}' references unknown fallback_locale '{fallback_locale}'"
                )
