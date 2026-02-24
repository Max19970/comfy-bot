from __future__ import annotations

import json
from pathlib import Path

from domain.ui_text import (
    UICopyProfile,
    UICopyProfileCatalog,
    UITextDefinition,
    UITextRegistry,
    normalize_locale_or_empty,
    normalize_profile_name,
    normalize_text_id,
)


class UITextResourceError(RuntimeError):
    pass


class FileSystemUITextRegistry(UITextRegistry):
    def __init__(self, registry_file_path: str) -> None:
        self._registry_path = Path(registry_file_path)
        self._definitions: dict[str, UITextDefinition] = {}
        self._load()

    def list_text_ids(self) -> list[str]:
        return sorted(self._definitions.keys())

    def get_definition(self, text_id: str) -> UITextDefinition | None:
        normalized_text_id = normalize_text_id(text_id)
        if not normalized_text_id:
            return None
        return self._definitions.get(normalized_text_id)

    def _load(self) -> None:
        payload = self._read_json_object(self._registry_path)
        if not isinstance(payload, dict):
            raise UITextResourceError(f"UI text registry must be an object: {self._registry_path}")

        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise UITextResourceError(
                f"Invalid schema_version in UI text registry: {self._registry_path}"
            )

        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise UITextResourceError(f"Registry entries must be an object: {self._registry_path}")

        definitions: dict[str, UITextDefinition] = {}
        for raw_text_id, raw_value in entries.items():
            text_id = normalize_text_id(str(raw_text_id or ""), default="")
            if not text_id:
                raise UITextResourceError(
                    f"Invalid text_id in registry '{self._registry_path}': {raw_text_id!r}"
                )

            if isinstance(raw_value, str):
                default_text = raw_value
                localization_key = text_id
            elif isinstance(raw_value, dict):
                default_text = str(raw_value.get("default") or "")
                if not default_text:
                    raise UITextResourceError(
                        f"Missing default text for '{text_id}' in registry: {self._registry_path}"
                    )
                localization_key_raw = str(raw_value.get("localization_key") or "").strip()
                localization_key = localization_key_raw or text_id
            else:
                raise UITextResourceError(
                    f"Invalid registry entry type for '{text_id}' in {self._registry_path}"
                )

            if text_id in definitions:
                raise UITextResourceError(
                    f"Duplicate text_id in registry '{self._registry_path}': {text_id}"
                )

            definitions[text_id] = UITextDefinition(
                text_id=text_id,
                default_text=default_text,
                localization_key=localization_key,
            )

        if not definitions:
            raise UITextResourceError(f"UI text registry is empty: {self._registry_path}")

        self._definitions = definitions

    def _read_json_object(self, path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as exc:
            raise UITextResourceError(f"UI text resource file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise UITextResourceError(f"Invalid JSON in UI text resource file: {path}") from exc
        except OSError as exc:
            raise UITextResourceError(f"Failed to read UI text resource file: {path}") from exc


class FileSystemUICopyProfileCatalog(UICopyProfileCatalog):
    def __init__(self, profiles_root_path: str, *, filename_suffix: str = ".profile.json") -> None:
        self._profiles_root = Path(profiles_root_path)
        self._filename_suffix = filename_suffix
        self._profiles: dict[str, UICopyProfile] = {}
        self._default_profile = ""
        self._load()

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles.keys())

    def default_profile(self) -> str:
        return self._default_profile

    def get_profile(self, profile: str) -> UICopyProfile | None:
        profile_name = normalize_profile_name(profile)
        if not profile_name:
            return None
        return self._profiles.get(profile_name)

    def _load(self) -> None:
        if not self._profiles_root.exists() or not self._profiles_root.is_dir():
            raise UITextResourceError(f"UI text profiles root not found: {self._profiles_root}")

        loaded: dict[str, UICopyProfile] = {}
        for file_path in sorted(self._profiles_root.iterdir()):
            if not file_path.is_file() or not file_path.name.endswith(self._filename_suffix):
                continue
            profile = self._load_profile(file_path)
            if not profile.enabled:
                continue
            if profile.profile in loaded:
                raise UITextResourceError(
                    f"Duplicate copy profile '{profile.profile}' in {self._profiles_root}"
                )
            loaded[profile.profile] = profile

        if not loaded:
            raise UITextResourceError(
                f"No enabled UI text copy profiles found in: {self._profiles_root}"
            )

        defaults = [name for name, profile in loaded.items() if profile.is_default]
        if len(defaults) != 1:
            raise UITextResourceError(
                f"Exactly one enabled copy profile must have is_default=true in {self._profiles_root}"
            )

        self._profiles = loaded
        self._default_profile = defaults[0]

    def _load_profile(self, file_path: Path) -> UICopyProfile:
        payload = self._read_json_object(file_path)
        if not isinstance(payload, dict):
            raise UITextResourceError(f"Copy profile must be an object: {file_path}")

        schema_version = payload.get("schema_version")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise UITextResourceError(f"Invalid schema_version in copy profile: {file_path}")

        profile_name = normalize_profile_name(str(payload.get("profile") or ""), default="")
        if not profile_name:
            raise UITextResourceError(f"Invalid profile name in copy profile: {file_path}")

        overrides = self._parse_overrides(payload.get("overrides"), file_path=file_path)
        locale_overrides = self._parse_locale_overrides(
            payload.get("locale_overrides"),
            file_path=file_path,
        )

        return UICopyProfile(
            profile=profile_name,
            enabled=bool(payload.get("enabled", True)),
            is_default=bool(payload.get("is_default", False)),
            overrides=overrides,
            locale_overrides=locale_overrides,
        )

    def _parse_overrides(self, payload: object, *, file_path: Path) -> dict[str, str]:
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise UITextResourceError(f"Profile overrides must be an object: {file_path}")

        overrides: dict[str, str] = {}
        for raw_text_id, raw_value in payload.items():
            text_id = normalize_text_id(str(raw_text_id or ""), default="")
            if not text_id:
                raise UITextResourceError(
                    f"Invalid override text_id '{raw_text_id}' in {file_path}"
                )
            if not isinstance(raw_value, str):
                raise UITextResourceError(
                    f"Override value for '{text_id}' must be string in {file_path}"
                )
            overrides[text_id] = raw_value
        return overrides

    def _parse_locale_overrides(
        self, payload: object, *, file_path: Path
    ) -> dict[str, dict[str, str]]:
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise UITextResourceError(f"locale_overrides must be an object: {file_path}")

        locale_overrides: dict[str, dict[str, str]] = {}
        for raw_locale, raw_overrides in payload.items():
            locale_code = normalize_locale_or_empty(str(raw_locale or ""))
            if not locale_code:
                raise UITextResourceError(f"Invalid locale key '{raw_locale}' in {file_path}")
            parsed = self._parse_overrides(raw_overrides, file_path=file_path)
            locale_overrides[locale_code] = parsed
        return locale_overrides

    def _read_json_object(self, path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError as exc:
            raise UITextResourceError(f"UI text resource file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise UITextResourceError(f"Invalid JSON in UI text resource file: {path}") from exc
        except OSError as exc:
            raise UITextResourceError(f"Failed to read UI text resource file: {path}") from exc
