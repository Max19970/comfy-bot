from __future__ import annotations

import json
from pathlib import Path

import pytest

from infrastructure.ui_text_resources import (
    FileSystemUICopyProfileCatalog,
    FileSystemUITextRegistry,
    UITextResourceError,
)


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "entries": {
            "ui.start": {
                "default": "Start",
                "localization_key": "ui.start",
            },
            "ui.fallback": "Fallback",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_profile(
    path: Path,
    *,
    profile: str,
    is_default: bool,
    overrides: dict[str, str] | None = None,
    locale_overrides: dict[str, dict[str, str]] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "profile": profile,
        "enabled": True,
        "is_default": is_default,
        "overrides": overrides or {},
        "locale_overrides": locale_overrides or {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_filesystem_ui_text_registry_loads_entries() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = FileSystemUITextRegistry(str(root / "ui_text" / "registry.json"))

    assert "ui.start" in registry.list_text_ids()
    item = registry.get_definition("ui.start")
    assert item is not None
    assert item.localization_key == "ui.start"


def test_copy_profile_catalog_loads_default_and_overrides(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    _write_profile(
        profiles_dir / "base.profile.json",
        profile="base",
        is_default=True,
        overrides={"ui.start": "Base"},
    )
    _write_profile(
        profiles_dir / "brand.profile.json",
        profile="brand",
        is_default=False,
        locale_overrides={"ru": {"ui.start": "Бренд"}},
    )

    catalog = FileSystemUICopyProfileCatalog(str(profiles_dir))

    assert catalog.default_profile() == "base"
    assert catalog.list_profiles() == ["base", "brand"]
    brand = catalog.get_profile("brand")
    assert brand is not None
    assert brand.locale_overrides["ru"]["ui.start"] == "Бренд"


def test_copy_profile_catalog_requires_single_default(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    _write_profile(profiles_dir / "one.profile.json", profile="one", is_default=True)
    _write_profile(profiles_dir / "two.profile.json", profile="two", is_default=True)

    with pytest.raises(UITextResourceError):
        FileSystemUICopyProfileCatalog(str(profiles_dir))
