from __future__ import annotations

import json
from pathlib import Path

from infrastructure.locale_maintenance import (
    RuntimeKeyUsage,
    audit_locales,
    audit_summary,
    build_canonical_defaults,
    default_locale_from_bundles,
    discover_locale_bundles,
    extract_runtime_key_usage,
    load_ui_text_registry_defaults,
    sync_locales,
)


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
    meta = {
        "schema_version": 1,
        "locale": locale,
        "name": locale.upper(),
        "native_name": locale.upper(),
        "messages_file": "messages.json",
        "is_default": is_default,
        "enabled": True,
        "fallback_locale": fallback_locale,
    }
    (locale_dir / "locale.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (locale_dir / "messages.json").write_text(json.dumps(messages), encoding="utf-8")


def test_extract_runtime_key_usage_finds_helper_and_direct_calls(tmp_path: Path) -> None:
    app_dir = tmp_path / "application"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "sample.py").write_text(
        """
def x(localization, translate, text_service):
    localization.t(\"sample.direct\", default=\"Direct\")
    _t(localization, \"sample.helper\", default=\"Helper\")
    _tx(translate, \"sample.tx\", None, \"Tx fallback\")
    text_service.text(\"sample.block\", default=\"Block\")
""".strip(),
        encoding="utf-8",
    )

    usage = extract_runtime_key_usage(tmp_path)

    assert "sample.direct" in usage
    assert "sample.helper" in usage
    assert "sample.tx" in usage
    assert "sample.block" in usage
    assert "Direct" in usage["sample.direct"].defaults
    assert "Helper" in usage["sample.helper"].defaults
    assert "Tx fallback" in usage["sample.tx"].defaults


def test_sync_locales_adds_missing_keys_with_strategy(tmp_path: Path) -> None:
    locales_root = tmp_path / "locales"
    _write_locale(
        locales_root,
        "ru",
        is_default=True,
        fallback_locale=None,
        messages={"existing": "Есть"},
    )
    _write_locale(
        locales_root,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={"existing": "Exists"},
    )

    bundles = discover_locale_bundles(locales_root)
    default_locale = default_locale_from_bundles(bundles)
    canonical_defaults = {
        "existing": "Есть",
        "added.key": "Новое значение",
    }

    result = sync_locales(
        bundles=bundles,
        default_locale=default_locale,
        canonical_defaults=canonical_defaults,
        missing_strategy="todo",
        write=False,
    )

    assert result.added_by_locale["ru"] == 1
    assert result.added_by_locale["en"] == 1
    assert bundles["ru"].messages["added.key"] == "Новое значение"
    assert bundles["en"].messages["added.key"] == "[[TODO:en]] added.key"


def test_audit_locales_detects_placeholder_and_mixed_script(tmp_path: Path) -> None:
    locales_root = tmp_path / "locales"
    _write_locale(
        locales_root,
        "ru",
        is_default=True,
        fallback_locale=None,
        messages={"ui.greeting": "Привет, {name}!"},
    )
    _write_locale(
        locales_root,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={"ui.greeting": "Привет, {user}!"},
    )

    bundles = discover_locale_bundles(locales_root)
    default_locale = default_locale_from_bundles(bundles)
    canonical_defaults = {"ui.greeting": "Привет, {name}!", "ui.extra": "x"}

    report = audit_locales(
        bundles=bundles,
        default_locale=default_locale,
        canonical_defaults=canonical_defaults,
    )
    totals = audit_summary(report)

    en_entry = next(entry for entry in report.entries if entry.locale == "en")
    assert "ui.extra" in en_entry.missing_keys
    assert "ui.greeting" in en_entry.placeholder_mismatches
    assert "ui.greeting" in en_entry.mixed_script_keys
    assert totals["missing_keys"] >= 1


def test_build_canonical_defaults_merges_registry_runtime_and_default_locale(
    tmp_path: Path,
) -> None:
    locales_root = tmp_path / "locales"
    _write_locale(
        locales_root,
        "ru",
        is_default=True,
        fallback_locale=None,
        messages={"from.locale": "Locale"},
    )
    _write_locale(
        locales_root,
        "en",
        is_default=False,
        fallback_locale="ru",
        messages={"from.locale": "Locale EN"},
    )

    bundles = discover_locale_bundles(locales_root)
    runtime_usage = extract_runtime_key_usage(tmp_path)
    runtime_usage["from.runtime"] = RuntimeKeyUsage(
        key="from.runtime",
        defaults={"Runtime"},
        locations=[],
    )

    registry_path = tmp_path / "ui_text" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": {
                    "ui.sample": {
                        "default": "Registry",
                        "localization_key": "from.registry",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    registry_defaults = load_ui_text_registry_defaults(registry_path)

    canonical = build_canonical_defaults(
        bundles=bundles,
        default_locale="ru",
        runtime_usage=runtime_usage,
        registry_defaults=registry_defaults,
    )

    assert canonical["from.locale"] == "Locale"
    assert canonical["from.runtime"] == "Runtime"
    assert canonical["from.registry"] == "Registry"
