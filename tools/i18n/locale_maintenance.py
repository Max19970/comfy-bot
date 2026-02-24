from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _maintenance_module() -> object:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return importlib.import_module("infrastructure.locale_maintenance")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    locales_root = (project_root / args.locales_root).resolve()
    registry_path = (project_root / args.ui_text_registry).resolve()

    maintenance = _maintenance_module()

    discover_locale_bundles = getattr(maintenance, "discover_locale_bundles")
    default_locale_from_bundles = getattr(maintenance, "default_locale_from_bundles")
    extract_runtime_key_usage = getattr(maintenance, "extract_runtime_key_usage")
    load_ui_text_registry_defaults = getattr(maintenance, "load_ui_text_registry_defaults")
    build_canonical_defaults = getattr(maintenance, "build_canonical_defaults")
    audit_locales = getattr(maintenance, "audit_locales")
    audit_summary = getattr(maintenance, "audit_summary")
    sync_locales = getattr(maintenance, "sync_locales")

    bundles = discover_locale_bundles(locales_root)
    default_locale = default_locale_from_bundles(bundles)

    runtime_usage = extract_runtime_key_usage(project_root)
    registry_defaults = load_ui_text_registry_defaults(registry_path)
    canonical_defaults = build_canonical_defaults(
        bundles=bundles,
        default_locale=default_locale,
        runtime_usage=runtime_usage,
        registry_defaults=registry_defaults,
    )

    if args.command == "scan-runtime":
        return _run_scan_runtime(runtime_usage)

    if args.command == "audit":
        report = audit_locales(
            bundles=bundles,
            default_locale=default_locale,
            canonical_defaults=canonical_defaults,
        )
        totals = audit_summary(report)
        _print_audit(report, totals, maintenance=maintenance)
        if args.json_output:
            payload = {
                "default_locale": report.default_locale,
                "canonical_key_count": report.canonical_key_count,
                "totals": totals,
                "entries": [
                    {
                        "locale": entry.locale,
                        "missing_keys": entry.missing_keys,
                        "orphan_keys": entry.orphan_keys,
                        "placeholder_mismatches": entry.placeholder_mismatches,
                        "mixed_script_keys": entry.mixed_script_keys,
                        "untranslated_keys": entry.untranslated_keys,
                    }
                    for entry in report.entries
                ],
            }
            Path(args.json_output).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if args.strict and (
            totals.get("missing_keys", 0) > 0
            or totals.get("orphan_keys", 0) > 0
            or totals.get("placeholder_mismatches", 0) > 0
        ):
            return 2
        if args.fail_on_mixed_script and totals.get("mixed_script_keys", 0) > 0:
            return 3
        if args.fail_on_untranslated and totals.get("untranslated_keys", 0) > 0:
            return 2
        return 0

    if args.command == "sync":
        result = sync_locales(
            bundles=bundles,
            default_locale=default_locale,
            canonical_defaults=canonical_defaults,
            missing_strategy=args.missing_strategy,
            write=args.write,
        )
        _print_sync_result(result, write=args.write, maintenance=maintenance)
        return 0

    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locale maintenance: runtime key scan, audit and key sync.",
    )
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument(
        "--locales-root", default="locales", help="Locales root relative to project"
    )
    parser.add_argument(
        "--ui-text-registry",
        default="ui_text/registry.json",
        help="Path to UI text registry relative to project root",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan-runtime", help="List translation keys discovered in runtime code")

    audit = sub.add_parser("audit", help="Audit locale consistency and quality")
    audit.add_argument("--json-output", default="", help="Optional path to write JSON report")
    audit.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Return non-zero when locale parity issues are found "
            "(missing/orphan/placeholder mismatches)."
        ),
    )
    audit.add_argument(
        "--fail-on-mixed-script",
        action="store_true",
        help="Also fail when mixed-script values are detected.",
    )
    audit.add_argument(
        "--fail-on-untranslated",
        action="store_true",
        help="Also fail when untranslated values are detected.",
    )

    sync = sub.add_parser("sync", help="Sync missing canonical keys into all locales")
    sync.add_argument(
        "--missing-strategy",
        choices=("source", "key", "todo"),
        default="source",
        help="Value strategy for missing keys in non-default locales",
    )
    sync.add_argument(
        "--write",
        action="store_true",
        help="Write changes to locale files (default is dry-run)",
    )

    return parser


def _run_scan_runtime(runtime_usage: Mapping[str, object]) -> int:
    keys = sorted(runtime_usage.keys())
    print(f"Runtime keys discovered: {len(keys)}")
    for key in keys:
        print(key)
    return 0


def _print_audit(report: object, totals: dict[str, int], *, maintenance: object) -> None:
    typed: Any = report

    print(f"Default locale: {typed.default_locale}")
    print(f"Canonical keys: {typed.canonical_key_count}")
    print(
        "Totals: "
        f"missing={totals.get('missing_keys', 0)}, "
        f"orphan={totals.get('orphan_keys', 0)}, "
        f"placeholder_mismatch={totals.get('placeholder_mismatches', 0)}, "
        f"mixed_script={totals.get('mixed_script_keys', 0)}, "
        f"untranslated={totals.get('untranslated_keys', 0)}"
    )
    for entry in typed.entries:
        print(
            f"- {entry.locale}: "
            f"missing={len(entry.missing_keys)}, "
            f"orphan={len(entry.orphan_keys)}, "
            f"placeholder_mismatch={len(entry.placeholder_mismatches)}, "
            f"mixed_script={len(entry.mixed_script_keys)}, "
            f"untranslated={len(entry.untranslated_keys)}"
        )


def _print_sync_result(result: object, *, write: bool, maintenance: object) -> None:
    typed: Any = result
    mode = "WRITE" if write else "DRY-RUN"
    print(f"Sync mode: {mode}")
    print(f"Default locale: {typed.default_locale}")
    print(f"Canonical keys: {typed.canonical_key_count}")
    for locale in sorted(typed.added_by_locale.keys()):
        print(f"- {locale}: add {typed.added_by_locale[locale]} keys")


if __name__ == "__main__":
    raise SystemExit(main())
