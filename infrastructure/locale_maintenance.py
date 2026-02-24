from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TEXT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")

SKIP_PARTS = {".git", ".venv", "__pycache__", ".mypy_cache"}

LATIN_SCRIPT_LOCALES = {
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "sv",
    "no",
    "da",
    "fi",
    "pl",
    "tr",
    "cs",
    "sk",
    "sl",
    "ro",
    "hu",
}


@dataclass(slots=True)
class RuntimeKeyUsage:
    key: str
    defaults: set[str]
    locations: list[str]


@dataclass(slots=True)
class LocaleBundle:
    locale: str
    is_default: bool
    fallback_locale: str | None
    messages_path: Path
    messages: dict[str, str]


@dataclass(slots=True)
class LocaleAuditEntry:
    locale: str
    missing_keys: list[str]
    orphan_keys: list[str]
    placeholder_mismatches: list[str]
    mixed_script_keys: list[str]
    untranslated_keys: list[str]


@dataclass(slots=True)
class LocaleAuditReport:
    default_locale: str
    canonical_key_count: int
    entries: list[LocaleAuditEntry]


@dataclass(slots=True)
class LocaleSyncResult:
    default_locale: str
    canonical_key_count: int
    added_by_locale: dict[str, int]
    changed_locales: list[str]


def discover_locale_bundles(locales_root: Path) -> dict[str, LocaleBundle]:
    if not locales_root.exists() or not locales_root.is_dir():
        raise RuntimeError(f"Locales root not found: {locales_root}")

    bundles: dict[str, LocaleBundle] = {}
    default_locales: list[str] = []

    for locale_dir in sorted(path for path in locales_root.iterdir() if path.is_dir()):
        metadata_path = locale_dir / "locale.meta.json"
        if not metadata_path.exists():
            continue

        metadata = _read_json_object(metadata_path)
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Locale metadata must be an object: {metadata_path}")
        locale = str(metadata.get("locale") or "").strip().lower()
        if not locale:
            raise RuntimeError(f"Invalid locale in metadata: {metadata_path}")

        messages_file = str(metadata.get("messages_file") or "messages.json").strip()
        messages_path = locale_dir / messages_file
        messages = _read_json_object(messages_path)

        if not isinstance(messages, dict):
            raise RuntimeError(f"Messages file must be an object: {messages_path}")

        normalized_messages: dict[str, str] = {}
        for key, value in messages.items():
            str_key = str(key)
            if not isinstance(value, str):
                raise RuntimeError(
                    f"Message value must be string for key '{str_key}' in {messages_path}"
                )
            normalized_messages[str_key] = value

        is_default = bool(metadata.get("is_default", False))
        if is_default:
            default_locales.append(locale)

        fallback_locale_raw = str(metadata.get("fallback_locale") or "").strip().lower()
        fallback_locale = fallback_locale_raw or None

        bundles[locale] = LocaleBundle(
            locale=locale,
            is_default=is_default,
            fallback_locale=fallback_locale,
            messages_path=messages_path,
            messages=normalized_messages,
        )

    if not bundles:
        raise RuntimeError(f"No locales discovered in: {locales_root}")
    if len(default_locales) != 1:
        raise RuntimeError(
            f"Expected exactly one default locale, discovered {len(default_locales)} in {locales_root}"
        )

    return bundles


def default_locale_from_bundles(bundles: dict[str, LocaleBundle]) -> str:
    defaults = [locale for locale, bundle in bundles.items() if bundle.is_default]
    if len(defaults) != 1:
        raise RuntimeError(f"Expected exactly one default locale, got: {defaults}")
    return defaults[0]


def load_ui_text_registry_defaults(registry_path: Path) -> dict[str, str]:
    if not registry_path.exists():
        return {}
    payload = _read_json_object(registry_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"UI text registry must be an object: {registry_path}")
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"UI text registry entries must be an object: {registry_path}")

    resolved: dict[str, str] = {}
    for raw_text_id, raw_entry in entries.items():
        text_id = str(raw_text_id or "").strip().lower()
        if not text_id:
            continue
        if isinstance(raw_entry, str):
            default_text = raw_entry
            localization_key = text_id
        elif isinstance(raw_entry, dict):
            default_text = str(raw_entry.get("default") or "")
            localization_key = str(raw_entry.get("localization_key") or text_id).strip() or text_id
        else:
            continue
        if not default_text:
            continue
        if _is_text_key(localization_key):
            resolved[localization_key] = default_text
    return resolved


def extract_runtime_key_usage(project_root: Path) -> dict[str, RuntimeKeyUsage]:
    usages: dict[str, RuntimeKeyUsage] = {}
    scan_paths: list[Path] = []

    for name in ("application", "core", "domain", "handlers", "infrastructure"):
        path = project_root / name
        if path.exists() and path.is_dir():
            scan_paths.append(path)

    bot_file = project_root / "bot.py"
    if bot_file.exists() and bot_file.is_file():
        scan_paths.append(bot_file)

    for path in scan_paths:
        python_files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for file_path in python_files:
            if any(part in SKIP_PARTS for part in file_path.parts):
                continue
            source = file_path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                pair = _extract_key_and_default_from_call(node)
                if pair is None:
                    continue
                key, default_value = pair
                if not _is_text_key(key):
                    continue

                usage = usages.get(key)
                if usage is None:
                    usage = RuntimeKeyUsage(key=key, defaults=set(), locations=[])
                    usages[key] = usage

                usage.locations.append(f"{file_path.relative_to(project_root)}:{node.lineno}")
                if default_value:
                    usage.defaults.add(default_value)

    return usages


def build_canonical_defaults(
    *,
    bundles: dict[str, LocaleBundle],
    default_locale: str,
    runtime_usage: dict[str, RuntimeKeyUsage],
    registry_defaults: dict[str, str],
) -> dict[str, str]:
    if default_locale not in bundles:
        raise RuntimeError(f"Default locale '{default_locale}' not found in bundles")

    canonical: dict[str, str] = dict(bundles[default_locale].messages)

    for key, usage in runtime_usage.items():
        if key in canonical:
            continue
        fallback = next(iter(sorted(usage.defaults)), key)
        canonical[key] = fallback

    for key, value in registry_defaults.items():
        canonical.setdefault(key, value)

    return dict(sorted(canonical.items()))


def audit_locales(
    *,
    bundles: dict[str, LocaleBundle],
    default_locale: str,
    canonical_defaults: dict[str, str],
) -> LocaleAuditReport:
    canonical_keys = set(canonical_defaults.keys())
    entries: list[LocaleAuditEntry] = []

    for locale, bundle in sorted(bundles.items()):
        locale_keys = set(bundle.messages.keys())
        missing = sorted(canonical_keys - locale_keys)
        orphan = sorted(locale_keys - canonical_keys)

        placeholder_mismatches: list[str] = []
        mixed_script_keys: list[str] = []
        untranslated_keys: list[str] = []

        for key in sorted(canonical_keys.intersection(locale_keys)):
            value = bundle.messages[key]
            canonical_value = canonical_defaults[key]

            if _placeholder_set(value) != _placeholder_set(canonical_value):
                placeholder_mismatches.append(key)

            if _is_mixed_script(locale, value):
                mixed_script_keys.append(key)

            if locale != default_locale and value == canonical_value and ALPHA_RE.search(value):
                untranslated_keys.append(key)

        entries.append(
            LocaleAuditEntry(
                locale=locale,
                missing_keys=missing,
                orphan_keys=orphan,
                placeholder_mismatches=placeholder_mismatches,
                mixed_script_keys=mixed_script_keys,
                untranslated_keys=untranslated_keys,
            )
        )

    return LocaleAuditReport(
        default_locale=default_locale,
        canonical_key_count=len(canonical_defaults),
        entries=entries,
    )


def sync_locales(
    *,
    bundles: dict[str, LocaleBundle],
    default_locale: str,
    canonical_defaults: dict[str, str],
    missing_strategy: Literal["source", "key", "todo"] = "source",
    write: bool = False,
) -> LocaleSyncResult:
    canonical_keys = set(canonical_defaults.keys())
    added_by_locale: dict[str, int] = {}
    changed_locales: list[str] = []

    for locale, bundle in sorted(bundles.items()):
        missing_keys = sorted(canonical_keys - set(bundle.messages.keys()))
        if not missing_keys:
            added_by_locale[locale] = 0
            continue

        for key in missing_keys:
            bundle.messages[key] = _build_missing_value(
                locale=locale,
                key=key,
                default_locale=default_locale,
                canonical_defaults=canonical_defaults,
                strategy=missing_strategy,
            )

        bundle.messages = dict(sorted(bundle.messages.items()))
        added_by_locale[locale] = len(missing_keys)
        changed_locales.append(locale)

        if write:
            _write_json_object(bundle.messages_path, bundle.messages)

    return LocaleSyncResult(
        default_locale=default_locale,
        canonical_key_count=len(canonical_defaults),
        added_by_locale=added_by_locale,
        changed_locales=changed_locales,
    )


def audit_summary(report: LocaleAuditReport) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for entry in report.entries:
        totals["missing_keys"] += len(entry.missing_keys)
        totals["orphan_keys"] += len(entry.orphan_keys)
        totals["placeholder_mismatches"] += len(entry.placeholder_mismatches)
        totals["mixed_script_keys"] += len(entry.mixed_script_keys)
        totals["untranslated_keys"] += len(entry.untranslated_keys)
    return dict(totals)


def _extract_key_and_default_from_call(node: ast.Call) -> tuple[str, str | None] | None:
    func_name = _call_name(node)
    key: str | None = None
    default: str | None = None

    if func_name in {"t", "text"}:
        key = _str_arg(node.args, 0) or _str_kwarg(node, "key") or _str_kwarg(node, "text_id")
        default = _str_kwarg(node, "default")
    elif func_name in {"_t", "_tr"}:
        key = _str_arg(node.args, 1) or _str_kwarg(node, "key")
        default = _str_kwarg(node, "default") or _str_arg(node.args, 2)
    elif func_name == "_tx":
        key = _str_arg(node.args, 1) or _str_kwarg(node, "key")
        default = _str_kwarg(node, "default") or _str_arg(node.args, 3)
    elif func_name == "ui_text":
        key = _str_arg(node.args, 0) or _str_kwarg(node, "text_id")
        default = _str_arg(node.args, 1) or _str_kwarg(node, "default_text")

    if not key:
        return None
    return key.strip(), default


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _str_arg(args: list[ast.expr], index: int) -> str | None:
    if index >= len(args):
        return None
    arg = args[index]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _str_kwarg(node: ast.Call, name: str) -> str | None:
    for kw in node.keywords:
        if kw.arg != name:
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _placeholder_set(value: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(value))


def _is_text_key(value: str) -> bool:
    return bool(TEXT_KEY_RE.match(str(value or "").strip()))


def _is_mixed_script(locale: str, value: str) -> bool:
    base = locale.split("-", maxsplit=1)[0].lower()
    if base not in LATIN_SCRIPT_LOCALES:
        return False
    return bool(CYRILLIC_RE.search(value))


def _build_missing_value(
    *,
    locale: str,
    key: str,
    default_locale: str,
    canonical_defaults: dict[str, str],
    strategy: Literal["source", "key", "todo"],
) -> str:
    source = canonical_defaults.get(key, key)
    if locale == default_locale:
        return source
    if strategy == "key":
        return key
    if strategy == "todo":
        return f"[[TODO:{locale}]] {key}"
    return source


def _read_json_object(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON file: {path}") from exc


def _write_json_object(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
