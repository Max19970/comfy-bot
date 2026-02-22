from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stage-0 guardrails: strict rules for new layered modules + no core->handlers dependency.
FORBIDDEN_IMPORTS_BY_ROOT: dict[str, set[str]] = {
    "core": {"handlers"},
    "domain": {
        "application",
        "core",
        "handlers",
        "infrastructure",
        "presentation",
    },
    "application": {
        "handlers",
        "presentation",
    },
    "infrastructure": {
        "handlers",
        "presentation",
    },
    "presentation": {
        "infrastructure",
    },
}

# Stage-8 tightening: new application modules should not depend on legacy helper packages.
FORBIDDEN_ABSOLUTE_IMPORTS_BY_ROOT: dict[str, set[str]] = {
    "application": {
        "core.download_filters",
    },
}


def _iter_python_files(root_name: str) -> list[Path]:
    root = PROJECT_ROOT / root_name
    if not root.exists() or not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _absolute_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root:
                    roots.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            root = node.module.split(".", maxsplit=1)[0]
            if root:
                roots.add(root)
    return roots


def _absolute_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.strip()
                if name:
                    modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                continue
            modules.add(node.module)
    return modules


def test_layer_import_boundaries() -> None:
    violations: list[str] = []

    for root_name, forbidden_roots in FORBIDDEN_IMPORTS_BY_ROOT.items():
        for file_path in _iter_python_files(root_name):
            imported_roots = _absolute_import_roots(file_path)
            blocked = sorted(imported_roots.intersection(forbidden_roots))
            if not blocked:
                continue
            rel_path = file_path.relative_to(PROJECT_ROOT)
            violations.append(f"{rel_path}: {', '.join(blocked)}")

    assert not violations, "Forbidden cross-layer imports detected:\n" + "\n".join(violations)


def test_layer_absolute_import_boundaries() -> None:
    violations: list[str] = []

    for root_name, forbidden_modules in FORBIDDEN_ABSOLUTE_IMPORTS_BY_ROOT.items():
        for file_path in _iter_python_files(root_name):
            imported_modules = _absolute_import_modules(file_path)
            blocked = sorted(
                module
                for module in imported_modules
                for forbidden in forbidden_modules
                if module == forbidden or module.startswith(f"{forbidden}.")
            )
            if not blocked:
                continue
            rel_path = file_path.relative_to(PROJECT_ROOT)
            violations.append(f"{rel_path}: {', '.join(blocked)}")

    assert not violations, "Forbidden absolute imports detected:\n" + "\n".join(violations)
