from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TARGET_UI_MODULES = [
    PROJECT_ROOT / "core" / "ui_copy.py",
    PROJECT_ROOT / "core" / "ui_kit" / "buttons.py",
    PROJECT_ROOT / "core" / "ui_kit" / "dialogs.py",
    PROJECT_ROOT / "core" / "ui_kit" / "pagination.py",
]

ALPHA_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")


def _iter_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_literal_text(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    for kw in call.keywords:
        if (
            kw.arg == "text"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return None


def test_ui_modules_do_not_use_literal_user_facing_strings_in_buttons() -> None:
    violations: list[str] = []

    guarded_calls = {
        "button",
        "noop_button",
        "InlineKeyboardButton",
    }
    for path in TARGET_UI_MODULES:
        for call in _iter_calls(path):
            call_name = _call_name(call)
            if call_name not in guarded_calls:
                continue
            literal = _extract_literal_text(call)
            if literal is None:
                continue
            if not ALPHA_RE.search(literal):
                continue
            rel = path.relative_to(PROJECT_ROOT)
            violations.append(f"{rel}:{call.lineno}: {literal}")

    assert not violations, "Literal UI strings found in guarded UI modules:\n" + "\n".join(
        violations
    )


def test_main_menu_and_start_text_calls_use_text_service_in_handlers() -> None:
    targets = {
        PROJECT_ROOT / "handlers" / "prompt_editor_handlers_send.py": "start_text",
    }
    violations: list[str] = []

    for path, fn_name in targets.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found_with_text_service = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            if call_name != fn_name:
                continue
            if any(kw.arg == "text_service" for kw in node.keywords):
                found_with_text_service = True
                break

        if not found_with_text_service:
            violations.append(str(path.relative_to(PROJECT_ROOT)))

    assert not violations, "Missing text_service in guarded handler calls: " + ", ".join(violations)
