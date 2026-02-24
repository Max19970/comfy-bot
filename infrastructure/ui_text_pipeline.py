from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import cast

from domain.ui_text import UITextModifier


class UITextPipelineError(RuntimeError):
    pass


def build_ui_text_modifiers(
    factories_csv: str,
    *,
    dependencies: Mapping[str, object],
) -> tuple[UITextModifier, ...]:
    factory_paths = _parse_factories_csv(factories_csv)
    if not factory_paths:
        raise UITextPipelineError("UI text modifier pipeline is empty")

    modifiers: list[UITextModifier] = []
    for factory_path in factory_paths:
        factory = _import_factory(factory_path)
        modifier = _call_factory(factory, dependencies=dependencies, factory_path=factory_path)
        modifiers.append(modifier)
    return tuple(modifiers)


def _parse_factories_csv(value: str) -> tuple[str, ...]:
    parts = [item.strip() for item in str(value or "").split(",")]
    return tuple(item for item in parts if item)


def _import_factory(path: str) -> Callable[..., object]:
    if ":" not in path:
        raise UITextPipelineError(
            f"Invalid modifier factory path '{path}', expected 'module.submodule:factory_name'"
        )
    module_path, factory_name = path.split(":", maxsplit=1)
    if not module_path or not factory_name:
        raise UITextPipelineError(
            f"Invalid modifier factory path '{path}', expected 'module.submodule:factory_name'"
        )
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - defensive import diagnostics
        raise UITextPipelineError(
            f"Failed to import UI text modifier module '{module_path}'"
        ) from exc

    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise UITextPipelineError(f"UI text modifier factory '{path}' is not callable")
    return cast(Callable[..., object], factory)


def _call_factory(
    factory: Callable[..., object],
    *,
    dependencies: Mapping[str, object],
    factory_path: str,
) -> UITextModifier:
    signature = inspect.signature(factory)
    kwargs: dict[str, object] = {}
    accepts_var_kw = False

    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise UITextPipelineError(
                f"UI text modifier factory '{factory_path}' uses unsupported signature"
            )

        if param.name in dependencies:
            kwargs[param.name] = dependencies[param.name]
            continue

        if param.default is inspect.Signature.empty:
            raise UITextPipelineError(
                f"UI text modifier factory '{factory_path}' requires missing dependency '{param.name}'"
            )

    if accepts_var_kw:
        kwargs = {**dependencies, **kwargs}

    instance = factory(**kwargs)
    if not hasattr(instance, "modify") or not callable(getattr(instance, "modify")):
        raise UITextPipelineError(
            f"UI text modifier factory '{factory_path}' did not return a modifier with method 'modify'"
        )
    return instance
