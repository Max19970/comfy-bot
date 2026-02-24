from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from application.model_source_providers import (
    ModelSourceProviderBuildContext,
    ModelSourceProviderRegistry,
)

DEFAULT_MODEL_SOURCE_PROVIDER_PACKAGE = "application.model_source_plugins.builtin"


class ModelSourceProviderLoaderError(RuntimeError):
    pass


def parse_model_source_provider_packages(value: str) -> tuple[str, ...]:
    raw_items = [item.strip() for item in str(value or "").split(",")]
    package_names: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item or item in seen:
            continue
        seen.add(item)
        package_names.append(item)
    return tuple(package_names)


def discover_provider_modules(package_name: str) -> list[ModuleType]:
    normalized = package_name.strip()
    if not normalized:
        raise ModelSourceProviderLoaderError("Model source provider package must not be empty")

    try:
        package = importlib.import_module(normalized)
    except Exception as exc:  # pragma: no cover - defensive import diagnostics
        raise ModelSourceProviderLoaderError(
            f"Failed to import model source provider package '{normalized}'"
        ) from exc

    modules = [package]
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return modules

    module_names = sorted(
        module_info.name
        for module_info in pkgutil.iter_modules(package_paths, prefix=f"{normalized}.")
    )
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:  # pragma: no cover - defensive import diagnostics
            raise ModelSourceProviderLoaderError(
                f"Failed to import model source provider module '{module_name}'"
            ) from exc
    return modules


def _register_from_module(
    module: ModuleType,
    registry: ModelSourceProviderRegistry,
    context: ModelSourceProviderBuildContext,
) -> None:
    register_providers = getattr(module, "register_providers", None)
    if register_providers is None:
        return
    if not callable(register_providers):
        raise ModelSourceProviderLoaderError(
            f"Module '{module.__name__}' defines non-callable register_providers"
        )

    try:
        register_providers(registry, context)
    except ValueError as exc:
        raise ModelSourceProviderLoaderError(str(exc)) from exc
    except Exception as exc:
        raise ModelSourceProviderLoaderError(
            f"Failed to register model source providers from module '{module.__name__}'"
        ) from exc


def _register_from_packages(
    package_names: tuple[str, ...],
    *,
    registry: ModelSourceProviderRegistry,
    context: ModelSourceProviderBuildContext,
) -> None:
    for package_name in package_names:
        for module in discover_provider_modules(package_name):
            _register_from_module(module, registry, context)


def load_model_source_provider_registry(
    packages_csv: str,
    *,
    context: ModelSourceProviderBuildContext,
    default_package: str = DEFAULT_MODEL_SOURCE_PROVIDER_PACKAGE,
) -> ModelSourceProviderRegistry:
    registry = ModelSourceProviderRegistry()

    package_names = parse_model_source_provider_packages(packages_csv)
    if package_names:
        _register_from_packages(package_names, registry=registry, context=context)

    if registry.sources():
        return registry

    fallback_packages = parse_model_source_provider_packages(default_package)
    if fallback_packages:
        _register_from_packages(fallback_packages, registry=registry, context=context)

    if not registry.sources():
        raise ModelSourceProviderLoaderError("No model source providers discovered")

    return registry
