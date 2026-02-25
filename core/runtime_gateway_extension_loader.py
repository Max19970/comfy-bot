from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

from core.runtime import RuntimeStore
from core.runtime_gateways import RuntimeGateways

DEFAULT_RUNTIME_GATEWAY_EXTENSION_PACKAGE = "core.runtime_gateway_plugins.builtin"


class RuntimeGatewayExtensionLoaderError(RuntimeError):
    pass


def parse_runtime_gateway_extension_packages(value: str) -> tuple[str, ...]:
    raw_items = [item.strip() for item in str(value or "").split(",")]
    package_names: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if not item or item in seen:
            continue
        seen.add(item)
        package_names.append(item)
    return tuple(package_names)


def discover_runtime_gateway_extension_modules(package_name: str) -> list[ModuleType]:
    normalized = package_name.strip()
    if not normalized:
        raise RuntimeGatewayExtensionLoaderError(
            "Runtime gateway extension package name must not be empty"
        )

    try:
        package = importlib.import_module(normalized)
    except Exception as exc:  # pragma: no cover - defensive import diagnostics
        raise RuntimeGatewayExtensionLoaderError(
            f"Failed to import runtime gateway extension package '{normalized}'"
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
            raise RuntimeGatewayExtensionLoaderError(
                f"Failed to import runtime gateway extension module '{module_name}'"
            ) from exc
    return modules


def _register_module_extensions(
    module: ModuleType,
    *,
    gateways: RuntimeGateways,
    runtime: RuntimeStore,
) -> None:
    register_extensions = getattr(module, "register_runtime_gateway_extensions", None)
    if register_extensions is None:
        return
    if not callable(register_extensions):
        raise RuntimeGatewayExtensionLoaderError(
            f"Module '{module.__name__}' defines non-callable register_runtime_gateway_extensions"
        )

    try:
        register_extensions(gateways, runtime)
    except ValueError as exc:
        raise RuntimeGatewayExtensionLoaderError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeGatewayExtensionLoaderError(
            f"Failed to register runtime gateway extensions from module '{module.__name__}'"
        ) from exc


def load_runtime_gateway_extensions(
    packages_csv: str,
    *,
    gateways: RuntimeGateways,
    default_package: str = DEFAULT_RUNTIME_GATEWAY_EXTENSION_PACKAGE,
) -> tuple[str, ...]:
    package_names = parse_runtime_gateway_extension_packages(packages_csv)
    if not package_names:
        package_names = parse_runtime_gateway_extension_packages(default_package)

    runtime = gateways.runtime
    for package_name in package_names:
        for module in discover_runtime_gateway_extension_modules(package_name):
            _register_module_extensions(module, gateways=gateways, runtime=runtime)

    return package_names
