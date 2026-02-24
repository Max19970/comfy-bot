from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Iterable

from .contracts import SUPPORTED_HANDLER_PLUGIN_API, HandlerPlugin, PluginApiVersion, PluginRegistry


class HandlerPluginLoaderError(RuntimeError):
    pass


def parse_plugin_packages(value: str) -> tuple[str, ...]:
    raw_items = [item.strip() for item in str(value or "").split(",")]
    seen: set[str] = set()
    package_names: list[str] = []
    for item in raw_items:
        if not item or item in seen:
            continue
        seen.add(item)
        package_names.append(item)
    return tuple(package_names)


def discover_plugin_modules(package_name: str) -> list[ModuleType]:
    normalized_name = package_name.strip()
    if not normalized_name:
        raise HandlerPluginLoaderError("Plugin package name must not be empty")

    try:
        package = importlib.import_module(normalized_name)
    except Exception as exc:  # pragma: no cover - defensive import diagnostics
        raise HandlerPluginLoaderError(
            f"Failed to import handler plugin package '{normalized_name}'"
        ) from exc

    modules = [package]
    package_paths = getattr(package, "__path__", None)
    if package_paths is None:
        return modules

    module_names = sorted(
        module_info.name
        for module_info in pkgutil.iter_modules(package_paths, prefix=f"{normalized_name}.")
    )
    for module_name in module_names:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:  # pragma: no cover - defensive import diagnostics
            raise HandlerPluginLoaderError(
                f"Failed to import handler plugin module '{module_name}'"
            ) from exc
    return modules


def ensure_plugin_api_compatible(
    plugin_api: PluginApiVersion,
    *,
    supported_api: PluginApiVersion,
    plugin_id: str,
) -> None:
    if plugin_api.major != supported_api.major:
        raise HandlerPluginLoaderError(
            f"Plugin '{plugin_id}' uses unsupported API {plugin_api}; "
            f"expected major {supported_api.major}"
        )
    if plugin_api.minor > supported_api.minor:
        raise HandlerPluginLoaderError(
            f"Plugin '{plugin_id}' requires API {plugin_api}, "
            f"but runtime supports up to {supported_api}"
        )


def _register_module_plugins(module: ModuleType, registry: PluginRegistry) -> None:
    register_plugins = getattr(module, "register_plugins", None)
    if register_plugins is None:
        return
    if not callable(register_plugins):
        raise HandlerPluginLoaderError(
            f"Module '{module.__name__}' defines non-callable register_plugins"
        )

    try:
        result = register_plugins(registry)
    except ValueError as exc:
        raise HandlerPluginLoaderError(str(exc)) from exc
    except Exception as exc:
        raise HandlerPluginLoaderError(
            f"Failed to register plugins from module '{module.__name__}'"
        ) from exc

    if result is None:
        return

    try:
        for plugin in _iter_registered_plugins(result):
            registry.register(plugin)
    except Exception as exc:
        raise HandlerPluginLoaderError(
            f"Module '{module.__name__}' returned invalid plugin collection"
        ) from exc


def _iter_registered_plugins(result: object) -> Iterable[HandlerPlugin]:
    if isinstance(result, tuple) or isinstance(result, list):
        return result
    if hasattr(result, "__iter__"):
        return result  # type: ignore[return-value]
    raise TypeError("register_plugins result is not iterable")


def load_handler_plugins_from_package(
    package_name: str,
    *,
    supported_api: PluginApiVersion = SUPPORTED_HANDLER_PLUGIN_API,
) -> tuple[HandlerPlugin, ...]:
    registry = PluginRegistry()
    for module in discover_plugin_modules(package_name):
        _register_module_plugins(module, registry)

    plugins = registry.ordered()
    for plugin in plugins:
        descriptor = plugin.descriptor
        ensure_plugin_api_compatible(
            descriptor.api_version,
            supported_api=supported_api,
            plugin_id=descriptor.plugin_id,
        )
    return plugins


def load_handler_plugins_from_packages(
    packages_csv: str,
    *,
    supported_api: PluginApiVersion = SUPPORTED_HANDLER_PLUGIN_API,
) -> tuple[HandlerPlugin, ...]:
    package_names = parse_plugin_packages(packages_csv)
    if not package_names:
        return ()

    registry = PluginRegistry()
    for package_name in package_names:
        for plugin in load_handler_plugins_from_package(package_name, supported_api=supported_api):
            try:
                registry.register(plugin)
            except ValueError as exc:
                raise HandlerPluginLoaderError(str(exc)) from exc

    return registry.ordered()
