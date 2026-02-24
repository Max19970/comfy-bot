from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

HANDLER_CAPABILITY_REGISTRATION = "handlers.registration"


@dataclass(frozen=True, order=True, slots=True)
class PluginApiVersion:
    major: int
    minor: int = 0

    def __post_init__(self) -> None:
        if self.major < 0 or self.minor < 0:
            raise ValueError("Plugin API version must use non-negative numbers")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


SUPPORTED_HANDLER_PLUGIN_API = PluginApiVersion(major=1, minor=0)


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    plugin_id: str
    display_name: str
    api_version: PluginApiVersion = SUPPORTED_HANDLER_PLUGIN_API
    order: int = 100
    capabilities: tuple[str, ...] = (HANDLER_CAPABILITY_REGISTRATION,)
    enabled_by_default: bool = True

    def __post_init__(self) -> None:
        plugin_id = self.plugin_id.strip()
        display_name = self.display_name.strip()
        if not plugin_id:
            raise ValueError("Plugin descriptor requires non-empty plugin_id")
        if not display_name:
            raise ValueError("Plugin descriptor requires non-empty display_name")

        normalized_capabilities = tuple(
            capability.strip() for capability in self.capabilities if capability.strip()
        )

        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "capabilities", normalized_capabilities)


@dataclass(slots=True)
class HandlerPluginContext:
    router: Any
    deps: Any
    shared: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class HandlerPlugin(Protocol):
    descriptor: PluginDescriptor

    def register(self, context: HandlerPluginContext) -> object | None: ...


def plugin_descriptor(plugin: HandlerPlugin) -> PluginDescriptor:
    descriptor = getattr(plugin, "descriptor", None)
    if not isinstance(descriptor, PluginDescriptor):
        raise TypeError("Handler plugin must expose `descriptor: PluginDescriptor`")
    return descriptor


@dataclass(slots=True)
class PluginRegistry:
    _plugins: list[HandlerPlugin] = field(default_factory=list)

    def register(self, plugin: HandlerPlugin) -> HandlerPlugin:
        descriptor = plugin_descriptor(plugin)
        if any(existing.descriptor.plugin_id == descriptor.plugin_id for existing in self._plugins):
            raise ValueError(f"Duplicate plugin id: {descriptor.plugin_id}")
        self._plugins.append(plugin)
        return plugin

    def ordered(self) -> tuple[HandlerPlugin, ...]:
        return tuple(
            sorted(
                self._plugins,
                key=lambda plugin: (plugin.descriptor.order, plugin.descriptor.plugin_id),
            )
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(plugin.descriptor.plugin_id for plugin in self.ordered())
