from __future__ import annotations

from application.model_source_providers import (
    ModelSourceProviderBuildContext,
    ModelSourceProviderRegistry,
    register_default_model_source_providers,
)


def register_providers(
    registry: ModelSourceProviderRegistry,
    context: ModelSourceProviderBuildContext,
) -> None:
    register_default_model_source_providers(registry, context)
