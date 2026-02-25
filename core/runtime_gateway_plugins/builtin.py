from __future__ import annotations

from core.runtime import RuntimeStore
from core.runtime_gateways import RuntimeGateways


def register_runtime_gateway_extensions(
    gateways: RuntimeGateways,
    runtime: RuntimeStore,
) -> None:
    _ = gateways, runtime
