from __future__ import annotations

from typing import Any

import aiohttp

from core.config import Config
from infrastructure.comfyui_client import ComfyUIClient


class _NoopTransport:
    async def get_session(self) -> aiohttp.ClientSession:
        raise AssertionError("not used in this test")

    async def close(self) -> None:
        return None

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        raise AssertionError("not used in this test")

    async def post_json(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        data: Any = None,
        timeout: float = 30,
    ) -> Any:
        raise AssertionError("not used in this test")

    async def post_status(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        timeout: float = 10,
    ) -> int:
        raise AssertionError("not used in this test")

    async def get_bytes(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> bytes:
        raise AssertionError("not used in this test")

    async def ws_connect(
        self,
        url: str,
        *,
        heartbeat: float = 30,
        timeout: float = 20,
    ) -> aiohttp.ClientWebSocketResponse:
        raise AssertionError("not used in this test")


class _LocalizationStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str]] = []

    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: dict[str, object] | None = None,
        default: str = "",
    ) -> str:
        self.calls.append((key, locale, default))
        return f"translated:{key}"


def _client(localization: _LocalizationStub | None = None) -> ComfyUIClient:
    cfg = Config(telegram_token="token", comfyui_url="http://127.0.0.1:8188")
    return ComfyUIClient(
        cfg,
        transport=_NoopTransport(),
        localization=localization,
        locale="en",
    )


def test_stage_name_uses_node_metadata_localization() -> None:
    localization = _LocalizationStub()
    client = _client(localization)

    stage_name = client._stage_name("UpscaleModelLoader")

    assert stage_name == "translated:comfyui.progress.stage.upscale_model_loader"
    assert localization.calls[-1] == (
        "comfyui.progress.stage.upscale_model_loader",
        "en",
        "загрузка апскейлера",
    )


def test_stage_name_falls_back_to_class_type_for_unknown_node() -> None:
    client = _client()

    assert client._stage_name("CustomNode") == "CustomNode"


def test_stage_name_uses_node_execution_label_for_empty_class_type() -> None:
    localization = _LocalizationStub()
    client = _client(localization)

    stage_name = client._stage_name("")

    assert stage_name == "translated:comfyui.progress.stage.node_execution"
    assert localization.calls[-1] == (
        "comfyui.progress.stage.node_execution",
        "en",
        "выполнение узла",
    )
