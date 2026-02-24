from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import aiohttp

from core.config import Config
from infrastructure.comfyui_client import ComfyUIClient


class _FakeTransport:
    def __init__(self) -> None:
        self.get_json_calls: list[tuple[str, float, dict[str, Any] | None]] = []
        self.post_json_calls: list[tuple[str, float, Any | None, Any]] = []
        self.post_status_calls: list[tuple[str, float, Any | None]] = []
        self.get_bytes_calls: list[tuple[str, float, dict[str, Any] | None]] = []
        self.closed = False

        self.get_json_result: Any = {}
        self.post_json_result: Any = {}
        self.post_status_result = 200
        self.get_bytes_result = b""

    async def get_session(self) -> aiohttp.ClientSession:
        raise AssertionError("get_session is not expected in this test")

    async def close(self) -> None:
        self.closed = True

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        self.get_json_calls.append((path, timeout, params))
        return self.get_json_result

    async def post_json(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        data: Any = None,
        timeout: float = 30,
    ) -> Any:
        self.post_json_calls.append((path, timeout, json_payload, data))
        return self.post_json_result

    async def post_status(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        timeout: float = 10,
    ) -> int:
        self.post_status_calls.append((path, timeout, json_payload))
        return self.post_status_result

    async def get_bytes(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> bytes:
        self.get_bytes_calls.append((path, timeout, params))
        return self.get_bytes_result

    async def ws_connect(
        self,
        url: str,
        *,
        heartbeat: float = 30,
        timeout: float = 20,
    ) -> aiohttp.ClientWebSocketResponse:
        raise AssertionError("ws_connect is not expected in this test")


def _client_with_fake_transport(fake: _FakeTransport) -> ComfyUIClient:
    cfg = Config(telegram_token="token", comfyui_url="http://127.0.0.1:8188")
    return ComfyUIClient(cfg, transport=fake)


def test_queue_prompt_uses_transport_post_json_contract() -> None:
    fake = _FakeTransport()
    fake.post_json_result = {"prompt_id": "prompt-123"}
    client = _client_with_fake_transport(fake)

    workflow = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyBot"}}}
    prompt_id = asyncio.run(client.queue_prompt(workflow, client_id="client-abc"))

    assert prompt_id == "prompt-123"
    assert fake.post_json_calls == [
        (
            "/prompt",
            30,
            {"prompt": workflow, "client_id": "client-abc"},
            None,
        )
    ]


def test_upload_input_image_uses_transport_and_returns_comfy_name() -> None:
    fake = _FakeTransport()
    fake.post_json_result = {"name": "img.png", "subfolder": "input"}
    client = _client_with_fake_transport(fake)

    uploaded_name = asyncio.run(client.upload_input_image(b"fake-image"))

    assert uploaded_name == "input/img.png"
    assert fake.post_json_calls
    path, timeout, json_payload, data = fake.post_json_calls[0]
    assert path == "/upload/image"
    assert timeout == 60
    assert json_payload is None
    assert isinstance(data, aiohttp.FormData)


def test_get_queue_status_uses_transport_get_json_contract() -> None:
    fake = _FakeTransport()
    fake.get_json_result = {"queue_running": [], "queue_pending": []}
    client = _client_with_fake_transport(fake)

    status = asyncio.run(client.get_queue_status())

    assert status == {"queue_running": [], "queue_pending": []}
    assert fake.get_json_calls == [("/queue", 5, None)]


def test_close_delegates_to_transport_close() -> None:
    fake = _FakeTransport()
    client = _client_with_fake_transport(fake)

    asyncio.run(client.close())

    assert fake.closed is True


def test_upscale_image_only_uses_node_based_builder() -> None:
    fake = _FakeTransport()
    fake.post_json_result = {"name": "img.png", "subfolder": "input"}
    client = _client_with_fake_transport(fake)

    captured_workflow: dict[str, Any] = {}

    async def _fake_run_workflow_and_collect(
        workflow: dict[str, Any],
        *,
        progress_cb: Any = None,
        prompt_id_cb: Any = None,
        image_cb: Any = None,
    ) -> list[bytes]:
        captured_workflow.clear()
        captured_workflow.update(workflow)
        return [b"ok"]

    client._run_workflow_and_collect = _fake_run_workflow_and_collect  # type: ignore[method-assign]

    expected_workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "input/img.png"}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x.pth"}},
        "3": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": "ComfyBot"},
        },
    }

    with patch(
        "infrastructure.comfyui_client.build_comfy_upscale_workflow",
        return_value=expected_workflow,
    ) as build_mock:
        result = asyncio.run(
            client.upscale_image_only(
                image_bytes=b"image",
                upscale_model="4x.pth",
            )
        )

    assert result == [b"ok"]
    build_mock.assert_called_once_with(
        reference_image_name="input/img.png",
        upscale_model="4x.pth",
    )
    assert captured_workflow == expected_workflow

