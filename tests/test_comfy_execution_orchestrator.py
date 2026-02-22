from __future__ import annotations

import asyncio
from typing import Any

from infrastructure.comfy_execution_orchestrator import ComfyExecutionOrchestrator


class _FakeGateway:
    def __init__(self) -> None:
        self.queue_calls: list[tuple[dict[str, Any], str | None]] = []
        self.wait_realtime_calls: list[dict[str, Any]] = []
        self.wait_poll_calls: list[dict[str, Any]] = []
        self.get_images_calls: list[dict[str, Any]] = []

        self.prompt_id = "prompt-1"
        self.history: dict[str, Any] = {"status": "ok"}
        self.images_with_keys: list[tuple[str, bytes]] = []

    async def queue_prompt(
        self,
        workflow: dict[str, Any],
        *,
        client_id: str | None = None,
    ) -> str:
        self.queue_calls.append((workflow, client_id))
        return self.prompt_id

    async def wait_for_completion_realtime(
        self,
        prompt_id: str,
        *,
        client_id: str,
        workflow: dict[str, Any],
        timeout: float = 600,
        poll_interval: float = 1.5,
        progress_cb: Any = None,
        image_cb: Any = None,
        delivered_image_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        self.wait_realtime_calls.append(
            {
                "prompt_id": prompt_id,
                "client_id": client_id,
                "workflow": workflow,
                "timeout": timeout,
                "poll_interval": poll_interval,
            }
        )

        if image_cb is not None and delivered_image_keys is not None:
            delivered_image_keys.add("streamed-key")
            await image_cb(b"streamed")

        return self.history

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 1.5,
        progress_cb: Any = None,
    ) -> dict[str, Any]:
        self.wait_poll_calls.append(
            {
                "prompt_id": prompt_id,
                "timeout": timeout,
                "poll_interval": poll_interval,
            }
        )
        return self.history

    async def get_images_with_keys(
        self,
        history_entry: dict[str, Any],
    ) -> list[tuple[str, bytes]]:
        self.get_images_calls.append(history_entry)
        return list(self.images_with_keys)


def test_orchestrator_uses_realtime_path_and_deduplicates_streamed_images() -> None:
    gateway = _FakeGateway()
    gateway.images_with_keys = [
        ("streamed-key", b"from-history-duplicate"),
        ("history-key", b"from-history"),
    ]
    orchestrator = ComfyExecutionOrchestrator(gateway)

    workflow = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyBot"}}}
    progress_events: list[tuple[int, int, str]] = []
    delivered_images: list[bytes] = []
    prompt_ids: list[str] = []

    async def _progress_cb(current: int, total: int, text: str) -> None:
        progress_events.append((current, total, text))

    async def _image_cb(image_bytes: bytes) -> None:
        delivered_images.append(image_bytes)

    async def _prompt_id_cb(prompt_id: str) -> None:
        prompt_ids.append(prompt_id)

    result = asyncio.run(
        orchestrator.run_workflow_and_collect(
            workflow,
            progress_cb=_progress_cb,
            prompt_id_cb=_prompt_id_cb,
            image_cb=_image_cb,
        )
    )

    assert result == []
    assert prompt_ids == ["prompt-1"]
    assert progress_events[0] == (0, 0, "Промпт отправлен в очередь ComfyUI...")
    assert progress_events[-1] == (1, 1, "Генерация завершена. Получаю изображения...")
    assert delivered_images == [b"streamed", b"from-history"]

    assert len(gateway.queue_calls) == 1
    queued_workflow, queued_client_id = gateway.queue_calls[0]
    assert queued_workflow == workflow
    assert isinstance(queued_client_id, str)
    assert queued_client_id != ""

    assert len(gateway.wait_realtime_calls) == 1
    assert gateway.wait_poll_calls == []
    assert gateway.get_images_calls == [gateway.history]


def test_orchestrator_uses_polling_path_without_progress_callback() -> None:
    gateway = _FakeGateway()
    gateway.images_with_keys = [("k1", b"image-1"), ("k2", b"image-2")]
    orchestrator = ComfyExecutionOrchestrator(gateway)

    workflow = {"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyBot"}}}
    images = asyncio.run(orchestrator.run_workflow_and_collect(workflow))

    assert images == [b"image-1", b"image-2"]
    assert len(gateway.queue_calls) == 1
    assert gateway.queue_calls[0][1] is None
    assert gateway.wait_realtime_calls == []
    assert len(gateway.wait_poll_calls) == 1
    assert gateway.get_images_calls == [gateway.history]
