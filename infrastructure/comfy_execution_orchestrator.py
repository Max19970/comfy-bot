from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

GenerationProgressCallback = Callable[[int, int, str], Awaitable[None]]
GenerationImageCallback = Callable[[bytes], Awaitable[None]]
PromptIdCallback = Callable[[str], Awaitable[None]]


class ComfyExecutionGateway(Protocol):
    async def queue_prompt(
        self,
        workflow: dict[str, Any],
        *,
        client_id: str | None = None,
    ) -> str: ...

    async def wait_for_completion_realtime(
        self,
        prompt_id: str,
        *,
        client_id: str,
        workflow: dict[str, Any],
        timeout: float = 600,
        poll_interval: float = 1.5,
        progress_cb: GenerationProgressCallback | None = None,
        image_cb: GenerationImageCallback | None = None,
        delivered_image_keys: set[str] | None = None,
    ) -> dict[str, Any]: ...

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 1.5,
        progress_cb: GenerationProgressCallback | None = None,
    ) -> dict[str, Any]: ...

    async def get_images_with_keys(
        self,
        history_entry: dict[str, Any],
    ) -> list[tuple[str, bytes]]: ...


class ComfyExecutionOrchestrator:
    def __init__(self, gateway: ComfyExecutionGateway) -> None:
        self._gateway = gateway

    async def _emit_progress(
        self,
        progress_cb: GenerationProgressCallback | None,
        current: int,
        total: int,
        text: str,
    ) -> None:
        if progress_cb is None:
            return
        await progress_cb(current, total, text)

    async def run_workflow_and_collect(
        self,
        workflow: dict[str, Any],
        *,
        progress_cb: GenerationProgressCallback | None = None,
        prompt_id_cb: PromptIdCallback | None = None,
        image_cb: GenerationImageCallback | None = None,
    ) -> list[bytes]:
        client_id = uuid.uuid4().hex if progress_cb else None
        delivered_image_keys: set[str] = set()

        prompt_id = await self._gateway.queue_prompt(workflow, client_id=client_id)
        if prompt_id_cb:
            await prompt_id_cb(prompt_id)

        await self._emit_progress(progress_cb, 0, 0, "Промпт отправлен в очередь ComfyUI...")

        if progress_cb and client_id:
            history = await self._gateway.wait_for_completion_realtime(
                prompt_id,
                client_id=client_id,
                workflow=workflow,
                progress_cb=progress_cb,
                image_cb=image_cb,
                delivered_image_keys=delivered_image_keys,
            )
        else:
            history = await self._gateway.wait_for_completion(prompt_id)

        await self._emit_progress(progress_cb, 1, 1, "Генерация завершена. Получаю изображения...")

        images_with_keys = await self._gateway.get_images_with_keys(history)
        if image_cb:
            for key, image_bytes in images_with_keys:
                if key in delivered_image_keys:
                    continue
                delivered_image_keys.add(key)
                await image_cb(image_bytes)
            return []
        return [image_bytes for _, image_bytes in images_with_keys]
