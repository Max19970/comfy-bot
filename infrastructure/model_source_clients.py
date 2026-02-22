from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

SessionProvider = Callable[[], Awaitable[aiohttp.ClientSession]]
DownloadProgressCallback = Callable[[int, int], Awaitable[None]]


class CivitaiApiClient:
    def __init__(self, session_provider: SessionProvider) -> None:
        self._session_provider = session_provider

    async def search_models(
        self,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float = 20,
    ) -> dict[str, Any]:
        session = await self._session_provider()
        async with session.get(
            "https://civitai.com/api/v1/models",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload if isinstance(payload, dict) else {}

    async def fetch_model(
        self,
        model_id: str,
        *,
        headers: dict[str, str],
        timeout: float = 20,
    ) -> dict[str, Any]:
        session = await self._session_provider()
        async with session.get(
            f"https://civitai.com/api/v1/models/{model_id}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload if isinstance(payload, dict) else {}


class HuggingFaceApiClient:
    def __init__(self, session_provider: SessionProvider) -> None:
        self._session_provider = session_provider

    async def search_models(
        self,
        *,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float = 20,
    ) -> list[dict[str, Any]]:
        session = await self._session_provider()
        async with session.get(
            "https://huggingface.co/api/models",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_model(
        self,
        repo_id: str,
        *,
        headers: dict[str, str],
        timeout: float = 20,
    ) -> dict[str, Any]:
        session = await self._session_provider()
        async with session.get(
            f"https://huggingface.co/api/models/{repo_id}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return payload if isinstance(payload, dict) else {}

    async def list_files(
        self,
        repo_id: str,
        *,
        headers: dict[str, str],
        timeout: float = 15,
    ) -> list[dict[str, Any]]:
        session = await self._session_provider()
        async with session.get(
            f"https://huggingface.co/api/models/{repo_id}/tree/main",
            headers=headers,
            params={"recursive": "true", "expand": "true"},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                return []
            payload = await resp.json()
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]


@dataclass(slots=True)
class FileDownloadResult:
    downloaded_bytes: int
    total_bytes: int


class RemoteFileDownloader:
    def __init__(self, session_provider: SessionProvider) -> None:
        self._session_provider = session_provider

    async def download_to_file(
        self,
        *,
        url: str,
        tmp_path: str,
        headers: dict[str, str],
        expected_size: int = 0,
        timeout_total: float = 3600,
        timeout_sock_read: float = 120,
        chunk_size: int = 1024 * 256,
        progress_cb: DownloadProgressCallback | None = None,
    ) -> FileDownloadResult:
        session = await self._session_provider()
        downloaded = 0
        total = int(expected_size or 0)

        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_total, sock_read=timeout_sock_read),
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", expected_size or 0))

            with open(tmp_path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(chunk_size):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb is not None:
                        await progress_cb(downloaded, total)

        return FileDownloadResult(downloaded_bytes=downloaded, total_bytes=total)
