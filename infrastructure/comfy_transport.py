from __future__ import annotations

from typing import Any, Protocol

import aiohttp


class ComfyTransportProtocol(Protocol):
    async def get_session(self) -> aiohttp.ClientSession: ...

    async def close(self) -> None: ...

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any: ...

    async def post_json(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        data: Any = None,
        timeout: float = 30,
    ) -> Any: ...

    async def post_status(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        timeout: float = 10,
    ) -> int: ...

    async def get_bytes(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> bytes: ...

    async def ws_connect(
        self,
        url: str,
        *,
        heartbeat: float = 30,
        timeout: float = 20,
    ) -> aiohttp.ClientWebSocketResponse: ...


class ComfyHttpTransport:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _url(self, path: str) -> str:
        if path.startswith("/"):
            return f"{self._base_url}{path}"
        return f"{self._base_url}/{path}"

    async def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
    ) -> Any:
        session = await self.get_session()
        async with session.get(
            self._url(path),
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def post_json(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        data: Any = None,
        timeout: float = 30,
    ) -> Any:
        session = await self.get_session()
        async with session.post(
            self._url(path),
            json=json_payload,
            data=data,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def post_status(
        self,
        path: str,
        *,
        json_payload: Any | None = None,
        timeout: float = 10,
    ) -> int:
        session = await self.get_session()
        async with session.post(
            self._url(path),
            json=json_payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            return resp.status

    async def get_bytes(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> bytes:
        session = await self.get_session()
        async with session.get(
            self._url(path),
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def ws_connect(
        self,
        url: str,
        *,
        heartbeat: float = 30,
        timeout: float = 20,
    ) -> aiohttp.ClientWebSocketResponse:
        session = await self.get_session()
        return await session.ws_connect(
            url,
            heartbeat=heartbeat,
            timeout=timeout,
        )
