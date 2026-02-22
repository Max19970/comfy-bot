from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from infrastructure.model_source_clients import (
    CivitaiApiClient,
    HuggingFaceApiClient,
    RemoteFileDownloader,
    SessionProvider,
)


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def iter_chunked(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        json_payload: Any = None,
        read_payload: bytes = b"",
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self._json_payload = json_payload
        self._read_payload = read_payload
        self.headers = headers or {}
        if chunks is None:
            chunks = [read_payload] if read_payload else []
        self.content = _FakeContent(chunks)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self) -> Any:
        return self._json_payload

    async def read(self) -> bytes:
        return self._read_payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self._responses:
            raise AssertionError("No fake response prepared")
        return self._responses.pop(0)


def test_civitai_client_search_models_uses_expected_endpoint() -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                json_payload={
                    "items": [{"id": 1}],
                    "metadata": {"nextCursor": "abc"},
                }
            )
        ]
    )

    async def _session_provider() -> _FakeSession:
        return session

    client = CivitaiApiClient(cast(SessionProvider, _session_provider))
    payload = asyncio.run(
        client.search_models(
            params={"query": "anime", "limit": 8},
            headers={"Authorization": "Bearer token"},
            timeout=12,
        )
    )

    assert payload["metadata"]["nextCursor"] == "abc"
    assert session.calls[0]["url"] == "https://civitai.com/api/v1/models"
    assert session.calls[0]["params"] == {"query": "anime", "limit": 8}
    assert session.calls[0]["headers"] == {"Authorization": "Bearer token"}


def test_huggingface_list_files_handles_non_200_without_exception() -> None:
    session = _FakeSession([_FakeResponse(status=404, json_payload={"error": "not found"})])

    async def _session_provider() -> _FakeSession:
        return session

    client = HuggingFaceApiClient(cast(SessionProvider, _session_provider))
    files = asyncio.run(client.list_files("org/repo", headers={}, timeout=10))

    assert files == []
    assert session.calls[0]["url"].endswith("/api/models/org/repo/tree/main")


def test_remote_file_downloader_writes_chunks_and_reports_progress(tmp_path: Path) -> None:
    session = _FakeSession(
        [
            _FakeResponse(
                headers={"Content-Length": "6"},
                chunks=[b"ab", b"cdef"],
            )
        ]
    )

    async def _session_provider() -> _FakeSession:
        return session

    downloader = RemoteFileDownloader(cast(SessionProvider, _session_provider))
    target = tmp_path / "model.tmp"
    progress_events: list[tuple[int, int]] = []

    async def _progress(downloaded: int, total: int) -> None:
        progress_events.append((downloaded, total))

    result = asyncio.run(
        downloader.download_to_file(
            url="https://example.com/model",
            tmp_path=str(target),
            headers={"Authorization": "Bearer token"},
            expected_size=0,
            progress_cb=_progress,
        )
    )

    assert target.read_bytes() == b"abcdef"
    assert result.downloaded_bytes == 6
    assert result.total_bytes == 6
    assert progress_events == [(2, 6), (6, 6)]
