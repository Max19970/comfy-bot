"""Model search/downloader with metadata and compatibility support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from config import Config
from core.formatting import human_size, short_number

logger = logging.getLogger(__name__)


# Internal model type -> CivitAI "types" parameter
CIVITAI_TYPE_MAP = {
    "checkpoint": "Checkpoint",
    "lora": "LORA",
    "upscaler": "",  # No dedicated API type, selected by file extension.
    "embedding": "TextualInversion",
    "controlnet": "Controlnet",
    "vae": "",  # Keep query+extension mode for broader compatibility.
}


# Internal model type -> ComfyUI models subfolder
MODEL_SUBFOLDER = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "upscaler": "upscale_models",
    "embedding": "embeddings",
    "controlnet": "controlnet",
    "vae": "vae",
}


# Internal model type -> accepted file extensions
MODEL_FILE_EXTENSIONS = {
    "checkpoint": (".safetensors", ".ckpt"),
    "lora": (".safetensors", ".pt", ".ckpt"),
    "upscaler": (".pth", ".pt", ".safetensors", ".onnx", ".bin"),
    "embedding": (".pt", ".bin", ".safetensors", ".ckpt"),
    "controlnet": (".safetensors", ".pth", ".pt", ".ckpt", ".bin"),
    "vae": (".safetensors", ".vae", ".pt", ".ckpt"),
}


# ProgressCallback: async func(downloaded_bytes, total_bytes, status_text)
ProgressCallback = Callable[[int, int, str], Awaitable[None]]


@dataclass
class SearchVersionOption:
    """Concrete downloadable version/file option for a model."""

    version_id: str = ""
    version_name: str = ""
    base_model: str = ""
    filename: str = ""
    download_url: str = ""
    size_bytes: int = 0
    trained_words: list[str] = field(default_factory=list)
    fp: str = ""
    file_format: str = ""
    created_at: str = ""
    preview_url: str = ""
    download_count: int = 0


@dataclass
class SearchResult:
    """A single search result from CivitAI or HuggingFace."""

    name: str = ""
    source: str = ""  # "civitai" or "huggingface"
    model_id: str = ""  # CivitAI model ID or HF repo_id
    version_id: str = ""  # CivitAI modelVersionId / HF branch
    version_name: str = ""
    filename: str = ""  # target filename
    download_url: str = ""
    size_bytes: int = 0
    download_count: int = 0
    description: str = ""  # short result line
    model_type: str = ""  # checkpoint / lora / upscaler / embedding / controlnet / vae

    # Rich metadata used in confirmations and compatibility checks.
    model_url: str = ""
    base_model: str = ""
    trained_words: list[str] = field(default_factory=list)
    creator: str = ""
    rating: float = 0.0
    favorite_count: int = 0
    tags: list[str] = field(default_factory=list)
    preview_url: str = ""
    file_format: str = ""
    file_fp: str = ""
    nsfw: bool = False
    search_text: str = ""

    # CivitAI can return multiple versions, users can pick later.
    available_versions: list[SearchVersionOption] = field(default_factory=list)


def _human_size(size_bytes: int) -> str:
    return human_size(size_bytes)


def _query_tokens(query: str) -> list[str]:
    tokens = [token for token in re.split(r"[^\w]+", query.lower()) if token]
    cleaned = [token for token in tokens if len(token) >= 2]
    return cleaned if cleaned else tokens


def _result_search_text(result: SearchResult) -> str:
    parts: list[str] = [
        result.name,
        result.description,
        result.search_text,
        result.filename,
        result.model_id,
        result.version_name,
        result.base_model,
        result.creator,
        " ".join(result.trained_words),
        " ".join(result.tags),
    ]
    return " ".join(part for part in parts if part).lower()


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _matches_author_filters(result: SearchResult, author_filters: list[str]) -> bool:
    if not author_filters:
        return True
    creator = result.creator.strip().lstrip("@")
    if not creator:
        return False
    creator_l = creator.lower()
    creator_n = _normalize_token(creator)
    for raw in author_filters:
        needle = raw.strip().lstrip("@").lower()
        if not needle:
            continue
        if needle in creator_l:
            return True
        if _normalize_token(needle) and _normalize_token(needle) in creator_n:
            return True
    return False


def _matches_base_models(result: SearchResult, base_models: list[str] | None) -> bool:
    if not base_models:
        return True
    base = (result.base_model or "").strip()
    if not base:
        return False
    base_l = base.lower()
    base_n = _normalize_token(base)
    for candidate in base_models:
        token = str(candidate or "").strip().lower()
        if not token:
            continue
        if token in base_l:
            return True
        token_n = _normalize_token(token)
        if token_n and token_n in base_n:
            return True
    return False


def _matches_query_text(result: SearchResult, query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return True
    haystack = _result_search_text(result)
    if q in haystack:
        return True
    tokens = _query_tokens(q)
    if not tokens:
        return True
    return all(token in haystack for token in tokens)


def _result_unique_key(result: SearchResult) -> str:
    return f"{result.source}:{result.model_id}:{result.version_id}:{result.download_url}"


def _merge_results(
    base: list[SearchResult],
    extra: list[SearchResult],
    *,
    limit: int,
) -> list[SearchResult]:
    if not extra:
        return base
    merged = list(base)
    seen = {_result_unique_key(item) for item in base}
    for item in extra:
        key = _result_unique_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max(1, limit):
            break
    return merged


def apply_version_option(
    result: SearchResult,
    option: SearchVersionOption | dict[str, Any],
) -> SearchResult:
    """Apply a chosen version option to a SearchResult in-place."""
    if isinstance(option, dict):
        version = SearchVersionOption(**option)
    else:
        version = option

    result.version_id = version.version_id or result.version_id
    result.version_name = version.version_name or result.version_name
    result.base_model = version.base_model or result.base_model
    result.filename = version.filename or result.filename
    result.download_url = version.download_url or result.download_url
    result.size_bytes = version.size_bytes or result.size_bytes
    result.trained_words = list(version.trained_words or result.trained_words)
    result.file_fp = version.fp or result.file_fp
    result.file_format = version.file_format or result.file_format
    result.preview_url = version.preview_url or result.preview_url
    if version.download_count > 0:
        result.download_count = version.download_count
    return result


def _clean_words(words: Any, *, limit: int = 16) -> list[str]:
    if not isinstance(words, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in words:
        word = str(item).strip()
        if not word:
            continue
        key = word.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(word)
        if len(cleaned) >= limit:
            break
    return cleaned


def _civitai_to_internal_type(model_type: str) -> str:
    value = model_type.strip().lower()
    mapping = {
        "checkpoint": "checkpoint",
        "lora": "lora",
        "textualinversion": "embedding",
        "controlnet": "controlnet",
        "vae": "vae",
    }
    return mapping.get(value, "")


def _short_number(value: int) -> str:
    return short_number(value)


def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ModelDownloader:
    """Async downloader with CivitAI/HuggingFace search and local metadata index."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._session: aiohttp.ClientSession | None = None

        self._metadata_path = os.path.join(
            self.cfg.comfyui_models_path,
            ".comfybot_model_index.json",
        )
        self._metadata_lock = asyncio.Lock()
        self._metadata: dict[str, dict[str, Any]] = self._load_metadata()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ===================================================================
    # Local metadata index
    # ===================================================================

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        path = self._metadata_path
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict) and isinstance(payload.get("models"), dict):
                payload = payload["models"]
            if not isinstance(payload, dict):
                return {}

            normalized: dict[str, dict[str, Any]] = {}
            for key, value in payload.items():
                if isinstance(value, dict):
                    normalized[str(key)] = value
            return normalized
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.exception("Failed to load model metadata index: %s", path)
            return {}

    async def _save_metadata(self) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "models": self._metadata,
        }
        os.makedirs(self.cfg.comfyui_models_path, exist_ok=True)
        tmp_path = self._metadata_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._metadata_path)

    def _metadata_key(self, filename: str) -> str:
        return os.path.basename(str(filename or "")).strip().casefold()

    def get_model_metadata(
        self,
        filename: str,
        *,
        model_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Return stored metadata for a local filename."""
        key = self._metadata_key(filename)
        if not key:
            return None
        raw = self._metadata.get(key)
        if not raw:
            return None
        if model_type and str(raw.get("model_type", "")).strip() != model_type:
            return None
        return dict(raw)

    def get_lora_trained_words(self, lora_name: str) -> list[str]:
        meta = self.get_model_metadata(lora_name, model_type="lora")
        if not meta:
            return []
        return _clean_words(meta.get("trained_words", []), limit=24)

    def infer_base_model(self, text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return ""
        if "pony" in value:
            return "Pony"
        if "flux" in value:
            return "Flux"
        if "illustrious" in value:
            return "Illustrious"
        if "sdxl" in value or "stable diffusion xl" in value:
            return "SDXL"
        if "sd 2.1" in value or "sd2.1" in value or "stable diffusion 2.1" in value:
            return "SD 2.1"
        if "sd 2" in value or "sd2" in value:
            return "SD 2.x"
        if (
            "sd 1.5" in value
            or "sd1.5" in value
            or "stable diffusion 1.5" in value
            or "_15" in value
        ):
            return "SD 1.5"
        if "sd 1" in value or "stable diffusion 1" in value:
            return "SD 1.x"
        return ""

    def _base_family(self, base_model: str) -> str:
        value = str(base_model or "").strip().lower()
        if not value:
            return ""
        if "pony" in value:
            return "pony"
        if "flux" in value:
            return "flux"
        if "illustrious" in value:
            return "illustrious"
        if "sdxl" in value or "stable diffusion xl" in value:
            return "sdxl"
        if "sd 2" in value or "stable diffusion 2" in value:
            return "sd2"
        if "sd 1.5" in value or "sd1.5" in value or "stable diffusion 1.5" in value:
            return "sd15"
        if "sd 1" in value or "stable diffusion 1" in value:
            return "sd1"
        return value

    def base_models_compatible(self, checkpoint_base: str, lora_base: str) -> bool:
        """Loose compatibility check used for warnings/filtering."""
        ck = self._base_family(checkpoint_base)
        lr = self._base_family(lora_base)
        if not ck or not lr:
            return True
        if ck == lr:
            return True

        # Sometimes Illustrious and SDXL are close enough for practical use.
        if {ck, lr} <= {"sdxl", "illustrious"}:
            return True
        return False

    async def _record_download_metadata(
        self,
        result: SearchResult,
        *,
        safe_filename: str,
        target_path: str,
    ) -> None:
        entry = {
            "filename": safe_filename,
            "original_filename": result.filename,
            "path": target_path,
            "name": result.name,
            "source": result.source,
            "model_type": result.model_type,
            "model_id": result.model_id,
            "version_id": result.version_id,
            "version_name": result.version_name,
            "model_url": result.model_url,
            "download_url": result.download_url,
            "size_bytes": int(result.size_bytes or 0),
            "download_count": int(result.download_count or 0),
            "base_model": result.base_model,
            "trained_words": _clean_words(result.trained_words),
            "creator": result.creator,
            "rating": float(result.rating or 0.0),
            "favorite_count": int(result.favorite_count or 0),
            "tags": [str(tag) for tag in (result.tags or [])[:30]],
            "preview_url": result.preview_url,
            "file_format": result.file_format,
            "file_fp": result.file_fp,
            "nsfw": bool(result.nsfw),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        key = self._metadata_key(safe_filename)
        original_key = self._metadata_key(result.filename)

        async with self._metadata_lock:
            self._metadata[key] = entry
            if original_key and original_key != key:
                self._metadata[original_key] = entry
            await self._save_metadata()

    # ===================================================================
    # URL parsing helpers
    # ===================================================================

    def _extract_civitai_model_id(self, query: str) -> str:
        # Supports: https://civitai.com/models/12345/... or civitai.com/models/12345
        match = re.search(r"civitai\.com/models/(\d+)", query, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    def _extract_hf_repo_id(self, query: str) -> str:
        if "huggingface.co/" not in query.lower():
            return ""
        try:
            parsed = urlsplit(query.strip())
            path = parsed.path.strip("/")
            if not path:
                return ""
            parts = path.split("/")
            if len(parts) < 2:
                return ""
            # /{org}/{repo}[/(tree|blob)/...]
            return f"{parts[0]}/{parts[1]}"
        except ValueError:
            return ""

    # ===================================================================
    # CivitAI
    # ===================================================================

    def _file_matches_type(self, filename: str, model_type: str) -> bool:
        name = filename.strip().lower()
        if not name:
            return False
        exts = MODEL_FILE_EXTENSIONS.get(model_type, ())
        if not exts:
            return True
        return any(name.endswith(ext) for ext in exts)

    def _pick_civitai_file(self, files: list[dict[str, Any]], model_type: str) -> dict | None:
        if not files:
            return None

        matching = [
            item for item in files if self._file_matches_type(item.get("name", ""), model_type)
        ]
        if not matching:
            return None
        candidates = matching

        def _score(item: dict[str, Any]) -> tuple[int, int, float]:
            metadata = item.get("metadata") or {}
            fmt = str(metadata.get("format", "")).lower()
            name = str(item.get("name", "")).lower()
            is_safe = int("safetensor" in fmt or name.endswith(".safetensors"))
            primary = int(bool(item.get("primary")))
            size_kb = float(item.get("sizeKB") or 0)
            return primary, is_safe, size_kb

        return sorted(candidates, key=_score, reverse=True)[0]

    def _pick_preview_url(self, images: list[dict[str, Any]]) -> str:
        if not images:
            return ""

        def _is_nsfw(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() not in {"", "0", "false", "none"}

        for image in images:
            if not _is_nsfw(image.get("nsfw")) and image.get("url"):
                return str(image.get("url"))
        first = images[0]
        return str(first.get("url", ""))

    def _build_description(
        self,
        *,
        source: str,
        base_model: str,
        size_bytes: int,
        downloads: int,
        rating: float,
    ) -> str:
        parts = []
        if source == "huggingface":
            parts.append("HF")
        if base_model:
            parts.append(base_model)
        if size_bytes > 0:
            parts.append(_human_size(size_bytes))
        if rating > 0:
            parts.append(f"⭐ {rating:.2f}")
        parts.append(f"📥 {_short_number(max(0, int(downloads)))}")
        return " | ".join(parts)

    def _build_civitai_version_option(
        self,
        version: dict[str, Any],
        *,
        model_type: str,
    ) -> SearchVersionOption | None:
        chosen_file = self._pick_civitai_file(version.get("files", []) or [], model_type)
        if not chosen_file:
            return None

        size_kb = float(chosen_file.get("sizeKB") or 0)
        size_bytes = int(size_kb * 1024)
        version_id = str(version.get("id") or "")
        download_url = str(chosen_file.get("downloadUrl") or "")
        if not download_url and version_id:
            download_url = f"https://civitai.com/api/download/models/{version_id}"

        metadata = chosen_file.get("metadata") or {}
        version_stats = version.get("stats") or {}

        return SearchVersionOption(
            version_id=version_id,
            version_name=str(version.get("name") or "").strip(),
            base_model=str(version.get("baseModel") or "").strip(),
            filename=str(chosen_file.get("name") or "model.safetensors"),
            download_url=download_url,
            size_bytes=size_bytes,
            trained_words=_clean_words(version.get("trainedWords", []), limit=24),
            fp=str(metadata.get("fp") or "").strip(),
            file_format=str(metadata.get("format") or "").strip(),
            created_at=str(version.get("createdAt") or "").strip(),
            preview_url=self._pick_preview_url(version.get("images", []) or []),
            download_count=int(version_stats.get("downloadCount") or 0),
        )

    def _build_civitai_result(
        self,
        item: dict[str, Any],
        *,
        model_type: str,
    ) -> SearchResult | None:
        versions_raw = item.get("modelVersions") or []
        if not versions_raw:
            return None

        options: list[SearchVersionOption] = []
        for version in versions_raw:
            option = self._build_civitai_version_option(version, model_type=model_type)
            if option:
                options.append(option)
        if not options:
            return None

        selected = options[0]
        stats = item.get("stats") or {}
        creator = item.get("creator") or {}
        rating = float(stats.get("rating") or 0.0)
        downloads = int(stats.get("downloadCount") or 0)

        model_id = str(item.get("id") or "")
        model_url = f"https://civitai.com/models/{model_id}" if model_id else ""

        description = self._build_description(
            source="civitai",
            base_model=selected.base_model,
            size_bytes=selected.size_bytes,
            downloads=downloads,
            rating=rating,
        )

        civitai_desc = _strip_html(str(item.get("description") or ""))
        tags = [str(tag) for tag in (item.get("tags") or []) if str(tag).strip()]

        if civitai_desc:
            snippet = civitai_desc[:160]
            if len(civitai_desc) > 160:
                snippet += "..."
            if snippet:
                description = f"{description} | {snippet}"

        search_chunks = [
            str(item.get("name") or ""),
            civitai_desc,
            " ".join(tags),
            str(creator.get("username") or ""),
            selected.version_name,
            selected.filename,
            selected.base_model,
            " ".join(selected.trained_words),
        ]
        search_text = " ".join(chunk for chunk in search_chunks if chunk).strip()

        return SearchResult(
            name=str(item.get("name") or "Unknown"),
            source="civitai",
            model_id=model_id,
            version_id=selected.version_id,
            version_name=selected.version_name,
            filename=selected.filename,
            download_url=selected.download_url,
            size_bytes=selected.size_bytes,
            download_count=downloads,
            description=description,
            model_type=model_type,
            model_url=model_url,
            base_model=selected.base_model,
            trained_words=list(selected.trained_words),
            creator=str(creator.get("username") or "").strip(),
            rating=rating,
            favorite_count=int(stats.get("favoriteCount") or 0),
            tags=tags[:20],
            preview_url=selected.preview_url,
            file_format=selected.file_format,
            file_fp=selected.fp,
            nsfw=bool(item.get("nsfw")),
            search_text=search_text,
            available_versions=options,
        )

    async def search_civitai(
        self,
        query: str,
        model_type: str,
        limit: int = 10,
        *,
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        base_models: list[str] | None = None,
        include_nsfw: bool = False,
        authors: list[str] | None = None,
        strict_type: bool = True,
    ) -> list[SearchResult]:
        """Search CivitAI for models with rich metadata."""
        session = await self._get_session()
        params: dict[str, Any] = {
            "query": query,
            "limit": max(1, min(limit, 100)),
            "sort": sort,
            "period": period,
            "primaryFileOnly": "false",
        }
        if not include_nsfw:
            params["nsfw"] = "false"

        civitai_type = CIVITAI_TYPE_MAP.get(model_type, "")
        if strict_type and civitai_type:
            params["types"] = civitai_type
        if base_models:
            params["baseModels"] = base_models
        author_filters = [
            item.strip().lstrip("@").lower() for item in (authors or []) if item.strip().lstrip("@")
        ]
        if author_filters:
            params["username"] = author_filters[0]

        headers: dict[str, str] = {}
        if self.cfg.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.civitai_api_key}"

        async def _fetch_page(
            request_params: dict[str, Any],
        ) -> tuple[list[dict[str, Any]], str | None]:
            async with session.get(
                "https://civitai.com/api/v1/models",
                params=request_params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            metadata = data.get("metadata") if isinstance(data, dict) else {}
            items = data.get("items", [])
            next_cursor_raw: Any = None
            if isinstance(metadata, dict):
                next_cursor_raw = metadata.get("nextCursor")
            next_cursor = (
                str(next_cursor_raw).strip() if next_cursor_raw not in {None, ""} else None
            )
            return (items if isinstance(items, list) else []), next_cursor

        def _collect_results(
            items_payload: list[dict[str, Any]],
            *,
            results: list[SearchResult],
            seen_ids: set[str],
        ) -> None:
            for item in items_payload:
                result = self._build_civitai_result(item, model_type=model_type)
                if not result:
                    continue

                # If user asks specific type, avoid obviously mismatched assets.
                civitai_item_type = _civitai_to_internal_type(str(item.get("type") or ""))
                if strict_type and civitai_item_type and civitai_item_type != model_type:
                    # Keep upscaler queries permissive because there is no dedicated API type
                    # and authors can misclassify models.
                    if model_type != "upscaler":
                        continue
                if not _matches_author_filters(result, author_filters):
                    continue
                if not _matches_base_models(result, base_models):
                    continue
                if not _matches_query_text(result, query):
                    continue
                unique_id = _result_unique_key(result)
                if unique_id in seen_ids:
                    continue
                seen_ids.add(unique_id)
                results.append(result)

        async def _search_pages(
            base_params: dict[str, Any],
            *,
            max_pages: int,
        ) -> list[SearchResult]:
            results: list[SearchResult] = []
            seen_ids: set[str] = set()
            cursor: str | None = None
            for _ in range(max(1, max_pages)):
                request_params = dict(base_params)
                if cursor:
                    request_params["cursor"] = cursor
                try:
                    items, next_cursor = await _fetch_page(request_params)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                    logger.exception("CivitAI search page failed")
                    break
                _collect_results(items, results=results, seen_ids=seen_ids)
                if len(results) >= max(1, limit):
                    return results
                if not next_cursor:
                    break
                cursor = next_cursor
            return results

        try:
            results = await _search_pages(dict(params), max_pages=8)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.exception("CivitAI search failed")
            raise

        if len(results) >= max(1, limit):
            return results

        # Fallback: some CivitAI searches miss obvious matches with strict query,
        # especially when combined with author/base filters. Retry without `query`
        # and apply robust local text matching.
        if query.strip():
            relaxed_params = dict(params)
            relaxed_params.pop("query", None)
            relaxed_params["limit"] = max(20, min(limit * 8, 100))
            relaxed_results = await _search_pages(relaxed_params, max_pages=10)
            results = _merge_results(results, relaxed_results, limit=limit)
            if len(results) >= max(1, limit):
                return results

            # Last-resort: remove API-side author/base filters completely and
            # rely on local robust matching.
            broad_params = dict(relaxed_params)
            broad_params.pop("baseModels", None)
            broad_params.pop("username", None)
            broad_params["limit"] = max(30, min(limit * 10, 100))
            broad_results = await _search_pages(broad_params, max_pages=12)
            results = _merge_results(results, broad_results, limit=limit)

        return results

    async def fetch_civitai_model(
        self,
        model_id: str,
        *,
        model_type: str,
    ) -> list[SearchResult]:
        """Fetch one CivitAI model by model ID."""
        session = await self._get_session()
        headers: dict[str, str] = {}
        if self.cfg.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.civitai_api_key}"

        url = f"https://civitai.com/api/v1/models/{model_id}"
        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                item = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.exception("CivitAI model fetch failed: %s", model_id)
            raise

        result = self._build_civitai_result(item, model_type=model_type)
        if not result:
            return []
        return [result]

    # ===================================================================
    # HuggingFace
    # ===================================================================

    async def _hf_list_files(self, repo_id: str, headers: dict[str, str]) -> list[dict]:
        """List files in a HuggingFace repo (recursive tree)."""
        session = await self._get_session()
        url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
        try:
            async with session.get(
                url,
                headers=headers,
                params={"recursive": "true", "expand": "true"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return []

    def _pick_hf_file(self, files: list[dict[str, Any]], model_type: str) -> dict[str, Any] | None:
        if not files:
            return None

        candidates = [
            item
            for item in files
            if str(item.get("type") or "") == "file"
            and self._file_matches_type(str(item.get("path") or ""), model_type)
        ]
        candidates = [item for item in candidates if self._hf_file_is_compatible(item, model_type)]
        if not candidates:
            return None

        hints = {
            "checkpoint": ("checkpoint", "model", "fp16", "fp32"),
            "lora": ("lora", "adapter", "lycoris", "locon", "loha"),
            "upscaler": ("upscale", "esrgan", "realesrgan", "4x", "x4", "x2"),
            "embedding": ("embedding", "textual", "ti"),
            "controlnet": ("controlnet", "control", "cn"),
            "vae": ("vae",),
        }.get(model_type, ())

        def _size_of(item: dict[str, Any]) -> int:
            raw = int(item.get("size") or 0)
            lfs = item.get("lfs")
            if isinstance(lfs, dict):
                return int(lfs.get("size") or raw)
            return raw

        def _score(item: dict[str, Any]) -> tuple[int, int, int]:
            path = str(item.get("path") or "").lower()
            name = os.path.basename(path)
            hint_score = int(any(token in name for token in hints))
            safetensor_score = int(name.endswith(".safetensors"))
            size = _size_of(item)

            if model_type in {"lora", "embedding"}:
                # Smaller artifacts are often better signals for LoRA/embeddings.
                target = 200 * 1024 * 1024 if model_type == "lora" else 20 * 1024 * 1024
                size_score = -abs(size - target)
            else:
                size_score = size
            return hint_score, safetensor_score, size_score

        return sorted(candidates, key=_score, reverse=True)[0]

    def _hf_file_is_compatible(self, item: dict[str, Any], model_type: str) -> bool:
        path = str(item.get("path") or "").strip().lower()
        if not path:
            return False
        name = os.path.basename(path)

        if name.endswith((".json", ".txt", ".yaml", ".yml")):
            return False
        if any(token in path for token in ("/tokenizer", "/scheduler", "/feature_extractor")):
            return False

        size = int(item.get("size") or 0)
        lfs = item.get("lfs")
        if isinstance(lfs, dict):
            size = int(lfs.get("size") or size)

        if model_type == "checkpoint":
            # Avoid diffusers internals and sharded artifacts that are not single-file checkpoints.
            blocked_tokens = (
                "diffusion_pytorch_model",
                "pytorch_model",
                "unet",
                "text_encoder",
                "vae/",
                "/vae",
                "prior",
                "clip",
            )
            if any(token in path for token in blocked_tokens):
                return False
            if "-0000" in name and "-of-" in name:
                return False
            # Checkpoints are usually large.
            if size > 0 and size < 700 * 1024 * 1024:
                return False

        if model_type == "lora":
            if size > 0 and size > 2 * 1024 * 1024 * 1024:
                return False

        if model_type == "embedding":
            if size > 0 and size > 256 * 1024 * 1024:
                return False

        return True

    def list_local_models(self, model_type: str) -> list[str]:
        subfolder = MODEL_SUBFOLDER.get(model_type)
        if not subfolder:
            return []
        folder = os.path.join(self.cfg.comfyui_models_path, subfolder)
        if not os.path.isdir(folder):
            return []

        entries: list[str] = []
        for item in os.listdir(folder):
            path = os.path.join(folder, item)
            if not os.path.isfile(path):
                continue
            if not self._file_matches_type(item, model_type):
                continue
            entries.append(item)
        return sorted(entries, key=str.casefold)

    async def delete_local_model(self, model_type: str, filename: str) -> str:
        safe_name = os.path.basename(str(filename or "")).strip()
        if not safe_name:
            raise ValueError("Empty filename")

        subfolder = MODEL_SUBFOLDER.get(model_type)
        if not subfolder:
            raise ValueError(f"Unsupported model type: {model_type}")

        if not self._file_matches_type(safe_name, model_type):
            raise ValueError(f"File extension does not match type: {safe_name}")

        target_dir = os.path.join(self.cfg.comfyui_models_path, subfolder)
        target_path = os.path.join(target_dir, safe_name)
        real_target = os.path.realpath(target_path)
        real_dir = os.path.realpath(target_dir)
        try:
            common = os.path.commonpath([real_target, real_dir])
        except ValueError:
            common = ""
        if common != real_dir:
            raise ValueError("Unsafe filename")

        if not os.path.exists(target_path):
            raise FileNotFoundError(target_path)
        os.remove(target_path)

        key = self._metadata_key(safe_name)
        async with self._metadata_lock:
            self._metadata.pop(key, None)
            await self._save_metadata()

        return target_path

    def _build_hf_result(
        self,
        model: dict[str, Any],
        *,
        model_type: str,
        chosen_file: dict[str, Any],
    ) -> SearchResult:
        repo_id = str(model.get("id") or "")
        card = model.get("cardData") or {}
        tags = model.get("tags")
        tags_list = [str(tag) for tag in tags] if isinstance(tags, list) else []

        size = int(chosen_file.get("size") or 0)
        lfs = chosen_file.get("lfs")
        if isinstance(lfs, dict):
            size = int(lfs.get("size") or size)

        filename = str(chosen_file.get("path") or "")
        base_model = str(
            card.get("base_model") or card.get("baseModel") or card.get("base") or ""
        ).strip()
        rating = float(model.get("likes") or 0.0)
        downloads = int(model.get("downloads") or 0)

        description = self._build_description(
            source="huggingface",
            base_model=base_model,
            size_bytes=size,
            downloads=downloads,
            rating=rating,
        )

        card_text_parts = [
            str(card.get("model_description") or ""),
            str(card.get("description") or ""),
            str(card.get("widget") or ""),
            str(card.get("language") or ""),
        ]
        search_text = " ".join(
            part
            for part in [repo_id, filename, base_model, " ".join(tags_list), *card_text_parts]
            if part
        ).strip()

        return SearchResult(
            name=f"{repo_id} / {filename}",
            source="huggingface",
            model_id=repo_id,
            version_id="main",
            version_name="main",
            filename=filename,
            download_url=f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            size_bytes=size,
            download_count=downloads,
            description=description,
            model_type=model_type,
            model_url=f"https://huggingface.co/{repo_id}",
            base_model=base_model,
            trained_words=[],
            creator=repo_id.split("/", 1)[0] if "/" in repo_id else repo_id,
            rating=rating,
            favorite_count=int(model.get("likes") or 0),
            tags=tags_list[:20],
            preview_url="",
            file_format="SafeTensor" if filename.lower().endswith(".safetensors") else "",
            file_fp="",
            nsfw=False,
            search_text=search_text,
            available_versions=[],
        )

    async def search_huggingface(
        self,
        query: str,
        model_type: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search HuggingFace for model artifacts."""
        session = await self._get_session()
        params: dict[str, Any] = {
            "search": query,
            "sort": "downloads",
            "direction": "-1",
            "limit": max(1, min(limit, 100)),
        }
        if model_type == "checkpoint":
            params["pipeline_tag"] = "text-to-image"

        headers: dict[str, str] = {}
        if self.cfg.huggingface_token:
            headers["Authorization"] = f"Bearer {self.cfg.huggingface_token}"

        async def _fetch_models(fetch_params: dict[str, Any]) -> list[dict[str, Any]]:
            async with session.get(
                "https://huggingface.co/api/models",
                params=fetch_params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
            return payload if isinstance(payload, list) else []

        try:
            models = await _fetch_models(dict(params))
            if model_type == "checkpoint" and not models:
                relaxed = dict(params)
                relaxed.pop("pipeline_tag", None)
                models = await _fetch_models(relaxed)
            if not models and " " in query.strip():
                tokens = [token for token in re.split(r"\s+", query.strip()) if len(token) >= 3]
                if tokens:
                    token_params = dict(params)
                    token_params["search"] = max(tokens, key=len)
                    token_params.pop("pipeline_tag", None)
                    models = await _fetch_models(token_params)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.exception("HuggingFace search failed")
            raise

        results: list[SearchResult] = []
        for model in models[:limit]:
            repo_id = str(model.get("id") or "")
            if not repo_id:
                continue

            files = await self._hf_list_files(repo_id, headers)
            chosen = self._pick_hf_file(files, model_type)
            if not chosen:
                continue

            result = self._build_hf_result(model, model_type=model_type, chosen_file=chosen)
            if not _matches_query_text(result, query):
                continue
            results.append(result)
        return results

    async def fetch_huggingface_repo(
        self,
        repo_id: str,
        *,
        model_type: str,
    ) -> list[SearchResult]:
        """Fetch one HuggingFace repo and pick the best matching file."""
        session = await self._get_session()
        headers: dict[str, str] = {}
        if self.cfg.huggingface_token:
            headers["Authorization"] = f"Bearer {self.cfg.huggingface_token}"

        url = f"https://huggingface.co/api/models/{repo_id}"
        try:
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                model = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.exception("HuggingFace repo fetch failed: %s", repo_id)
            raise

        files = await self._hf_list_files(repo_id, headers)
        chosen = self._pick_hf_file(files, model_type)
        if not chosen:
            return []
        return [self._build_hf_result(model, model_type=model_type, chosen_file=chosen)]

    # ===================================================================
    # Download
    # ===================================================================

    async def download_model(
        self,
        result: SearchResult,
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> str:
        """
        Download model file to the corresponding ComfyUI folder.
        Returns the full local path.
        """
        subfolder = MODEL_SUBFOLDER.get(result.model_type, "checkpoints")
        target_dir = os.path.join(self.cfg.comfyui_models_path, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        safe_name = re.sub(r'[<>:"/\\|?*]', "_", result.filename).strip()
        safe_name = os.path.basename(safe_name).strip(" .")
        if not safe_name:
            safe_name = "model.safetensors"
        if safe_name.upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }:
            safe_name = f"_{safe_name}"
        target_path = os.path.join(target_dir, safe_name)

        # Prevent path traversal
        real_target = os.path.realpath(target_path)
        real_dir = os.path.realpath(target_dir)
        try:
            common = os.path.commonpath([real_target, real_dir])
        except ValueError:
            common = ""
        if common != real_dir:
            raise ValueError(f"Unsafe filename: {result.filename}")

        if os.path.exists(target_path):
            raise FileExistsError(f"File already exists: {target_path}")

        session = await self._get_session()
        headers: dict[str, str] = {}
        if result.source == "civitai" and self.cfg.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.civitai_api_key}"
        elif result.source == "huggingface" and self.cfg.huggingface_token:
            headers["Authorization"] = f"Bearer {self.cfg.huggingface_token}"

        if progress_cb:
            await progress_cb(0, result.size_bytes, "📥 Начинаю скачивание...")

        tmp_path = target_path + ".tmp"
        total = int(result.size_bytes or 0)
        downloaded = 0
        last_report_pct = -1
        last_report_bytes = 0

        try:
            async with session.get(
                result.download_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=3600, sock_read=120),
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", result.size_bytes or 0))

                with open(tmp_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(1024 * 256):
                        fh.write(chunk)
                        downloaded += len(chunk)

                        if progress_cb and total > 0:
                            pct = (downloaded * 100) // total
                            bytes_delta = downloaded - last_report_bytes
                            if pct >= last_report_pct + 5 or bytes_delta >= 10 * 1024 * 1024:
                                last_report_pct = pct
                                last_report_bytes = downloaded
                                await progress_cb(
                                    downloaded,
                                    total,
                                    f"📥 {pct}% ({_human_size(downloaded)} / {_human_size(total)})",
                                )

            os.replace(tmp_path, target_path)
            await self._record_download_metadata(
                result,
                safe_filename=safe_name,
                target_path=target_path,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        if progress_cb:
            await progress_cb(total, total, f"✅ Скачано: {safe_name} ({_human_size(total)})")

        logger.info("Downloaded %s to %s", result.filename, target_path)
        return target_path

    # ===================================================================
    # Combined search
    # ===================================================================

    async def _search_direct_url(
        self,
        query: str,
        *,
        model_type: str,
    ) -> list[SearchResult]:
        civitai_model_id = self._extract_civitai_model_id(query)
        if civitai_model_id:
            return await self.fetch_civitai_model(civitai_model_id, model_type=model_type)

        hf_repo_id = self._extract_hf_repo_id(query)
        if hf_repo_id:
            return await self.fetch_huggingface_repo(hf_repo_id, model_type=model_type)

        return []

    async def search(
        self,
        query: str,
        model_type: str,
        source: str = "all",
        limit: int = 8,
        *,
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        base_models: list[str] | None = None,
        include_nsfw: bool = False,
        civitai_author: str = "",
        civitai_authors: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Search across sources.

        source: "civitai", "huggingface", or "all"
        """
        query = query.strip()
        if not query:
            return []

        # URL mode: direct model link from CivitAI/HuggingFace.
        try:
            direct_results = await self._search_direct_url(query, model_type=model_type)
            if direct_results:
                return direct_results
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RuntimeError) as exc:
            logger.warning("Direct URL search failed: %s", exc)

        author_filters = [
            item.strip().lstrip("@").lower()
            for item in (civitai_authors or [])
            if item.strip().lstrip("@")
        ]
        if not author_filters and civitai_author.strip():
            author_filters = [civitai_author.strip().lstrip("@").lower()]

        tasks = []
        if source in ("civitai", "all"):
            tasks.append(
                self.search_civitai(
                    query,
                    model_type,
                    limit,
                    sort=sort,
                    period=period,
                    base_models=base_models,
                    include_nsfw=include_nsfw,
                    authors=author_filters,
                )
            )
        if source in ("huggingface", "all"):
            tasks.append(self.search_huggingface(query, model_type, limit))

        results: list[SearchResult] = []
        for coro in asyncio.as_completed(tasks):
            try:
                batch = await coro
                results.extend(batch)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RuntimeError) as exc:
                logger.warning("Search source error: %s", exc)

        if not results:
            relaxed_tasks = []
            if source in ("civitai", "all"):
                relaxed_tasks.append(
                    self.search_civitai(
                        query,
                        model_type,
                        limit,
                        sort="Most Downloaded",
                        period="AllTime",
                        base_models=None,
                        include_nsfw=include_nsfw,
                        authors=[],
                        strict_type=False,
                    )
                )
            if source in ("huggingface", "all"):
                relaxed_tasks.append(self.search_huggingface(query, model_type, limit))

            for coro in asyncio.as_completed(relaxed_tasks):
                try:
                    batch = await coro
                    results.extend(batch)
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, RuntimeError) as exc:
                    logger.warning("Relaxed search source error: %s", exc)

        # Rank primarily by downloads, secondarily by rating.
        results.sort(
            key=lambda item: (
                int(item.download_count or 0),
                float(item.rating or 0.0),
            ),
            reverse=True,
        )
        return results[: max(limit * 2, limit)]
