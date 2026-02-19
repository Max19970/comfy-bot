"""Model search/downloader with metadata and compatibility support."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import aiohttp

from config import Config

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

    # CivitAI can return multiple versions, users can pick later.
    available_versions: list[SearchVersionOption] = field(default_factory=list)


def _human_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "unknown"
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


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
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}K"
    return f"{value / 1_000_000:.1f}M"


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
            with open(path, "r", encoding="utf-8") as fh:
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
        except Exception:
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
        except Exception:
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

    def _pick_civitai_file(
        self, files: list[dict[str, Any]], model_type: str
    ) -> dict | None:
        if not files:
            return None

        matching = [
            item
            for item in files
            if self._file_matches_type(item.get("name", ""), model_type)
        ]
        candidates = matching or files

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
        chosen_file = self._pick_civitai_file(
            version.get("files", []) or [], model_type
        )
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
        if civitai_type:
            params["types"] = civitai_type
        if base_models:
            params["baseModels"] = base_models

        headers: dict[str, str] = {}
        if self.cfg.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.cfg.civitai_api_key}"

        try:
            async with session.get(
                "https://civitai.com/api/v1/models",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            logger.exception("CivitAI search failed")
            raise

        results: list[SearchResult] = []
        for item in data.get("items", []):
            result = self._build_civitai_result(item, model_type=model_type)
            if not result:
                continue

            # If user asks specific type, avoid obviously mismatched assets.
            civitai_item_type = _civitai_to_internal_type(str(item.get("type") or ""))
            if civitai_item_type and civitai_item_type != model_type:
                # Keep upscaler queries permissive because there is no dedicated API type
                # and authors can misclassify models.
                if model_type != "upscaler":
                    continue
            results.append(result)
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
        except Exception:
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
        except Exception:
            return []

    def _pick_hf_file(
        self, files: list[dict[str, Any]], model_type: str
    ) -> dict[str, Any] | None:
        if not files:
            return None

        candidates = [
            item
            for item in files
            if str(item.get("type") or "") == "file"
            and self._file_matches_type(str(item.get("path") or ""), model_type)
        ]
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
            file_format="SafeTensor"
            if filename.lower().endswith(".safetensors")
            else "",
            file_fp="",
            nsfw=False,
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

        try:
            async with session.get(
                "https://huggingface.co/api/models",
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                models = await resp.json()
        except Exception:
            logger.exception("HuggingFace search failed")
            raise

        if not isinstance(models, list):
            return []

        results: list[SearchResult] = []
        for model in models[:limit]:
            repo_id = str(model.get("id") or "")
            if not repo_id:
                continue

            files = await self._hf_list_files(repo_id, headers)
            chosen = self._pick_hf_file(files, model_type)
            if not chosen:
                continue

            results.append(
                self._build_hf_result(model, model_type=model_type, chosen_file=chosen)
            )
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
        except Exception:
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
                            if (
                                pct >= last_report_pct + 5
                                or bytes_delta >= 10 * 1024 * 1024
                            ):
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
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        if progress_cb:
            await progress_cb(
                total, total, f"✅ Скачано: {safe_name} ({_human_size(total)})"
            )

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
            return await self.fetch_civitai_model(
                civitai_model_id, model_type=model_type
            )

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
        except Exception as exc:
            logger.warning("Direct URL search failed: %s", exc)

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
                )
            )
        if source in ("huggingface", "all"):
            tasks.append(self.search_huggingface(query, model_type, limit))

        results: list[SearchResult] = []
        for coro in asyncio.as_completed(tasks):
            try:
                batch = await coro
                results.extend(batch)
            except Exception as exc:
                logger.warning("Search source error: %s", exc)

        # Rank primarily by downloads, secondarily by rating.
        results.sort(
            key=lambda item: (
                int(item.download_count or 0),
                float(item.rating or 0.0),
            ),
            reverse=True,
        )
        return results[: max(limit * 2, limit)]
