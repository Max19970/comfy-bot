"""
ComfyUI API client.

Handles communication with a local ComfyUI server:
- Fetches available checkpoints, LoRAs, upscale models, samplers, schedulers
- Builds API-format workflows (prompt payloads)
- Queues prompts, polls for completion, downloads resulting images
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
from PIL import Image, ImageOps

from config import Config
from core.models import GenerationParams
from core.queue_utils import queue_item_prompt_id
from infrastructure.comfy_transport import ComfyHttpTransport, ComfyTransportProtocol
from infrastructure.comfy_workflow_builder import build_comfy_workflow

logger = logging.getLogger(__name__)


_resampling = getattr(Image, "Resampling", None)
LANCZOS_RESAMPLE = getattr(_resampling, "LANCZOS", 1)
_NO_DEFAULT = object()

GenerationProgressCallback = Callable[[int, int, str], Awaitable[None]]
GenerationImageCallback = Callable[[bytes], Awaitable[None]]


def _compose_reference_image(
    reference_images: list[bytes],
    *,
    width: int,
    height: int,
) -> bytes:
    loaded: list[Image.Image] = []
    for image_bytes in reference_images:
        with Image.open(BytesIO(image_bytes)) as image:
            loaded.append(image.convert("RGB"))

    if not loaded:
        raise ValueError("reference_images is empty")

    if len(loaded) == 1:
        composed = ImageOps.fit(loaded[0], (width, height), method=LANCZOS_RESAMPLE)
    else:
        cols = math.ceil(math.sqrt(len(loaded)))
        rows = math.ceil(len(loaded) / cols)
        cell_w = max(1, width // cols)
        cell_h = max(1, height // rows)
        composed = Image.new("RGB", (width, height), (255, 255, 255))

        for index, source in enumerate(loaded):
            row = index // cols
            col = index % cols
            left = col * cell_w
            top = row * cell_h
            right = width if col == cols - 1 else left + cell_w
            bottom = height if row == rows - 1 else top + cell_h
            tile_size = (max(1, right - left), max(1, bottom - top))
            tile = ImageOps.fit(source, tile_size, method=LANCZOS_RESAMPLE)
            composed.paste(tile, (left, top))

    buffer = BytesIO()
    composed.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComfyUIInfo:
    """Cached information about available resources on the ComfyUI server."""

    checkpoints: list[str] = field(default_factory=list)
    loras: list[str] = field(default_factory=list)
    embeddings: list[str] = field(default_factory=list)
    upscale_models: list[str] = field(default_factory=list)
    vaes: list[str] = field(default_factory=list)
    controlnets: list[str] = field(default_factory=list)
    samplers: list[str] = field(default_factory=list)
    schedulers: list[str] = field(default_factory=list)
    clip_vision_models: list[str] = field(default_factory=list)
    ipadapter_models: list[str] = field(default_factory=list)
    ipadapter_apply_node: str = ""
    ipadapter_supported: bool = False
    freeu_supported: bool = False
    pag_supported: bool = False
    tiled_diffusion_supported: bool = False


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ComfyUIClient:
    """Async client for the ComfyUI HTTP API."""

    def __init__(
        self,
        config: Config,
        *,
        transport: ComfyTransportProtocol | None = None,
    ) -> None:
        self.base_url = config.comfyui_url
        self._transport: ComfyTransportProtocol = transport or ComfyHttpTransport(self.base_url)
        self.info = ComfyUIInfo()
        self._object_info: dict[str, Any] = {}

    # -- session management --------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        return await self._transport.get_session()

    async def close(self) -> None:
        await self._transport.close()

    # -- fetch server info ---------------------------------------------------

    @staticmethod
    def _extract_combo_options(input_val: list) -> list[str]:
        """
        Extract option list from a ComfyUI input definition.

        Handles both formats:
          Old: [["option1", "option2", ...]]
          New: ["COMBO", {"options": ["option1", "option2", ...]}]
        """
        if not input_val:
            return []
        first = input_val[0]
        # Old format: first element is a list of strings
        if isinstance(first, list):
            return first
        # New format: first element is "COMBO" (or another type string),
        # second element is a dict with "options"
        if isinstance(first, str) and len(input_val) > 1 and isinstance(input_val[1], dict):
            return input_val[1].get("options", [])
        return []

    @classmethod
    def _default_from_input_spec(cls, input_spec: Any) -> Any:
        if not isinstance(input_spec, list):
            return _NO_DEFAULT

        if len(input_spec) > 1 and isinstance(input_spec[1], dict):
            meta = input_spec[1]
            if "default" in meta:
                return meta["default"]

        combo = cls._extract_combo_options(input_spec)
        if combo:
            return combo[0]

        if not input_spec:
            return _NO_DEFAULT

        kind = input_spec[0]
        if kind == "BOOLEAN":
            return False
        if kind == "INT":
            return 0
        if kind == "FLOAT":
            return 0.0
        if kind == "STRING":
            return ""
        return _NO_DEFAULT

    @staticmethod
    def _select_field_name(
        field_specs: dict[str, Any],
        aliases: tuple[str, ...],
    ) -> str | None:
        for alias in aliases:
            if alias in field_specs:
                return alias
        return None

    @staticmethod
    def _select_ipadapter_apply_node(data: dict[str, Any]) -> str:
        for class_name in ("IPAdapterApply", "IPAdapterApplyAdvanced"):
            if class_name in data:
                return class_name
        return ""

    def _required_input_specs(self, class_name: str) -> dict[str, Any]:
        node = self._object_info.get(class_name, {})
        if not isinstance(node, dict):
            return {}
        inputs = node.get("input", {})
        if not isinstance(inputs, dict):
            return {}
        required = inputs.get("required", {})
        return required if isinstance(required, dict) else {}

    def _required_input_defaults(self, class_name: str) -> dict[str, Any]:
        required = self._required_input_specs(class_name)
        defaults: dict[str, Any] = {}
        for field_name, field_spec in required.items():
            default_value = self._default_from_input_spec(field_spec)
            defaults[field_name] = None if default_value is _NO_DEFAULT else default_value
        return defaults

    @staticmethod
    def _extract_embedding_options_from_object_info(data: dict[str, Any]) -> list[str]:
        options: set[str] = set()
        for node in data.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("input")
            if not isinstance(inputs, dict):
                continue
            required = inputs.get("required")
            if not isinstance(required, dict):
                continue

            for field_name, field_spec in required.items():
                key = str(field_name).strip().lower()
                if "embedding" not in key:
                    continue
                if not isinstance(field_spec, list):
                    continue
                combo = ComfyUIClient._extract_combo_options(field_spec)
                for item in combo:
                    name = str(item).strip()
                    if name:
                        options.add(name)
        return sorted(options)

    def supports_ipadapter(self) -> bool:
        if not self.info.ipadapter_supported:
            return False

        apply_node = self.info.ipadapter_apply_node
        if not apply_node:
            return False

        required = self._required_input_specs(apply_node)
        has_model = self._select_field_name(required, ("model",)) is not None
        has_image = self._select_field_name(required, ("image", "images")) is not None
        has_clip = self._select_field_name(required, ("clip_vision", "clipvision")) is not None
        has_ipadapter = (
            self._select_field_name(required, ("ipadapter", "ipadapter_model")) is not None
        )

        return has_model and has_image and has_clip and has_ipadapter

    def resolve_reference_mode(self, has_reference_images: bool) -> str:
        if not has_reference_images:
            return "none"
        return "ipadapter" if self.supports_ipadapter() else "img2img"

    async def refresh_info(self) -> ComfyUIInfo:
        """Query ComfyUI /object_info and populate available resources."""
        try:
            raw_data = await self._transport.get_json("/object_info", timeout=30)
            if not isinstance(raw_data, dict):
                raise ValueError("Invalid /object_info payload type")
            data: dict[str, Any] = raw_data
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            logger.exception("Failed to fetch /object_info")
            raise

        self._object_info = data

        def _get_input(node_name: str, field_name: str) -> list[str]:
            node = data.get(node_name, {})
            raw = node.get("input", {}).get("required", {}).get(field_name, [])
            return self._extract_combo_options(raw)

        self.info.checkpoints = sorted(_get_input("CheckpointLoaderSimple", "ckpt_name"))
        self.info.loras = sorted(_get_input("LoraLoader", "lora_name"))
        self.info.embeddings = self._extract_embedding_options_from_object_info(data)
        self.info.upscale_models = sorted(_get_input("UpscaleModelLoader", "model_name"))
        self.info.vaes = sorted(_get_input("VAELoader", "vae_name"))
        self.info.controlnets = sorted(
            _get_input("ControlNetLoader", "control_net_name")
            or _get_input("ControlNetLoader", "controlnet_name")
        )
        self.info.samplers = sorted(_get_input("KSampler", "sampler_name"))
        self.info.schedulers = sorted(_get_input("KSampler", "scheduler"))
        self.info.clip_vision_models = sorted(_get_input("CLIPVisionLoader", "clip_name"))
        self.info.ipadapter_models = sorted(
            _get_input("IPAdapterModelLoader", "ipadapter_file")
            or _get_input("IPAdapterModelLoader", "model_name")
        )
        self.info.ipadapter_apply_node = self._select_ipadapter_apply_node(data)
        self.info.ipadapter_supported = bool(
            self.info.clip_vision_models
            and self.info.ipadapter_models
            and self.info.ipadapter_apply_node
        )
        self.info.freeu_supported = "FreeU_V2" in data
        self.info.pag_supported = "PerturbedAttentionGuidance" in data
        self.info.tiled_diffusion_supported = "HyperTile" in data

        logger.info(
            "ComfyUI info refreshed: %d checkpoints, %d loras, %d embeddings, %d upscalers, "
            "%d vae, %d controlnet, %d samplers, %d schedulers, "
            "%d clip-vision, %d ipadapter-models, ipadapter=%s, "
            "freeu=%s, pag=%s, tiled_diffusion=%s",
            len(self.info.checkpoints),
            len(self.info.loras),
            len(self.info.embeddings),
            len(self.info.upscale_models),
            len(self.info.vaes),
            len(self.info.controlnets),
            len(self.info.samplers),
            len(self.info.schedulers),
            len(self.info.clip_vision_models),
            len(self.info.ipadapter_models),
            "yes" if self.supports_ipadapter() else "no",
            "yes" if self.info.freeu_supported else "no",
            "yes" if self.info.pag_supported else "no",
            "yes" if self.info.tiled_diffusion_supported else "no",
        )
        return self.info

    # -- workflow builder ----------------------------------------------------

    def build_workflow(
        self,
        params: GenerationParams,
        *,
        reference_image_name: str | None = None,
        reference_mode: str = "none",
    ) -> dict[str, Any]:
        """
        Build a ComfyUI API-format workflow (dict of nodes).

        Graph structure:
          CheckpointLoader -> (optional LoRA) -> CLIP Text Encode x2 ->
          (optional IP-Adapter on model) ->
          (EmptyLatentImage OR LoadImage->VAEEncode) ->
          KSampler -> VAE Decode -> (optional Upscale) -> SaveImage
        """
        return build_comfy_workflow(
            params,
            reference_image_name=reference_image_name,
            reference_mode=reference_mode,
            object_info=self._object_info,
            info=self.info,
            required_input_defaults=self._required_input_defaults,
            select_field_name=self._select_field_name,
        )

    async def upload_input_image(self, image_bytes: bytes) -> str:
        """Upload an image to ComfyUI input folder and return its workflow name."""
        filename = f"comfybot_ref_{uuid.uuid4().hex}.png"
        form = aiohttp.FormData()
        form.add_field(
            "image",
            image_bytes,
            filename=filename,
            content_type="image/png",
        )
        form.add_field("type", "input")
        form.add_field("overwrite", "true")

        raw_data = await self._transport.post_json(
            "/upload/image",
            data=form,
            timeout=60,
        )
        if not isinstance(raw_data, dict):
            raise ValueError("Invalid /upload/image payload type")
        data: dict[str, Any] = raw_data

        name = str(data.get("name") or filename)
        subfolder = str(data.get("subfolder") or "")
        return f"{subfolder}/{name}" if subfolder else name

    # -- queue & poll --------------------------------------------------------

    async def queue_prompt(
        self,
        workflow: dict[str, Any],
        *,
        client_id: str | None = None,
    ) -> str:
        """Send a workflow to the queue. Returns the prompt_id."""
        if not client_id:
            client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}

        data = await self._transport.post_json(
            "/prompt",
            json_payload=payload,
            timeout=30,
        )
        if not isinstance(data, dict):
            raise ValueError("Invalid /prompt payload type")

        prompt_id = data.get("prompt_id", "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        logger.info("Queued prompt %s", prompt_id)
        return prompt_id

    async def cancel_prompt(self, prompt_id: str) -> None:
        """Interrupt a running prompt or remove it from the queue."""
        # 1. Interrupt current execution
        try:
            status = await self._transport.post_status(
                "/interrupt",
                timeout=10,
            )
            if status != 200:
                logger.warning("Failed to interrupt: %s", status)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("Failed to interrupt prompt %s", prompt_id, exc_info=True)

        # 2. Remove from queue (if pending)
        try:
            status = await self._transport.post_status(
                "/queue",
                json_payload={"delete": [prompt_id]},
                timeout=10,
            )
            if status != 200:
                logger.warning("Failed to delete from queue: %s", status)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("Failed to delete prompt %s from queue", prompt_id, exc_info=True)

    @staticmethod
    def _ws_url(base_url: str, client_id: str) -> str:
        parsed = urlsplit(base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path.rstrip("/")
        ws_path = f"{path}/ws" if path else "/ws"
        query = urlencode({"clientId": client_id})
        return urlunsplit((scheme, parsed.netloc, ws_path, query, ""))

    @staticmethod
    def _workflow_node_types(workflow: dict[str, Any]) -> dict[str, str]:
        node_types: dict[str, str] = {}
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue
            class_type = node_data.get("class_type")
            if isinstance(class_type, str) and class_type:
                node_types[str(node_id)] = class_type
        return node_types

    @staticmethod
    def _stage_name(class_type: str) -> str:
        labels = {
            "CheckpointLoaderSimple": "подготовка модели",
            "LoraLoader": "применение LoRA",
            "VAELoader": "загрузка VAE",
            "CLIPTextEncode": "кодирование промпта",
            "ControlNetLoader": "загрузка ControlNet",
            "ControlNetApply": "применение ControlNet",
            "ControlNetApplyAdvanced": "применение ControlNet",
            "CLIPVisionLoader": "загрузка CLIP Vision",
            "IPAdapterModelLoader": "загрузка IP-Adapter",
            "IPAdapterApply": "применение IP-Adapter",
            "IPAdapterApplyAdvanced": "применение IP-Adapter",
            "LoadImage": "загрузка изображения",
            "VAEEncode": "кодирование референса",
            "EmptyLatentImage": "подготовка латента",
            "RepeatLatentBatch": "подготовка батча",
            "KSampler": "сэмплинг",
            "VAEDecode": "декодирование изображения",
            "UpscaleModelLoader": "загрузка апскейлера",
            "ImageUpscaleWithModel": "апскейл",
            "SaveImage": "сохранение результата",
            "FreeU_V2": "применение FreeU",
            "PerturbedAttentionGuidance": "применение PAG",
            "LatentUpscale": "Hi-res: апскейл латента",
            "HyperTile": "настройка HyperTile",
            "VAEDecodeTiled": "тайловое декодирование",
            "VAEEncodeTiled": "тайловое кодирование",
        }
        return labels.get(class_type, class_type or "выполнение узла")

    @staticmethod
    def _status_queue_remaining(data: Any) -> int | None:
        if not isinstance(data, dict):
            return None
        status = data.get("status")
        if not isinstance(status, dict):
            return None
        exec_info = status.get("exec_info")
        if not isinstance(exec_info, dict):
            return None
        value = exec_info.get("queue_remaining")
        if not isinstance(value, (int, float)):
            return None
        return max(0, int(value))

    @staticmethod
    def _image_info_key(img_info: Any) -> str | None:
        if not isinstance(img_info, dict):
            return None
        filename = str(img_info.get("filename") or "").strip()
        if not filename:
            return None
        subfolder = str(img_info.get("subfolder") or "").strip()
        img_type = str(img_info.get("type") or "output").strip() or "output"
        return f"{img_type}:{subfolder}:{filename}"

    @staticmethod
    def _history_image_entries(history_entry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        outputs = history_entry.get("outputs", {})
        if not isinstance(outputs, dict):
            return []

        def node_sort_key(node_id: str) -> tuple[int, str]:
            try:
                return (0, f"{int(node_id):09d}")
            except (TypeError, ValueError):
                return (1, str(node_id))

        entries: list[tuple[str, dict[str, Any]]] = []
        for node_id in sorted(outputs.keys(), key=node_sort_key):
            node_output = outputs.get(node_id)
            if not isinstance(node_output, dict):
                continue
            node_images = node_output.get("images", [])
            if not isinstance(node_images, list):
                continue
            for img_info in node_images:
                if not isinstance(img_info, dict):
                    continue
                key = ComfyUIClient._image_info_key(img_info)
                if key:
                    entries.append((key, img_info))
        return entries

    async def _download_image_from_info(self, img_info: dict[str, Any]) -> bytes | None:
        filename = str(img_info.get("filename") or "")
        subfolder = str(img_info.get("subfolder") or "")
        img_type = str(img_info.get("type") or "output")
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        }
        try:
            return await self._transport.get_bytes(
                "/view",
                params=params,
                timeout=60,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            logger.exception("Failed to download image %s", filename)
            return None

    async def _stream_progress_via_websocket(
        self,
        prompt_id: str,
        *,
        client_id: str,
        workflow: dict[str, Any],
        timeout: float,
        progress_cb: GenerationProgressCallback,
        image_cb: GenerationImageCallback | None = None,
        delivered_image_keys: set[str] | None = None,
    ) -> bool:
        ws_url = self._ws_url(self.base_url, client_id)
        node_types = self._workflow_node_types(workflow)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        last_progress_key: tuple[Any, ...] | None = None
        prompt_started = False

        async def report_progress(
            key: tuple[Any, ...],
            current: int,
            total: int,
            text: str,
        ) -> None:
            nonlocal last_progress_key
            if key == last_progress_key:
                return
            last_progress_key = key
            await progress_cb(current, total, text)

        try:
            ws = await self._transport.ws_connect(
                ws_url,
                heartbeat=30,
                timeout=20,
            )
            async with ws:
                await report_progress(
                    ("ws_connected",),
                    0,
                    0,
                    "🔌 Live-прогресс подключен",
                )

                while loop.time() - started_at < timeout:
                    remaining = timeout - (loop.time() - started_at)
                    if remaining <= 0:
                        break

                    try:
                        message = await ws.receive(timeout=min(3.0, remaining))
                    except asyncio.TimeoutError:
                        continue

                    if message.type == aiohttp.WSMsgType.TEXT:
                        try:
                            payload = json.loads(message.data)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(payload, dict):
                            continue

                        event_type = payload.get("type")
                        data = payload.get("data", {})
                        if not isinstance(data, dict):
                            data = {}

                        msg_prompt_id = data.get("prompt_id")
                        if isinstance(msg_prompt_id, str):
                            if msg_prompt_id != prompt_id:
                                continue
                            prompt_started = True
                        elif event_type in {
                            "execution_start",
                            "execution_cached",
                            "executing",
                            "executed",
                            "progress",
                            "execution_success",
                            "execution_error",
                            "execution_interrupted",
                        }:
                            prompt_started = True
                        elif event_type != "status" and not prompt_started:
                            continue

                        if event_type == "status" and not prompt_started:
                            queue_remaining = self._status_queue_remaining(data)
                            if queue_remaining is not None:
                                if queue_remaining > 0:
                                    await report_progress(
                                        ("queue_remaining", queue_remaining),
                                        0,
                                        0,
                                        f"⏳ В очереди ComfyUI: осталось задач {queue_remaining}",
                                    )
                                else:
                                    await report_progress(
                                        ("queue_remaining", 0),
                                        0,
                                        0,
                                        "⏳ Ожидаю запуск workflow...",
                                    )
                            continue

                        if event_type == "execution_start":
                            await report_progress(
                                ("execution_start",),
                                0,
                                0,
                                "▶️ Workflow запущен...",
                            )
                            continue

                        if event_type == "execution_cached":
                            cached_nodes = data.get("nodes", [])
                            cached_count = (
                                len(cached_nodes) if isinstance(cached_nodes, list) else 0
                            )
                            if cached_count > 0:
                                await report_progress(
                                    ("execution_cached", cached_count),
                                    0,
                                    0,
                                    f"💾 Использую кэш ({cached_count} узл.)...",
                                )
                            continue

                        if event_type == "executing":
                            node = data.get("node")
                            if node is None:
                                await report_progress(
                                    ("executing_done",),
                                    1,
                                    1,
                                    "✅ Генерация завершена. Финализирую...",
                                )
                                return True

                            node_id = str(node)
                            class_type = node_types.get(node_id, node_id)
                            await report_progress(
                                ("executing", node_id),
                                0,
                                0,
                                f"⚙️ {self._stage_name(class_type)}...",
                            )
                            continue

                        if event_type == "progress":
                            value = data.get("value")
                            max_value = data.get("max")
                            if not isinstance(value, (int, float)) or not isinstance(
                                max_value,
                                (int, float),
                            ):
                                continue

                            total = int(max_value)
                            if total <= 0:
                                continue
                            current = max(0, min(int(value), total))

                            node_raw = data.get("node")
                            node_id = str(node_raw) if node_raw is not None else ""
                            class_type = node_types.get(node_id, "KSampler")
                            stage_name = self._stage_name(class_type)

                            await report_progress(
                                ("progress", node_id, current, total),
                                current,
                                total,
                                f"🔄 {stage_name}: шаг {current}/{total}",
                            )
                            continue

                        if event_type == "executed":
                            if not image_cb:
                                continue
                            output = data.get("output")
                            if not isinstance(output, dict):
                                continue
                            node_images = output.get("images", [])
                            if not isinstance(node_images, list):
                                continue

                            for img_info in node_images:
                                key = self._image_info_key(img_info)
                                if not key:
                                    continue
                                if delivered_image_keys is not None and key in delivered_image_keys:
                                    continue
                                if not isinstance(img_info, dict):
                                    continue

                                image_bytes = await self._download_image_from_info(img_info)
                                if not image_bytes:
                                    continue

                                if delivered_image_keys is not None:
                                    delivered_image_keys.add(key)
                                await image_cb(image_bytes)

                            continue

                        if event_type == "execution_error":
                            details = (
                                data.get("exception_message")
                                or data.get("error")
                                or data.get("exception_type")
                                or "Unknown error"
                            )
                            raise RuntimeError(f"ComfyUI execution error: {details}")

                        if event_type == "execution_interrupted":
                            raise RuntimeError("ComfyUI execution interrupted")

                        if event_type == "execution_success":
                            await report_progress(
                                ("execution_success",),
                                1,
                                1,
                                "✅ Генерация завершена. Финализирую...",
                            )
                            return True

                    elif message.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                    ):
                        logger.debug("ComfyUI websocket closed for prompt %s", prompt_id)
                        return False
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        logger.debug("ComfyUI websocket error for prompt %s", prompt_id)
                        return False

        except asyncio.CancelledError:
            raise
        except RuntimeError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
            logger.debug(
                "Realtime websocket progress failed for prompt %s",
                prompt_id,
                exc_info=True,
            )
            return False

        return False

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
    ) -> dict[str, Any]:
        if not progress_cb:
            return await self.wait_for_completion(
                prompt_id,
                timeout=timeout,
                poll_interval=poll_interval,
            )

        history_task = asyncio.create_task(
            self.wait_for_completion(
                prompt_id,
                timeout=timeout,
                poll_interval=poll_interval,
                progress_cb=progress_cb,
            )
        )
        websocket_task = asyncio.create_task(
            self._stream_progress_via_websocket(
                prompt_id,
                client_id=client_id,
                workflow=workflow,
                timeout=timeout,
                progress_cb=progress_cb,
                image_cb=image_cb,
                delivered_image_keys=delivered_image_keys,
            )
        )

        try:
            done, _ = await asyncio.wait(
                {history_task, websocket_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if websocket_task in done:
                ws_error = websocket_task.exception()
                if isinstance(ws_error, RuntimeError):
                    history_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await history_task
                    raise ws_error

            if history_task in done:
                return history_task.result()

            return await history_task
        finally:
            if not websocket_task.done():
                websocket_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await websocket_task

    async def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 1.5,
        progress_cb: GenerationProgressCallback | None = None,
    ) -> dict[str, Any]:
        """
        Poll /history/{prompt_id} until the prompt finishes.
        Returns the history entry for the prompt.
        """
        session = await self._get_session()
        url = f"{self.base_url}/history/{prompt_id}"
        elapsed = 0.0
        last_progress_key: tuple[Any, ...] | None = None

        async def report_progress(
            key: tuple[Any, ...],
            current: int,
            total: int,
            text: str,
        ) -> None:
            nonlocal last_progress_key
            if not progress_cb:
                return
            if key == last_progress_key:
                return
            last_progress_key = key
            await progress_cb(current, total, text)

        while elapsed < timeout:
            try:
                queue_pending: list[Any] = []
                queue_running: list[Any] = []

                if progress_cb:
                    try:
                        queue_status = await self.get_queue_status()
                        raw_pending = queue_status.get("queue_pending", [])
                        raw_running = queue_status.get("queue_running", [])
                        if isinstance(raw_pending, list):
                            queue_pending = raw_pending
                        if isinstance(raw_running, list):
                            queue_running = raw_running
                    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                        logger.debug(
                            "Could not query queue status for prompt %s",
                            prompt_id,
                            exc_info=True,
                        )

                    pending_position = self._queue_pending_position(
                        queue_pending,
                        prompt_id,
                    )
                    if pending_position is not None:
                        pending_total = max(1, len(queue_pending))
                        await report_progress(
                            ("queue", pending_position, pending_total),
                            0,
                            0,
                            f"⏳ В очереди: позиция {pending_position}/{pending_total}",
                        )
                    elif self._queue_contains_prompt(queue_running, prompt_id):
                        await report_progress(
                            ("running",),
                            0,
                            0,
                            "⚙️ Выполняю workflow...",
                        )

                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if prompt_id in data:
                            entry = data[prompt_id]
                            status = entry.get("status", {})
                            messages = status.get("messages", [])

                            if progress_cb:
                                sampler_progress = self._extract_latest_sampler_progress(messages)
                                if sampler_progress:
                                    current_step, total_steps = sampler_progress
                                    await report_progress(
                                        ("sampler", current_step, total_steps),
                                        current_step,
                                        total_steps,
                                        "🔄 Выполняю шаги сэмплера...",
                                    )
                                elif self._queue_contains_prompt(
                                    queue_running,
                                    prompt_id,
                                ):
                                    await report_progress(
                                        ("running",),
                                        0,
                                        0,
                                        "⚙️ Выполняю workflow...",
                                    )

                            if (
                                status.get("completed", False)
                                or status.get("status_str") == "success"
                            ):
                                return entry
                            # Check for errors
                            for msg in messages:
                                if (
                                    isinstance(msg, list)
                                    and len(msg) >= 1
                                    and msg[0] == "execution_error"
                                ):
                                    error_info = msg[1] if len(msg) > 1 else "Unknown error"
                                    raise RuntimeError(f"ComfyUI execution error: {error_info}")
            except aiohttp.ClientError:
                logger.debug("Poll failed, retrying...")
            except RuntimeError:
                raise

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    @staticmethod
    def _queue_item_prompt_id(item: Any) -> str:
        return queue_item_prompt_id(item)

    @classmethod
    def _queue_contains_prompt(cls, queue_items: Any, prompt_id: str) -> bool:
        if not isinstance(queue_items, list):
            return False
        return any(cls._queue_item_prompt_id(item) == prompt_id for item in queue_items)

    @classmethod
    def _queue_pending_position(cls, queue_items: Any, prompt_id: str) -> int | None:
        if not isinstance(queue_items, list):
            return None
        for index, item in enumerate(queue_items, start=1):
            if cls._queue_item_prompt_id(item) == prompt_id:
                return index
        return None

    @staticmethod
    def _extract_latest_sampler_progress(messages: Any) -> tuple[int, int] | None:
        if not isinstance(messages, list):
            return None

        for message in reversed(messages):
            if not isinstance(message, (list, tuple)) or len(message) < 2:
                continue
            if message[0] != "progress":
                continue

            payload = message[1]
            if not isinstance(payload, dict):
                continue

            value = payload.get("value")
            max_value = payload.get("max")
            if not isinstance(value, (int, float)) or not isinstance(
                max_value,
                (int, float),
            ):
                continue

            total_steps = int(max_value)
            if total_steps <= 0:
                continue

            current_step = max(0, min(int(value), total_steps))
            return current_step, total_steps

        return None

    async def get_images_with_keys(
        self,
        history_entry: dict[str, Any],
    ) -> list[tuple[str, bytes]]:
        """Download all output images with stable dedup keys."""
        images: list[tuple[str, bytes]] = []
        for key, img_info in self._history_image_entries(history_entry):
            image_bytes = await self._download_image_from_info(img_info)
            if image_bytes:
                images.append((key, image_bytes))
        return images

    async def get_images(self, history_entry: dict[str, Any]) -> list[bytes]:
        """Download all output images from a completed prompt history entry."""
        images_with_keys = await self.get_images_with_keys(history_entry)
        return [image for _, image in images_with_keys]

    # -- convenience ---------------------------------------------------------

    async def _run_workflow_and_collect(
        self,
        workflow: dict[str, Any],
        *,
        progress_cb: GenerationProgressCallback | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: GenerationImageCallback | None = None,
    ) -> list[bytes]:
        client_id = uuid.uuid4().hex if progress_cb else None
        delivered_image_keys: set[str] = set()
        prompt_id = await self.queue_prompt(workflow, client_id=client_id)
        if prompt_id_cb:
            await prompt_id_cb(prompt_id)

        if progress_cb:
            await progress_cb(0, 0, "Промпт отправлен в очередь ComfyUI...")

        if progress_cb and client_id:
            history = await self.wait_for_completion_realtime(
                prompt_id,
                client_id=client_id,
                workflow=workflow,
                progress_cb=progress_cb,
                image_cb=image_cb,
                delivered_image_keys=delivered_image_keys,
            )
        else:
            history = await self.wait_for_completion(prompt_id)

        if progress_cb:
            await progress_cb(1, 1, "Генерация завершена. Получаю изображения...")

        images_with_keys = await self.get_images_with_keys(history)
        if image_cb:
            for key, image_bytes in images_with_keys:
                if key in delivered_image_keys:
                    continue
                delivered_image_keys.add(key)
                await image_cb(image_bytes)
            return []
        return [image_bytes for _, image_bytes in images_with_keys]

    async def generate_from_image(
        self,
        params: GenerationParams,
        *,
        image_bytes: bytes,
        progress_cb: GenerationProgressCallback | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: GenerationImageCallback | None = None,
    ) -> list[bytes]:
        """Run generation pipeline from an already existing image (img2img)."""
        reference_image_name = await self.upload_input_image(image_bytes)
        workflow = self.build_workflow(
            params,
            reference_image_name=reference_image_name,
            reference_mode="img2img",
        )
        return await self._run_workflow_and_collect(
            workflow,
            progress_cb=progress_cb,
            prompt_id_cb=prompt_id_cb,
            image_cb=image_cb,
        )

    async def upscale_image_only(
        self,
        *,
        image_bytes: bytes,
        upscale_model: str,
        progress_cb: GenerationProgressCallback | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: GenerationImageCallback | None = None,
    ) -> list[bytes]:
        """Run pure upscaler workflow for an existing image."""
        if not upscale_model:
            raise ValueError("upscale_model is required")

        reference_image_name = await self.upload_input_image(image_bytes)
        workflow: dict[str, Any] = {
            "1": {"class_type": "LoadImage", "inputs": {"image": reference_image_name}},
            "2": {
                "class_type": "UpscaleModelLoader",
                "inputs": {"model_name": upscale_model},
            },
            "3": {
                "class_type": "ImageUpscaleWithModel",
                "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {"images": ["3", 0], "filename_prefix": "ComfyBot"},
            },
        }

        return await self._run_workflow_and_collect(
            workflow,
            progress_cb=progress_cb,
            prompt_id_cb=prompt_id_cb,
            image_cb=image_cb,
        )

    async def generate(
        self,
        params: GenerationParams,
        *,
        reference_images: list[bytes] | None = None,
        progress_cb: GenerationProgressCallback | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: GenerationImageCallback | None = None,
    ) -> list[bytes]:
        """Full generation pipeline: build workflow -> queue -> wait -> download."""
        request = params.to_generation_request()
        reference_image_name: str | None = None
        reference_mode = "none"
        if reference_images:
            composed = _compose_reference_image(
                reference_images,
                width=request.image.width,
                height=request.image.height,
            )
            reference_image_name = await self.upload_input_image(composed)
            if request.models.controlnet_name:
                # For ControlNet we keep latent source as txt2img by default.
                # If IP-Adapter is available, it can be combined explicitly.
                reference_mode = "ipadapter" if self.supports_ipadapter() else "none"
            else:
                reference_mode = self.resolve_reference_mode(True)

        try:
            workflow = self.build_workflow(
                params,
                reference_image_name=reference_image_name,
                reference_mode=reference_mode,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            if reference_mode == "ipadapter":
                logger.warning(
                    "IP-Adapter workflow build failed (%s). Falling back to img2img.",
                    exc,
                )
                workflow = self.build_workflow(
                    params,
                    reference_image_name=reference_image_name,
                    reference_mode="img2img",
                )
            else:
                raise

        return await self._run_workflow_and_collect(
            workflow,
            progress_cb=progress_cb,
            prompt_id_cb=prompt_id_cb,
            image_cb=image_cb,
        )

    async def check_connection(self) -> bool:
        """Return True if ComfyUI is reachable."""
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.base_url}/system_stats",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def get_queue_status(self) -> dict[str, Any]:
        """Return current queue information."""
        data = await self._transport.get_json("/queue", timeout=5)
        if not isinstance(data, dict):
            raise ValueError("Invalid /queue payload type")
        return data
