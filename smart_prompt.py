from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import random
from dataclasses import dataclass
from typing import Any, cast

from application.smart_prompt_text import (
    dedupe_tags,
    detect_quality_preset,
    extract_anchor_tags,
    quality_tags_for_checkpoint,
    sanitize_final_prompt,
    split_tags,
)
from config import Config

logger = logging.getLogger(__name__)

_MAX_DESCRIPTION_LEN = 4000

_LENGTH_VALUES = {"very_short", "short", "long", "very_long"}

_FORMAT_KEYS: dict[str, str] = {
    "tag_only": "tag only (DTG mode)",
    "nl_only": "NL only (Tag to NL)",
    "both_tag_first": "Both, tag first (recommend)",
    "both_nl_first": "Both, NL first (recommend)",
    "both_generated": "Both + generated NL",
}


@dataclass
class SmartPromptResult:
    positive: str
    negative: str


class SmartPromptError(RuntimeError):
    pass


class SmartPromptService:
    def __init__(self, cfg: Config) -> None:
        self.provider = self._normalize_provider(cfg.smart_prompt_provider)
        raw_model = cfg.smart_prompt_model.strip()
        if raw_model in {"", "gpt-4o-mini", "gemini-2.0-flash", "llama3.2-vision"}:
            self.model = "KBlueLeaf/TIPO-500M-ft"
        else:
            self.model = raw_model
        self.timeout_seconds = max(10, int(cfg.smart_prompt_timeout))
        self.temperature = max(0.0, min(2.0, float(cfg.smart_prompt_temperature)))
        self.top_p = max(0.0, min(1.0, float(cfg.smart_prompt_top_p)))
        self.min_p = max(0.0, min(1.0, float(cfg.smart_prompt_min_p)))
        self.top_k = max(1, int(cfg.smart_prompt_top_k))
        self.device = self._normalize_device(cfg.smart_prompt_device)
        self.seed = int(cfg.smart_prompt_seed)
        self.tag_length = self._normalize_length(cfg.smart_prompt_tag_length)
        self.nl_length = self._normalize_length(cfg.smart_prompt_nl_length)
        self.output_format = self._normalize_format(cfg.smart_prompt_format)
        self.ban_tags = cfg.smart_prompt_ban_tags.strip()
        self.negative_base = cfg.smart_prompt_negative_base.strip()

        if not self.negative_base:
            self.negative_base = (
                "worst quality, low quality, bad quality, blurry, jpeg artifacts, "
                "bad anatomy, deformed, ugly, extra limbs, missing limbs, disfigured, "
                "mutilated, out of frame, duplicate, watermark, signature, text"
            )

        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self._backend_ready = False
        self._backend_error = ""
        self._loaded_model_key = ""
        self._runtime_device = ""
        self._supports_min_p = True

        self._torch: Any = None
        self._kgen_models: Any = None
        self._tipo_executor: Any = None
        self._separate_tags: Any = None
        self._apply_format: Any = None
        self._tipo_default_format: Any = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        value = provider.strip().lower()
        mapping = {
            "": "disabled",
            "none": "disabled",
            "off": "disabled",
            "disabled": "disabled",
            "tipo": "tipo",
            "openai": "tipo",
            "openai-compatible": "tipo",
            "openai_compatible": "tipo",
            "gemini": "tipo",
            "google": "tipo",
            "google-gemini": "tipo",
            "ollama": "tipo",
        }
        return mapping.get(value, value)

    @staticmethod
    def _normalize_device(device: str) -> str:
        value = device.strip().lower()
        if value in {"", "auto"}:
            return "auto"
        if value in {"cpu", "cuda"}:
            return value
        if value.startswith("cuda:"):
            return value
        return "auto"

    @staticmethod
    def _normalize_length(value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if normalized in _LENGTH_VALUES:
            return normalized
        return "long"

    @staticmethod
    def _normalize_format(value: str) -> str:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in _FORMAT_KEYS:
            return normalized
        return "both_tag_first"

    @staticmethod
    def detect_quality_preset(checkpoint: str) -> str:
        """Return a quality preset key based on checkpoint filename."""
        return detect_quality_preset(checkpoint)

    @staticmethod
    def _extract_anchor_tags(description: str) -> list[str]:
        return extract_anchor_tags(description)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        if self.provider != "tipo" or not self.model:
            return False
        if self._missing_dependencies():
            return False
        if self._backend_error:
            return False
        return True

    @staticmethod
    def _missing_dependencies() -> list[str]:
        required = {
            "torch": "torch",
            "transformers": "transformers",
            "kgen": "tipo-kgen",
        }
        missing: list[str] = []
        for module_name, package_name in required.items():
            if importlib.util.find_spec(module_name) is None:
                missing.append(package_name)
        return missing

    def configuration_hint(self) -> str:
        if self.provider == "disabled":
            return "Умный промпт выключен. Укажите SMART_PROMPT_PROVIDER=tipo в .env."
        if self.provider != "tipo":
            return "Неизвестный SMART_PROMPT_PROVIDER. Поддерживаются: tipo, disabled."
        if not self.model:
            return "Не задан SMART_PROMPT_MODEL для TIPO."

        missing = self._missing_dependencies()
        if missing:
            return f"Отсутствуют зависимости для TIPO. Установите: pip install {' '.join(missing)}"

        if self._backend_error:
            return self._backend_error
        return ""

    async def close(self) -> None:
        if self._kgen_models is not None:
            self._kgen_models.text_model = None
            self._kgen_models.tokenizer = None
        self._backend_ready = False

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    async def _ensure_backend_loaded(self) -> None:
        if self._backend_ready:
            return
        async with self._load_lock:
            if self._backend_ready:
                return
            await asyncio.to_thread(self._ensure_backend_loaded_sync)

    def _resolve_runtime_device(self, torch_module: Any) -> str:
        if self.device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch_module.cuda.is_available():
            return "cpu"
        return self.device

    def _load_tipo_model(self, *, device: str) -> tuple[Any, Any]:
        self._kgen_models.load_model(
            model_name=self.model,
            device=device,
        )
        model = self._kgen_models.text_model
        tokenizer = self._kgen_models.tokenizer
        if model is None or tokenizer is None:
            raise SmartPromptError(f"TIPO model {self.model} loaded without tokenizer/model")
        return model, tokenizer

    def _patch_min_p_compat(self) -> None:
        if self._supports_min_p:
            return
        if getattr(self._tipo_executor, "_comfybot_min_p_patch", False):
            return

        original_generate = self._tipo_executor.generate

        def compat_generate(*args: Any, **kwargs: Any):
            kwargs.pop("min_p", None)
            return original_generate(*args, **kwargs)

        self._tipo_executor.generate = compat_generate
        self._tipo_executor._comfybot_min_p_patch = True

    def _ensure_backend_loaded_sync(self) -> None:
        if self._backend_ready:
            return

        try:
            torch = importlib.import_module("torch")
            transformers_module = importlib.import_module("transformers")
            GenerationConfig = transformers_module.GenerationConfig
            kgen_models = importlib.import_module("kgen.models")
            tipo_executor = importlib.import_module("kgen.executor.tipo")
            kgen_formatter = importlib.import_module("kgen.formatter")
            kgen_metainfo = importlib.import_module("kgen.metainfo")
            seperate_tags = kgen_formatter.seperate_tags
            apply_format_fn = kgen_formatter.apply_format
            tipo_default_format = kgen_metainfo.TIPO_DEFAULT_FORMAT
        except (ImportError, AttributeError) as exc:
            missing = self._missing_dependencies()
            deps_hint = f" Установите: pip install {' '.join(missing)}" if missing else ""
            self._backend_error = "Не удалось загрузить зависимости TIPO." + deps_hint
            raise SmartPromptError(self._backend_error) from exc

        self._torch = torch
        self._kgen_models = kgen_models
        self._tipo_executor = tipo_executor
        self._separate_tags = seperate_tags
        self._apply_format = apply_format_fn
        self._tipo_default_format = tipo_default_format
        self._supports_min_p = hasattr(GenerationConfig(), "min_p")
        self._patch_min_p_compat()

        runtime_device = self._resolve_runtime_device(torch)
        if self.device.startswith("cuda") and runtime_device == "cpu":
            self._backend_error = (
                "SMART_PROMPT_DEVICE настроен на CUDA, но CUDA недоступна в PyTorch. "
                "Установите CUDA-сборку torch или переключите SMART_PROMPT_DEVICE=cpu."
            )
            raise SmartPromptError(self._backend_error)

        model_key = f"{self.model}@{runtime_device}"
        if self._loaded_model_key == model_key:
            self._backend_ready = True
            return

        kgen_models_any = cast(Any, self._kgen_models)
        kgen_models_any.text_model = None
        kgen_models_any.tokenizer = None

        try:
            model, tokenizer = self._load_tipo_model(device=runtime_device)
        except (OSError, RuntimeError, ValueError) as exc:
            allow_cpu_fallback = self.device == "auto" and runtime_device != "cpu"
            if allow_cpu_fallback:
                logger.warning(
                    "TIPO load on %s failed, retrying on CPU",
                    runtime_device,
                    exc_info=True,
                )
                runtime_device = "cpu"
                try:
                    model, tokenizer = self._load_tipo_model(device=runtime_device)
                except (OSError, RuntimeError, ValueError) as cpu_exc:
                    self._backend_error = (
                        f"Не удалось загрузить TIPO модель {self.model} (CUDA и CPU): {cpu_exc}"
                    )
                    raise SmartPromptError(self._backend_error) from cpu_exc
            else:
                self._backend_error = f"Не удалось загрузить TIPO модель {self.model}: {exc}"
                raise SmartPromptError(self._backend_error) from exc

        kgen_models_any.text_model = model
        kgen_models_any.tokenizer = tokenizer
        kgen_models_any.current_model_name = self.model.split("/")[-1]

        self._loaded_model_key = f"{self.model}@{runtime_device}"
        self._runtime_device = runtime_device
        self._backend_ready = True
        self._backend_error = ""

        logger.info(
            "TIPO model loaded: %s on %s",
            self.model,
            runtime_device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_prompts(
        self,
        *,
        description: str,
        checkpoint: str,
        reference_images: list[bytes] | None = None,
    ) -> SmartPromptResult:
        del reference_images

        if not self.is_enabled():
            hint = self.configuration_hint()
            raise SmartPromptError(hint or "Сервис умного промпта недоступен.")

        user_description = description.strip()
        if not user_description:
            raise SmartPromptError("Пустое описание запроса.")
        if len(user_description) > _MAX_DESCRIPTION_LEN:
            raise SmartPromptError(
                f"Описание слишком длинное. Сократите до {_MAX_DESCRIPTION_LEN} символов."
            )

        await self._ensure_backend_loaded()

        try:
            async with self._infer_lock:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._generate_prompts_sync,
                        description=user_description,
                        checkpoint=checkpoint.strip(),
                    ),
                    timeout=self.timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            raise SmartPromptError(
                "TIPO не успел обработать запрос по таймауту. "
                "Попробуйте сократить описание или увеличить SMART_PROMPT_TIMEOUT."
            ) from exc

    # ------------------------------------------------------------------
    # Core generation logic
    # ------------------------------------------------------------------

    def _generate_prompts_sync(
        self,
        *,
        description: str,
        checkpoint: str,
    ) -> SmartPromptResult:
        if not self._backend_ready:
            self._ensure_backend_loaded_sync()

        # 1. Determine quality tags based on checkpoint
        quality_tags = quality_tags_for_checkpoint(checkpoint)

        # 2. Extract anchor keywords from the NL description.
        #    This is the critical step: without concrete general tags,
        #    parse_tipo_request would choose 'long_to_tag' (wrong pipeline)
        #    and TIPO would generate random/unrelated anime tags.
        anchor_tags = self._extract_anchor_tags(description)

        # 3. Build seed tags = quality + NL-extracted anchors, then separate
        seed_tags = self._dedupe_tags(quality_tags + anchor_tags)
        tag_map = self._separate_tags(seed_tags)

        # 4. Use parse_tipo_request for multi-step pipeline.
        #    With non-empty general tags + NL prompt it picks the optimal chain:
        #    short_to_tag → tag_to_long (tags first, then NL description)
        nl_prompt = description
        want_nl = self.output_format != "tag_only"

        meta, operations, general, parsed_nl = self._tipo_executor.parse_tipo_request(
            tag_map,
            nl_prompt,
            expand_tags=True,
            expand_prompt=want_nl,
            generate_extra_nl_prompt=False,
            tag_first=True,
            tag_length_target=self.tag_length,
            nl_length_target=self.nl_length,
        )
        meta["aspect_ratio"] = "1.0"

        # 5. Set banned tags
        self._tipo_executor.BAN_TAGS = self._split_tags(self.ban_tags)

        # 6. Run the TIPO pipeline with retry logic
        seed = self.seed if self.seed >= 0 else random.randint(0, 2**31 - 1)
        runner_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": seed,
            "max_retry": 4,
            "max_same_output": 3,
        }
        if self._supports_min_p:
            runner_kwargs["min_p"] = self.min_p

        try:
            result_map, _ = self._tipo_executor.tipo_runner(
                meta,
                operations,
                general,
                parsed_nl,
                **runner_kwargs,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise SmartPromptError(f"Ошибка TIPO: {exc}") from exc

        # 7. Ensure quality tags are present in result_map
        existing_quality = result_map.get("quality", [])
        if isinstance(existing_quality, str):
            existing_quality = self._split_tags(existing_quality)
        merged_quality = self._dedupe_tags(quality_tags + existing_quality)
        result_map["quality"] = merged_quality

        # 8. Build positive prompt via apply_format
        positive = self._build_positive_prompt(result_map)
        if not positive:
            raise SmartPromptError("TIPO вернул пустой промпт.")
        negative = self._build_negative_prompt()
        return SmartPromptResult(positive=positive, negative=negative)

    # ------------------------------------------------------------------
    # Tag utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _split_tags(raw: str) -> list[str]:
        return split_tags(raw)

    @staticmethod
    def _dedupe_tags(tags: list[str]) -> list[str]:
        return dedupe_tags(tags)

    @staticmethod
    def _sanitize_final_prompt(text: str) -> str:
        """Remove TIPO control markers that may leak into the final output."""
        return sanitize_final_prompt(text)

    def _build_positive_prompt(self, result_map: dict[str, Any]) -> str:
        """Build positive prompt using kgen's apply_format with templates."""
        format_key = _FORMAT_KEYS.get(self.output_format, _FORMAT_KEYS["both_tag_first"])
        template = self._tipo_default_format[format_key]

        prompt = self._apply_format(result_map, template)

        # Apply ban-list filtering
        banned = {tag.casefold() for tag in self._split_tags(self.ban_tags)}
        if banned:
            # Split into tag parts and NL sentences, filter tags only
            # Sentences (containing '.') are passed through as-is
            parts = prompt.split(", ")
            filtered: list[str] = []
            for p in parts:
                p_stripped = p.strip()
                if not p_stripped:
                    continue
                # If this looks like a sentence (the NL part), keep it
                if (
                    "." in p_stripped
                    and len(p_stripped) > 40
                    or p_stripped.casefold() not in banned
                ):
                    filtered.append(p_stripped)
            prompt = ", ".join(filtered)

        # Sanitize any control markers that leaked through
        prompt = self._sanitize_final_prompt(prompt)
        return prompt

    def _build_negative_prompt(self) -> str:
        tags = self._split_tags(self.negative_base)
        tags.extend(self._split_tags(self.ban_tags))
        return ", ".join(self._dedupe_tags(tags)).strip().strip(",")
