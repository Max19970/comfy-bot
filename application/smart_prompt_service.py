from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Mapping

from application.smart_prompt_text import (
    dedupe_tags,
    detect_quality_preset,
    extract_anchor_tags,
    quality_tags_for_checkpoint,
    sanitize_final_prompt,
    split_tags,
)
from core.config import Config
from domain.localization import LocalizationService
from infrastructure.tipo_backend import TipoBackend, TipoBackendError, TipoBackendProtocol

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
    def __init__(
        self,
        cfg: Config,
        *,
        backend: TipoBackendProtocol | None = None,
        localization: LocalizationService | None = None,
        locale: str | None = None,
    ) -> None:
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
        self.localization = localization
        self.locale = locale

        if not self.negative_base:
            self.negative_base = (
                "worst quality, low quality, bad quality, blurry, jpeg artifacts, "
                "bad anatomy, deformed, ugly, extra limbs, missing limbs, disfigured, "
                "mutilated, out of frame, duplicate, watermark, signature, text"
            )

        self._infer_lock = asyncio.Lock()
        self._backend: TipoBackendProtocol = backend or TipoBackend(
            model=self.model,
            device=self.device,
        )

    def _t(
        self,
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        if self.localization is None:
            text = default
        else:
            locale = self.locale or self.localization.default_locale()
            text = self.localization.t(key, locale=locale, params=params, default=default)
        if params:
            try:
                return text.format(**params)
            except Exception:
                return text
        return text

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
        if self._backend.backend_error:
            return False
        return True

    def _missing_dependencies(self) -> list[str]:
        return self._backend.missing_dependencies()

    def configuration_hint(self) -> str:
        if self.provider == "disabled":
            return self._t(
                "smart_prompt.error.provider_disabled",
                "Умный промпт выключен. Укажите SMART_PROMPT_PROVIDER=tipo в .env.",
            )
        if self.provider != "tipo":
            return self._t(
                "smart_prompt.error.provider_invalid",
                "Неизвестный SMART_PROMPT_PROVIDER. Поддерживаются: tipo, disabled.",
            )
        if not self.model:
            return self._t(
                "smart_prompt.error.model_missing",
                "Не задан SMART_PROMPT_MODEL для TIPO.",
            )

        missing = self._missing_dependencies()
        if missing:
            return self._t(
                "smart_prompt.error.dependencies_missing",
                "Отсутствуют зависимости для TIPO. Установите: pip install {packages}",
                params={"packages": " ".join(missing)},
            )

        if self._backend.backend_error:
            return self._backend.backend_error
        return ""

    async def close(self) -> None:
        await self._backend.close()

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    async def _ensure_backend_loaded(self) -> None:
        try:
            await self._backend.ensure_loaded()
        except TipoBackendError as exc:
            raise SmartPromptError(str(exc)) from exc

    def _ensure_backend_loaded_sync(self) -> None:
        try:
            self._backend.ensure_loaded_sync()
        except TipoBackendError as exc:
            raise SmartPromptError(str(exc)) from exc

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
            raise SmartPromptError(
                hint
                or self._t(
                    "smart_prompt.error.service_unavailable",
                    "Сервис умного промпта недоступен.",
                )
            )

        user_description = description.strip()
        if not user_description:
            raise SmartPromptError(
                self._t(
                    "smart_prompt.error.empty_description",
                    "Пустое описание запроса.",
                )
            )
        if len(user_description) > _MAX_DESCRIPTION_LEN:
            raise SmartPromptError(
                self._t(
                    "smart_prompt.error.description_too_long",
                    "Описание слишком длинное. Сократите до {max_chars} символов.",
                    params={"max_chars": _MAX_DESCRIPTION_LEN},
                )
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
                self._t(
                    "smart_prompt.error.timeout",
                    "TIPO не успел обработать запрос по таймауту. Попробуйте сократить описание или увеличить SMART_PROMPT_TIMEOUT.",
                )
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
        if not self._backend.is_ready():
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
        tag_map = self._backend.separate_tags(seed_tags)

        # 4. Use parse_tipo_request for multi-step pipeline.
        #    With non-empty general tags + NL prompt it picks the optimal chain:
        #    short_to_tag → tag_to_long (tags first, then NL description)
        nl_prompt = description
        want_nl = self.output_format != "tag_only"

        meta, operations, general, parsed_nl = self._backend.parse_request(
            tag_map=tag_map,
            nl_prompt=nl_prompt,
            want_nl=want_nl,
            tag_length=self.tag_length,
            nl_length=self.nl_length,
        )

        # 5. Set banned tags
        self._backend.set_ban_tags(self._split_tags(self.ban_tags))

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
        if self._backend.supports_min_p:
            runner_kwargs["min_p"] = self.min_p

        try:
            result_map, _ = self._backend.run(
                meta,
                operations,
                general,
                parsed_nl,
                runner_kwargs=runner_kwargs,
            )
        except TipoBackendError as exc:
            raise SmartPromptError(str(exc)) from exc

        # 7. Ensure quality tags are present in result_map
        existing_quality = result_map.get("quality", [])
        if isinstance(existing_quality, str):
            existing_quality = self._split_tags(existing_quality)
        merged_quality = self._dedupe_tags(quality_tags + existing_quality)
        result_map["quality"] = merged_quality

        # 8. Build positive prompt via apply_format
        positive = self._build_positive_prompt(result_map)
        if not positive:
            raise SmartPromptError(
                self._t(
                    "smart_prompt.error.empty_result",
                    "TIPO вернул пустой промпт.",
                )
            )
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
        try:
            prompt = self._backend.apply_format(result_map, format_key)
        except TipoBackendError as exc:
            raise SmartPromptError(str(exc)) from exc

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
