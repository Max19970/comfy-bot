from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

TranslateText = Callable[[str, str | None, str], str]


class TipoBackendError(RuntimeError):
    pass


class TipoBackendProtocol(Protocol):
    def missing_dependencies(self) -> list[str]: ...

    def is_ready(self) -> bool: ...

    @property
    def backend_error(self) -> str: ...

    @property
    def supports_min_p(self) -> bool: ...

    async def ensure_loaded(self) -> None: ...

    def ensure_loaded_sync(self) -> None: ...

    async def close(self) -> None: ...

    def separate_tags(self, tags: list[str]) -> Any: ...

    def parse_request(
        self,
        *,
        tag_map: Any,
        nl_prompt: str,
        want_nl: bool,
        tag_length: str,
        nl_length: str,
    ) -> tuple[dict[str, Any], Any, Any, Any]: ...

    def set_ban_tags(self, tags: list[str]) -> None: ...

    def run(
        self,
        meta: dict[str, Any],
        operations: Any,
        general: Any,
        parsed_nl: Any,
        *,
        runner_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]: ...

    def apply_format(self, result_map: dict[str, Any], format_key: str) -> str: ...


class TipoBackend:
    def __init__(
        self,
        *,
        model: str,
        device: str,
        translate: TranslateText | None = None,
        locale: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self._translate = translate
        self._locale = locale

        self._load_lock = asyncio.Lock()
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

    def _t(self, key: str, default: str) -> str:
        if self._translate is None:
            return default
        return self._translate(key, self._locale, default)

    def _tf(self, key: str, default: str, **params: object) -> str:
        template = self._t(key, default)
        try:
            return template.format(**params)
        except (KeyError, ValueError, TypeError):
            return default.format(**params)

    def missing_dependencies(self) -> list[str]:
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

    def is_ready(self) -> bool:
        return self._backend_ready

    @property
    def backend_error(self) -> str:
        return self._backend_error

    @property
    def supports_min_p(self) -> bool:
        return self._supports_min_p

    async def close(self) -> None:
        if self._kgen_models is not None:
            self._kgen_models.text_model = None
            self._kgen_models.tokenizer = None
        self._backend_ready = False

    async def ensure_loaded(self) -> None:
        if self._backend_ready:
            return
        async with self._load_lock:
            if self._backend_ready:
                return
            await asyncio.to_thread(self.ensure_loaded_sync)

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
            raise TipoBackendError(f"TIPO model {self.model} loaded without tokenizer/model")
        return model, tokenizer

    def _patch_min_p_compat(self) -> None:
        if self._supports_min_p:
            return
        if getattr(self._tipo_executor, "_comfybot_min_p_patch", False):
            return

        original_generate = self._tipo_executor.generate

        def compat_generate(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("min_p", None)
            return original_generate(*args, **kwargs)

        self._tipo_executor.generate = compat_generate
        self._tipo_executor._comfybot_min_p_patch = True

    def ensure_loaded_sync(self) -> None:
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
            separate_tags = kgen_formatter.seperate_tags
            apply_format_fn = kgen_formatter.apply_format
            tipo_default_format = kgen_metainfo.TIPO_DEFAULT_FORMAT
        except (ImportError, AttributeError) as exc:
            missing = self.missing_dependencies()
            deps_hint = (
                self._tf(
                    "infrastructure.tipo_backend.error.dependencies_install_hint",
                    " Установите: pip install {packages}",
                    packages=" ".join(missing),
                )
                if missing
                else ""
            )
            self._backend_error = (
                self._t(
                    "infrastructure.tipo_backend.error.dependencies_load_failed",
                    "Не удалось загрузить зависимости TIPO.",
                )
                + deps_hint
            )
            raise TipoBackendError(self._backend_error) from exc

        self._torch = torch
        self._kgen_models = kgen_models
        self._tipo_executor = tipo_executor
        self._separate_tags = separate_tags
        self._apply_format = apply_format_fn
        self._tipo_default_format = tipo_default_format
        self._supports_min_p = hasattr(GenerationConfig(), "min_p")
        self._patch_min_p_compat()

        runtime_device = self._resolve_runtime_device(torch)
        if self.device.startswith("cuda") and runtime_device == "cpu":
            self._backend_error = self._t(
                "infrastructure.tipo_backend.error.cuda_unavailable",
                "SMART_PROMPT_DEVICE настроен на CUDA, но CUDA недоступна в PyTorch. "
                "Установите CUDA-сборку torch или переключите SMART_PROMPT_DEVICE=cpu.",
            )
            raise TipoBackendError(self._backend_error)

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
                    self._backend_error = self._tf(
                        "infrastructure.tipo_backend.error.model_load_failed_cuda_and_cpu",
                        "Не удалось загрузить TIPO модель {model} (CUDA и CPU): {error}",
                        model=self.model,
                        error=cpu_exc,
                    )
                    raise TipoBackendError(self._backend_error) from cpu_exc
            else:
                self._backend_error = self._tf(
                    "infrastructure.tipo_backend.error.model_load_failed",
                    "Не удалось загрузить TIPO модель {model}: {error}",
                    model=self.model,
                    error=exc,
                )
                raise TipoBackendError(self._backend_error) from exc

        kgen_models_any.text_model = model
        kgen_models_any.tokenizer = tokenizer
        kgen_models_any.current_model_name = self.model.split("/")[-1]

        self._loaded_model_key = f"{self.model}@{runtime_device}"
        self._runtime_device = runtime_device
        self._backend_ready = True
        self._backend_error = ""

        logger.info("TIPO model loaded: %s on %s", self.model, runtime_device)

    def separate_tags(self, tags: list[str]) -> Any:
        if not self._backend_ready:
            self.ensure_loaded_sync()
        return self._separate_tags(tags)

    def parse_request(
        self,
        *,
        tag_map: Any,
        nl_prompt: str,
        want_nl: bool,
        tag_length: str,
        nl_length: str,
    ) -> tuple[dict[str, Any], Any, Any, Any]:
        if not self._backend_ready:
            self.ensure_loaded_sync()
        meta, operations, general, parsed_nl = self._tipo_executor.parse_tipo_request(
            tag_map,
            nl_prompt,
            expand_tags=True,
            expand_prompt=want_nl,
            generate_extra_nl_prompt=False,
            tag_first=True,
            tag_length_target=tag_length,
            nl_length_target=nl_length,
        )
        if not isinstance(meta, dict):
            raise TipoBackendError(
                self._t(
                    "infrastructure.tipo_backend.error.invalid_meta_payload",
                    "TIPO parse_tipo_request вернул некорректный meta payload",
                )
            )
        meta["aspect_ratio"] = "1.0"
        return meta, operations, general, parsed_nl

    def set_ban_tags(self, tags: list[str]) -> None:
        if not self._backend_ready:
            self.ensure_loaded_sync()
        self._tipo_executor.BAN_TAGS = tags

    def run(
        self,
        meta: dict[str, Any],
        operations: Any,
        general: Any,
        parsed_nl: Any,
        *,
        runner_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Any]:
        if not self._backend_ready:
            self.ensure_loaded_sync()
        try:
            result_map, extra = self._tipo_executor.tipo_runner(
                meta,
                operations,
                general,
                parsed_nl,
                **runner_kwargs,
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise TipoBackendError(
                self._tf(
                    "infrastructure.tipo_backend.error.run_failed",
                    "Ошибка TIPO: {error}",
                    error=exc,
                )
            ) from exc
        if not isinstance(result_map, dict):
            raise TipoBackendError(
                self._t(
                    "infrastructure.tipo_backend.error.invalid_result_map",
                    "TIPO вернул некорректный result_map",
                )
            )
        return result_map, extra

    def apply_format(self, result_map: dict[str, Any], format_key: str) -> str:
        if not self._backend_ready:
            self.ensure_loaded_sync()
        template = self._tipo_default_format[format_key]
        prompt = self._apply_format(result_map, template)
        return str(prompt or "")
