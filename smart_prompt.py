from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from config import Config

logger = logging.getLogger(__name__)

_MAX_DESCRIPTION_LEN = 4000

# ---------------------------------------------------------------------------
# Checkpoint-aware quality tag presets
# ---------------------------------------------------------------------------
_QUALITY_PRESETS: dict[str, tuple[str, ...]] = {
    "pony": ("score_9", "score_8_up", "score_7_up", "source_anime"),
    "illustrious": ("masterpiece", "best quality", "amazing quality", "newest"),
    "default": ("masterpiece", "best quality", "highly detailed"),
}

# Substrings in checkpoint filenames that hint at the base model family.
_CHECKPOINT_HINTS: list[tuple[str, str]] = [
    ("pony", "pony"),
    ("pdxl", "pony"),
    ("autismmix", "pony"),
    ("illustrious", "illustrious"),
    ("wai", "illustrious"),
    ("noob", "illustrious"),
    ("hassaku", "illustrious"),
]

_LENGTH_VALUES = {"very_short", "short", "long", "very_long"}

_FORMAT_KEYS: dict[str, str] = {
    "tag_only": "tag only (DTG mode)",
    "nl_only": "NL only (Tag to NL)",
    "both_tag_first": "Both, tag first (recommend)",
    "both_nl_first": "Both, NL first (recommend)",
    "both_generated": "Both + generated NL",
}

# Control markers that should never leak into the final user-visible prompt.
_CONTROL_MARKERS = (
    "<|",
    "|>",
    "<0x",
    "target:",
    "target :",
)

# Stopwords filtered out when extracting anchor tags from NL descriptions.
_ANCHOR_STOPWORDS = {
    # English
    "a", "an", "and", "or", "the", "this", "that", "these", "those",
    "with", "without", "from", "into", "onto", "over", "under", "through",
    "for", "of", "to", "in", "on", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "become", "becoming", "becomes",
    "get", "gets", "getting", "got", "make", "makes", "making", "made",
    "as", "while", "during", "after", "before", "very", "more", "less",
    "some", "any", "each", "every", "all", "both", "few", "many", "much",
    "no", "not", "but", "so", "if", "then", "than", "too", "also",
    "it", "its", "he", "she", "his", "her", "they", "their", "we", "our",
    "my", "your", "who", "what", "where", "when", "how", "which",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "shall", "must",
    # Russian
    "и", "или", "на", "в", "во", "с", "со", "к", "по", "за", "для",
    "без", "под", "над", "из", "у", "при", "не", "ни", "это", "тот",
    "эта", "эти", "быть", "стал", "стала", "становится",
    "он", "она", "оно", "они", "его", "её", "их", "мой", "моя", "наш",
    "ваш", "который", "которая", "которые", "что", "как", "где", "когда",
    "очень", "более", "менее", "тоже", "также", "но", "а", "же",
}

# Lightweight Russian→English map for common visual/art description words.
# Lets TIPO understand Russian input without adding translation dependencies.
_RU_EN_VISUAL: dict[str, str] = {
    # Subjects
    "девушка": "girl", "девочка": "girl", "женщина": "woman",
    "парень": "boy", "мальчик": "boy", "мужчина": "man",
    "кот": "cat", "кошка": "cat", "котенок": "kitten", "котёнок": "kitten",
    "собака": "dog", "щенок": "puppy", "пёс": "dog", "пес": "dog",
    "лошадь": "horse", "конь": "horse", "птица": "bird",
    "дракон": "dragon", "волк": "wolf", "лиса": "fox", "лис": "fox",
    "кролик": "rabbit", "заяц": "rabbit", "медведь": "bear",
    "рыцарь": "knight", "воин": "warrior", "маг": "mage", "ведьма": "witch",
    "принцесса": "princess", "принц": "prince", "король": "king", "королева": "queen",
    "робот": "robot", "ангел": "angel", "демон": "demon",
    # Body / appearance
    "волосы": "hair", "глаза": "eyes", "лицо": "face", "улыбка": "smile",
    "длинные": "long", "короткие": "short", "большие": "large", "маленькие": "small",
    # Colors
    "красный": "red", "красная": "red", "красное": "red", "красные": "red",
    "синий": "blue", "синяя": "blue", "синее": "blue", "синие": "blue",
    "голубой": "light blue", "голубая": "light blue", "голубые": "light blue",
    "зелёный": "green", "зеленый": "green", "зелёная": "green", "зеленая": "green",
    "жёлтый": "yellow", "желтый": "yellow", "жёлтая": "yellow", "желтая": "yellow",
    "оранжевый": "orange", "оранжевая": "orange",
    "фиолетовый": "purple", "фиолетовая": "purple",
    "розовый": "pink", "розовая": "pink", "розовые": "pink",
    "белый": "white", "белая": "white", "белое": "white", "белые": "white",
    "чёрный": "black", "черный": "black", "чёрная": "black", "черная": "black",
    "серый": "grey", "серая": "grey", "золотой": "golden", "золотая": "golden",
    "серебряный": "silver", "серебряная": "silver",
    # Clothing / accessories
    "платье": "dress", "юбка": "skirt", "шляпа": "hat", "шляпе": "hat",
    "корона": "crown", "плащ": "cape", "доспехи": "armor", "броня": "armor",
    "очки": "glasses", "маска": "mask", "перчатки": "gloves", "сапоги": "boots",
    "униформа": "uniform", "костюм": "suit", "рубашка": "shirt",
    "шарф": "scarf", "капюшон": "hood", "крылья": "wings", "хвост": "tail",
    # Scenes / environment
    "город": "city", "лес": "forest", "море": "sea", "океан": "ocean",
    "гора": "mountain", "горы": "mountains", "река": "river", "озеро": "lake",
    "небо": "sky", "облака": "clouds", "звёзды": "stars", "звезды": "stars",
    "луна": "moon", "солнце": "sun", "закат": "sunset", "рассвет": "sunrise",
    "ночь": "night", "ночью": "night", "день": "day", "днём": "day",
    "дождь": "rain", "снег": "snow", "туман": "fog", "гроза": "storm",
    "поле": "field", "сад": "garden", "цветы": "flowers", "цветок": "flower",
    "дерево": "tree", "деревья": "trees", "трава": "grass",
    "замок": "castle", "дворец": "palace", "храм": "temple", "башня": "tower",
    "улица": "street", "мост": "bridge", "дом": "house", "комната": "room",
    "окно": "window", "подоконник": "windowsill", "подоконнике": "windowsill",
    "пляж": "beach", "пустыня": "desert", "джунгли": "jungle",
    "космос": "space", "планета": "planet",
    # Actions / poses
    "сидит": "sitting", "стоит": "standing", "лежит": "lying down",
    "бежит": "running", "идёт": "walking", "идет": "walking",
    "летит": "flying", "плывёт": "swimming", "плывет": "swimming",
    "держит": "holding", "смотрит": "looking", "читает": "reading",
    "играет": "playing", "танцует": "dancing", "спит": "sleeping",
    # Mood / style
    "красивый": "beautiful", "красивая": "beautiful",
    "милый": "cute", "милая": "cute", "грустный": "sad", "грустная": "sad",
    "счастливый": "happy", "счастливая": "happy",
    "тёмный": "dark", "темный": "dark", "тёмная": "dark", "темная": "dark",
    "светлый": "bright", "светлая": "bright",
    "магический": "magical", "магическая": "magical",
    "мрачный": "gloomy", "мрачная": "gloomy",
    "эпический": "epic", "эпичный": "epic",
    "фэнтези": "fantasy", "фентези": "fantasy",
    "киберпанк": "cyberpunk", "стимпанк": "steampunk",
    "реалистичный": "realistic", "реалистичная": "realistic",
    "аниме": "anime",
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
        name = checkpoint.lower()
        for hint, preset in _CHECKPOINT_HINTS:
            if hint in name:
                return preset
        return "default"

    @staticmethod
    def _extract_anchor_tags(description: str) -> list[str]:
        """Extract meaningful keywords from an NL description to use as
        seed general tags.  This gives TIPO concrete concepts to expand on
        instead of generating random anime tags.

        Russian words are translated to English via _RU_EN_VISUAL so that
        TIPO (trained on English/Japanese Danbooru data) can understand them.
        """
        # Split on whitespace, punctuation, and common separators
        words = re.split(r"[\s,;:!?.()\[\]{}\"/]+", description.lower())
        anchors: list[str] = []
        for word in words:
            word = word.strip().strip("'-")
            if not word or len(word) < 2:
                continue
            if word in _ANCHOR_STOPWORDS:
                continue
            if word.isdigit():
                continue
            # Translate Russian words to English equivalents
            translated = _RU_EN_VISUAL.get(word)
            if translated:
                anchors.append(translated)
            else:
                anchors.append(word.replace("_", " "))
        # Also look for multi-word phrases that map to common Danbooru tags
        # by joining adjacent non-stopwords (bigrams)
        raw_words = re.split(r"[\s,;:!?.]+", description.lower())
        raw_words = [w.strip().strip("'-") for w in raw_words if w.strip()]
        for i in range(len(raw_words) - 1):
            w1, w2 = raw_words[i], raw_words[i + 1]
            if w1 in _ANCHOR_STOPWORDS or w2 in _ANCHOR_STOPWORDS:
                continue
            if len(w1) < 2 or len(w2) < 2:
                continue
            # Translate each word in the bigram
            t1 = _RU_EN_VISUAL.get(w1, w1)
            t2 = _RU_EN_VISUAL.get(w2, w2)
            bigram = f"{t1} {t2}"
            anchors.append(bigram)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for tag in anchors:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result[:32]

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
            return (
                "Отсутствуют зависимости для TIPO. Установите: "
                f"pip install {' '.join(missing)}"
            )

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
            raise SmartPromptError(
                f"TIPO model {self.model} loaded without tokenizer/model"
            )
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
            GenerationConfig = getattr(transformers_module, "GenerationConfig")
            kgen_models = importlib.import_module("kgen.models")
            tipo_executor = importlib.import_module("kgen.executor.tipo")
            kgen_formatter = importlib.import_module("kgen.formatter")
            kgen_metainfo = importlib.import_module("kgen.metainfo")
            seperate_tags = getattr(kgen_formatter, "seperate_tags")
            apply_format_fn = getattr(kgen_formatter, "apply_format")
            tipo_default_format = getattr(kgen_metainfo, "TIPO_DEFAULT_FORMAT")
        except Exception as exc:
            missing = self._missing_dependencies()
            deps_hint = (
                f" Установите: pip install {' '.join(missing)}" if missing else ""
            )
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

        self._kgen_models.text_model = None
        self._kgen_models.tokenizer = None

        try:
            model, tokenizer = self._load_tipo_model(device=runtime_device)
        except Exception as exc:
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
                except Exception as cpu_exc:
                    self._backend_error = (
                        "Не удалось загрузить TIPO модель "
                        f"{self.model} (CUDA и CPU): {cpu_exc}"
                    )
                    raise SmartPromptError(self._backend_error) from cpu_exc
            else:
                self._backend_error = (
                    f"Не удалось загрузить TIPO модель {self.model}: {exc}"
                )
                raise SmartPromptError(self._backend_error) from exc

        self._kgen_models.text_model = model
        self._kgen_models.tokenizer = tokenizer
        self._kgen_models.current_model_name = self.model.split("/")[-1]

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
        quality_preset = self.detect_quality_preset(checkpoint)
        quality_tags = list(
            _QUALITY_PRESETS.get(quality_preset, _QUALITY_PRESETS["default"])
        )

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

        meta, operations, general, parsed_nl = (
            self._tipo_executor.parse_tipo_request(
                tag_map,
                nl_prompt,
                expand_tags=True,
                expand_prompt=want_nl,
                generate_extra_nl_prompt=False,
                tag_first=True,
                tag_length_target=self.tag_length,
                nl_length_target=self.nl_length,
            )
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
        except Exception as exc:
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
        if not raw:
            return []
        parts = re.split(r"[,;\n]", raw)
        tags: list[str] = []
        for part in parts:
            cleaned = part.strip().strip("-").strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned:
                tags.append(cleaned)
        return tags

    @staticmethod
    def _dedupe_tags(tags: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(tag)
        return result

    @staticmethod
    def _sanitize_final_prompt(text: str) -> str:
        """Remove TIPO control markers that may leak into the final output."""
        # Remove any <|...|> tokens
        text = re.sub(r"<\|[^|]*\|>", "", text)
        # Remove lines containing control markers
        lines = text.split("\n")
        clean_lines: list[str] = []
        for line in lines:
            lowered = line.casefold()
            if any(marker in lowered for marker in _CONTROL_MARKERS):
                continue
            clean_lines.append(line)
        text = "\n".join(clean_lines)
        # Clean up extra whitespace and commas
        text = re.sub(r"\s*,\s*,+\s*", ", ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        text = text.strip().strip(",").strip()
        return text

    def _build_positive_prompt(self, result_map: dict[str, Any]) -> str:
        """Build positive prompt using kgen's apply_format with templates."""
        format_key = _FORMAT_KEYS.get(
            self.output_format, _FORMAT_KEYS["both_tag_first"]
        )
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
                if "." in p_stripped and len(p_stripped) > 40:
                    filtered.append(p_stripped)
                elif p_stripped.casefold() not in banned:
                    filtered.append(p_stripped)
            prompt = ", ".join(filtered)

        # Sanitize any control markers that leaked through
        prompt = self._sanitize_final_prompt(prompt)
        return prompt

    def _build_negative_prompt(self) -> str:
        tags = self._split_tags(self.negative_base)
        tags.extend(self._split_tags(self.ban_tags))
        return ", ".join(self._dedupe_tags(tags)).strip().strip(",")
