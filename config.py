import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int_safe(name: str, default: int) -> int:
    raw = _env(name, str(default)).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_csv_ints(name: str) -> list[int]:
    raw = _env(name, "")
    if not raw.strip():
        return []
    return [int(uid.strip()) for uid in raw.split(",") if uid.strip()]


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


@dataclass
class Config:
    telegram_token: str = ""
    comfyui_url: str = "http://127.0.0.1:8000"
    allowed_users: list[int] = field(default_factory=list)

    # Paths
    comfyui_models_path: str = r"C:\Programs\ComfyUI\models"

    # API keys (optional)
    civitai_api_key: str = ""
    huggingface_token: str = ""

    # Smart prompt settings
    smart_prompt_provider: str = "tipo"
    smart_prompt_model: str = "KBlueLeaf/TIPO-500M-ft"
    smart_prompt_timeout: int = 120
    smart_prompt_temperature: float = 0.35
    smart_prompt_top_p: float = 0.95
    smart_prompt_min_p: float = 0.05
    smart_prompt_top_k: int = 80
    smart_prompt_device: str = "auto"
    smart_prompt_seed: int = -1
    smart_prompt_tag_length: str = "long"
    smart_prompt_nl_length: str = "short"
    smart_prompt_format: str = "both_tag_first"
    smart_prompt_ban_tags: str = "text, watermark, signature"
    smart_prompt_negative_base: str = (
        "worst quality, low quality, bad quality, blurry, jpeg artifacts, "
        "bad anatomy, deformed, ugly, extra limbs, missing limbs, disfigured, "
        "mutilated, out of frame, duplicate, watermark, signature, text"
    )

    # Default generation parameters
    default_width: int = 512
    default_height: int = 512
    default_steps: int = 20
    default_cfg: float = 7.0
    default_sampler: str = "euler"
    default_scheduler: str = "normal"
    default_denoise: float = 1.0

    @classmethod
    def from_env(cls) -> "Config":
        token = _env("TELEGRAM_BOT_TOKEN", "")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")

        return cls(
            telegram_token=token,
            comfyui_url=_strip_trailing_slash(
                _env("COMFYUI_URL", "http://127.0.0.1:8000")
            ),
            allowed_users=_env_csv_ints("ALLOWED_USERS"),
            comfyui_models_path=_env(
                "COMFYUI_MODELS_PATH",
                r"C:\Programs\ComfyUI\models",
            ),
            civitai_api_key=_env("CIVITAI_API_KEY", ""),
            huggingface_token=(
                _env("HUGGINGFACE_TOKEN", "").strip() or _env("HF_TOKEN", "").strip()
            ),
            smart_prompt_provider=_env("SMART_PROMPT_PROVIDER", "tipo"),
            smart_prompt_model=_env("SMART_PROMPT_MODEL", "KBlueLeaf/TIPO-500M-ft"),
            smart_prompt_timeout=_env_int("SMART_PROMPT_TIMEOUT", 120),
            smart_prompt_temperature=_env_float("SMART_PROMPT_TEMPERATURE", 0.35),
            smart_prompt_top_p=_env_float("SMART_PROMPT_TOP_P", 0.95),
            smart_prompt_min_p=_env_float("SMART_PROMPT_MIN_P", 0.05),
            smart_prompt_top_k=_env_int("SMART_PROMPT_TOP_K", 80),
            smart_prompt_device=_env("SMART_PROMPT_DEVICE", "auto"),
            smart_prompt_seed=_env_int_safe("SMART_PROMPT_SEED", -1),
            smart_prompt_tag_length=_env("SMART_PROMPT_TAG_LENGTH", "long"),
            smart_prompt_nl_length=_env("SMART_PROMPT_NL_LENGTH", "short"),
            smart_prompt_format=_env("SMART_PROMPT_FORMAT", "both_tag_first"),
            smart_prompt_ban_tags=_env(
                "SMART_PROMPT_BAN_TAGS",
                "text, watermark, signature",
            ),
            smart_prompt_negative_base=_env(
                "SMART_PROMPT_NEGATIVE_BASE",
                "worst quality, low quality, bad quality, blurry, jpeg artifacts, "
                "bad anatomy, deformed, ugly, extra limbs, missing limbs, disfigured, "
                "mutilated, out of frame, duplicate, watermark, signature, text",
            ),
            default_width=_env_int("DEFAULT_WIDTH", 512),
            default_height=_env_int("DEFAULT_HEIGHT", 512),
            default_steps=_env_int("DEFAULT_STEPS", 20),
            default_cfg=_env_float("DEFAULT_CFG", 7.0),
            default_sampler=_env("DEFAULT_SAMPLER", "euler"),
            default_scheduler=_env("DEFAULT_SCHEDULER", "normal"),
            default_denoise=_env_float("DEFAULT_DENOISE", 1.0),
        )

    @property
    def smart_prompt_enabled(self) -> bool:
        provider = self.smart_prompt_provider.strip().lower()
        return provider not in {"", "off", "none", "disabled"}


config = Config.from_env() if os.getenv("TELEGRAM_BOT_TOKEN") else Config()
