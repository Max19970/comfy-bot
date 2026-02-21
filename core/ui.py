from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.models import GenerationParams
from core.ui_kit import (
    build_keyboard,
    build_page_window,
    custom_value_button,
    item_callback,
    page_nav_row,
)

from .html_utils import h, truncate

SIZES = [
    ("512x512", 512, 512),
    ("512x768", 512, 768),
    ("768x512", 768, 512),
    ("768x768", 768, 768),
    ("768x1024", 768, 1024),
    ("1024x768", 1024, 768),
    ("1024x1024", 1024, 1024),
    ("1024x1536", 1024, 1536),
    ("1536x1024", 1536, 1024),
]

PAGE_SIZE = 8
MAX_REFERENCE_IMAGES = 9


def paginated_keyboard(
    items: list[str],
    page: int,
    prefix: str,
    *,
    extra: list[list[InlineKeyboardButton]] | None = None,
) -> InlineKeyboardMarkup:
    window = build_page_window(items, page, PAGE_SIZE)

    rows: list[list[InlineKeyboardButton]] = []
    for i, item in enumerate(window.items):
        short = item if len(item) <= 40 else item[:37] + "..."
        rows.append(
            [
                InlineKeyboardButton(
                    text=short,
                    callback_data=item_callback(prefix, window.start_index + i),
                )
            ]
        )

    rows.append(page_nav_row(prefix, window.page, window.total_pages))
    if extra:
        rows.extend(extra)
    return build_keyboard(rows)


def custom_btn(callback_data: str) -> list[InlineKeyboardButton]:
    return [custom_value_button(callback_data)]


def loras_text(loras: list[tuple[str, float]]) -> str:
    if not loras:
        return "\u2014"
    return ", ".join(f"{h(name)} <code>{strength}</code>" for name, strength in loras)


def params_summary(params: GenerationParams) -> str:
    """Rich HTML summary for the prompt editor message (full / pro mode)."""
    return params_summary_for_mode(params, pro_mode=True)


def params_summary_simple(params: GenerationParams) -> str:
    """Compact summary for beginner mode: only key info."""
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "\u2014"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>\u2014</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>\u2014</i>"

    enh_parts: list[str] = []
    if params.enable_hires_fix:
        enh_parts.append("Hi-res")
    if params.enable_freeu:
        enh_parts.append("FreeU")
    if params.enable_pag:
        enh_parts.append("PAG")
    if params.enable_tiled_diffusion:
        enh_parts.append("Tiled")
    if params.upscale_model:
        enh_parts.append("Upscale")
    if params.vae_name:
        enh_parts.append("VAE")
    if params.controlnet_name:
        enh_parts.append("ControlNet")
    if params.embedding_name:
        enh_parts.append("Embedding")
    enh_line = ""
    if enh_parts:
        enh_line = f"\n\u2728 <b>Улучшения:</b> {', '.join(enh_parts)}"

    lora_count = len(params.loras)
    lora_line = ""
    if lora_count:
        lora_line = f"\n\U0001f4ce <b>LoRA:</b> {loras_text(params.loras)}"

    return (
        f"\U0001f3a8 <b>{ckpt_short}</b> | {params.width}\u00d7{params.height}\n"
        "\n"
        f"\U0001f7e2 <b>Positive:</b> {pos_preview}\n"
        f"\U0001f534 <b>Negative:</b> {neg_preview}"
        f"{lora_line}"
        f"{enh_line}"
    )


def params_summary_full(params: GenerationParams) -> str:
    """Full HTML summary for pro mode."""
    seed_str = f"<code>{params.seed}</code>" if params.seed >= 0 else "\U0001f3b2 random"
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "\u2014"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>\u2014</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>\u2014</i>"
    upscaler = h(params.upscale_model) if params.upscale_model else "\u2014"
    vae = h(params.vae_name) if params.vae_name else "\u2014"
    controlnet = h(params.controlnet_name) if params.controlnet_name else "\u2014"
    embedding = h(params.embedding_name) if params.embedding_name else "\u2014"
    ref_count = len(params.reference_images)

    enhancements: list[str] = []
    time_warnings: list[str] = []
    if params.enable_hires_fix:
        enhancements.append("\U0001f527 Hi-res")
        time_warnings.append(f"Hi-res \u00d7{params.hires_scale}")
    if params.enable_freeu:
        enhancements.append("\u26a1 FreeU")
    if params.enable_pag:
        enhancements.append(f"\U0001f3af PAG {params.pag_scale}")
    if params.upscale_model:
        time_warnings.append("Upscale")
    if params.enable_tiled_diffusion:
        enhancements.append(f"\U0001f9e9 HyperTile {params.tile_size}")
        time_warnings.append("HyperTile")

    enhancement_line = ""
    if enhancements or time_warnings:
        parts = "  ".join(enhancements) if enhancements else ""
        if parts:
            enhancement_line += f"\n{parts}"
        if time_warnings:
            enhancement_line += f"\n\u26a0\ufe0f <i>+время: {', '.join(time_warnings)}</i>"

    return (
        f"\U0001f3a8 <b>{ckpt_short}</b> | {params.width}\u00d7{params.height} | Steps {params.steps}\n"
        "\n"
        f"\U0001f7e2 <b>Positive:</b> {pos_preview}\n"
        f"\U0001f534 <b>Negative:</b> {neg_preview}\n"
        "\n"
        f"\u2699\ufe0f <b>Sampler:</b> <code>{h(params.sampler)}</code>  "
        f"<b>Sched:</b> <code>{h(params.scheduler)}</code>\n"
        f"<b>CFG:</b> <code>{params.cfg}</code>  "
        f"<b>Denoise:</b> <code>{params.denoise}</code>  "
        f"<b>Seed:</b> {seed_str}\n"
        f"\U0001f4ce <b>LoRA:</b> {loras_text(params.loras)}  "
        f"<b>Batch:</b> <code>{params.batch_size}</code>\n"
        f"\U0001f5bc <b>Ref:</b> {ref_count}/{MAX_REFERENCE_IMAGES} "
        f"(str <code>{params.reference_strength}</code>)  "
        f"<b>Upscale:</b> {upscaler}\n"
        f"🧬 <b>VAE:</b> {vae}  <b>ControlNet:</b> {controlnet}\n"
        f"🔤 <b>Embedding:</b> {embedding}"
        f"{enhancement_line}"
    )


def params_summary_for_mode(params: GenerationParams, *, pro_mode: bool) -> str:
    """Return summary for selected UI mode."""
    if pro_mode:
        return params_summary_full(params)
    return params_summary_simple(params)
