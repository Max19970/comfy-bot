from __future__ import annotations

from core.models import GenerationParams

from .html_utils import h, truncate

MAX_REFERENCE_IMAGES_DEFAULT = 9


def loras_text(loras: list[tuple[str, float]]) -> str:
    if not loras:
        return "—"
    return ", ".join(f"{h(name)} <code>{strength}</code>" for name, strength in loras)


def params_summary(params: GenerationParams) -> str:
    return params_summary_for_mode(params, pro_mode=True)


def params_summary_simple(params: GenerationParams) -> str:
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "—"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>—</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>—</i>"

    enhancements = _simple_enhancements(params)
    enhancement_line = f"\n✨ <b>Улучшения:</b> {', '.join(enhancements)}" if enhancements else ""

    lora_count = len(params.loras)
    lora_line = f"\n📎 <b>LoRA:</b> {loras_text(params.loras)}" if lora_count else ""

    return (
        f"🎨 <b>{ckpt_short}</b> | {params.width}×{params.height}\n"
        "\n"
        f"🟢 <b>Positive:</b> {pos_preview}\n"
        f"🔴 <b>Negative:</b> {neg_preview}"
        f"{lora_line}"
        f"{enhancement_line}"
    )


def params_summary_full(
    params: GenerationParams,
    *,
    max_reference_images: int = MAX_REFERENCE_IMAGES_DEFAULT,
) -> str:
    seed_str = f"<code>{params.seed}</code>" if params.seed >= 0 else "🎲 random"
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "—"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>—</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>—</i>"
    upscaler = h(params.upscale_model) if params.upscale_model else "—"
    vae = h(params.vae_name) if params.vae_name else "—"
    controlnet = h(params.controlnet_name) if params.controlnet_name else "—"
    embedding = h(params.embedding_name) if params.embedding_name else "—"
    ref_count = len(params.reference_images)

    enhancement_icons, time_warnings = _pro_enhancements(params)

    enhancement_line = ""
    if enhancement_icons or time_warnings:
        icon_text = "  ".join(enhancement_icons)
        if icon_text:
            enhancement_line += f"\n{icon_text}"
        if time_warnings:
            enhancement_line += f"\n⚠️ <i>+время: {', '.join(time_warnings)}</i>"

    return (
        f"🎨 <b>{ckpt_short}</b> | {params.width}×{params.height} | Steps {params.steps}\n"
        "\n"
        f"🟢 <b>Positive:</b> {pos_preview}\n"
        f"🔴 <b>Negative:</b> {neg_preview}\n"
        "\n"
        f"⚙️ <b>Sampler:</b> <code>{h(params.sampler)}</code>  "
        f"<b>Sched:</b> <code>{h(params.scheduler)}</code>\n"
        f"<b>CFG:</b> <code>{params.cfg}</code>  "
        f"<b>Denoise:</b> <code>{params.denoise}</code>  "
        f"<b>Seed:</b> {seed_str}\n"
        f"📎 <b>LoRA:</b> {loras_text(params.loras)}  "
        f"<b>Batch:</b> <code>{params.batch_size}</code>\n"
        f"🖼 <b>Ref:</b> {ref_count}/{max_reference_images} "
        f"(str <code>{params.reference_strength}</code>)  "
        f"<b>Upscale:</b> {upscaler}\n"
        f"🧬 <b>VAE:</b> {vae}  <b>ControlNet:</b> {controlnet}\n"
        f"🔤 <b>Embedding:</b> {embedding}"
        f"{enhancement_line}"
    )


def params_summary_for_mode(params: GenerationParams, *, pro_mode: bool) -> str:
    if pro_mode:
        return params_summary_full(params)
    return params_summary_simple(params)


def _simple_enhancements(params: GenerationParams) -> list[str]:
    items: list[str] = []
    if params.enable_hires_fix:
        items.append("Hi-res")
    if params.enable_freeu:
        items.append("FreeU")
    if params.enable_pag:
        items.append("PAG")
    if params.enable_tiled_diffusion:
        items.append("Tiled")
    if params.upscale_model:
        items.append("Upscale")
    if params.vae_name:
        items.append("VAE")
    if params.controlnet_name:
        items.append("ControlNet")
    if params.embedding_name:
        items.append("Embedding")
    return items


def _pro_enhancements(params: GenerationParams) -> tuple[list[str], list[str]]:
    icons: list[str] = []
    time_warnings: list[str] = []

    if params.enable_hires_fix:
        icons.append(f"🔧 Hi-res ×{params.hires_scale}")
        time_warnings.append(f"Hi-res ×{params.hires_scale}")
    if params.enable_freeu:
        icons.append("⚡ FreeU")
    if params.enable_pag:
        icons.append(f"🎯 PAG {params.pag_scale}")
    if params.upscale_model:
        time_warnings.append("Upscale")
    if params.enable_tiled_diffusion:
        icons.append(f"🧩 HyperTile {params.tile_size}")
        time_warnings.append("HyperTile")

    return icons, time_warnings
