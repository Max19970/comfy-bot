from __future__ import annotations

from core.models import GenerationParams
from domain.localization import LocalizationService

from .html_utils import h, truncate

MAX_REFERENCE_IMAGES_DEFAULT = 9


def _t(
    localization: LocalizationService | None,
    key: str,
    *,
    locale: str | None,
    default: str,
) -> str:
    if localization is None:
        return default
    return localization.t(key, locale=locale, default=default)


def loras_text(
    loras: list[tuple[str, float]],
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    if not loras:
        return "—"
    return ", ".join(f"{h(name)} <code>{strength}</code>" for name, strength in loras)


def params_summary(
    params: GenerationParams,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    return params_summary_for_mode(params, pro_mode=True, localization=localization, locale=locale)


def params_summary_simple(
    params: GenerationParams,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "—"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>—</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>—</i>"
    positive_label = _t(
        localization,
        "core.ui_summary.positive_label",
        locale=locale,
        default="Positive",
    )
    negative_label = _t(
        localization,
        "core.ui_summary.negative_label",
        locale=locale,
        default="Negative",
    )
    enhancements_label = _t(
        localization,
        "core.ui_summary.enhancements_label",
        locale=locale,
        default="Улучшения",
    )
    lora_label = _t(
        localization,
        "common.delete.model_type.lora",
        locale=locale,
        default="LoRA",
    )

    enhancements = _simple_enhancements(params, localization=localization, locale=locale)
    enhancement_line = (
        f"\n✨ <b>{h(enhancements_label)}:</b> {', '.join(enhancements)}" if enhancements else ""
    )

    lora_count = len(params.loras)
    lora_line = (
        f"\n📎 <b>{h(lora_label)}:</b> {loras_text(params.loras, localization=localization, locale=locale)}"
        if lora_count
        else ""
    )

    return (
        f"🎨 <b>{ckpt_short}</b> | {params.width}×{params.height}\n"
        "\n"
        f"🟢 <b>{h(positive_label)}:</b> {pos_preview}\n"
        f"🔴 <b>{h(negative_label)}:</b> {neg_preview}"
        f"{lora_line}"
        f"{enhancement_line}"
    )


def params_summary_full(
    params: GenerationParams,
    *,
    max_reference_images: int = MAX_REFERENCE_IMAGES_DEFAULT,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    random_value = _t(localization, "common.value.random", locale=locale, default="random")
    steps_label = _t(
        localization,
        "common.settings.generation.value.steps",
        locale=locale,
        default="Steps",
    )
    positive_label = _t(
        localization,
        "core.ui_summary.positive_label",
        locale=locale,
        default="Positive",
    )
    negative_label = _t(
        localization,
        "core.ui_summary.negative_label",
        locale=locale,
        default="Negative",
    )
    sampler_label = _t(
        localization,
        "common.settings.generation.value.sampler",
        locale=locale,
        default="Sampler",
    )
    scheduler_label = _t(
        localization,
        "common.settings.generation.value.scheduler",
        locale=locale,
        default="Sched",
    )
    cfg_label = _t(
        localization,
        "common.settings.generation.value.cfg",
        locale=locale,
        default="CFG",
    )
    denoise_label = _t(
        localization,
        "common.settings.generation.value.denoise",
        locale=locale,
        default="Denoise",
    )
    seed_label = _t(
        localization,
        "common.settings.generation.value.seed",
        locale=locale,
        default="Seed",
    )
    lora_label = _t(
        localization,
        "common.delete.model_type.lora",
        locale=locale,
        default="LoRA",
    )
    batch_label = _t(
        localization,
        "common.settings.generation.value.batch",
        locale=locale,
        default="Batch",
    )
    ref_label = _t(localization, "core.ui_summary.references_label", locale=locale, default="Ref")
    strength_label = _t(
        localization, "core.ui_summary.strength_short", locale=locale, default="str"
    )
    upscale_label = _t(
        localization,
        "common.models.section.upscale",
        locale=locale,
        default="Upscale",
    )
    vae_label = _t(localization, "core.ui_summary.vae_label", locale=locale, default="VAE")
    controlnet_label = _t(
        localization,
        "core.ui_summary.controlnet_label",
        locale=locale,
        default="ControlNet",
    )
    embedding_label = _t(
        localization,
        "core.ui_summary.embedding_label",
        locale=locale,
        default="Embedding",
    )

    seed_str = f"<code>{params.seed}</code>" if params.seed >= 0 else f"🎲 {h(random_value)}"
    ckpt_short = h(truncate(params.checkpoint, 35)) if params.checkpoint else "—"
    pos_preview = h(truncate(params.positive, 60)) if params.positive.strip() else "<i>—</i>"
    neg_preview = h(truncate(params.negative, 60)) if params.negative.strip() else "<i>—</i>"
    upscaler = h(params.upscale_model) if params.upscale_model else "—"
    vae = h(params.vae_name) if params.vae_name else "—"
    controlnet = h(params.controlnet_name) if params.controlnet_name else "—"
    embedding = h(params.embedding_name) if params.embedding_name else "—"
    ref_count = len(params.reference_images)

    enhancement_icons, time_warnings = _pro_enhancements(
        params, localization=localization, locale=locale
    )

    enhancement_line = ""
    if enhancement_icons or time_warnings:
        icon_text = "  ".join(enhancement_icons)
        if icon_text:
            enhancement_line += f"\n{icon_text}"
        if time_warnings:
            time_label = _t(
                localization,
                "core.ui_summary.time_warning_prefix",
                locale=locale,
                default="+время",
            )
            enhancement_line += f"\n⚠️ <i>{h(time_label)}: {', '.join(time_warnings)}</i>"

    return (
        f"🎨 <b>{ckpt_short}</b> | {params.width}×{params.height} | {h(steps_label)} {params.steps}\n"
        "\n"
        f"🟢 <b>{h(positive_label)}:</b> {pos_preview}\n"
        f"🔴 <b>{h(negative_label)}:</b> {neg_preview}\n"
        "\n"
        f"⚙️ <b>{h(sampler_label)}:</b> <code>{h(params.sampler)}</code>  "
        f"<b>{h(scheduler_label)}:</b> <code>{h(params.scheduler)}</code>\n"
        f"<b>{h(cfg_label)}:</b> <code>{params.cfg}</code>  "
        f"<b>{h(denoise_label)}:</b> <code>{params.denoise}</code>  "
        f"<b>{h(seed_label)}:</b> {seed_str}\n"
        f"📎 <b>{h(lora_label)}:</b> {loras_text(params.loras, localization=localization, locale=locale)}  "
        f"<b>{h(batch_label)}:</b> <code>{params.batch_size}</code>\n"
        f"🖼 <b>{h(ref_label)}:</b> {ref_count}/{max_reference_images} "
        f"({h(strength_label)} <code>{params.reference_strength}</code>)  "
        f"<b>{h(upscale_label)}:</b> {upscaler}\n"
        f"🧬 <b>{h(vae_label)}:</b> {vae}  <b>{h(controlnet_label)}:</b> {controlnet}\n"
        f"🔤 <b>{h(embedding_label)}:</b> {embedding}"
        f"{enhancement_line}"
    )


def params_summary_for_mode(
    params: GenerationParams,
    *,
    pro_mode: bool,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> str:
    if pro_mode:
        return params_summary_full(params, localization=localization, locale=locale)
    return params_summary_simple(params, localization=localization, locale=locale)


def _simple_enhancements(
    params: GenerationParams,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[str]:
    items: list[str] = []
    if params.enable_hires_fix:
        items.append(
            _t(localization, "core.ui_summary.enhancement.hires", locale=locale, default="Hi-res")
        )
    if params.enable_freeu:
        items.append(
            _t(localization, "core.ui_summary.enhancement.freeu", locale=locale, default="FreeU")
        )
    if params.enable_pag:
        items.append(
            _t(localization, "core.ui_summary.enhancement.pag", locale=locale, default="PAG")
        )
    if params.enable_tiled_diffusion:
        items.append(
            _t(localization, "core.ui_summary.enhancement.tiled", locale=locale, default="Tiled")
        )
    if params.upscale_model:
        items.append(
            _t(
                localization,
                "common.models.section.upscale",
                locale=locale,
                default="Upscale",
            )
        )
    if params.vae_name:
        items.append(_t(localization, "core.ui_summary.vae_label", locale=locale, default="VAE"))
    if params.controlnet_name:
        items.append(
            _t(
                localization,
                "core.ui_summary.controlnet_label",
                locale=locale,
                default="ControlNet",
            )
        )
    if params.embedding_name:
        items.append(
            _t(localization, "core.ui_summary.embedding_label", locale=locale, default="Embedding")
        )
    return items


def _pro_enhancements(
    params: GenerationParams,
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> tuple[list[str], list[str]]:
    icons: list[str] = []
    time_warnings: list[str] = []
    hires_label = _t(
        localization, "core.ui_summary.enhancement.hires", locale=locale, default="Hi-res"
    )
    freeu_label = _t(
        localization, "core.ui_summary.enhancement.freeu", locale=locale, default="FreeU"
    )
    pag_label = _t(localization, "core.ui_summary.enhancement.pag", locale=locale, default="PAG")
    upscale_label = _t(
        localization,
        "common.models.section.upscale",
        locale=locale,
        default="Upscale",
    )
    hypertile_label = _t(
        localization,
        "core.ui_summary.enhancement.hypertile",
        locale=locale,
        default="HyperTile",
    )

    if params.enable_hires_fix:
        icons.append(f"🔧 {h(hires_label)} ×{params.hires_scale}")
        time_warnings.append(f"{h(hires_label)} ×{params.hires_scale}")
    if params.enable_freeu:
        icons.append(f"⚡ {h(freeu_label)}")
    if params.enable_pag:
        icons.append(f"🎯 {h(pag_label)} {params.pag_scale}")
    if params.upscale_model:
        time_warnings.append(h(upscale_label))
    if params.enable_tiled_diffusion:
        icons.append(f"🧩 {h(hypertile_label)} {params.tile_size}")
        time_warnings.append(h(hypertile_label))

    return icons, time_warnings
