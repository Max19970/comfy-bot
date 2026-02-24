from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Protocol

from core.models import GenerationParams


class ComfyInfoLike(Protocol):
    ipadapter_apply_node: str
    clip_vision_models: list[str]
    ipadapter_models: list[str]


def build_comfy_workflow(
    params: GenerationParams,
    *,
    reference_image_name: str | None,
    reference_mode: str,
    object_info: dict[str, Any],
    info: ComfyInfoLike,
    required_input_defaults: Callable[[str], dict[str, Any]],
    select_field_name: Callable[[dict[str, Any], tuple[str, ...]], str | None],
    skip_base_sampler_pass: bool = False,
) -> dict[str, Any]:
    request = params.to_generation_request()
    prompt = request.prompt
    models = request.models
    sampling = request.sampling
    image = request.image
    enhancements = request.enhancements

    seed = sampling.seed if sampling.seed >= 0 else _random_seed()

    workflow: dict[str, Any] = {}
    next_id = 1

    def _node(class_type: str, inputs: dict[str, Any]) -> str:
        nonlocal next_id
        nid = str(next_id)
        next_id += 1
        workflow[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    ckpt_id = _node("CheckpointLoaderSimple", {"ckpt_name": models.checkpoint})
    model_out = [ckpt_id, 0]
    clip_out = [ckpt_id, 1]
    vae_out = [ckpt_id, 2]

    if models.vae_name:
        vae_loader_id = _node("VAELoader", {"vae_name": models.vae_name})
        vae_out = [vae_loader_id, 0]

    for attachment in models.loras:
        lora_id = _node(
            "LoraLoader",
            {
                "lora_name": attachment.lora_name,
                "strength_model": attachment.strength_model,
                "strength_clip": attachment.strength_clip,
                "model": model_out,
                "clip": clip_out,
            },
        )
        model_out = [lora_id, 0]
        clip_out = [lora_id, 1]

    positive_text = prompt.positive
    negative_text = prompt.negative
    if models.embedding_name:
        token = f"embedding:{models.embedding_name}"
        if negative_text.strip():
            negative_text = f"{negative_text}, {token}"
        else:
            negative_text = token

    pos_id = _node(
        "CLIPTextEncode",
        {
            "text": positive_text,
            "clip": clip_out,
        },
    )

    neg_id = _node(
        "CLIPTextEncode",
        {
            "text": negative_text,
            "clip": clip_out,
        },
    )
    positive_ref: list[Any] = [pos_id, 0]
    negative_ref: list[Any] = [neg_id, 0]

    if models.controlnet_name and reference_image_name:
        control_loader_inputs = required_input_defaults("ControlNetLoader")
        control_name_field = select_field_name(
            control_loader_inputs,
            ("control_net_name", "controlnet_name"),
        )
        if control_name_field:
            control_loader_inputs[control_name_field] = models.controlnet_name
            unresolved_control_loader = [
                name for name, value in control_loader_inputs.items() if value is None
            ]
            if unresolved_control_loader:
                raise RuntimeError(
                    "ControlNetLoader has unresolved required fields: "
                    + ", ".join(unresolved_control_loader)
                )

            control_loader_id = _node("ControlNetLoader", control_loader_inputs)
            control_image_id = _node("LoadImage", {"image": reference_image_name})

            apply_class = ""
            if "ControlNetApplyAdvanced" in object_info:
                apply_class = "ControlNetApplyAdvanced"
            elif "ControlNetApply" in object_info:
                apply_class = "ControlNetApply"

            if apply_class:
                control_apply_inputs = required_input_defaults(apply_class)
                pos_field = select_field_name(
                    control_apply_inputs,
                    ("positive", "conditioning"),
                )
                neg_field = select_field_name(
                    control_apply_inputs,
                    ("negative",),
                )
                controlnet_field = select_field_name(
                    control_apply_inputs,
                    ("control_net", "controlnet"),
                )
                image_field = select_field_name(
                    control_apply_inputs,
                    ("image", "hint", "control_image"),
                )
                strength_field = select_field_name(
                    control_apply_inputs,
                    ("strength",),
                )

                if pos_field and controlnet_field and image_field:
                    control_apply_inputs[pos_field] = positive_ref
                    if neg_field:
                        control_apply_inputs[neg_field] = negative_ref
                    control_apply_inputs[controlnet_field] = [control_loader_id, 0]
                    control_apply_inputs[image_field] = [control_image_id, 0]
                    if strength_field:
                        control_apply_inputs[strength_field] = models.controlnet_strength

                    for field_name, field_value in (
                        ("start_percent", 0.0),
                        ("end_percent", 1.0),
                        ("start", 0.0),
                        ("end", 1.0),
                    ):
                        if field_name in control_apply_inputs:
                            control_apply_inputs[field_name] = field_value

                    unresolved_control_apply = [
                        name for name, value in control_apply_inputs.items() if value is None
                    ]
                    if unresolved_control_apply:
                        raise RuntimeError(
                            f"{apply_class} has unresolved required fields: "
                            + ", ".join(unresolved_control_apply)
                        )

                    control_apply_id = _node(apply_class, control_apply_inputs)
                    positive_ref = [control_apply_id, 0]
                    if neg_field:
                        negative_ref = [control_apply_id, 1]

    if reference_mode == "ipadapter" and reference_image_name:
        apply_class = info.ipadapter_apply_node
        if not apply_class:
            raise RuntimeError("IP-Adapter apply node not available")

        clip_loader_inputs = required_input_defaults("CLIPVisionLoader")
        clip_name_field = select_field_name(
            clip_loader_inputs,
            ("clip_name", "model_name"),
        )
        if not clip_name_field or not info.clip_vision_models:
            raise RuntimeError("CLIP Vision loader is not configured")
        clip_loader_inputs[clip_name_field] = info.clip_vision_models[0]
        unresolved_clip_loader = [
            name for name, value in clip_loader_inputs.items() if value is None
        ]
        if unresolved_clip_loader:
            raise RuntimeError(
                "CLIPVisionLoader has unresolved required fields: "
                + ", ".join(unresolved_clip_loader)
            )

        ip_model_loader_inputs = required_input_defaults("IPAdapterModelLoader")
        ip_model_field = select_field_name(
            ip_model_loader_inputs,
            ("ipadapter_file", "model_name", "ipadapter_name"),
        )
        if not ip_model_field or not info.ipadapter_models:
            raise RuntimeError("IP-Adapter model loader is not configured")
        ip_model_loader_inputs[ip_model_field] = info.ipadapter_models[0]
        unresolved_ip_loader = [
            name for name, value in ip_model_loader_inputs.items() if value is None
        ]
        if unresolved_ip_loader:
            raise RuntimeError(
                "IPAdapterModelLoader has unresolved required fields: "
                + ", ".join(unresolved_ip_loader)
            )

        clip_loader_id = _node("CLIPVisionLoader", clip_loader_inputs)
        ip_model_loader_id = _node("IPAdapterModelLoader", ip_model_loader_inputs)
        ref_image_id = _node("LoadImage", {"image": reference_image_name})

        apply_inputs = required_input_defaults(apply_class)
        model_field = select_field_name(apply_inputs, ("model",))
        image_field = select_field_name(apply_inputs, ("image", "images"))
        clip_field = select_field_name(
            apply_inputs,
            ("clip_vision", "clipvision"),
        )
        ipadapter_field = select_field_name(
            apply_inputs,
            ("ipadapter", "ipadapter_model"),
        )
        if not model_field or not image_field or not clip_field or not ipadapter_field:
            raise RuntimeError("IP-Adapter apply node has unsupported input schema")

        apply_inputs[model_field] = model_out
        apply_inputs[image_field] = [ref_image_id, 0]
        apply_inputs[clip_field] = [clip_loader_id, 0]
        apply_inputs[ipadapter_field] = [ip_model_loader_id, 0]

        strength_field = select_field_name(
            apply_inputs,
            ("weight", "strength", "scale"),
        )
        if strength_field:
            apply_inputs[strength_field] = image.reference_strength

        for field_name, field_value in (
            ("start_at", 0.0),
            ("end_at", 1.0),
            ("start_at_percent", 0.0),
            ("end_at_percent", 1.0),
        ):
            if field_name in apply_inputs:
                apply_inputs[field_name] = field_value

        unresolved_apply = [name for name, value in apply_inputs.items() if value is None]
        if unresolved_apply:
            raise RuntimeError(
                f"{apply_class} has unresolved required fields: " + ", ".join(unresolved_apply)
            )

        ip_apply_id = _node(apply_class, apply_inputs)
        model_out = [ip_apply_id, 0]

    if enhancements.enable_freeu:
        freeu_id = _node(
            "FreeU_V2",
            {
                "model": model_out,
                "b1": 1.3,
                "b2": 1.4,
                "s1": 0.9,
                "s2": 0.2,
            },
        )
        model_out = [freeu_id, 0]

    if enhancements.enable_pag:
        pag_id = _node(
            "PerturbedAttentionGuidance",
            {
                "model": model_out,
                "scale": enhancements.pag_scale,
            },
        )
        model_out = [pag_id, 0]

    if enhancements.enable_tiled_diffusion:
        ht_id = _node(
            "HyperTile",
            {
                "model": model_out,
                "tile_size": enhancements.tile_size,
                "swap_size": 2,
                "max_depth": 0,
                "scale_depth": False,
            },
        )
        model_out = [ht_id, 0]

    w = (image.width // 8) * 8 or 8
    h = (image.height // 8) * 8 or 8

    if reference_mode == "img2img" and reference_image_name:
        load_image_id = _node("LoadImage", {"image": reference_image_name})
        if enhancements.enable_tiled_diffusion:
            latent_id = _node(
                "VAEEncodeTiled",
                {
                    "pixels": [load_image_id, 0],
                    "vae": vae_out,
                    "tile_size": enhancements.vae_tile_size,
                    "overlap": enhancements.tile_overlap,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            )
        else:
            latent_id = _node(
                "VAEEncode",
                {
                    "pixels": [load_image_id, 0],
                    "vae": vae_out,
                },
            )
        if sampling.batch_size > 1:
            latent_id = _node(
                "RepeatLatentBatch",
                {
                    "samples": [latent_id, 0],
                    "amount": sampling.batch_size,
                },
            )
    else:
        latent_id = _node(
            "EmptyLatentImage",
            {
                "width": w,
                "height": h,
                "batch_size": sampling.batch_size,
            },
        )

    def _ksampler_inputs(*, latent_image: list[Any], denoise: float) -> dict[str, Any]:
        return {
            "model": model_out,
            "positive": positive_ref,
            "negative": negative_ref,
            "latent_image": latent_image,
            "seed": seed,
            "steps": sampling.steps,
            "cfg": sampling.cfg,
            "sampler_name": sampling.sampler,
            "scheduler": sampling.scheduler,
            "denoise": denoise,
        }

    params_skip_flag = bool(getattr(params, "_skip_base_sampler_pass", False))
    skip_base_sampling = (
        skip_base_sampler_pass or params_skip_flag
    ) and enhancements.enable_hires_fix
    base_sampler_id: str | None = None
    if not skip_base_sampling:
        base_sampler_id = _node(
            "KSampler",
            _ksampler_inputs(latent_image=[latent_id, 0], denoise=sampling.denoise),
        )

    if enhancements.enable_hires_fix:
        if skip_base_sampling:
            hires_sampler_source = [latent_id, 0]
        elif base_sampler_id is not None:
            hires_sampler_source = [base_sampler_id, 0]
        else:
            hires_sampler_source = [latent_id, 0]
        hires_latent_id = _node(
            "LatentUpscale",
            {
                "samples": hires_sampler_source,
                "upscale_method": "bislerp",
                "width": max(64, ((int(w * enhancements.hires_scale) + 7) // 8) * 8),
                "height": max(64, ((int(h * enhancements.hires_scale) + 7) // 8) * 8),
                "crop": "disabled",
            },
        )
        sampler_id = _node(
            "KSampler",
            _ksampler_inputs(latent_image=[hires_latent_id, 0], denoise=enhancements.hires_denoise),
        )
    else:
        if base_sampler_id is None:
            base_sampler_id = _node(
                "KSampler",
                _ksampler_inputs(latent_image=[latent_id, 0], denoise=sampling.denoise),
            )
        sampler_id = base_sampler_id

    if enhancements.enable_tiled_diffusion:
        decode_id = _node(
            "VAEDecodeTiled",
            {
                "samples": [sampler_id, 0],
                "vae": vae_out,
                "tile_size": enhancements.vae_tile_size,
                "overlap": enhancements.tile_overlap,
                "temporal_size": 64,
                "temporal_overlap": 8,
            },
        )
    else:
        decode_id = _node(
            "VAEDecode",
            {
                "samples": [sampler_id, 0],
                "vae": vae_out,
            },
        )

    image_out = [decode_id, 0]

    if models.upscale_model:
        upscale_loader_id = _node(
            "UpscaleModelLoader",
            {
                "model_name": models.upscale_model,
            },
        )
        upscale_id = _node(
            "ImageUpscaleWithModel",
            {
                "upscale_model": [upscale_loader_id, 0],
                "image": image_out,
            },
        )
        image_out = [upscale_id, 0]

    _node(
        "SaveImage",
        {
            "images": image_out,
            "filename_prefix": "ComfyBot",
        },
    )

    return workflow


def _random_seed() -> int:
    return random.randint(0, 2**63 - 1)
