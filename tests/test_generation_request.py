from __future__ import annotations

from core.models import GenerationParams


def test_generation_params_to_generation_request_maps_nested_sections() -> None:
    params = GenerationParams(
        positive="a cinematic portrait",
        negative="blurry",
        checkpoint="sdxl.safetensors",
        loras=[("anime.safetensors", 0.8)],
        vae_name="vae.safetensors",
        controlnet_name="controlnet.safetensors",
        controlnet_strength=0.65,
        embedding_name="easynegative",
        upscale_model="4x-ultrasharp.pth",
        width=1024,
        height=768,
        steps=30,
        cfg=6.5,
        sampler="dpmpp_2m",
        scheduler="karras",
        denoise=0.55,
        seed=42,
        batch_size=2,
        reference_strength=0.7,
        reference_images=[{"id": "ref_1", "file_id": "file_abc"}],
        enable_hires_fix=True,
        hires_scale=1.75,
        hires_denoise=0.4,
        enable_freeu=True,
        enable_pag=True,
        pag_scale=2.8,
        enable_tiled_diffusion=True,
        tile_size=384,
        tile_overlap=96,
        vae_tile_size=640,
    )

    request = params.to_generation_request()

    assert request.prompt.positive == "a cinematic portrait"
    assert request.prompt.negative == "blurry"
    assert request.models.checkpoint == "sdxl.safetensors"
    assert [(item.lora_name, item.strength_model) for item in request.models.loras] == [
        ("anime.safetensors", 0.8),
    ]
    assert request.models.controlnet_name == "controlnet.safetensors"
    assert request.sampling.steps == 30
    assert request.sampling.batch_size == 2
    assert request.image.width == 1024
    assert request.image.reference_images[0].file_id == "file_abc"
    assert request.enhancements.enable_hires_fix is True
    assert request.enhancements.tile_overlap == 96


def test_generation_params_from_generation_request_roundtrip() -> None:
    src = GenerationParams(
        positive="portrait",
        negative="bad hands",
        checkpoint="model.safetensors",
        loras=[("style.safetensors", 1.1)],
        width=896,
        height=512,
        seed=777,
        reference_images=[{"id": "ref_1", "file_id": "file_1"}],
        enable_pag=True,
        pag_scale=3.2,
    )

    restored = GenerationParams.from_generation_request(src.to_generation_request())

    assert restored.positive == src.positive
    assert restored.negative == src.negative
    assert restored.checkpoint == src.checkpoint
    assert restored.loras == src.loras
    assert restored.width == src.width
    assert restored.height == src.height
    assert restored.seed == src.seed
    assert restored.reference_images == src.reference_images
    assert restored.enable_pag is True
    assert restored.pag_scale == 3.2
