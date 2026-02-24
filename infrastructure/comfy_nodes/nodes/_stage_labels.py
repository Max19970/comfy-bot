from __future__ import annotations

from ..contracts import WorkflowStageLabel

_ALL_LABELS = {
    "CheckpointLoaderSimple": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.checkpoint_loader",
        default_text="подготовка модели",
    ),
    "LoraLoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.lora_loader",
        default_text="применение LoRA",
    ),
    "VAELoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.vae_loader",
        default_text="загрузка VAE",
    ),
    "CLIPTextEncode": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.clip_text_encode",
        default_text="кодирование промпта",
    ),
    "ControlNetLoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.controlnet_loader",
        default_text="загрузка ControlNet",
    ),
    "ControlNetApply": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.controlnet_apply",
        default_text="применение ControlNet",
    ),
    "ControlNetApplyAdvanced": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.controlnet_apply",
        default_text="применение ControlNet",
    ),
    "CLIPVisionLoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.clip_vision_loader",
        default_text="загрузка CLIP Vision",
    ),
    "IPAdapterModelLoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.ipadapter_model_loader",
        default_text="загрузка IP-Adapter",
    ),
    "IPAdapterApply": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.ipadapter_apply",
        default_text="применение IP-Adapter",
    ),
    "IPAdapterApplyAdvanced": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.ipadapter_apply",
        default_text="применение IP-Adapter",
    ),
    "LoadImage": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.load_image",
        default_text="загрузка изображения",
    ),
    "VAEEncode": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.vae_encode",
        default_text="кодирование референса",
    ),
    "EmptyLatentImage": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.empty_latent_image",
        default_text="подготовка латента",
    ),
    "RepeatLatentBatch": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.repeat_latent_batch",
        default_text="подготовка батча",
    ),
    "KSampler": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.ksampler",
        default_text="сэмплинг",
    ),
    "VAEDecode": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.vae_decode",
        default_text="декодирование изображения",
    ),
    "UpscaleModelLoader": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.upscale_model_loader",
        default_text="загрузка апскейлера",
    ),
    "ImageUpscaleWithModel": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.image_upscale_with_model",
        default_text="апскейл",
    ),
    "SaveImage": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.save_image",
        default_text="сохранение результата",
    ),
    "FreeU_V2": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.freeu",
        default_text="применение FreeU",
    ),
    "PerturbedAttentionGuidance": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.pag",
        default_text="применение PAG",
    ),
    "LatentUpscale": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.latent_upscale",
        default_text="Hi-res: апскейл латента",
    ),
    "HyperTile": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.hypertile",
        default_text="настройка HyperTile",
    ),
    "VAEDecodeTiled": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.vae_decode_tiled",
        default_text="тайловое декодирование",
    ),
    "VAEEncodeTiled": WorkflowStageLabel(
        localization_key="comfyui.progress.stage.vae_encode_tiled",
        default_text="тайловое кодирование",
    ),
}


def stage_labels_for(*class_types: str) -> dict[str, WorkflowStageLabel]:
    labels: dict[str, WorkflowStageLabel] = {}
    for class_type in class_types:
        label = _ALL_LABELS.get(class_type)
        if label is not None:
            labels[class_type] = label
    return labels
