from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class GenStates(StatesGroup):
    generating = State()


class PresetStates(StatesGroup):
    entering_name = State()
    confirming_overwrite = State()
    browsing = State()


class DownloadStates(StatesGroup):
    choosing_type = State()
    choosing_source = State()
    choosing_filters = State()
    entering_author = State()
    entering_query = State()
    choosing_result = State()
    choosing_version = State()
    confirming_download = State()
    downloading = State()


class DeleteModelStates(StatesGroup):
    choosing_type = State()
    choosing_file = State()
    confirming_delete = State()


class ServiceSettingsStates(StatesGroup):
    entering_generation_value = State()
    entering_download_author = State()


class PromptEditorStates(StatesGroup):
    editing = State()
    entering_positive = State()
    entering_negative = State()
    entering_shared_prompt = State()
    entering_smart_prompt = State()
    confirming_smart_prompt_result = State()
    entering_smart_result_positive = State()
    entering_smart_result_negative = State()
    entering_custom_size = State()
    entering_custom_steps = State()
    entering_custom_cfg = State()
    entering_custom_denoise = State()
    entering_custom_reference_strength = State()
    entering_custom_seed = State()
    entering_custom_batch = State()
    choosing_lora_strength = State()
    confirming_lora_triggers = State()
    entering_custom_lora_strength = State()
    entering_preset_name = State()
    entering_custom_hires_scale = State()
    entering_custom_hires_denoise = State()
    entering_custom_pag_scale = State()
    entering_custom_controlnet_strength = State()
    entering_custom_tile_size = State()
    entering_custom_vae_tile_size = State()
    entering_custom_tile_overlap = State()
