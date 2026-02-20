from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    Message,
)

from comfyui_client import ComfyUIClient
from config import Config
from core.models import GenerationParams
from core.runtime import (
    RuntimeStore,
    get_user_pro_mode,
    set_user_pro_mode,
)
from core.telegram import callback_user_id, message_user_id
from core.ui import (
    MAX_REFERENCE_IMAGES,
    SIZES,
    paginated_keyboard,
    params_summary_for_mode,
)
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

from .prompt_editor_generation import (
    PromptEditorGenerationDeps,
)
from .prompt_editor_generation import (
    run_generate_operation as run_generate_operation_impl,
)
from .prompt_editor_lora import (
    checkpoint_base_model as _checkpoint_base_model,
)
from .prompt_editor_lora import (
    incompatible_loras as _incompatible_loras,
)
from .prompt_editor_lora import (
    lora_compatibility as _lora_compatibility,
)
from .prompt_editor_lora import (
    lora_picker_items as _lora_picker_items,
)
from .prompt_editor_lora import (
    lora_trained_words as _lora_trained_words,
)
from .prompt_editor_lora import (
    merge_prompt_with_words as _merge_prompt_with_words,
)
from .prompt_editor_lora import (
    show_lora_menu as _show_lora_menu,
)
from .prompt_editor_ops import (
    change_paginated_choice_page as change_paginated_choice_page_impl,
)
from .prompt_editor_ops import (
    changed_params_count as changed_params_count_impl,
)
from .prompt_editor_ops import (
    open_paginated_choice as open_paginated_choice_impl,
)
from .prompt_editor_ops import (
    set_prompt_param_from_callback as set_prompt_param_from_callback_impl,
)
from .prompt_editor_references import (
    add_reference_from_message as _add_reference_from_message,
)
from .prompt_editor_references import (
    collect_reference_images as _collect_reference_images,
)
from .prompt_editor_references import (
    denoise_from_reference_strength as _denoise_from_reference_strength,
)
from .prompt_editor_references import (
    make_reference_image as _make_reference_image,
)
from .prompt_editor_references import (
    show_reference_menu as _show_reference_menu,
)
from .prompt_editor_send import (
    deliver_generated_images as _deliver_generated_images,
)
from .prompt_editor_send import (
    generation_result_keyboard as _generation_result_keyboard,
)
from .prompt_editor_send import (
    preview_image_keyboard as _preview_image_keyboard,
)
from .prompt_editor_session import (
    build_default_params,
    build_default_params_for_user,
    ensure_models_available,
    move_prompt_panel_to_bottom,
    normalize_generation_params,
    remember_prompt_panel,
    safe_delete_user_message,
    show_prompt_panel,
)
from .prompt_editor_session import (
    require_prompt_request_for_callback as require_prompt_request_for_callback_session,
)
from .prompt_editor_session import (
    require_prompt_request_for_message as require_prompt_request_for_message_session,
)
from .prompt_editor_smart import (
    merge_prompt_text as _merge_prompt_text,
)
from .prompt_editor_smart import (
    prompt_input_text as _prompt_input_text,
)
from .prompt_editor_smart import (
    prompt_preview as _prompt_preview,
)
from .prompt_editor_smart import (
    smart_prompt_result_keyboard as _smart_prompt_result_keyboard,
)
from .prompt_editor_smart_result import (
    clear_smart_prompt_result_data as clear_smart_prompt_result_data_impl,
)
from .prompt_editor_smart_result import (
    show_smart_prompt_result_confirmation as show_smart_prompt_result_confirmation_impl,
)
from .prompt_editor_subhandlers import (
    PromptEditorSubhandlersDeps,
    register_prompt_editor_subhandlers,
)
from .prompt_editor_ui import (
    back_keyboard as _back_keyboard,
)
from .prompt_editor_ui import (
    editor_keyboard as _editor_keyboard,
)
from .prompt_editor_view import (
    PromptEditorViewDeps,
)
from .prompt_editor_view import (
    open_prompt_request as open_prompt_request_impl,
)
from .prompt_editor_view import (
    show_prompt_editor as show_prompt_editor_impl,
)

logger = logging.getLogger(__name__)

SMART_PROMPT_INPUT_MAX_CHARS = 4000


@dataclass
class PromptEditorService:
    ensure_models: Callable[[Message], Awaitable[bool]]
    normalize_params: Callable[[GenerationParams], GenerationParams]
    show_prompt_editor: Callable[..., Awaitable[None]]


def register_prompt_editor_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    smart_prompt: SmartPromptService | None = None,
) -> PromptEditorService:
    _show_prompt_panel = partial(show_prompt_panel, runtime)
    _move_prompt_panel_to_bottom = partial(move_prompt_panel_to_bottom, runtime)

    ensure_models = partial(ensure_models_available, client)
    default_params = partial(build_default_params, cfg)
    default_params_for_user = partial(build_default_params_for_user, cfg, runtime)
    require_prompt_request_for_message = partial(
        require_prompt_request_for_message_session, runtime
    )
    require_prompt_request_for_callback = partial(
        require_prompt_request_for_callback_session, runtime
    )
    cleanup_user_message = safe_delete_user_message
    open_paginated_choice = partial(
        open_paginated_choice_impl, paginated_keyboard=paginated_keyboard
    )
    change_paginated_choice_page = partial(
        change_paginated_choice_page_impl,
        paginated_keyboard=paginated_keyboard,
    )

    async def set_prompt_param_from_callback(
        cb: CallbackQuery,
        state: FSMContext,
        *,
        field: str,
        value: Any,
        notice: str,
    ) -> None:
        await set_prompt_param_from_callback_impl(
            cb,
            state,
            field=field,
            value=value,
            notice=notice,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            show_prompt_editor=show_prompt_editor,
        )

    normalize_params = partial(
        normalize_generation_params,
        client=client,
        max_reference_images=MAX_REFERENCE_IMAGES,
    )
    prompt_input_text = _prompt_input_text
    back_keyboard = _back_keyboard
    merge_prompt_text = _merge_prompt_text
    prompt_preview = _prompt_preview
    smart_prompt_result_keyboard = _smart_prompt_result_keyboard
    make_reference_image = _make_reference_image
    collect_reference_images = _collect_reference_images
    denoise_from_reference_strength = _denoise_from_reference_strength

    def smart_prompt_is_enabled() -> bool:
        return bool(smart_prompt and smart_prompt.is_enabled())

    checkpoint_base_model = partial(_checkpoint_base_model, downloader=downloader)
    lora_trained_words = partial(_lora_trained_words, downloader=downloader)
    lora_compatibility = partial(_lora_compatibility, downloader=downloader)
    incompatible_loras = partial(_incompatible_loras, downloader=downloader)
    lora_picker_items = partial(_lora_picker_items, client=client, downloader=downloader)
    merge_prompt_with_words = _merge_prompt_with_words
    changed_params_count = partial(
        changed_params_count_impl,
        normalize_params=normalize_params,
        default_params=default_params,
    )
    clear_smart_prompt_result_data = clear_smart_prompt_result_data_impl

    add_reference_from_message = _add_reference_from_message
    editor_keyboard = _editor_keyboard

    view_deps = PromptEditorViewDeps(
        runtime=runtime,
        normalize_params=normalize_params,
        get_user_pro_mode=get_user_pro_mode,
        params_summary_for_mode=params_summary_for_mode,
        smart_prompt_is_enabled=smart_prompt_is_enabled,
        checkpoint_base_model=checkpoint_base_model,
        incompatible_loras=incompatible_loras,
        editor_keyboard=editor_keyboard,
        show_prompt_panel=_show_prompt_panel,
    )

    show_prompt_editor = partial(show_prompt_editor_impl, deps=view_deps)
    open_prompt_request = partial(
        open_prompt_request_impl,
        deps=view_deps,
        show_prompt_editor_fn=show_prompt_editor,
    )

    show_smart_prompt_result_confirmation = partial(
        show_smart_prompt_result_confirmation_impl,
        runtime=runtime,
        show_prompt_editor=show_prompt_editor,
        prompt_preview=prompt_preview,
        smart_prompt_result_keyboard=smart_prompt_result_keyboard,
        show_prompt_panel=_show_prompt_panel,
    )

    generation_result_keyboard = _generation_result_keyboard
    preview_image_keyboard = _preview_image_keyboard
    deliver_generated_images = _deliver_generated_images

    generation_deps = PromptEditorGenerationDeps(
        runtime=runtime,
        logger=logger,
        client=client,
        normalize_params=normalize_params,
        incompatible_loras=incompatible_loras,
        collect_reference_images=collect_reference_images,
        denoise_from_reference_strength=denoise_from_reference_strength,
        show_prompt_editor=show_prompt_editor,
        show_prompt_panel=_show_prompt_panel,
        move_prompt_panel_to_bottom=_move_prompt_panel_to_bottom,
        remember_prompt_panel=partial(remember_prompt_panel, runtime),
        generation_result_keyboard=generation_result_keyboard,
        preview_image_keyboard=preview_image_keyboard,
        deliver_generated_images=deliver_generated_images,
        prune_preview_artifacts=runtime.prune_preview_artifacts,
    )

    run_generate_operation = partial(run_generate_operation_impl, deps=generation_deps)

    show_lora_menu = partial(_show_lora_menu, runtime=runtime, client=client, downloader=downloader)
    show_reference_menu = partial(_show_reference_menu, runtime=runtime, client=client)

    register_prompt_editor_subhandlers(
        PromptEditorSubhandlersDeps(
            router=router,
            runtime=runtime,
            client=client,
            smart_prompt=smart_prompt,
            logger=logger,
            smart_prompt_input_max_chars=SMART_PROMPT_INPUT_MAX_CHARS,
            max_reference_images=MAX_REFERENCE_IMAGES,
            sizes=SIZES,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            get_user_pro_mode=get_user_pro_mode,
            set_user_pro_mode=set_user_pro_mode,
            ensure_models=ensure_models,
            default_params_for_user=default_params_for_user,
            open_prompt_request=open_prompt_request,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            show_prompt_editor=show_prompt_editor,
            changed_params_count=changed_params_count,
            run_generate_operation=run_generate_operation,
            clear_smart_prompt_result_data=clear_smart_prompt_result_data,
            show_smart_prompt_result_confirmation=show_smart_prompt_result_confirmation,
            smart_prompt_is_enabled=smart_prompt_is_enabled,
            add_reference_from_message=add_reference_from_message,
            merge_prompt_text=merge_prompt_text,
            prompt_input_text=prompt_input_text,
            back_keyboard=back_keyboard,
            cleanup_user_message=cleanup_user_message,
            open_paginated_choice=open_paginated_choice,
            change_paginated_choice_page=change_paginated_choice_page,
            set_prompt_param_from_callback=set_prompt_param_from_callback,
            incompatible_loras=incompatible_loras,
            normalize_params=normalize_params,
            is_freeu_supported=lambda: client.info.freeu_supported,
            is_pag_supported=lambda: client.info.pag_supported,
            is_tiled_diffusion_supported=lambda: client.info.tiled_diffusion_supported,
            show_lora_menu=show_lora_menu,
            lora_picker_items=lora_picker_items,
            lora_compatibility=lora_compatibility,
            lora_trained_words=lora_trained_words,
            merge_prompt_with_words=merge_prompt_with_words,
            list_available_loras=lambda: list(client.info.loras),
            show_reference_menu=show_reference_menu,
            make_reference_image=make_reference_image,
            deliver_generated_images=deliver_generated_images,
            generation_result_keyboard=generation_result_keyboard,
            preview_image_keyboard=preview_image_keyboard,
        )
    )

    return PromptEditorService(
        ensure_models=ensure_models,
        normalize_params=normalize_params,
        show_prompt_editor=show_prompt_editor,
    )
