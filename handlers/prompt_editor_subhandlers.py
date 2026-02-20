from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, InlineKeyboardMarkup, Message

from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from smart_prompt import SmartPromptService

from .prompt_editor_handlers_edit import (
    PromptEditorEditHandlersDeps,
    register_prompt_editor_edit_handlers,
)
from .prompt_editor_handlers_exchange import (
    PromptEditorExchangeHandlersDeps,
    register_prompt_editor_exchange_handlers,
)
from .prompt_editor_handlers_flow import (
    PromptEditorFlowHandlersDeps,
    register_prompt_editor_flow_handlers,
)
from .prompt_editor_handlers_lora import (
    PromptEditorLoraHandlersDeps,
    register_prompt_editor_lora_handlers,
)
from .prompt_editor_handlers_references import (
    PromptEditorReferenceHandlersDeps,
    register_prompt_editor_reference_handlers,
)
from .prompt_editor_handlers_send import (
    PromptEditorSendHandlersDeps,
    register_prompt_editor_send_handlers,
)
from .prompt_editor_handlers_smart import (
    PromptEditorSmartHandlersDeps,
    register_prompt_editor_smart_handlers,
)
from .prompt_editor_handlers_thematic import (
    PromptEditorThematicHandlersDeps,
    register_prompt_editor_thematic_handlers,
)


@dataclass
class PromptEditorSubhandlersDeps:
    router: Router
    runtime: RuntimeStore
    client: Any
    smart_prompt: SmartPromptService | None
    logger: Any
    smart_prompt_input_max_chars: int
    max_reference_images: int
    sizes: list[tuple[str, int, int]]
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    get_user_pro_mode: Callable[[RuntimeStore, int], bool]
    set_user_pro_mode: Callable[[RuntimeStore, int, bool], None]
    ensure_models: Callable[[Message], Awaitable[bool]]
    default_params_for_user: Callable[[int], GenerationParams]
    open_prompt_request: Callable[..., Awaitable[None]]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    changed_params_count: Callable[[GenerationParams], int]
    run_generate_operation: Callable[..., Awaitable[None]]
    clear_smart_prompt_result_data: Callable[[FSMContext], Awaitable[None]]
    show_smart_prompt_result_confirmation: Callable[..., Awaitable[None]]
    smart_prompt_is_enabled: Callable[[], bool]
    add_reference_from_message: Callable[[Message, list[dict[str, str]]], tuple[int, bool]]
    merge_prompt_text: Callable[[str, str], str]
    prompt_input_text: Callable[..., str]
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    cleanup_user_message: Callable[[Message], Awaitable[None]]
    open_paginated_choice: Callable[..., Awaitable[None]]
    change_paginated_choice_page: Callable[..., Awaitable[None]]
    set_prompt_param_from_callback: Callable[..., Awaitable[None]]
    incompatible_loras: Callable[[GenerationParams], list[tuple[str, str, str]]]
    normalize_params: Callable[[GenerationParams], GenerationParams]
    is_freeu_supported: Callable[[], bool]
    is_pag_supported: Callable[[], bool]
    is_tiled_diffusion_supported: Callable[[], bool]
    show_lora_menu: Callable[..., Awaitable[None]]
    lora_picker_items: Callable[[str], tuple[list[str], list[str]]]
    lora_compatibility: Callable[[str, str], tuple[str, str, str]]
    lora_trained_words: Callable[[str], list[str]]
    merge_prompt_with_words: Callable[[str, list[str]], str]
    list_available_loras: Callable[[], list[str]]
    show_reference_menu: Callable[..., Awaitable[None]]
    make_reference_image: Callable[[str], dict[str, str]]
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    generation_result_keyboard: Callable[[], InlineKeyboardMarkup]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]


def register_prompt_editor_subhandlers(deps: PromptEditorSubhandlersDeps) -> None:
    register_prompt_editor_flow_handlers(
        deps.router,
        PromptEditorFlowHandlersDeps(
            runtime=deps.runtime,
            client=deps.client,
            callback_user_id=deps.callback_user_id,
            message_user_id=deps.message_user_id,
            ensure_models=deps.ensure_models,
            default_params_for_user=deps.default_params_for_user,
            open_prompt_request=deps.open_prompt_request,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            show_prompt_editor=deps.show_prompt_editor,
            changed_params_count=deps.changed_params_count,
            run_generate_operation=deps.run_generate_operation,
        ),
    )

    register_prompt_editor_smart_handlers(
        deps.router,
        PromptEditorSmartHandlersDeps(
            smart_prompt=deps.smart_prompt,
            smart_prompt_input_max_chars=deps.smart_prompt_input_max_chars,
            max_reference_images=deps.max_reference_images,
            logger=deps.logger,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            clear_smart_prompt_result_data=deps.clear_smart_prompt_result_data,
            show_smart_prompt_result_confirmation=deps.show_smart_prompt_result_confirmation,
            show_prompt_editor=deps.show_prompt_editor,
            smart_prompt_is_enabled=deps.smart_prompt_is_enabled,
            add_reference_from_message=deps.add_reference_from_message,
            merge_prompt_text=deps.merge_prompt_text,
            prompt_input_text=deps.prompt_input_text,
            back_keyboard=deps.back_keyboard,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_edit_handlers(
        deps.router,
        PromptEditorEditHandlersDeps(
            runtime=deps.runtime,
            client=deps.client,
            sizes=deps.sizes,
            callback_user_id=deps.callback_user_id,
            get_user_pro_mode=deps.get_user_pro_mode,
            set_user_pro_mode=deps.set_user_pro_mode,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            open_paginated_choice=deps.open_paginated_choice,
            change_paginated_choice_page=deps.change_paginated_choice_page,
            set_prompt_param_from_callback=deps.set_prompt_param_from_callback,
            show_prompt_editor=deps.show_prompt_editor,
            prompt_input_text=deps.prompt_input_text,
            back_keyboard=deps.back_keyboard,
            incompatible_loras=deps.incompatible_loras,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_exchange_handlers(
        deps.router,
        PromptEditorExchangeHandlersDeps(
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            show_prompt_editor=deps.show_prompt_editor,
            normalize_params=deps.normalize_params,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_thematic_handlers(
        deps.router,
        PromptEditorThematicHandlersDeps(
            max_reference_images=deps.max_reference_images,
            is_freeu_supported=deps.is_freeu_supported,
            is_pag_supported=deps.is_pag_supported,
            is_tiled_diffusion_supported=deps.is_tiled_diffusion_supported,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            show_prompt_editor=deps.show_prompt_editor,
            back_keyboard=deps.back_keyboard,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_lora_handlers(
        deps.router,
        PromptEditorLoraHandlersDeps(
            runtime=deps.runtime,
            callback_user_id=deps.callback_user_id,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            show_lora_menu=deps.show_lora_menu,
            lora_picker_items=deps.lora_picker_items,
            lora_compatibility=deps.lora_compatibility,
            lora_trained_words=deps.lora_trained_words,
            merge_prompt_with_words=deps.merge_prompt_with_words,
            open_paginated_choice=deps.open_paginated_choice,
            change_paginated_choice_page=deps.change_paginated_choice_page,
            back_keyboard=deps.back_keyboard,
            list_available_loras=deps.list_available_loras,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_reference_handlers(
        deps.router,
        PromptEditorReferenceHandlersDeps(
            runtime=deps.runtime,
            callback_user_id=deps.callback_user_id,
            max_reference_images=deps.max_reference_images,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            require_prompt_request_for_message=deps.require_prompt_request_for_message,
            show_reference_menu=deps.show_reference_menu,
            make_reference_image=deps.make_reference_image,
            cleanup_user_message=deps.cleanup_user_message,
        ),
    )

    register_prompt_editor_send_handlers(
        deps.router,
        PromptEditorSendHandlersDeps(
            logger=deps.logger,
            runtime=deps.runtime,
            client=deps.client,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
            show_prompt_editor=deps.show_prompt_editor,
            deliver_generated_images=deps.deliver_generated_images,
            generation_result_keyboard=deps.generation_result_keyboard,
            preview_image_keyboard=deps.preview_image_keyboard,
        ),
    )
