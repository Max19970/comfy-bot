from __future__ import annotations

from dataclasses import dataclass

from aiogram import Router

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
    flow: PromptEditorFlowHandlersDeps
    smart: PromptEditorSmartHandlersDeps
    edit: PromptEditorEditHandlersDeps
    exchange: PromptEditorExchangeHandlersDeps
    thematic: PromptEditorThematicHandlersDeps
    lora: PromptEditorLoraHandlersDeps
    references: PromptEditorReferenceHandlersDeps
    send: PromptEditorSendHandlersDeps


def register_prompt_editor_subhandlers(deps: PromptEditorSubhandlersDeps) -> None:
    register_prompt_editor_flow_handlers(deps.router, deps.flow)
    register_prompt_editor_smart_handlers(deps.router, deps.smart)
    register_prompt_editor_edit_handlers(deps.router, deps.edit)
    register_prompt_editor_exchange_handlers(deps.router, deps.exchange)
    register_prompt_editor_thematic_handlers(deps.router, deps.thematic)
    register_prompt_editor_lora_handlers(deps.router, deps.lora)
    register_prompt_editor_reference_handlers(deps.router, deps.references)
    register_prompt_editor_send_handlers(deps.router, deps.send)
