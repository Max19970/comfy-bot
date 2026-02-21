from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, Message

from core.html_utils import h
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates


@dataclass
class PromptEditorViewDeps:
    runtime: RuntimeStore
    normalize_params: Callable[[GenerationParams], GenerationParams]
    get_user_pro_mode: Callable[[RuntimeStore, int], bool]
    params_summary_for_mode: Callable[..., str]
    smart_prompt_is_enabled: Callable[[], bool]
    checkpoint_base_model: Callable[[str], str]
    incompatible_loras: Callable[[GenerationParams], list[tuple[str, str, str]]]
    editor_keyboard: Callable[..., InlineKeyboardMarkup]
    show_prompt_panel: Callable[..., Awaitable[Message]]


def build_prompt_editor_text(
    req: PromptRequest,
    *,
    notice: str,
    pro_mode: bool,
    deps: PromptEditorViewDeps,
) -> str:
    lines: list[str] = []
    if notice:
        lines.append(f"💬 <i>{h(notice)}</i>")
        lines.append("")
    lines.append(deps.params_summary_for_mode(req.params, pro_mode=pro_mode))

    if deps.smart_prompt_is_enabled():
        lines.append("\n🧠 <b>Smart Prompt (TIPO):</b> вкл.")

    ckpt_base = deps.checkpoint_base_model(req.params.checkpoint)
    if ckpt_base:
        lines.append(f"\n🧬 <b>Checkpoint base:</b> <code>{h(ckpt_base)}</code>")

    bad_loras = deps.incompatible_loras(req.params)
    if bad_loras:
        names = ", ".join(name for name, _, _ in bad_loras[:3])
        suffix = "" if len(bad_loras) <= 3 else f" и ещё {len(bad_loras) - 3}"
        lines.append(
            f"⚠️ <b>LoRA compatibility:</b> {len(bad_loras)} потенциально несовместимы ({h(names)}{h(suffix)})"
        )

    return "\n".join(lines)


async def show_prompt_editor(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    edit: bool = False,
    notice: str = "",
    deps: PromptEditorViewDeps,
) -> None:
    req = deps.runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer("❌ Активный запрос не найден. Используйте /generate.")
        return

    req.params = deps.normalize_params(req.params)
    await state.set_state(PromptEditorStates.editing)
    pro_mode = deps.get_user_pro_mode(deps.runtime, uid)
    text = build_prompt_editor_text(req, notice=notice, pro_mode=pro_mode, deps=deps)
    kb = deps.editor_keyboard(
        req,
        smart_prompt_enabled=deps.smart_prompt_is_enabled(),
        pro_mode=pro_mode,
    )
    await deps.show_prompt_panel(
        message,
        req,
        text,
        kb,
        prefer_edit=edit,
    )


async def open_prompt_request(
    message: Message,
    state: FSMContext,
    uid: int,
    params: GenerationParams,
    *,
    operation: str,
    notice: str,
    deps: PromptEditorViewDeps,
    show_prompt_editor_fn: Callable[..., Awaitable[None]],
) -> None:
    deps.runtime.active_prompt_requests[uid] = PromptRequest(
        params=deps.normalize_params(params),
        operation=operation,
    )
    await state.clear()
    await show_prompt_editor_fn(message, state, uid, edit=False, notice=notice)
