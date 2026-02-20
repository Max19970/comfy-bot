from __future__ import annotations

from collections.abc import Awaitable, Callable

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, Message

from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates


async def clear_smart_prompt_result_data(state: FSMContext) -> None:
    await state.update_data(
        pe_smart_prev_positive=None,
        pe_smart_prev_negative=None,
        pe_smart_generated_positive=None,
        pe_smart_generated_negative=None,
        pe_smart_notice=None,
    )


async def show_smart_prompt_result_confirmation(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    edit: bool,
    notice: str,
    runtime: RuntimeStore,
    show_prompt_editor: Callable[..., Awaitable[None]],
    prompt_preview: Callable[[str], str],
    smart_prompt_result_keyboard: Callable[[], InlineKeyboardMarkup],
    show_prompt_panel: Callable[..., Awaitable[Message]],
) -> None:
    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await state.clear()
        await message.answer("⚠️ Активный запрос не найден. Используйте /generate.")
        return

    data = await state.get_data()
    generated_positive = str(data.get("pe_smart_generated_positive") or "").strip()
    generated_negative = str(data.get("pe_smart_generated_negative") or "").strip()

    if not generated_positive and not generated_negative:
        await show_prompt_editor(
            message,
            state,
            uid,
            edit=edit,
            notice="⚠️ Результат Smart Prompt не найден.",
        )
        return

    lines: list[str] = ["🧠 <b>Smart Prompt (TIPO) готов</b>"]
    if notice:
        lines.extend(["", f"💬 <i>{h(notice)}</i>"])

    lines.extend(
        [
            "",
            "🟢 <b>Новый Positive:</b>",
            f"<code>{h(prompt_preview(generated_positive))}</code>",
            "",
            "🔴 <b>Новый Negative:</b>",
            f"<code>{h(prompt_preview(generated_negative))}</code>",
        ]
    )

    await state.set_state(PromptEditorStates.confirming_smart_prompt_result)
    text = "\n".join(lines)
    kb = smart_prompt_result_keyboard()
    await show_prompt_panel(
        message,
        req,
        text,
        kb,
        prefer_edit=edit,
    )
