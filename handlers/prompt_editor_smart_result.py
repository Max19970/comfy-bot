from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, Message

from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService


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
    localization: LocalizationService,
    resolve_user_locale: Callable[..., str],
) -> None:
    telegram_locale = message.from_user.language_code if message.from_user else None

    def _resolved_locale() -> str:
        prefs = runtime.user_preferences.get(uid, {})
        selected_locale = read_user_locale(
            prefs,
            default_locale=localization.default_locale(),
        )
        return resolve_user_locale(
            user_locale=selected_locale,
            telegram_locale=telegram_locale,
        )

    def _t(
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        locale = _resolved_locale()
        return localization.t(key, locale=locale, params=params, default=default)

    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await state.clear()
        await message.answer(
            _t(
                "prompt_editor.smart_result.error.active_request_not_found",
                "⚠️ Active request not found. Use /generate.",
            )
        )
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
            notice=_t(
                "prompt_editor.smart_result.error.result_not_found",
                "⚠️ Smart Prompt result not found.",
            ),
        )
        return

    lines: list[str] = [
        _t(
            "prompt_editor.smart_result.title",
            "🧠 <b>Smart Prompt (TIPO) is ready</b>",
        )
    ]
    if notice:
        lines.extend(["", f"💬 <i>{h(notice)}</i>"])

    lines.extend(
        [
            "",
            _t(
                "prompt_editor.smart_result.positive_label",
                "🟢 <b>New Positive:</b>",
            ),
            f"<code>{h(prompt_preview(generated_positive))}</code>",
            "",
            _t(
                "prompt_editor.smart_result.negative_label",
                "🔴 <b>New Negative:</b>",
            ),
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
