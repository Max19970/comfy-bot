from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, Message

from core.html_utils import h
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService


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
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]


def _resolved_locale(
    deps: PromptEditorViewDeps,
    uid: int,
    *,
    telegram_locale: str | None,
) -> str:
    prefs = deps.runtime.user_preferences.get(uid, {})
    selected_locale = read_user_locale(
        prefs,
        default_locale=deps.localization.default_locale(),
    )
    return deps.resolve_user_locale(
        user_locale=selected_locale,
        telegram_locale=telegram_locale,
    )


def _t(
    deps: PromptEditorViewDeps,
    uid: int,
    key: str,
    default: str,
    *,
    telegram_locale: str | None,
    params: Mapping[str, object] | None = None,
) -> str:
    locale = _resolved_locale(deps, uid, telegram_locale=telegram_locale)
    return deps.localization.t(key, locale=locale, params=params, default=default)


def build_prompt_editor_text(
    req: PromptRequest,
    *,
    uid: int,
    telegram_locale: str | None,
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
        lines.append(
            _t(
                deps,
                uid,
                "prompt_editor.view.smart_prompt.enabled",
                "\n🧠 <b>Smart Prompt (TIPO):</b> on.",
                telegram_locale=telegram_locale,
            )
        )

    ckpt_base = deps.checkpoint_base_model(req.params.checkpoint)
    if ckpt_base:
        lines.append(
            _t(
                deps,
                uid,
                "prompt_editor.view.checkpoint_base",
                "\n🧬 <b>Checkpoint base:</b> <code>{base}</code>",
                telegram_locale=telegram_locale,
                params={"base": h(ckpt_base)},
            )
        )

    bad_loras = deps.incompatible_loras(req.params)
    if bad_loras:
        names = ", ".join(name for name, _, _ in bad_loras[:3])
        suffix = ""
        if len(bad_loras) > 3:
            suffix = _t(
                deps,
                uid,
                "prompt_editor.view.lora.more_suffix",
                " and {count} more",
                telegram_locale=telegram_locale,
                params={"count": len(bad_loras) - 3},
            )
        lines.append(
            _t(
                deps,
                uid,
                "prompt_editor.view.lora.incompatibility",
                "⚠️ <b>LoRA compatibility:</b> {count} potentially incompatible ({names}{suffix})",
                telegram_locale=telegram_locale,
                params={
                    "count": len(bad_loras),
                    "names": h(names),
                    "suffix": h(suffix),
                },
            )
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
        await message.answer(
            _t(
                deps,
                uid,
                "prompt_editor.view.error.active_request_not_found",
                "❌ Active request not found. Use /generate.",
                telegram_locale=message.from_user.language_code if message.from_user else None,
            )
        )
        return

    req.params = deps.normalize_params(req.params)
    req.sync_editor_loras_from_params()
    await state.set_state(PromptEditorStates.editing)
    pro_mode = deps.get_user_pro_mode(deps.runtime, uid)
    telegram_locale = message.from_user.language_code if message.from_user else None
    locale = _resolved_locale(deps, uid, telegram_locale=telegram_locale)

    def _ui_translate(key: str, target_locale: str | None, default: str) -> str:
        return deps.localization.t(key, locale=target_locale or locale, default=default)

    text = build_prompt_editor_text(
        req,
        uid=uid,
        telegram_locale=telegram_locale,
        notice=notice,
        pro_mode=pro_mode,
        deps=deps,
    )
    kb = deps.editor_keyboard(
        req,
        smart_prompt_enabled=deps.smart_prompt_is_enabled(),
        pro_mode=pro_mode,
        translate=_ui_translate,
        locale=locale,
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
