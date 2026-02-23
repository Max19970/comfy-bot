from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardMarkup,
    Message,
)

from core.html_utils import h
from core.interaction import require_callback_message
from core.runtime import PromptRequest
from core.states import PromptEditorStates
from domain.localization import LocalizationService
from smart_prompt import SmartPromptError, SmartPromptService


@dataclass
class PromptEditorSmartHandlersDeps:
    smart_prompt: SmartPromptService | None
    smart_prompt_input_max_chars: int
    max_reference_images: int
    logger: logging.Logger
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    clear_smart_prompt_result_data: Callable[[FSMContext], Awaitable[None]]
    show_smart_prompt_result_confirmation: Callable[..., Awaitable[None]]
    show_prompt_editor: Callable[..., Awaitable[None]]
    smart_prompt_is_enabled: Callable[[], bool]
    add_reference_from_message: Callable[[Message, list[dict[str, str]]], tuple[int, bool]]
    merge_prompt_text: Callable[[str, str], str]
    prompt_input_text: Callable[..., str]
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    cleanup_user_message: Callable[[Message], Awaitable[None]]
    localization: LocalizationService | None = None
    resolve_user_locale: Callable[..., str] | None = None


def register_prompt_editor_smart_handlers(
    router: Router,
    deps: PromptEditorSmartHandlersDeps,
) -> None:
    def _resolved_locale(uid: int, *, telegram_locale: str | None) -> str | None:
        _ = uid
        if deps.localization is None:
            return None
        selected_locale = deps.localization.default_locale()
        if deps.resolve_user_locale is None:
            return selected_locale
        return deps.resolve_user_locale(
            user_locale=selected_locale,
            telegram_locale=telegram_locale,
        )

    def _t(
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: Mapping[str, object] | None = None,
    ) -> str:
        if deps.localization is None:
            text = default
        else:
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            text = deps.localization.t(key, locale=locale, params=params, default=default)
        if params:
            try:
                return text.format(**params)
            except Exception:
                return text
        return text

    def _t_cb(
        cb: CallbackQuery,
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        uid = cb.from_user.id if cb.from_user else 0
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        return _t(uid, key, default, telegram_locale=telegram_locale, params=params)

    @router.callback_query(F.data == "pe:smart:start")
    async def pe_smart_prompt_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        await deps.clear_smart_prompt_result_data(state)
        if not deps.smart_prompt_is_enabled() or deps.smart_prompt is None:
            hint = (
                deps.smart_prompt.configuration_hint()
                if deps.smart_prompt is not None
                else _t_cb(
                    cb,
                    "prompt_editor.smart.error.service_unavailable",
                    "Сервис умного промпта недоступен.",
                )
            )
            await cb.answer(hint, show_alert=True)
            return

        lines = [
            _t_cb(cb, "prompt_editor.smart.start.title", "🧠 <b>Smart Prompt</b>"),
            "",
            _t_cb(
                cb,
                "prompt_editor.smart.start.description_line1",
                "Опишите желаемую картинку естественным языком.",
            ),
            _t_cb(
                cb,
                "prompt_editor.smart.start.description_line2",
                "TIPO преобразует текст в теговый Positive/Negative.",
            ),
            _t_cb(
                cb,
                "prompt_editor.smart.start.description_line3",
                "Можно приложить фото с подписью - фото добавится в референсы.",
            ),
            "",
            _t_cb(
                cb,
                "prompt_editor.smart.start.checkpoint",
                "🧪 <b>Checkpoint:</b> <code>{checkpoint}</code>",
                params={
                    "checkpoint": h(
                        req.params.checkpoint
                        or _t_cb(
                            cb,
                            "prompt_editor.smart.start.checkpoint_not_selected",
                            "(не выбран)",
                        )
                    )
                },
            ),
        ]
        await state.set_state(PromptEditorStates.entering_smart_prompt)
        await message.edit_text("\n".join(lines), reply_markup=deps.back_keyboard("pe:back"))
        await cb.answer()

    @router.message(PromptEditorStates.entering_smart_prompt)
    async def pe_smart_prompt_entered(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        if not deps.smart_prompt_is_enabled() or deps.smart_prompt is None:
            await deps.show_prompt_editor(
                msg,
                state,
                uid,
                notice=(
                    deps.smart_prompt.configuration_hint()
                    if deps.smart_prompt is not None
                    else _t(
                        uid,
                        "prompt_editor.smart.error.service_unavailable",
                        "Сервис умного промпта недоступен.",
                        telegram_locale=telegram_locale,
                    )
                ),
            )
            return

        description = (msg.text or msg.caption or "").strip()
        if not description:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.smart.input.description_required",
                    "Отправьте текстовое описание. Можно прикрепить изображение с подписью.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        if len(description) > deps.smart_prompt_input_max_chars:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.smart.input.description_too_long",
                    "Описание слишком длинное. Лимит: {limit} символов.",
                    telegram_locale=telegram_locale,
                    params={"limit": deps.smart_prompt_input_max_chars},
                )
            )
            return

        added_refs, refs_limit_hit = deps.add_reference_from_message(
            msg,
            req.params.reference_images,
        )

        status_msg = await msg.answer(
            _t(
                uid,
                "prompt_editor.smart.status.optimizing",
                "🧠 <b>TIPO:</b> оптимизирую в теги…",
                telegram_locale=telegram_locale,
            )
        )

        try:
            result = await deps.smart_prompt.generate_prompts(
                description=description,
                checkpoint=req.params.checkpoint,
            )
        except SmartPromptError as exc:
            try:
                await status_msg.delete()
            except TelegramBadRequest:
                pass
            await deps.show_prompt_editor(
                msg,
                state,
                uid,
                notice=_t(
                    uid,
                    "prompt_editor.smart.error.execution_failed",
                    "Умный промпт не выполнен: {error}",
                    telegram_locale=telegram_locale,
                    params={"error": exc},
                ),
            )
            return
        except (RuntimeError, ValueError, OSError, asyncio.TimeoutError) as exc:
            deps.logger.exception("Smart prompt failed")
            try:
                await status_msg.delete()
            except TelegramBadRequest:
                pass
            await deps.show_prompt_editor(
                msg,
                state,
                uid,
                notice=_t(
                    uid,
                    "prompt_editor.smart.error.runtime_failed",
                    "❌ Умный промпт не выполнен: {error}",
                    telegram_locale=telegram_locale,
                    params={"error": exc},
                ),
            )
            return

        notice_lines = [
            _t(
                uid,
                "prompt_editor.smart.notice.ready",
                "✅ TIPO: теговые промпты готовы к применению.",
                telegram_locale=telegram_locale,
            )
        ]
        if added_refs:
            notice_lines.append(
                _t(
                    uid,
                    "prompt_editor.smart.notice.refs_added",
                    "Добавлено референсов из сообщения: {count}.",
                    telegram_locale=telegram_locale,
                    params={"count": added_refs},
                )
            )
        if refs_limit_hit:
            notice_lines.append(
                _t(
                    uid,
                    "prompt_editor.smart.notice.refs_limit_hit",
                    "Лимит референсов ({limit}) достигнут: часть файлов не добавлена.",
                    telegram_locale=telegram_locale,
                    params={"limit": deps.max_reference_images},
                )
            )

        await state.update_data(
            pe_smart_prev_positive=req.params.positive,
            pe_smart_prev_negative=req.params.negative,
            pe_smart_generated_positive=result.positive.strip(),
            pe_smart_generated_negative=result.negative.strip(),
            pe_smart_notice="\n".join(notice_lines),
        )
        await deps.cleanup_user_message(msg)
        await deps.show_smart_prompt_result_confirmation(
            status_msg,
            state,
            uid,
            edit=True,
            notice="\n".join(notice_lines),
        )

    @router.callback_query(F.data == "pe:smart:guide")
    async def pe_smart_guide_disabled(cb: CallbackQuery):
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.smart.guide.disabled",
                "Prompting Guide отключён и больше не используется.",
            ),
            show_alert=True,
        )

    @router.callback_query(F.data == "pe:smart:apply")
    async def pe_smart_apply(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        data = await state.get_data()
        req.params.positive = str(data.get("pe_smart_generated_positive") or "").strip()
        req.params.negative = str(data.get("pe_smart_generated_negative") or "").strip()
        notice = str(
            data.get("pe_smart_notice")
            or _t_cb(
                cb,
                "prompt_editor.smart.notice.applied",
                "✅ TIPO-промпт применён.",
            )
        ).strip()

        await deps.clear_smart_prompt_result_data(state)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:smart:merge")
    async def pe_smart_merge(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        data = await state.get_data()
        generated_positive = str(data.get("pe_smart_generated_positive") or "").strip()
        generated_negative = str(data.get("pe_smart_generated_negative") or "").strip()

        req.params.positive = deps.merge_prompt_text(req.params.positive, generated_positive)
        req.params.negative = deps.merge_prompt_text(req.params.negative, generated_negative)

        meta_notice = str(data.get("pe_smart_notice") or "").strip()
        notice = _t_cb(
            cb,
            "prompt_editor.smart.notice.merged",
            "📝 TIPO-промпт объединён с текущими Positive/Negative.",
        )
        if meta_notice:
            notice += "\n" + meta_notice

        await deps.clear_smart_prompt_result_data(state)
        await deps.show_prompt_editor(message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:smart:restore")
    async def pe_smart_restore(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        data = await state.get_data()
        req.params.positive = str(data.get("pe_smart_prev_positive") or "")
        req.params.negative = str(data.get("pe_smart_prev_negative") or "")

        await deps.clear_smart_prompt_result_data(state)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t_cb(
                cb,
                "prompt_editor.smart.notice.restored",
                "↩️ Возвращены прежние Positive/Negative.",
            ),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:smart:cancel")
    async def pe_smart_cancel(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await deps.clear_smart_prompt_result_data(state)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t_cb(
                cb,
                "prompt_editor.smart.notice.not_applied",
                "ℹ️ Результат TIPO не применён.",
            ),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:smart:edit")
    async def pe_smart_edit(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, _ = payload
        data = await state.get_data()
        generated_positive = str(data.get("pe_smart_generated_positive") or "").strip()

        await state.set_state(PromptEditorStates.entering_smart_result_positive)
        await message.edit_text(
            deps.prompt_input_text("positive", generated_positive),
            reply_markup=deps.back_keyboard("pe:smart:result:back"),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:smart:result:back")
    async def pe_smart_result_back(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await deps.show_smart_prompt_result_confirmation(
            message,
            state,
            uid,
            edit=True,
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_smart_result_positive, F.text)
    async def pe_smart_result_positive(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        await state.update_data(pe_smart_generated_positive=(msg.text or "").strip())
        data = await state.get_data()
        generated_negative = str(data.get("pe_smart_generated_negative") or "").strip()

        await deps.cleanup_user_message(msg)

        await state.set_state(PromptEditorStates.entering_smart_result_negative)
        await msg.answer(
            deps.prompt_input_text("negative", generated_negative, allow_clear=True),
            reply_markup=deps.back_keyboard("pe:smart:result:back"),
        )

    @router.message(PromptEditorStates.entering_smart_result_negative, F.text)
    async def pe_smart_result_negative(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, _ = payload
        raw = (msg.text or "").strip()
        await state.update_data(pe_smart_generated_negative="" if raw == "-" else raw)
        await deps.cleanup_user_message(msg)
        await deps.show_smart_prompt_result_confirmation(
            msg,
            state,
            uid,
            edit=False,
            notice=_t(
                uid,
                "prompt_editor.smart.notice.edited",
                "✏️ Результат TIPO отредактирован.",
                telegram_locale=msg.from_user.language_code if msg.from_user else None,
            ),
        )
