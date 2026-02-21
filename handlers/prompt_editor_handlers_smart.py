from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
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


def register_prompt_editor_smart_handlers(
    router: Router,
    deps: PromptEditorSmartHandlersDeps,
) -> None:
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
                else "Сервис умного промпта недоступен."
            )
            await cb.answer(hint, show_alert=True)
            return

        lines = [
            "🧠 <b>Smart Prompt</b>",
            "",
            "Опишите желаемую картинку естественным языком.",
            "TIPO преобразует текст в теговый Positive/Negative.",
            "Можно приложить фото с подписью - фото добавится в референсы.",
            "",
            f"🧪 <b>Checkpoint:</b> <code>{h(req.params.checkpoint or '(не выбран)')}</code>",
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
        if not deps.smart_prompt_is_enabled() or deps.smart_prompt is None:
            await deps.show_prompt_editor(
                msg,
                state,
                uid,
                notice=(
                    deps.smart_prompt.configuration_hint()
                    if deps.smart_prompt is not None
                    else "Сервис умного промпта недоступен."
                ),
            )
            return

        description = (msg.text or msg.caption or "").strip()
        if not description:
            await msg.answer(
                "Отправьте текстовое описание. Можно прикрепить изображение с подписью."
            )
            return

        if len(description) > deps.smart_prompt_input_max_chars:
            await msg.answer(
                f"Описание слишком длинное. Лимит: {deps.smart_prompt_input_max_chars} символов."
            )
            return

        added_refs, refs_limit_hit = deps.add_reference_from_message(
            msg,
            req.params.reference_images,
        )

        status_msg = await msg.answer("🧠 <b>TIPO:</b> оптимизирую в теги…")

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
                notice=f"Умный промпт не выполнен: {exc}",
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
                notice=f"❌ Умный промпт не выполнен: {exc}",
            )
            return

        notice_lines = ["✅ TIPO: теговые промпты готовы к применению."]
        if added_refs:
            notice_lines.append(f"Добавлено референсов из сообщения: {added_refs}.")
        if refs_limit_hit:
            notice_lines.append(
                f"Лимит референсов ({deps.max_reference_images}) достигнут: часть файлов не добавлена."
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
            "Prompting Guide отключён и больше не используется.",
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
        notice = str(data.get("pe_smart_notice") or "✅ TIPO-промпт применён.").strip()

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
        notice = "📝 TIPO-промпт объединён с текущими Positive/Negative."
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
            notice="↩️ Возвращены прежние Positive/Negative.",
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
            notice="ℹ️ Результат TIPO не применён.",
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
            notice="✏️ Результат TIPO отредактирован.",
        )
