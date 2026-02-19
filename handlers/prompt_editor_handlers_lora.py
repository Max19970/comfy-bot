from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, cast

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from core.html_utils import h
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui import custom_btn


@dataclass
class PromptEditorLoraHandlersDeps:
    runtime: RuntimeStore
    callback_user_id: Callable[[CallbackQuery], int]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_lora_menu: Callable[..., Awaitable[None]]
    lora_picker_items: Callable[[str], tuple[list[str], list[str]]]
    lora_compatibility: Callable[[str, str], tuple[str, str, str]]
    lora_trained_words: Callable[[str], list[str]]
    merge_prompt_with_words: Callable[[str, list[str]], str]
    open_paginated_choice: Callable[..., Awaitable[None]]
    change_paginated_choice_page: Callable[..., Awaitable[None]]
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    list_available_loras: Callable[[], list[str]]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def register_prompt_editor_lora_handlers(
    router: Router,
    deps: PromptEditorLoraHandlersDeps,
) -> None:
    async def offer_lora_trigger_prompt(
        message: Message,
        state: FSMContext,
        uid: int,
        req: PromptRequest,
        *,
        lora_name: str,
        edit: bool,
    ) -> None:
        words = deps.lora_trained_words(lora_name)
        if not words:
            await deps.show_lora_menu(
                message,
                state,
                uid,
                edit=edit,
                notice="✅ LoRA добавлена.",
            )
            return

        words_to_offer = [
            word
            for word in words
            if word.casefold() not in req.params.positive.casefold()
        ]
        if not words_to_offer:
            await deps.show_lora_menu(
                message,
                state,
                uid,
                edit=edit,
                notice="✅ LoRA добавлена. Триггер-слова уже присутствуют в prompt.",
            )
            return

        await state.update_data(
            pe_pending_trigger_words=words_to_offer,
            pe_pending_trigger_lora=lora_name,
        )
        await state.set_state(PromptEditorStates.confirming_lora_triggers)

        preview = ", ".join(words_to_offer[:8])
        if len(words_to_offer) > 8:
            preview += "..."

        text = (
            f"✅ LoRA <code>{h(lora_name)}</code> добавлена.\n"
            f"🔑 Триггер-слова: <code>{h(preview)}</code>\n"
            "Добавить их в Positive prompt?"
        )
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="➕ Добавить",
                        callback_data="pe:lora:trg:add",
                    ),
                    InlineKeyboardButton(
                        text="Пропустить",
                        callback_data="pe:lora:trg:skip",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="⬅️ К LoRA меню",
                        callback_data="pe:lora:trg:skip",
                    )
                ],
            ]
        )
        sent: Message | None = None
        if req.ui_chat_id is not None and req.ui_message_id is not None and message.bot:
            try:
                edited = await message.bot.edit_message_text(
                    text=text,
                    chat_id=req.ui_chat_id,
                    message_id=req.ui_message_id,
                    reply_markup=kb,
                )
                if isinstance(edited, Message):
                    req.ui_chat_id = edited.chat.id
                    req.ui_message_id = edited.message_id
                    deps.runtime.user_ui_panels[uid] = {
                        "chat_id": edited.chat.id,
                        "message_id": edited.message_id,
                    }
                    return
            except Exception:
                pass

        if edit:
            try:
                edited = await message.edit_text(text, reply_markup=kb)
                if isinstance(edited, Message):
                    req.ui_chat_id = edited.chat.id
                    req.ui_message_id = edited.message_id
                    deps.runtime.user_ui_panels[uid] = {
                        "chat_id": edited.chat.id,
                        "message_id": edited.message_id,
                    }
                    return
            except Exception:
                pass

        sent = await message.answer(text, reply_markup=kb)
        req.ui_chat_id = sent.chat.id
        req.ui_message_id = sent.message_id
        deps.runtime.user_ui_panels[uid] = {
            "chat_id": sent.chat.id,
            "message_id": sent.message_id,
        }

    @router.callback_query(F.data == "pe:edit:lora")
    async def pe_edit_lora(cb: CallbackQuery, state: FSMContext):
        uid = deps.callback_user_id(cb)
        await deps.show_lora_menu(cb.message, state, uid, edit=True)
        await cb.answer()

    @router.callback_query(F.data == "pe:lora:add")
    async def pe_lora_add(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        names, labels = deps.lora_picker_items(req.params.checkpoint)
        if not names:
            await cb.answer("На сервере нет доступных LoRA.", show_alert=True)
            return

        await state.update_data(pe_lora_pick_names=names, pe_lora_pick_labels=labels)
        await deps.open_paginated_choice(
            cb,
            title="Выберите LoRA для добавления:",
            items=labels,
            prefix="pe_lora_pick",
            back_callback="pe:edit:lora",
        )

    @router.callback_query(F.data.startswith("pe_lora_pick_page:"))
    async def pe_lora_pick_page(cb: CallbackQuery, state: FSMContext):
        data = await state.get_data()
        labels = data.get("pe_lora_pick_labels")
        if not isinstance(labels, list) or not labels:
            labels = deps.list_available_loras()
        await deps.change_paginated_choice_page(
            cb,
            items=labels,
            prefix="pe_lora_pick",
            back_callback="pe:edit:lora",
        )

    @router.callback_query(F.data.startswith("pe_lora_pick:"))
    async def pe_lora_pick(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload

        idx = int(cb.data.split(":", 1)[1])
        data = await state.get_data()
        items = data.get("pe_lora_pick_names")
        if not isinstance(items, list) or not items:
            items = deps.list_available_loras()
        if idx < 0 or idx >= len(items):
            await cb.answer("❌ Неверный индекс.", show_alert=True)
            return

        chosen = items[idx]
        await state.update_data(pe_pending_lora=chosen)
        await state.set_state(PromptEditorStates.choosing_lora_strength)

        status, ckpt_base, lora_base = deps.lora_compatibility(
            req.params.checkpoint, chosen
        )
        compatibility_line = ""
        if status == "compatible" and ckpt_base and lora_base:
            compatibility_line = (
                f"✅ Совместимо: checkpoint <code>{h(ckpt_base)}</code> "
                f"↔ LoRA <code>{h(lora_base)}</code>"
            )
        elif status == "incompatible" and ckpt_base and lora_base:
            compatibility_line = (
                f"⚠️ Возможна несовместимость: checkpoint <code>{h(ckpt_base)}</code> "
                f"↔ LoRA <code>{h(lora_base)}</code>"
            )
        elif lora_base:
            compatibility_line = f"ℹ️ Base LoRA: <code>{h(lora_base)}</code>"

        trigger_words = deps.lora_trained_words(chosen)
        trigger_line = ""
        if trigger_words:
            trigger_line = (
                f"🔑 Триггер-слова: <code>{h(', '.join(trigger_words[:8]))}</code>"
            )

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="0.5", callback_data="pe_lstr:0.5"),
                    InlineKeyboardButton(text="0.7", callback_data="pe_lstr:0.7"),
                    InlineKeyboardButton(text="0.8", callback_data="pe_lstr:0.8"),
                ],
                [
                    InlineKeyboardButton(text="0.9", callback_data="pe_lstr:0.9"),
                    InlineKeyboardButton(text="1.0", callback_data="pe_lstr:1.0"),
                    InlineKeyboardButton(text="1.2", callback_data="pe_lstr:1.2"),
                ],
                custom_btn("pe_lstr:custom"),
                [
                    InlineKeyboardButton(
                        text="⬅️ Назад",
                        callback_data="pe:edit:lora",
                    )
                ],
            ]
        )

        text_lines = [f"LoRA: <code>{h(chosen)}</code>"]
        if compatibility_line:
            text_lines.append(compatibility_line)
        if trigger_line:
            text_lines.append(trigger_line)
        text_lines.append("Выберите strength:")

        await cb.message.edit_text("\n".join(text_lines), reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_lstr:"))
    async def pe_lora_strength(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_lora_strength)
            await cb.message.edit_text(
                "Введите LoRA strength (0.0 - 2.0):",
                reply_markup=deps.back_keyboard(),
            )
            await cb.answer()
            return

        data = await state.get_data()
        lora_name = data.get("pe_pending_lora")
        if not lora_name:
            await cb.answer("LoRA не выбрана.", show_alert=True)
            return

        req.params.loras.append((lora_name, float(value)))
        await state.update_data(pe_pending_lora=None)
        await offer_lora_trigger_prompt(
            cast(Message, cb.message),
            state,
            uid,
            req,
            lora_name=lora_name,
            edit=True,
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_custom_lora_strength, F.text)
    async def pe_custom_lora_strength(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            if not 0.0 <= value <= 2.0:
                raise ValueError
        except ValueError:
            await msg.answer("Некорректное значение (0.0 - 2.0):")
            return

        data = await state.get_data()
        lora_name = data.get("pe_pending_lora")
        if not lora_name:
            await msg.answer("LoRA не выбрана. Начните добавление заново.")
            await deps.show_lora_menu(msg, state, uid, edit=False)
            return

        req.params.loras.append((lora_name, value))
        await state.update_data(pe_pending_lora=None)
        await deps.cleanup_user_message(msg)
        await offer_lora_trigger_prompt(
            msg,
            state,
            uid,
            req,
            lora_name=lora_name,
            edit=False,
        )

    @router.callback_query(
        PromptEditorStates.confirming_lora_triggers,
        F.data.startswith("pe:lora:trg:"),
    )
    async def pe_lora_trigger_choice(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        action = cb.data.split(":")[-1]

        data = await state.get_data()
        words_raw = data.get("pe_pending_trigger_words")
        lora_name = str(data.get("pe_pending_trigger_lora") or "").strip()
        words = [str(word).strip() for word in words_raw or [] if str(word).strip()]

        notice = "✅ LoRA добавлена."
        if action == "add" and words:
            req.params.positive = deps.merge_prompt_with_words(
                req.params.positive, words
            )
            notice = f"✅ Триггер-слова для {lora_name or 'LoRA'} добавлены в Positive prompt."
        elif words:
            notice = f"ℹ️ Триггер-слова для {lora_name or 'LoRA'} не добавлены."

        await state.update_data(
            pe_pending_trigger_words=None,
            pe_pending_trigger_lora=None,
        )
        await deps.show_lora_menu(cb.message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:lora:remove_last")
    async def pe_lora_remove_last(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        notice = "ℹ️ Цепочка LoRA не изменилась."
        if req.params.loras:
            req.params.loras.pop()
            notice = "✅ Последняя LoRA удалена."
        await deps.show_lora_menu(cb.message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:lora:clear")
    async def pe_lora_clear(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        count = len(req.params.loras)
        if count == 0:
            await deps.show_lora_menu(
                cb.message,
                state,
                uid,
                edit=True,
                notice="ℹ️ Цепочка LoRA уже пуста.",
            )
            await cb.answer()
            return

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Удалить",
                        callback_data="pe:lora:clear:yes",
                    ),
                    InlineKeyboardButton(
                        text="❌ Отмена",
                        callback_data="pe:lora:clear:no",
                    ),
                ]
            ]
        )
        await cb.message.edit_text(
            f"⚠️ Удалить {count} LoRA из цепочки?",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:lora:clear:yes")
    async def pe_lora_clear_yes(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        req.params.loras = []
        await deps.show_lora_menu(
            cb.message,
            state,
            uid,
            edit=True,
            notice="✅ Цепочка LoRA очищена.",
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:lora:clear:no")
    async def pe_lora_clear_no(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await deps.show_lora_menu(
            cb.message,
            state,
            uid,
            edit=True,
            notice="↩️ Очистка LoRA отменена.",
        )
        await cb.answer()
