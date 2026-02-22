from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from core.interaction import require_callback_message
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui_kit import build_keyboard
from core.ui_kit.buttons import button

from .prompt_editor_handler_guards import require_message_and_request
from .prompt_editor_selection_utils import parse_value_selection


@dataclass
class PromptEditorReferenceHandlersDeps:
    runtime: RuntimeStore
    callback_user_id: Callable[[CallbackQuery], int]
    max_reference_images: int
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_reference_menu: Callable[..., Awaitable[None]]
    make_reference_image: Callable[[str], dict[str, str]]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def register_prompt_editor_reference_handlers(
    router: Router,
    deps: PromptEditorReferenceHandlersDeps,
) -> None:
    @router.callback_query(F.data == "pe:edit:refs")
    async def pe_edit_refs(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        await deps.show_reference_menu(message, state, uid, edit=True)
        await cb.answer()

    @router.message(PromptEditorStates.editing, F.photo)
    async def pe_add_reference_photo(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        if len(req.params.reference_images) >= deps.max_reference_images:
            await msg.answer(
                f"Достигнут лимит: {deps.max_reference_images} картинок. "
                'Удалите лишние в меню "Картинки".'
            )
            return

        if not msg.photo:
            return
        photo = msg.photo[-1]
        req.params.reference_images.append(deps.make_reference_image(photo.file_id))
        await deps.cleanup_user_message(msg)
        await deps.show_reference_menu(
            msg,
            state,
            uid,
            edit=False,
            notice=(
                f"Картинка добавлена. Теперь: "
                f"{len(req.params.reference_images)}/{deps.max_reference_images}."
            ),
        )

    @router.message(PromptEditorStates.editing, F.document)
    async def pe_add_reference_document(msg: Message, state: FSMContext):
        if not msg.document:
            return

        mime_type = msg.document.mime_type or ""
        if not mime_type.startswith("image/"):
            return

        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return

        uid, req = payload
        if len(req.params.reference_images) >= deps.max_reference_images:
            await msg.answer(
                f"Достигнут лимит: {deps.max_reference_images} картинок. "
                'Удалите лишние в меню "Картинки".'
            )
            return

        req.params.reference_images.append(deps.make_reference_image(msg.document.file_id))
        await deps.cleanup_user_message(msg)
        await deps.show_reference_menu(
            msg,
            state,
            uid,
            edit=False,
            notice=(
                f"Картинка добавлена как файл. Теперь: "
                f"{len(req.params.reference_images)}/{deps.max_reference_images}."
            ),
        )

    @router.callback_query(F.data == "pe:refs:view")
    async def pe_refs_view(cb: CallbackQuery):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, _, req = context
        refs = req.params.reference_images
        if not refs:
            await cb.answer("Список картинок пуст.", show_alert=True)
            return

        await cb.answer()
        await message.answer(
            "👁 <b>Превью референсов</b>\nНажмите кнопку под картинкой для удаления."
        )

        for index, ref in enumerate(refs, start=1):
            file_id = str(ref.get("file_id", "")).strip()
            ref_id = str(ref.get("id", "")).strip()
            if not file_id or not ref_id:
                continue

            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=f"🗑 #{index}",
                            callback_data=f"pe:refs:del:{ref_id}",
                        )
                    ]
                ]
            )
            try:
                await message.answer_photo(
                    photo=file_id,
                    caption=f"Референс #{index}",
                    reply_markup=kb,
                )
            except TelegramBadRequest:
                await message.answer_document(
                    document=file_id,
                    caption=f"Референс #{index}",
                    reply_markup=kb,
                )

    @router.callback_query(F.data == "pe:refs:remove_last")
    async def pe_refs_remove_last(cb: CallbackQuery, state: FSMContext):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, uid, req = context
        notice = "Список уже пуст."
        if req.params.reference_images:
            req.params.reference_images.pop()
            notice = "Последняя картинка удалена."

        await deps.show_reference_menu(message, state, uid, edit=True, notice=notice)
        await cb.answer()

    @router.callback_query(F.data == "pe:refs:clear")
    async def pe_refs_clear(cb: CallbackQuery, state: FSMContext):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, uid, req = context
        count = len(req.params.reference_images)
        if count == 0:
            await deps.show_reference_menu(
                message,
                state,
                uid,
                edit=True,
                notice="ℹ️ Список картинок уже пуст.",
            )
            await cb.answer()
            return

        kb = build_keyboard(
            [[button("✅ Удалить", "pe:refs:clear:yes"), button("❌ Отмена", "pe:refs:clear:no")]]
        )
        await message.edit_text(
            f"⚠️ Удалить все {count} референсов?",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:refs:clear:yes")
    async def pe_refs_clear_yes(cb: CallbackQuery, state: FSMContext):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, uid, req = context
        req.params.reference_images = []
        await deps.show_reference_menu(
            message,
            state,
            uid,
            edit=True,
            notice="🗑 Список картинок очищен.",
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:refs:clear:no")
    async def pe_refs_clear_no(cb: CallbackQuery, state: FSMContext):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, uid, _ = context
        await deps.show_reference_menu(
            message,
            state,
            uid,
            edit=True,
            notice="↩️ Очистка референсов отменена.",
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe:refs:del:"))
    async def pe_refs_del(cb: CallbackQuery):
        context = await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )
        if context is None:
            return

        message, _, req = context
        ref_id = await parse_value_selection(
            cb,
            prefix="pe:refs:del",
            invalid_text="Некорректный идентификатор референса.",
        )
        if ref_id is None:
            return
        if not ref_id:
            await cb.answer("Некорректный идентификатор референса.", show_alert=True)
            return

        before = len(req.params.reference_images)
        req.params.reference_images = [
            item for item in req.params.reference_images if item.get("id") != ref_id
        ]
        after = len(req.params.reference_images)

        if after == before:
            await cb.answer("Картинка уже удалена.", show_alert=True)
            return

        try:
            await message.delete()
        except TelegramBadRequest:
            try:
                await message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass

        await cb.answer(f"Удалено. Осталось: {after}")
