from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
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
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService

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
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]


def register_prompt_editor_reference_handlers(
    router: Router,
    deps: PromptEditorReferenceHandlersDeps,
) -> None:
    def _resolved_locale(uid: int, *, telegram_locale: str | None) -> str:
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
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: Mapping[str, object] | None = None,
    ) -> str:
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        return deps.localization.t(key, locale=locale, params=params, default=default)

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
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        if len(req.params.reference_images) >= deps.max_reference_images:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.references.limit_reached",
                    'Достигнут лимит: {max_count} картинок. Удалите лишние в меню "Картинки".',
                    telegram_locale=telegram_locale,
                    params={"max_count": deps.max_reference_images},
                )
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
            notice=_t(
                uid,
                "prompt_editor.references.notice.added_photo",
                "Картинка добавлена. Теперь: {count}/{max_count}.",
                telegram_locale=telegram_locale,
                params={
                    "count": len(req.params.reference_images),
                    "max_count": deps.max_reference_images,
                },
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
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        if len(req.params.reference_images) >= deps.max_reference_images:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.references.limit_reached",
                    'Достигнут лимит: {max_count} картинок. Удалите лишние в меню "Картинки".',
                    telegram_locale=telegram_locale,
                    params={"max_count": deps.max_reference_images},
                )
            )
            return

        req.params.reference_images.append(deps.make_reference_image(msg.document.file_id))
        await deps.cleanup_user_message(msg)
        await deps.show_reference_menu(
            msg,
            state,
            uid,
            edit=False,
            notice=_t(
                uid,
                "prompt_editor.references.notice.added_file",
                "Картинка добавлена как файл. Теперь: {count}/{max_count}.",
                telegram_locale=telegram_locale,
                params={
                    "count": len(req.params.reference_images),
                    "max_count": deps.max_reference_images,
                },
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
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.references.alert.empty_list",
                    "Список картинок пуст.",
                ),
                show_alert=True,
            )
            return

        await cb.answer()
        await message.answer(
            _t_cb(
                cb,
                "prompt_editor.references.preview.title",
                "👁 <b>Превью референсов</b>\nНажмите кнопку под картинкой для удаления.",
            )
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
                    caption=_t_cb(
                        cb,
                        "prompt_editor.references.preview.item_caption",
                        "Референс #{index}",
                        params={"index": index},
                    ),
                    reply_markup=kb,
                )
            except TelegramBadRequest:
                await message.answer_document(
                    document=file_id,
                    caption=_t_cb(
                        cb,
                        "prompt_editor.references.preview.item_caption",
                        "Референс #{index}",
                        params={"index": index},
                    ),
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
        notice = _t_cb(cb, "prompt_editor.references.notice.already_empty", "Список уже пуст.")
        if req.params.reference_images:
            req.params.reference_images.pop()
            notice = _t_cb(
                cb,
                "prompt_editor.references.notice.last_removed",
                "Последняя картинка удалена.",
            )

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
                notice=_t_cb(
                    cb,
                    "prompt_editor.references.notice.already_empty_verbose",
                    "ℹ️ Список картинок уже пуст.",
                ),
            )
            await cb.answer()
            return

        kb = build_keyboard(
            [
                [
                    button(
                        _t_cb(
                            cb,
                            "prompt_editor.references.button.clear_confirm",
                            "✅ Удалить",
                        ),
                        "pe:refs:clear:yes",
                    ),
                    button(
                        _t_cb(cb, "common.action.cancel", "❌ Отмена"),
                        "pe:refs:clear:no",
                    ),
                ]
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.references.confirm.clear_all",
                "⚠️ Удалить все {count} референсов?",
                params={"count": count},
            ),
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
            notice=_t_cb(
                cb,
                "prompt_editor.references.notice.cleared",
                "🗑 Список картинок очищен.",
            ),
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
            notice=_t_cb(
                cb,
                "prompt_editor.references.notice.clear_cancelled",
                "↩️ Очистка референсов отменена.",
            ),
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
            invalid_text=_t_cb(
                cb,
                "prompt_editor.references.error.invalid_reference_id",
                "Некорректный идентификатор референса.",
            ),
        )
        if ref_id is None:
            return
        if not ref_id:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.references.error.invalid_reference_id",
                    "Некорректный идентификатор референса.",
                ),
                show_alert=True,
            )
            return

        before = len(req.params.reference_images)
        req.params.reference_images = [
            item for item in req.params.reference_images if item.get("id") != ref_id
        ]
        after = len(req.params.reference_images)

        if after == before:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.references.alert.already_removed",
                    "Картинка уже удалена.",
                ),
                show_alert=True,
            )
            return

        try:
            await message.delete()
        except TelegramBadRequest:
            try:
                await message.edit_reply_markup(reply_markup=None)
            except TelegramBadRequest:
                pass

        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.references.alert.removed_left",
                "Удалено. Осталось: {count}",
                params={"count": after},
            )
        )
