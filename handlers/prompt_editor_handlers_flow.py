from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    Message,
)

from core.callbacks import ValueSelectionCallback
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService
from infrastructure.comfyui_client import ComfyUIClient


@dataclass
class PromptEditorFlowHandlersDeps:
    runtime: RuntimeStore
    client: ComfyUIClient
    callback_user_id: Callable[[CallbackQuery], int]
    message_user_id: Callable[[Message], int]
    ensure_models: Callable[[Message], Awaitable[bool]]
    default_params_for_user: Callable[[int], GenerationParams]
    open_prompt_request: Callable[..., Awaitable[None]]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    changed_params_count: Callable[[GenerationParams], int]
    run_generate_operation: Callable[[Message, FSMContext, int], Awaitable[None]]
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]


def register_prompt_editor_flow_handlers(
    router: Router,
    deps: PromptEditorFlowHandlersDeps,
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

    def _latest_user_generation(uid: int):
        candidates = [
            item for item in deps.runtime.active_generations.values() if item.owner_uid == uid
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.created_at, reverse=True)
        return candidates[0]

    async def _start_generate(msg: Message, state: FSMContext, uid: int) -> None:
        if not await deps.ensure_models(msg):
            return
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        await deps.open_prompt_request(
            msg,
            state,
            uid,
            deps.default_params_for_user(uid),
            operation="generate",
            notice=_t(
                uid,
                "prompt_editor.flow.notice.new_request",
                "✨ Новый запрос создан.",
                telegram_locale=telegram_locale,
            ),
        )

    async def _start_repeat(msg: Message, state: FSMContext, uid: int) -> None:
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        if uid not in deps.runtime.last_params:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.flow.error.no_previous_generation",
                    "❌ Нет предыдущей генерации. Используйте /generate.",
                    telegram_locale=telegram_locale,
                )
            )
            return
        if not await deps.ensure_models(msg):
            return

        params = GenerationParams(**asdict(deps.runtime.last_params[uid]))
        params.seed = -1
        await deps.open_prompt_request(
            msg,
            state,
            uid,
            params,
            operation="generate",
            notice=_t(
                uid,
                "prompt_editor.flow.notice.repeat_loaded",
                "🔁 Загружен последний запрос (seed = random).",
                telegram_locale=telegram_locale,
            ),
        )

    def _cancel_editor_keyboard(*, uid: int, telegram_locale: str | None):
        return build_keyboard(
            [
                [
                    button(
                        _t(
                            uid,
                            "prompt_editor.flow.button.confirm_cancel",
                            "✅ Отменить",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:cancel:confirm",
                    ),
                    button(
                        _t(
                            uid,
                            "prompt_editor.flow.button.save",
                            "💾 Сохранить",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:save",
                    ),
                ],
                [
                    back_button(
                        "pe:back",
                        text=_t(
                            uid,
                            "common.action.back",
                            "⬅️ Назад",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
            ]
        )

    def _empty_positive_confirm_keyboard(*, uid: int, telegram_locale: str | None):
        return build_keyboard(
            [
                [
                    button(
                        _t(
                            uid,
                            "prompt_editor.flow.button.yes",
                            "✅ Да",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:gen:empty:yes",
                    ),
                    button(
                        _t(
                            uid,
                            "prompt_editor.flow.button.no",
                            "❌ Нет",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:gen:empty:no",
                    ),
                ]
            ]
        )

    @router.message(Command("generate"))
    async def cmd_generate(msg: Message, state: FSMContext):
        await _start_generate(msg, state, deps.message_user_id(msg))

    @router.callback_query(F.data == "menu:generate")
    async def menu_generate(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        await _start_generate(message, state, deps.callback_user_id(cb))
        await cb.answer()

    @router.message(Command("repeat"))
    async def cmd_repeat(msg: Message, state: FSMContext):
        await _start_repeat(msg, state, deps.message_user_id(msg))

    @router.callback_query(F.data == "menu:repeat")
    async def menu_repeat(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        await _start_repeat(message, state, deps.callback_user_id(cb))
        await cb.answer()

    @router.callback_query(F.data == "pe:back")
    async def pe_back(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        await deps.show_prompt_editor(message, state, uid, edit=True)
        await cb.answer()

    @router.callback_query(F.data == "pe:cancel")
    async def pe_cancel(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        changed_count = deps.changed_params_count(req.params)
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        if changed_count > 0:
            await message.edit_text(
                _t(
                    uid,
                    "prompt_editor.flow.confirm.cancel_with_changes",
                    "⚠️ Вы изменили {count} параметров. Точно отменить редактор?",
                    telegram_locale=telegram_locale,
                    params={"count": changed_count},
                ),
                reply_markup=_cancel_editor_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            await cb.answer()
            return

        deps.runtime.active_prompt_requests.pop(uid, None)
        await state.clear()
        await message.edit_text(
            _t(
                uid,
                "prompt_editor.flow.notice.operation_cancelled",
                "❌ Операция отменена.",
                telegram_locale=telegram_locale,
            )
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:cancel:confirm")
    async def pe_cancel_confirm(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        uid = deps.callback_user_id(cb)
        deps.runtime.active_prompt_requests.pop(uid, None)
        await state.clear()
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.flow.notice.operation_cancelled",
                "❌ Операция отменена.",
            )
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe:gen:cancel"))
    async def pe_gen_cancel(cb: CallbackQuery, state: FSMContext):
        uid = deps.callback_user_id(cb)
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix="pe:gen:cancel")
        generation_id = parsed.value if parsed else ""

        gen = None
        if generation_id and generation_id != "pending":
            candidate = deps.runtime.active_generations.get(generation_id)
            if candidate and candidate.owner_uid == uid:
                gen = candidate
        if gen is None:
            gen = _latest_user_generation(uid)

        if gen:
            if gen.task is not None and not gen.task.done():
                gen.task.cancel()
            if gen.prompt_id:
                asyncio.create_task(deps.client.cancel_prompt(gen.prompt_id))
            await cb.answer(
                _t_cb(cb, "prompt_editor.flow.alert.cancelling", "❌ Отменяю..."),
                show_alert=False,
            )
        else:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.flow.alert.nothing_to_cancel",
                    "Нечего отменять.",
                ),
                show_alert=True,
            )

    @router.callback_query(F.data == "pe:proceed")
    async def pe_proceed(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        if req.operation == "generate":
            if not req.params.positive.strip():
                telegram_locale = cb.from_user.language_code if cb.from_user else None
                await message.edit_text(
                    _t(
                        uid,
                        "prompt_editor.flow.confirm.empty_positive",
                        "⚠️ Positive prompt пустой. Всё равно генерировать?",
                        telegram_locale=telegram_locale,
                    ),
                    reply_markup=_empty_positive_confirm_keyboard(
                        uid=uid,
                        telegram_locale=telegram_locale,
                    ),
                )
                await cb.answer()
                return

            await cb.answer()
            await deps.run_generate_operation(message, state, uid)
            return
        await cb.answer()
        await message.answer(
            _t(
                uid,
                "prompt_editor.flow.error.unknown_operation",
                "Неизвестная операция: {operation}",
                telegram_locale=cb.from_user.language_code if cb.from_user else None,
                params={"operation": req.operation},
            )
        )

    @router.callback_query(F.data == "pe:gen:empty:yes")
    async def pe_generate_empty_yes(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await cb.answer()
        await deps.run_generate_operation(message, state, uid)

    @router.callback_query(F.data == "pe:gen:empty:no")
    async def pe_generate_empty_no(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t(
                uid,
                "prompt_editor.flow.notice.generation_cancelled_fill_positive",
                "ℹ️ Генерация отменена: заполните Positive или подтвердите пустой prompt.",
                telegram_locale=cb.from_user.language_code if cb.from_user else None,
            ),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:gen:back")
    async def pe_generate_back_to_editor(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t(
                uid,
                "prompt_editor.flow.notice.back_to_editor",
                "↩️ Возвращаемся в редактор.",
                telegram_locale=cb.from_user.language_code if cb.from_user else None,
            ),
        )
        await cb.answer()
