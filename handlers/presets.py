from __future__ import annotations

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    Message,
)

from application.user_locale_resolver import DefaultUserLocaleResolver
from core.html_utils import h
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.panels import render_user_panel
from core.runtime import PromptRequest, RuntimeStore
from core.states import PresetStates, PromptEditorStates
from core.telegram import callback_user_id, message_user_id
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService

from .presets_flow import (
    cleanup_overwrite_messages,
    overwrite_state_prepare,
    overwrite_state_reset,
    parse_callback_data,
    remember_overwrite_messages,
)
from .presets_storage import (
    is_valid_preset_name,
    load_user_presets,
    normalize_preset_name,
    params_from_payload,
    params_payload_from,
    resolve_preset_name,
    save_user_presets,
)
from .presets_ui import (
    empty_presets_keyboard,
    overwrite_confirmation_keyboard,
    presets_list_keyboard,
    presets_title_text,
    result_save_keyboard,
    save_name_keyboard,
)
from .prompt_editor import PromptEditorService


def register_preset_handlers(
    router: Router,
    runtime: RuntimeStore,
    editor: PromptEditorService,
    localization: LocalizationService,
) -> None:
    locale_resolver = DefaultUserLocaleResolver(localization)

    def _resolve_locale(uid: int, *, telegram_locale: str | None) -> str:
        prefs = runtime.user_preferences.get(uid, {})
        user_locale = read_user_locale(
            prefs,
            default_locale=localization.default_locale(),
        )
        return locale_resolver.resolve(
            user_locale=user_locale,
            telegram_locale=telegram_locale,
        )

    def _t(
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: dict[str, object] | None = None,
    ) -> str:
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        return localization.t(key, locale=locale, params=params, default=default)

    def _ui_translate(key: str, locale: str | None, default: str) -> str:
        return localization.t(key, locale=locale, default=default)

    def _parse_callback_index(cb: CallbackQuery, *, prefix: str) -> int | None:
        raw_value = parse_callback_data(cb.data, prefix=prefix)
        if raw_value is None:
            return None
        try:
            return int(raw_value)
        except ValueError:
            return None

    async def _show_presets_panel(
        message: Message,
        state: FSMContext,
        *,
        uid: int,
        telegram_locale: str | None,
        prefer_edit: bool,
    ) -> None:
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        presets = load_user_presets(uid)
        if not presets:
            await render_user_panel(
                message,
                runtime,
                uid,
                _t(
                    uid,
                    "presets.empty",
                    "📂 Нет пресетов. Создайте через /generate.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=empty_presets_keyboard(translate=_ui_translate, locale=locale),
                prefer_edit=prefer_edit,
            )
            return

        names = sorted(presets.keys())
        await _remember_preset_snapshot(state, names)
        await state.set_state(PresetStates.browsing)
        await render_user_panel(
            message,
            runtime,
            uid,
            presets_title_text(translate=_ui_translate, locale=locale),
            reply_markup=presets_list_keyboard(
                uid,
                names,
                has_active_prompt_request=uid in runtime.active_prompt_requests,
                translate=_ui_translate,
                locale=locale,
            ),
            prefer_edit=prefer_edit,
        )

    async def _remember_preset_snapshot(state: FSMContext, names: list[str]) -> None:
        await state.update_data(
            preset_names_snapshot=names,
            preset_delete_confirm_index=None,
        )

    @router.callback_query(F.data == "pe:presets")
    async def pe_open_presets(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        locale = _resolve_locale(uid, telegram_locale=cb.from_user.language_code)
        presets = load_user_presets(uid)
        if not presets:
            await cb.answer(
                _t(
                    uid,
                    "presets.library_empty_alert",
                    "📂 Библиотека пуста.",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return

        names = sorted(presets.keys())
        await _remember_preset_snapshot(state, names)
        await state.set_state(PresetStates.browsing)
        await message.edit_text(
            presets_title_text(translate=_ui_translate, locale=locale),
            reply_markup=presets_list_keyboard(
                uid,
                names,
                has_active_prompt_request=uid in runtime.active_prompt_requests,
                translate=_ui_translate,
                locale=locale,
            ),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:save")
    async def pe_save_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        if uid not in runtime.active_prompt_requests:
            await cb.answer(
                _t(
                    uid,
                    "presets.no_active_request",
                    "⚠️ Нет активного запроса.",
                    telegram_locale=cb.from_user.language_code,
                ),
                show_alert=True,
            )
            return

        await state.update_data(**overwrite_state_reset(save_source="editor"))
        await state.set_state(PromptEditorStates.entering_preset_name)
        await message.edit_text(
            _t(
                uid,
                "presets.enter_name",
                "✏️ Введите название пресета:",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=save_name_keyboard(
                translate=_ui_translate,
                locale=_resolve_locale(uid, telegram_locale=cb.from_user.language_code),
            ),
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_preset_name, F.text)
    async def pe_save_name(msg: Message, state: FSMContext):
        uid = message_user_id(msg)
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        req = runtime.active_prompt_requests.get(uid)
        if not req:
            await state.clear()
            await msg.answer(
                _t(
                    uid,
                    "presets.active_request_not_found",
                    "⚠️ Активный запрос не найден. Используйте /generate.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        name = normalize_preset_name(msg.text or "")
        if not is_valid_preset_name(name):
            await msg.answer(
                _t(
                    uid,
                    "presets.name_invalid",
                    "⚠️ Название: 1–50 символов.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        presets = load_user_presets(uid)
        params_payload = params_payload_from(req.params)
        if name in presets:
            await state.update_data(
                **overwrite_state_prepare(
                    save_source="editor",
                    overwrite_source="editor",
                    preset_name=name,
                    preset_params=params_payload,
                )
            )
            await state.set_state(PresetStates.confirming_overwrite)
            confirm_msg = await msg.answer(
                _t(
                    uid,
                    "presets.overwrite_confirm",
                    "⚠️ Пресет «{name}» уже существует. Перезаписать?",
                    telegram_locale=telegram_locale,
                    params={"name": h(name)},
                ),
                reply_markup=overwrite_confirmation_keyboard(
                    translate=_ui_translate,
                    locale=locale,
                ),
            )
            await remember_overwrite_messages(
                state,
                input_message=msg,
                confirm_message=confirm_msg,
            )
            return

        presets[name] = params_payload_from(req.params)
        save_user_presets(uid, presets)
        try:
            await msg.delete()
        except TelegramBadRequest:
            pass
        await editor.show_prompt_editor(
            msg,
            state,
            uid,
            notice=_t(
                uid,
                "presets.saved",
                "✅ Пресет «{name}» сохранён.",
                telegram_locale=telegram_locale,
                params={"name": h(name)},
            ),
        )

    @router.callback_query(F.data == "save_preset")
    async def save_preset_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        await state.update_data(**overwrite_state_reset(save_source="result"))
        await state.set_state(PresetStates.entering_name)
        await message.edit_text(
            _t(
                uid,
                "presets.enter_name",
                "✏️ Введите название пресета:",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=save_name_keyboard(
                translate=_ui_translate,
                locale=_resolve_locale(uid, telegram_locale=cb.from_user.language_code),
            ),
        )
        await cb.answer()

    @router.callback_query(F.data == "preset:save:back")
    async def preset_save_back(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        data = await state.get_data()
        source = str(data.get("preset_save_source") or "editor")
        await state.update_data(**overwrite_state_reset())

        if source == "editor":
            uid = callback_user_id(cb)
            if uid not in runtime.active_prompt_requests:
                await state.clear()
                await message.edit_text(
                    _t(
                        uid,
                        "presets.active_request_not_found",
                        "⚠️ Активный запрос не найден. Используйте /generate.",
                        telegram_locale=cb.from_user.language_code,
                    )
                )
                await cb.answer()
                return

            await editor.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice=_t(
                    uid,
                    "presets.save_cancelled",
                    "↩️ Сохранение пресета отменено.",
                    telegram_locale=cb.from_user.language_code,
                ),
            )
            await cb.answer()
            return

        await state.clear()
        uid = callback_user_id(cb)
        await message.edit_text(
            _t(
                uid,
                "presets.save_from_result",
                "💾 Сохранить параметры как пресет?",
                telegram_locale=cb.from_user.language_code,
            ),
            reply_markup=result_save_keyboard(
                translate=_ui_translate,
                locale=_resolve_locale(uid, telegram_locale=cb.from_user.language_code),
            ),
        )
        await cb.answer()

    @router.message(PresetStates.entering_name, F.text)
    async def save_preset_name(msg: Message, state: FSMContext):
        name = normalize_preset_name(msg.text or "")
        uid = message_user_id(msg)
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        if not is_valid_preset_name(name):
            await msg.answer(
                _t(
                    uid,
                    "presets.name_invalid",
                    "⚠️ Название: 1–50 символов.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        data = await state.get_data()
        params = data.get("params")
        if not params and uid in runtime.active_prompt_requests:
            params = runtime.active_prompt_requests[uid].params
        if not params:
            await msg.answer(
                _t(
                    uid,
                    "presets.params_not_found",
                    "⚠️ Параметры не найдены.",
                    telegram_locale=telegram_locale,
                )
            )
            await state.clear()
            return

        if not isinstance(params, (GenerationParams, dict)):
            await msg.answer(
                _t(
                    uid,
                    "presets.params_not_found",
                    "⚠️ Параметры не найдены.",
                    telegram_locale=telegram_locale,
                )
            )
            await state.clear()
            return

        payload = params_payload_from(params)
        presets = load_user_presets(uid)
        if name in presets:
            await state.update_data(
                **overwrite_state_prepare(
                    save_source="result",
                    overwrite_source="result",
                    preset_name=name,
                    preset_params=payload,
                )
            )
            await state.set_state(PresetStates.confirming_overwrite)
            confirm_msg = await msg.answer(
                _t(
                    uid,
                    "presets.overwrite_confirm",
                    "⚠️ Пресет «{name}» уже существует. Перезаписать?",
                    telegram_locale=telegram_locale,
                    params={"name": h(name)},
                ),
                reply_markup=overwrite_confirmation_keyboard(
                    translate=_ui_translate,
                    locale=locale,
                ),
            )
            await remember_overwrite_messages(
                state,
                input_message=msg,
                confirm_message=confirm_msg,
            )
            return

        presets[name] = payload
        save_user_presets(uid, presets)
        await state.clear()
        await render_user_panel(
            msg,
            runtime,
            uid,
            _t(
                uid,
                "presets.saved_done",
                "✅ Пресет «{name}» сохранён!",
                telegram_locale=telegram_locale,
                params={"name": h(name)},
            ),
        )

    @router.callback_query(
        PresetStates.confirming_overwrite,
        F.data.startswith("preset:overwrite:"),
    )
    async def preset_overwrite(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        telegram_locale = cb.from_user.language_code
        decision = parse_callback_data(cb.data, prefix="preset:overwrite")
        if not decision:
            await cb.answer(
                _t(
                    uid,
                    "presets.invalid_response",
                    "⚠️ Некорректный ответ.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return
        data = await state.get_data()

        name = str(data.get("preset_overwrite_name") or "").strip()
        params_payload = data.get("preset_overwrite_params")
        source = str(data.get("preset_overwrite_source") or "result")

        if not name or not isinstance(params_payload, dict):
            await state.clear()
            await cleanup_overwrite_messages(message, data)
            await message.edit_text(
                _t(
                    uid,
                    "presets.restore_failed",
                    "⚠️ Не удалось восстановить данные для сохранения.",
                    telegram_locale=telegram_locale,
                )
            )
            await cb.answer()
            return

        if decision != "yes":
            if source == "editor":
                req = runtime.active_prompt_requests.get(uid)
                if req:
                    await editor.show_prompt_editor(
                        message,
                        state,
                        uid,
                        edit=True,
                        notice=_t(
                            uid,
                            "presets.save_cancelled",
                            "↩️ Сохранение пресета отменено.",
                            telegram_locale=telegram_locale,
                        ),
                    )
                    await cleanup_overwrite_messages(message, data)
                    await state.update_data(**overwrite_state_reset())
                    await cb.answer()
                    return

            await state.clear()
            await cleanup_overwrite_messages(message, data)
            await cb.answer(
                _t(
                    uid,
                    "presets.save_cancelled",
                    "↩️ Сохранение пресета отменено.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        presets = load_user_presets(uid)
        presets[name] = params_payload
        save_user_presets(uid, presets)

        if source == "editor":
            await editor.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice=_t(
                    uid,
                    "presets.overwritten",
                    "✅ Пресет «{name}» перезаписан.",
                    telegram_locale=telegram_locale,
                    params={"name": h(name)},
                ),
            )
            await cleanup_overwrite_messages(message, data)
            await state.update_data(**overwrite_state_reset())
            await cb.answer()
            return

        await state.clear()
        await cleanup_overwrite_messages(message, data)
        await cb.answer(
            _t(
                uid,
                "presets.overwritten",
                "✅ Пресет «{name}» перезаписан.",
                telegram_locale=telegram_locale,
                params={"name": h(name)},
            )
        )

    @router.message(Command("presets"))
    async def cmd_presets(msg: Message, state: FSMContext):
        uid = message_user_id(msg)
        await _show_presets_panel(
            msg,
            state,
            uid=uid,
            telegram_locale=msg.from_user.language_code if msg.from_user else None,
            prefer_edit=True,
        )

    @router.callback_query(F.data == "menu:presets")
    async def menu_presets(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        uid = callback_user_id(cb)
        await _show_presets_panel(
            message,
            state,
            uid=uid,
            telegram_locale=cb.from_user.language_code,
            prefer_edit=True,
        )
        await cb.answer()

    @router.callback_query(PresetStates.browsing, F.data.startswith("preset_load:"))
    async def preset_load(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        telegram_locale = cb.from_user.language_code
        index = _parse_callback_index(cb, prefix="preset_load")
        if index is None:
            await cb.answer(
                _t(
                    uid,
                    "presets.invalid_choice",
                    "⚠️ Некорректный выбор.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        presets = load_user_presets(uid)
        data = await state.get_data()
        name = resolve_preset_name(index, presets, data.get("preset_names_snapshot"))
        if not name:
            await cb.answer(
                _t(
                    uid,
                    "presets.not_found",
                    "⚠️ Не найден.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        await state.update_data(preset_delete_confirm_index=None)

        if not await editor.ensure_models(message):
            await cb.answer()
            return

        params = params_from_payload(presets[name])
        runtime.active_prompt_requests[uid] = PromptRequest(
            params=editor.normalize_params(params),
            operation="generate",
        )
        await editor.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t(
                uid,
                "presets.loaded",
                "📂 Загружен пресет «{name}».",
                telegram_locale=telegram_locale,
                params={"name": h(name)},
            ),
        )
        await cb.answer()

    @router.callback_query(PresetStates.browsing, F.data.startswith("preset_del:"))
    async def preset_del(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        telegram_locale = cb.from_user.language_code
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        index = _parse_callback_index(cb, prefix="preset_del")
        if index is None:
            await cb.answer(
                _t(
                    uid,
                    "presets.invalid_choice",
                    "⚠️ Некорректный выбор.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        presets = load_user_presets(uid)
        data = await state.get_data()
        name = resolve_preset_name(index, presets, data.get("preset_names_snapshot"))
        if not name:
            await cb.answer(
                _t(
                    uid,
                    "presets.not_found",
                    "⚠️ Не найден.",
                    telegram_locale=telegram_locale,
                ),
                show_alert=True,
            )
            return

        pending_index = data.get("preset_delete_confirm_index")
        if pending_index != index:
            await state.update_data(preset_delete_confirm_index=index)
            names = sorted(presets.keys())
            await state.update_data(preset_names_snapshot=names)
            rows = presets_list_keyboard(
                uid,
                names,
                has_active_prompt_request=uid in runtime.active_prompt_requests,
                confirm_delete_index=index,
                translate=_ui_translate,
                locale=locale,
            )
            await message.edit_text(
                presets_title_text(translate=_ui_translate, locale=locale),
                reply_markup=rows,
            )
            await cb.answer(
                _t(
                    uid,
                    "presets.delete_confirm_again",
                    "Нажмите ещё раз, чтобы подтвердить удаление.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        presets.pop(name, None)
        save_user_presets(uid, presets)
        await state.update_data(preset_delete_confirm_index=None)

        if not presets:
            await message.edit_text(
                _t(
                    uid,
                    "presets.all_deleted",
                    "📂 Все пресеты удалены.",
                    telegram_locale=telegram_locale,
                )
            )
            await state.clear()
            await cb.answer()
            return

        names = sorted(presets.keys())
        await state.update_data(preset_names_snapshot=names)
        rows = presets_list_keyboard(
            uid,
            names,
            has_active_prompt_request=uid in runtime.active_prompt_requests,
            translate=_ui_translate,
            locale=locale,
        )
        await message.edit_text(
            presets_title_text(translate=_ui_translate, locale=locale),
            reply_markup=rows,
        )
        await cb.answer(
            _t(
                uid,
                "presets.deleted",
                "🗑 «{name}» удалён.",
                telegram_locale=telegram_locale,
                params={"name": h(name)},
            )
        )
