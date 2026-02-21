from __future__ import annotations

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from core.callbacks import ValueSelectionCallback
from core.html_utils import h
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.panels import render_user_panel
from core.runtime import PromptRequest, RuntimeStore
from core.states import PresetStates, PromptEditorStates
from core.storage import dict_to_params, load_presets, params_to_dict, save_presets
from core.telegram import callback_user_id, message_user_id
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, menu_root_button

from .prompt_editor import PromptEditorService


def register_preset_handlers(
    router: Router,
    runtime: RuntimeStore,
    editor: PromptEditorService,
) -> None:
    presets_title = "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:"

    save_name_kb = build_keyboard([[back_button("preset:save:back")]])

    overwrite_kb = build_keyboard(
        [[button("✅ Перезаписать", "preset:overwrite:yes"), cancel_button("preset:overwrite:no")]]
    )

    result_save_kb = build_keyboard([[button("💾 Сохранить как пресет", "save_preset")]])

    def _empty_presets_keyboard() -> InlineKeyboardMarkup:
        return build_keyboard(
            [[back_button("menu:generation", text="⬅️ Генерация")], [menu_root_button()]]
        )

    def _presets_text() -> str:
        return presets_title

    def _parse_callback_value(cb: CallbackQuery, prefix: str) -> str | None:
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
        if parsed is None:
            return None
        return parsed.value

    async def _remember_overwrite_messages(
        state: FSMContext,
        *,
        input_message: Message,
        confirm_message: Message,
    ) -> None:
        await state.update_data(
            preset_overwrite_input_message_id=input_message.message_id,
            preset_overwrite_prompt_message_id=confirm_message.message_id,
            preset_overwrite_chat_id=input_message.chat.id,
        )

    async def _cleanup_overwrite_messages(
        source_message: Message,
        state_data: dict[str, object],
    ) -> None:
        bot = source_message.bot
        if bot is None:
            return

        chat_id = state_data.get("preset_overwrite_chat_id")
        if not isinstance(chat_id, int):
            chat_id = source_message.chat.id

        ids_to_delete: list[int] = []
        for key in ("preset_overwrite_input_message_id", "preset_overwrite_prompt_message_id"):
            value = state_data.get(key)
            if isinstance(value, int):
                ids_to_delete.append(value)

        if not ids_to_delete:
            return

        for message_id in set(ids_to_delete):
            try:
                await bot.delete_message(chat_id=chat_id, message_id=message_id)
            except TelegramBadRequest:
                pass

    def _preset_rows(
        uid: int,
        names: list[str],
        *,
        confirm_delete_index: int | None = None,
    ) -> list[list[InlineKeyboardButton]]:
        rows: list[list[InlineKeyboardButton]] = [
            [
                button(f"📂 {name}", f"preset_load:{index}"),
                button(
                    text=(
                        (f"⚠️ Удалить «{name if len(name) <= 18 else name[:15] + '...'}»?")
                        if confirm_delete_index == index
                        else "🗑"
                    ),
                    callback_data=f"preset_del:{index}",
                ),
            ]
            for index, name in enumerate(names)
        ]
        if uid in runtime.active_prompt_requests:
            rows.append([back_button("pe:back", text="⬅️ В редактор")])
        else:
            rows.append([back_button("menu:generation", text="⬅️ Генерация")])
        rows.append([menu_root_button()])
        return rows

    async def _show_presets_panel(
        message: Message,
        state: FSMContext,
        *,
        uid: int,
        prefer_edit: bool,
    ) -> None:
        presets = load_presets(uid)
        if not presets:
            await render_user_panel(
                message,
                runtime,
                uid,
                "📂 Нет пресетов. Создайте через /generate.",
                reply_markup=_empty_presets_keyboard(),
                prefer_edit=prefer_edit,
            )
            return

        names = sorted(presets.keys())
        await _remember_preset_snapshot(state, names)
        rows = _preset_rows(uid, names)
        await state.set_state(PresetStates.browsing)
        await render_user_panel(
            message,
            runtime,
            uid,
            _presets_text(),
            reply_markup=build_keyboard(rows),
            prefer_edit=prefer_edit,
        )

    async def _remember_preset_snapshot(state: FSMContext, names: list[str]) -> None:
        await state.update_data(
            preset_names_snapshot=names,
            preset_delete_confirm_index=None,
        )

    def _name_from_index(
        index: int,
        presets: dict[str, dict],
        snapshot: object,
    ) -> str | None:
        if isinstance(snapshot, list) and 0 <= index < len(snapshot):
            candidate = snapshot[index]
            if isinstance(candidate, str) and candidate in presets:
                return candidate

        names = sorted(presets.keys())
        if 0 <= index < len(names):
            return names[index]
        return None

    @router.callback_query(F.data == "pe:presets")
    async def pe_open_presets(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        presets = load_presets(uid)
        if not presets:
            await cb.answer("📂 Библиотека пуста.", show_alert=True)
            return

        names = sorted(presets.keys())
        await _remember_preset_snapshot(state, names)
        rows = _preset_rows(uid, names)
        await state.set_state(PresetStates.browsing)
        await message.edit_text(
            _presets_text(),
            reply_markup=build_keyboard(rows),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:save")
    async def pe_save_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = callback_user_id(cb)
        if uid not in runtime.active_prompt_requests:
            await cb.answer("⚠️ Нет активного запроса.", show_alert=True)
            return

        await state.update_data(
            preset_save_source="editor",
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
            preset_overwrite_input_message_id=None,
            preset_overwrite_prompt_message_id=None,
            preset_overwrite_chat_id=None,
        )
        await state.set_state(PromptEditorStates.entering_preset_name)
        await message.edit_text(
            "✏️ Введите название пресета:",
            reply_markup=save_name_kb,
        )
        await cb.answer()

    @router.message(PromptEditorStates.entering_preset_name, F.text)
    async def pe_save_name(msg: Message, state: FSMContext):
        uid = message_user_id(msg)
        req = runtime.active_prompt_requests.get(uid)
        if not req:
            await state.clear()
            await msg.answer("⚠️ Активный запрос не найден. Используйте /generate.")
            return

        name = (msg.text or "").strip()
        if not name or len(name) > 50:
            await msg.answer("⚠️ Название: 1–50 символов.")
            return

        presets = load_presets(uid)
        params_payload = params_to_dict(req.params)
        if name in presets:
            await state.update_data(
                preset_save_source="editor",
                preset_overwrite_name=name,
                preset_overwrite_params=params_payload,
                preset_overwrite_source="editor",
            )
            await state.set_state(PresetStates.confirming_overwrite)
            confirm_msg = await msg.answer(
                f"⚠️ Пресет «{h(name)}» уже существует. Перезаписать?",
                reply_markup=overwrite_kb,
            )
            await _remember_overwrite_messages(
                state,
                input_message=msg,
                confirm_message=confirm_msg,
            )
            return

        presets[name] = params_to_dict(req.params)
        save_presets(uid, presets)
        try:
            await msg.delete()
        except TelegramBadRequest:
            pass
        await editor.show_prompt_editor(
            msg,
            state,
            uid,
            notice=f"✅ Пресет «{h(name)}» сохранён.",
        )

    @router.callback_query(F.data == "save_preset")
    async def save_preset_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        await state.update_data(
            preset_save_source="result",
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
            preset_overwrite_input_message_id=None,
            preset_overwrite_prompt_message_id=None,
            preset_overwrite_chat_id=None,
        )
        await state.set_state(PresetStates.entering_name)
        await message.edit_text(
            "✏️ Введите название пресета:",
            reply_markup=save_name_kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "preset:save:back")
    async def preset_save_back(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        data = await state.get_data()
        source = str(data.get("preset_save_source") or "editor")
        await state.update_data(
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
            preset_overwrite_input_message_id=None,
            preset_overwrite_prompt_message_id=None,
            preset_overwrite_chat_id=None,
        )

        if source == "editor":
            uid = callback_user_id(cb)
            if uid not in runtime.active_prompt_requests:
                await state.clear()
                await message.edit_text("⚠️ Активный запрос не найден. Используйте /generate.")
                await cb.answer()
                return

            await editor.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice="↩️ Сохранение пресета отменено.",
            )
            await cb.answer()
            return

        await state.clear()
        await message.edit_text(
            "💾 Сохранить параметры как пресет?",
            reply_markup=result_save_kb,
        )
        await cb.answer()

    @router.message(PresetStates.entering_name, F.text)
    async def save_preset_name(msg: Message, state: FSMContext):
        name = (msg.text or "").strip()
        if not name or len(name) > 50:
            await msg.answer("⚠️ Название: 1–50 символов.")
            return

        uid = message_user_id(msg)
        data = await state.get_data()
        params = data.get("params")
        if not params and uid in runtime.active_prompt_requests:
            params = runtime.active_prompt_requests[uid].params
        if not params:
            await msg.answer("⚠️ Параметры не найдены.")
            await state.clear()
            return

        payload = params_to_dict(params) if isinstance(params, GenerationParams) else params
        presets = load_presets(uid)
        if name in presets:
            await state.update_data(
                preset_save_source="result",
                preset_overwrite_name=name,
                preset_overwrite_params=payload,
                preset_overwrite_source="result",
            )
            await state.set_state(PresetStates.confirming_overwrite)
            confirm_msg = await msg.answer(
                f"⚠️ Пресет «{h(name)}» уже существует. Перезаписать?",
                reply_markup=overwrite_kb,
            )
            await _remember_overwrite_messages(
                state,
                input_message=msg,
                confirm_message=confirm_msg,
            )
            return

        presets[name] = payload
        save_presets(uid, presets)
        await state.clear()
        await render_user_panel(
            msg,
            runtime,
            uid,
            f"✅ Пресет «{h(name)}» сохранён!",
        )

    @router.callback_query(
        PresetStates.confirming_overwrite,
        F.data.startswith("preset:overwrite:"),
    )
    async def preset_overwrite(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        decision = _parse_callback_value(cb, "preset:overwrite")
        if not decision:
            await cb.answer("⚠️ Некорректный ответ.", show_alert=True)
            return
        data = await state.get_data()

        name = str(data.get("preset_overwrite_name") or "").strip()
        params_payload = data.get("preset_overwrite_params")
        source = str(data.get("preset_overwrite_source") or "result")

        if not name or not isinstance(params_payload, dict):
            await state.clear()
            await _cleanup_overwrite_messages(message, data)
            await message.edit_text("⚠️ Не удалось восстановить данные для сохранения.")
            await cb.answer()
            return

        if decision != "yes":
            if source == "editor":
                uid = callback_user_id(cb)
                req = runtime.active_prompt_requests.get(uid)
                if req:
                    await editor.show_prompt_editor(
                        message,
                        state,
                        uid,
                        edit=True,
                        notice="↩️ Сохранение пресета отменено.",
                    )
                    await _cleanup_overwrite_messages(message, data)
                    await state.update_data(
                        preset_overwrite_input_message_id=None,
                        preset_overwrite_prompt_message_id=None,
                        preset_overwrite_chat_id=None,
                    )
                    await cb.answer()
                    return

            await state.clear()
            await _cleanup_overwrite_messages(message, data)
            await cb.answer("↩️ Сохранение пресета отменено.")
            return

        uid = callback_user_id(cb)
        presets = load_presets(uid)
        presets[name] = params_payload
        save_presets(uid, presets)

        if source == "editor":
            await editor.show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice=f"✅ Пресет «{h(name)}» перезаписан.",
            )
            await _cleanup_overwrite_messages(message, data)
            await state.update_data(
                preset_overwrite_input_message_id=None,
                preset_overwrite_prompt_message_id=None,
                preset_overwrite_chat_id=None,
            )
            await cb.answer()
            return

        await state.clear()
        await _cleanup_overwrite_messages(message, data)
        await cb.answer(f"✅ Пресет «{h(name)}» перезаписан.")

    @router.message(Command("presets"))
    async def cmd_presets(msg: Message, state: FSMContext):
        await _show_presets_panel(
            msg,
            state,
            uid=message_user_id(msg),
            prefer_edit=True,
        )

    @router.callback_query(F.data == "menu:presets")
    async def menu_presets(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        await _show_presets_panel(
            message,
            state,
            uid=callback_user_id(cb),
            prefer_edit=True,
        )
        await cb.answer()

    @router.callback_query(PresetStates.browsing, F.data.startswith("preset_load:"))
    async def preset_load(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        raw_index = _parse_callback_value(cb, "preset_load")
        if raw_index is None:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return
        try:
            index = int(raw_index)
        except ValueError:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return

        uid = callback_user_id(cb)
        presets = load_presets(uid)
        data = await state.get_data()
        name = _name_from_index(index, presets, data.get("preset_names_snapshot"))
        if not name:
            await cb.answer("⚠️ Не найден.", show_alert=True)
            return

        await state.update_data(preset_delete_confirm_index=None)

        if not await editor.ensure_models(message):
            await cb.answer()
            return

        params = dict_to_params(presets[name])
        runtime.active_prompt_requests[uid] = PromptRequest(
            params=editor.normalize_params(params),
            operation="generate",
        )
        await editor.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=f"📂 Загружен пресет «{h(name)}».",
        )
        await cb.answer()

    @router.callback_query(PresetStates.browsing, F.data.startswith("preset_del:"))
    async def preset_del(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        raw_index = _parse_callback_value(cb, "preset_del")
        if raw_index is None:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return
        try:
            index = int(raw_index)
        except ValueError:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return

        uid = callback_user_id(cb)
        presets = load_presets(uid)
        data = await state.get_data()
        name = _name_from_index(index, presets, data.get("preset_names_snapshot"))
        if not name:
            await cb.answer("⚠️ Не найден.", show_alert=True)
            return

        pending_index = data.get("preset_delete_confirm_index")
        if pending_index != index:
            await state.update_data(preset_delete_confirm_index=index)
            names = sorted(presets.keys())
            await state.update_data(preset_names_snapshot=names)
            rows = _preset_rows(
                uid,
                names,
                confirm_delete_index=index,
            )
            await message.edit_text(
                _presets_text(),
                reply_markup=build_keyboard(rows),
            )
            await cb.answer("Нажмите ещё раз, чтобы подтвердить удаление.")
            return

        presets.pop(name, None)
        save_presets(uid, presets)
        await state.update_data(preset_delete_confirm_index=None)

        if not presets:
            await message.edit_text("📂 Все пресеты удалены.")
            await state.clear()
            await cb.answer()
            return

        names = sorted(presets.keys())
        await state.update_data(preset_names_snapshot=names)
        rows = _preset_rows(uid, names)
        await message.edit_text(
            _presets_text(),
            reply_markup=build_keyboard(rows),
        )
        await cb.answer(f"🗑 «{name}» удалён.")
