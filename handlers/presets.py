from __future__ import annotations

from typing import cast

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

from core.html_utils import h
from core.models import GenerationParams
from core.panels import render_user_panel
from core.runtime import PromptRequest, RuntimeStore
from core.states import PresetStates, PromptEditorStates
from core.storage import dict_to_params, load_presets, params_to_dict, save_presets
from core.telegram import callback_user_id, message_user_id

from .prompt_editor import PromptEditorService


def register_preset_handlers(
    router: Router,
    runtime: RuntimeStore,
    editor: PromptEditorService,
) -> None:
    save_name_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data="preset:save:back",
                )
            ]
        ]
    )

    overwrite_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Перезаписать",
                    callback_data="preset:overwrite:yes",
                ),
                InlineKeyboardButton(
                    text="❌ Отмена",
                    callback_data="preset:overwrite:no",
                ),
            ]
        ]
    )

    def _preset_rows(
        uid: int,
        names: list[str],
        *,
        confirm_delete_index: int | None = None,
    ) -> list[list[InlineKeyboardButton]]:
        rows: list[list[InlineKeyboardButton]] = [
            [
                InlineKeyboardButton(
                    text=f"\U0001f4c2 {name}", callback_data=f"preset_load:{index}"
                ),
                InlineKeyboardButton(
                    text=(
                        (f"⚠️ Удалить «{name if len(name) <= 18 else name[:15] + '...'}»?")
                        if confirm_delete_index == index
                        else "\U0001f5d1"
                    ),
                    callback_data=f"preset_del:{index}",
                ),
            ]
            for index, name in enumerate(names)
        ]
        if uid in runtime.active_prompt_requests:
            rows.append(
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u0412 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440",
                        callback_data="pe:back",
                    )
                ]
            )
        else:
            rows.append(
                [
                    InlineKeyboardButton(
                        text="⬅️ Генерация",
                        callback_data="menu:generation",
                    )
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="🏠 В меню",
                    callback_data="menu:root",
                )
            ]
        )
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
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="⬅️ Генерация", callback_data="menu:generation")],
                        [InlineKeyboardButton(text="🏠 В меню", callback_data="menu:root")],
                    ]
                ),
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
            "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
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
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

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
            "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:save")
    async def pe_save_start(cb: CallbackQuery, state: FSMContext):
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        uid = callback_user_id(cb)
        if uid not in runtime.active_prompt_requests:
            await cb.answer("⚠️ Нет активного запроса.", show_alert=True)
            return

        await state.update_data(
            preset_save_source="editor",
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
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
            await msg.answer(
                f"⚠️ Пресет «{h(name)}» уже существует. Перезаписать?",
                reply_markup=overwrite_kb,
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
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        await state.update_data(
            preset_save_source="result",
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
        )
        await state.set_state(PresetStates.entering_name)
        await message.edit_text(
            "✏️ Введите название пресета:",
            reply_markup=save_name_kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "preset:save:back")
    async def preset_save_back(cb: CallbackQuery, state: FSMContext):
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        data = await state.get_data()
        source = str(data.get("preset_save_source") or "editor")
        await state.update_data(
            preset_overwrite_name=None,
            preset_overwrite_params=None,
            preset_overwrite_source=None,
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
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="💾 Сохранить как пресет",
                            callback_data="save_preset",
                        )
                    ]
                ]
            ),
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
            await msg.answer(
                f"⚠️ Пресет «{h(name)}» уже существует. Перезаписать?",
                reply_markup=overwrite_kb,
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
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        data_value = cb.data
        if not data_value:
            await cb.answer("⚠️ Некорректный ответ.", show_alert=True)
            return

        decision = data_value.split(":")[-1]
        data = await state.get_data()

        name = str(data.get("preset_overwrite_name") or "").strip()
        params_payload = data.get("preset_overwrite_params")
        source = str(data.get("preset_overwrite_source") or "result")

        if not name or not isinstance(params_payload, dict):
            await state.clear()
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
                    await cb.answer()
                    return

            await state.clear()
            await message.edit_text("↩️ Сохранение пресета отменено.")
            await cb.answer()
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
            await cb.answer()
            return

        await state.clear()
        await message.edit_text(f"✅ Пресет «{h(name)}» перезаписан.")
        await cb.answer()

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
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        await _show_presets_panel(
            cast(Message, cb.message),
            state,
            uid=callback_user_id(cb),
            prefer_edit=True,
        )
        await cb.answer()

    @router.callback_query(PresetStates.browsing, F.data.startswith("preset_load:"))
    async def preset_load(cb: CallbackQuery, state: FSMContext):
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        data_value = cb.data
        if not data_value:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return

        raw_index = data_value.split(":", 1)[1]
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
        if cb.message is None or not hasattr(cb.message, "edit_text"):
            await cb.answer("⚠️ Сообщение недоступно.", show_alert=True)
            return
        message = cast(Message, cb.message)

        data_value = cb.data
        if not data_value:
            await cb.answer("⚠️ Некорректный выбор.", show_alert=True)
            return

        raw_index = data_value.split(":", 1)[1]
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
                "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
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
            "📂 <b>Пресеты</b>\nНажмите для загрузки, 🗑 для удаления:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=rows),
        )
        await cb.answer(f"🗑 «{name}» удалён.")
