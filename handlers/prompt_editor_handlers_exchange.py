from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from io import BytesIO

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardMarkup,
    Message,
)

from core.html_utils import h, truncate
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.prompt_exchange import (
    PROMPT_EXCHANGE_PREFIX,
    PromptExchangeError,
    export_prompt_token,
    import_prompt_token,
)
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService

_MAX_IMPORT_FILE_BYTES = 128_000
_TOKEN_INLINE_LIMIT = 3500
_TOKEN_TEXT_PATTERN = r"CBOT_PROMPT_V\d+:"


@dataclass
class PromptEditorExchangeHandlersDeps:
    runtime: RuntimeStore
    localization: LocalizationService
    resolve_user_locale: Callable[..., str]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    normalize_params: Callable[[GenerationParams], GenerationParams]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def _preview_line(label: str, value: str) -> str:
    text = value.strip()
    if not text:
        return f"{label}: <i>—</i>"
    return f"{label}: <code>{h(truncate(text, 85))}</code>"


def register_prompt_editor_exchange_handlers(
    router: Router,
    deps: PromptEditorExchangeHandlersDeps,
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

    def _t_msg(
        msg: Message,
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        uid = msg.from_user.id if msg.from_user else 0
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        return _t(uid, key, default, telegram_locale=telegram_locale, params=params)

    def _exchange_menu_keyboard(*, uid: int, telegram_locale: str | None) -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [
                    button(
                        _t(
                            uid,
                            "prompt_editor.exchange.button.copy",
                            "📤 Копировать",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:exchange:export",
                    ),
                    button(
                        _t(
                            uid,
                            "prompt_editor.exchange.button.paste",
                            "📥 Вставить",
                            telegram_locale=telegram_locale,
                        ),
                        "pe:exchange:import",
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

    def _exchange_result_keyboard(*, uid: int, telegram_locale: str | None) -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [
                    back_button(
                        "pe:exchange",
                        text=_t(
                            uid,
                            "prompt_editor.exchange.button.back_to_exchange",
                            "⬅️ К обмену",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
                [
                    back_button(
                        "pe:back",
                        text=_t(
                            uid,
                            "prompt_editor.exchange.button.back_to_editor",
                            "⬅️ В редактор",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ],
            ]
        )

    def _exchange_import_keyboard(*, uid: int, telegram_locale: str | None) -> InlineKeyboardMarkup:
        return build_keyboard(
            [
                [
                    back_button(
                        "pe:exchange",
                        text=_t(
                            uid,
                            "prompt_editor.exchange.button.back_to_exchange",
                            "⬅️ К обмену",
                            telegram_locale=telegram_locale,
                        ),
                    )
                ]
            ]
        )

    async def _apply_import_from_text(
        message: Message,
        state: FSMContext,
        raw_text: str,
    ) -> None:
        payload = await deps.require_prompt_request_for_message(message, state)
        if not payload:
            return

        uid, req = payload
        telegram_locale = message.from_user.language_code if message.from_user else None
        try:
            imported_params = import_prompt_token(raw_text)
        except PromptExchangeError as exc:
            await message.answer(
                _t(
                    uid,
                    "prompt_editor.exchange.import.failed",
                    "❌ Не удалось импортировать код обмена.\n"
                    "<i>{error}</i>\n\n"
                    "Ожидается код вида <code>{prefix}...</code>",
                    telegram_locale=telegram_locale,
                    params={"error": h(exc), "prefix": PROMPT_EXCHANGE_PREFIX},
                ),
                reply_markup=_exchange_import_keyboard(
                    uid=uid,
                    telegram_locale=telegram_locale,
                ),
            )
            return

        req.params = deps.normalize_params(imported_params)
        await deps.cleanup_user_message(message)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            notice=_t(
                uid,
                "prompt_editor.exchange.import.success",
                "✅ Параметры загружены из кода обмена.",
                telegram_locale=telegram_locale,
            ),
        )

    @router.callback_query(F.data == "pe:exchange")
    async def pe_exchange_menu(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        text = "\n".join(
            [
                _t(
                    uid,
                    "prompt_editor.exchange.menu.title",
                    "📋 <b>Копирование и вставка промпта</b>",
                    telegram_locale=telegram_locale,
                ),
                "",
                _t(
                    uid,
                    "prompt_editor.exchange.menu.description_line1",
                    "Экспортируйте полный набор параметров редактора в код обмена и "
                    "отправьте его другому пользователю.",
                    telegram_locale=telegram_locale,
                ),
                "",
                _preview_line(
                    _t(
                        uid,
                        "prompt_editor.exchange.preview.positive_label",
                        "🟢 Positive",
                        telegram_locale=telegram_locale,
                    ),
                    req.params.positive,
                ),
                _preview_line(
                    _t(
                        uid,
                        "prompt_editor.exchange.preview.negative_label",
                        "🔴 Negative",
                        telegram_locale=telegram_locale,
                    ),
                    req.params.negative,
                ),
            ]
        )
        await message.edit_text(
            text,
            reply_markup=_exchange_menu_keyboard(uid=uid, telegram_locale=telegram_locale),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:exchange:export")
    async def pe_exchange_export(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, req = payload
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        token = export_prompt_token(req.params)

        if len(token) > _TOKEN_INLINE_LIMIT:
            await message.edit_text(
                _t(
                    uid,
                    "prompt_editor.exchange.export.file.ready",
                    "📤 <b>Код обмена готов</b>\n"
                    "\n"
                    "Текущий код слишком длинный для безопасного вывода в одном "
                    "сообщении. Отправил его .txt-файлом ниже.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=_exchange_result_keyboard(
                    uid=uid,
                    telegram_locale=telegram_locale,
                ),
            )
            await message.answer_document(
                BufferedInputFile(
                    (token + "\n").encode("utf-8"),
                    filename="comfybot_prompt_code.txt",
                ),
                caption=_t(
                    uid,
                    "prompt_editor.exchange.export.file.caption",
                    "📎 Код обмена. Скопируйте содержимое файла и вставьте через «📥 Вставить».",
                    telegram_locale=telegram_locale,
                ),
            )
            await cb.answer(
                _t(
                    uid,
                    "prompt_editor.exchange.export.file.answer",
                    "Код отправлен файлом.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        await message.edit_text(
            _t(
                uid,
                "prompt_editor.exchange.export.inline.ready",
                "📤 <b>Код обмена готов</b>\n"
                "\n"
                "Скопируйте строку целиком и отправьте её другому пользователю.\n"
                "Код содержит полный набор параметров редактора (prompt, модель, LoRA, "
                "sampler, размер, seed, референсы и улучшения).\n"
                "\n"
                "<code>{token}</code>",
                telegram_locale=telegram_locale,
                params={"token": h(token)},
            ),
            reply_markup=_exchange_result_keyboard(uid=uid, telegram_locale=telegram_locale),
        )
        await cb.answer(
            _t(
                uid,
                "prompt_editor.exchange.export.inline.answer",
                "Код обмена готов.",
                telegram_locale=telegram_locale,
            )
        )

    @router.callback_query(F.data == "pe:exchange:import")
    async def pe_exchange_import_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        uid, _ = payload
        telegram_locale = cb.from_user.language_code if cb.from_user else None
        await state.set_state(PromptEditorStates.entering_shared_prompt)
        await message.edit_text(
            _t(
                uid,
                "prompt_editor.exchange.import.start",
                "📥 <b>Вставка кода обмена</b>\n"
                "\n"
                "Отправьте код обмена сообщением.\n"
                "Можно вставить текст или отправить .txt-файл с кодом.\n"
                "\n"
                "Также можно просто отправить код прямо из главного экрана редактора.",
                telegram_locale=telegram_locale,
            ),
            reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
        )
        await cb.answer()

    @router.message(PromptEditorStates.editing, F.text.regexp(_TOKEN_TEXT_PATTERN))
    async def pe_exchange_import_inline(msg: Message, state: FSMContext):
        await _apply_import_from_text(msg, state, (msg.text or "").strip())

    @router.message(PromptEditorStates.entering_shared_prompt, F.text)
    async def pe_exchange_import_text(msg: Message, state: FSMContext):
        await _apply_import_from_text(msg, state, (msg.text or "").strip())

    @router.message(PromptEditorStates.entering_shared_prompt, F.document)
    async def pe_exchange_import_document(msg: Message, state: FSMContext):
        uid = msg.from_user.id if msg.from_user else 0
        telegram_locale = msg.from_user.language_code if msg.from_user else None

        document = msg.document
        if document is None:
            await msg.answer(
                _t_msg(
                    msg,
                    "prompt_editor.exchange.import.document_missing",
                    "⚠️ Документ не найден. Отправьте текст с кодом обмена.",
                ),
                reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            return

        if document.file_size and document.file_size > _MAX_IMPORT_FILE_BYTES:
            await msg.answer(
                _t_msg(
                    msg,
                    "prompt_editor.exchange.import.file_too_large",
                    "❌ Файл слишком большой. Отправьте файл до {size_kb} KB.",
                    params={"size_kb": _MAX_IMPORT_FILE_BYTES // 1024},
                ),
                reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            return

        bot = msg.bot
        if bot is None:
            await msg.answer(
                _t_msg(
                    msg,
                    "prompt_editor.exchange.import.bot_unavailable",
                    "❌ Бот недоступен в текущем сообщении.",
                ),
                reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            return

        try:
            buffer = BytesIO()
            stream = await bot.download(document.file_id, destination=buffer)
            if stream is None:
                raise RuntimeError("document stream is empty")
            raw_text = buffer.getvalue().decode("utf-8", errors="replace")
        except (TelegramBadRequest, RuntimeError, OSError):
            await msg.answer(
                _t_msg(
                    msg,
                    "prompt_editor.exchange.import.read_failed",
                    "❌ Не удалось прочитать документ. Отправьте код текстом.",
                ),
                reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
            )
            return

        await _apply_import_from_text(msg, state, raw_text.strip())

    @router.message(PromptEditorStates.entering_shared_prompt)
    async def pe_exchange_import_unsupported(msg: Message):
        uid = msg.from_user.id if msg.from_user else 0
        telegram_locale = msg.from_user.language_code if msg.from_user else None
        await msg.answer(
            _t_msg(
                msg,
                "prompt_editor.exchange.import.unsupported",
                "⚠️ Отправьте текстовый код обмена или .txt-файл.",
            ),
            reply_markup=_exchange_import_keyboard(uid=uid, telegram_locale=telegram_locale),
        )
