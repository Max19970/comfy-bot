from __future__ import annotations

from collections.abc import Awaitable, Callable
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
from core.runtime import PromptRequest
from core.states import PromptEditorStates
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button

_MAX_IMPORT_FILE_BYTES = 128_000
_TOKEN_INLINE_LIMIT = 3500
_TOKEN_TEXT_PATTERN = r"CBOT_PROMPT_V\d+:"


@dataclass
class PromptEditorExchangeHandlersDeps:
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    normalize_params: Callable[[GenerationParams], GenerationParams]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def _exchange_menu_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [
                button("📤 Копировать", "pe:exchange:export"),
                button("📥 Вставить", "pe:exchange:import"),
            ],
            [back_button("pe:back")],
        ]
    )


def _exchange_result_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [back_button("pe:exchange", text="⬅️ К обмену")],
            [back_button("pe:back", text="⬅️ В редактор")],
        ]
    )


def _exchange_import_keyboard() -> InlineKeyboardMarkup:
    return build_keyboard([[back_button("pe:exchange", text="⬅️ К обмену")]])


def _preview_line(label: str, value: str) -> str:
    text = value.strip()
    if not text:
        return f"{label}: <i>—</i>"
    return f"{label}: <code>{h(truncate(text, 85))}</code>"


def register_prompt_editor_exchange_handlers(
    router: Router,
    deps: PromptEditorExchangeHandlersDeps,
) -> None:
    async def _apply_import_from_text(
        message: Message,
        state: FSMContext,
        raw_text: str,
    ) -> None:
        payload = await deps.require_prompt_request_for_message(message, state)
        if not payload:
            return

        uid, req = payload
        try:
            imported_params = import_prompt_token(raw_text)
        except PromptExchangeError as exc:
            await message.answer(
                "❌ Не удалось импортировать код обмена.\n"
                f"<i>{h(exc)}</i>\n\n"
                f"Ожидается код вида <code>{PROMPT_EXCHANGE_PREFIX}...</code>",
                reply_markup=_exchange_import_keyboard(),
            )
            return

        req.params = deps.normalize_params(imported_params)
        await deps.cleanup_user_message(message)
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            notice="✅ Параметры загружены из кода обмена.",
        )

    @router.callback_query(F.data == "pe:exchange")
    async def pe_exchange_menu(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        text = "\n".join(
            [
                "📋 <b>Копирование и вставка промпта</b>",
                "",
                "Экспортируйте полный набор параметров редактора в код обмена и "
                "отправьте его другому пользователю.",
                "",
                _preview_line("🟢 Positive", req.params.positive),
                _preview_line("🔴 Negative", req.params.negative),
            ]
        )
        await message.edit_text(text, reply_markup=_exchange_menu_keyboard())
        await cb.answer()

    @router.callback_query(F.data == "pe:exchange:export")
    async def pe_exchange_export(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        _, req = payload
        token = export_prompt_token(req.params)

        if len(token) > _TOKEN_INLINE_LIMIT:
            await message.edit_text(
                "📤 <b>Код обмена готов</b>\n"
                "\n"
                "Текущий код слишком длинный для безопасного вывода в одном "
                "сообщении. Отправил его .txt-файлом ниже.",
                reply_markup=_exchange_result_keyboard(),
            )
            await message.answer_document(
                BufferedInputFile(
                    (token + "\n").encode("utf-8"),
                    filename="comfybot_prompt_code.txt",
                ),
                caption=(
                    "📎 Код обмена. Скопируйте содержимое файла и вставьте через «📥 Вставить»."
                ),
            )
            await cb.answer("Код отправлен файлом.")
            return

        await message.edit_text(
            "📤 <b>Код обмена готов</b>\n"
            "\n"
            "Скопируйте строку целиком и отправьте её другому пользователю.\n"
            "Код содержит полный набор параметров редактора (prompt, модель, LoRA, "
            "sampler, размер, seed, референсы и улучшения).\n"
            "\n"
            f"<code>{h(token)}</code>",
            reply_markup=_exchange_result_keyboard(),
        )
        await cb.answer("Код обмена готов.")

    @router.callback_query(F.data == "pe:exchange:import")
    async def pe_exchange_import_start(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return

        await state.set_state(PromptEditorStates.entering_shared_prompt)
        await message.edit_text(
            "📥 <b>Вставка кода обмена</b>\n"
            "\n"
            "Отправьте код обмена сообщением.\n"
            "Можно вставить текст или отправить .txt-файл с кодом.\n"
            "\n"
            "Также можно просто отправить код прямо из главного экрана редактора.",
            reply_markup=_exchange_import_keyboard(),
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
        document = msg.document
        if document is None:
            await msg.answer(
                "⚠️ Документ не найден. Отправьте текст с кодом обмена.",
                reply_markup=_exchange_import_keyboard(),
            )
            return

        if document.file_size and document.file_size > _MAX_IMPORT_FILE_BYTES:
            await msg.answer(
                "❌ Файл слишком большой. Отправьте файл до 128 KB.",
                reply_markup=_exchange_import_keyboard(),
            )
            return

        bot = msg.bot
        if bot is None:
            await msg.answer(
                "❌ Бот недоступен в текущем сообщении.",
                reply_markup=_exchange_import_keyboard(),
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
                "❌ Не удалось прочитать документ. Отправьте код текстом.",
                reply_markup=_exchange_import_keyboard(),
            )
            return

        await _apply_import_from_text(msg, state, raw_text.strip())

    @router.message(PromptEditorStates.entering_shared_prompt)
    async def pe_exchange_import_unsupported(msg: Message):
        await msg.answer(
            "⚠️ Отправьте текстовый код обмена или .txt-файл.",
            reply_markup=_exchange_import_keyboard(),
        )
