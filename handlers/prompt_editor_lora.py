from __future__ import annotations

from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from comfyui_client import ComfyUIClient, GenerationParams
from core.html_utils import h
from core.runtime import RuntimeStore
from core.states import PromptEditorStates
from core.ui import loras_text
from model_downloader import ModelDownloader


def checkpoint_base_model(checkpoint_name: str, downloader: ModelDownloader) -> str:
    name = checkpoint_name.strip()
    if not name:
        return ""

    meta = downloader.get_model_metadata(name, model_type="checkpoint")
    base = str(meta.get("base_model") if meta else "").strip()
    if base:
        return base
    return downloader.infer_base_model(name)


def lora_base_model(lora_name: str, downloader: ModelDownloader) -> str:
    name = lora_name.strip()
    if not name:
        return ""

    meta = downloader.get_model_metadata(name, model_type="lora")
    base = str(meta.get("base_model") if meta else "").strip()
    if base:
        return base
    return downloader.infer_base_model(name)


def lora_trained_words(lora_name: str, downloader: ModelDownloader) -> list[str]:
    return downloader.get_lora_trained_words(lora_name)[:12]


def lora_compatibility(
    checkpoint_name: str,
    lora_name: str,
    downloader: ModelDownloader,
) -> tuple[str, str, str]:
    ckpt_base = checkpoint_base_model(checkpoint_name, downloader)
    lora_base = lora_base_model(lora_name, downloader)
    if ckpt_base and lora_base:
        if downloader.base_models_compatible(ckpt_base, lora_base):
            return "compatible", ckpt_base, lora_base
        return "incompatible", ckpt_base, lora_base
    return "unknown", ckpt_base, lora_base


def incompatible_loras(
    params: GenerationParams,
    downloader: ModelDownloader,
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for lora_name, _ in params.loras:
        status, ckpt_base, lora_base = lora_compatibility(
            params.checkpoint,
            lora_name,
            downloader,
        )
        if status == "incompatible":
            result.append((lora_name, ckpt_base, lora_base))
    return result


def lora_picker_items(
    checkpoint_name: str,
    client: ComfyUIClient,
    downloader: ModelDownloader,
) -> tuple[list[str], list[str]]:
    entries: list[tuple[int, str, str, str]] = []
    for name in client.info.loras:
        status, _, lora_base = lora_compatibility(checkpoint_name, name, downloader)
        if status == "compatible":
            rank = 0
            icon = "✅"
        elif status == "incompatible":
            rank = 2
            icon = "⚠️"
        else:
            rank = 1
            icon = "❔"

        label = f"{icon} {name}"
        if lora_base:
            label += f" [{lora_base}]"
        entries.append((rank, name.casefold(), name, label))

    entries.sort(key=lambda item: (item[0], item[1]))
    names = [item[2] for item in entries]
    labels = [item[3] for item in entries]
    return names, labels


def merge_prompt_with_words(prompt: str, words: list[str]) -> str:
    base = prompt.strip()
    if not words:
        return base

    normalized_base = base.casefold()
    missing = [word for word in words if word.casefold() not in normalized_base]
    if not missing:
        return base

    addon = ", ".join(missing)
    if not base:
        return addon
    return f"{base}, {addon}"


async def show_lora_menu(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    runtime: RuntimeStore,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    edit: bool = True,
    notice: str = "",
) -> None:
    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer("Активный запрос не найден. Используйте /generate.")
        return

    lines: list[str] = []
    if notice:
        lines.append(f"💬 <i>{h(notice)}</i>")
        lines.append("")

    lines.append("🧲 <b>LoRA chain</b>")
    lines.append(f"Текущая цепочка: {loras_text(req.params.loras)}")

    ckpt_base = checkpoint_base_model(req.params.checkpoint, downloader)
    if ckpt_base:
        lines.append(f"Checkpoint base: <code>{h(ckpt_base)}</code>")

    bad = incompatible_loras(req.params, downloader)
    if bad:
        bad_names = ", ".join(name for name, _, _ in bad[:3])
        suffix = "" if len(bad) <= 3 else f" и ещё {len(bad) - 3}"
        lines.append(f"⚠️ Возможна несовместимость LoRA: {h(bad_names)}{h(suffix)}")

    text = "\n".join(lines)
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="➕ Добавить",
                    callback_data="pe:lora:add",
                )
            ],
            [
                InlineKeyboardButton(
                    text="➖ Удалить последнюю",
                    callback_data="pe:lora:remove_last",
                )
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Очистить все",
                    callback_data="pe:lora:clear",
                )
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data="pe:back",
                )
            ],
        ]
    )
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
                runtime.user_ui_panels[uid] = {
                    "chat_id": edited.chat.id,
                    "message_id": edited.message_id,
                }
                await state.set_state(PromptEditorStates.editing)
                return
        except Exception:
            pass

    if edit:
        try:
            edited = await message.edit_text(text, reply_markup=kb)
            if isinstance(edited, Message):
                req.ui_chat_id = edited.chat.id
                req.ui_message_id = edited.message_id
                runtime.user_ui_panels[uid] = {
                    "chat_id": edited.chat.id,
                    "message_id": edited.message_id,
                }
                await state.set_state(PromptEditorStates.editing)
                return
        except Exception:
            pass

    sent = await message.answer(text, reply_markup=kb)
    req.ui_chat_id = sent.chat.id
    req.ui_message_id = sent.message_id
    runtime.user_ui_panels[uid] = {
        "chat_id": sent.chat.id,
        "message_id": sent.message_id,
    }
    await state.set_state(PromptEditorStates.editing)
