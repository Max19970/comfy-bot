from __future__ import annotations

import logging
import asyncio
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import ComfyUIClient, GenerationParams
from config import Config
from core.html_utils import h, progress_bar, truncate
from core.image_utils import image_dimensions
from core.runtime import (
    ActiveGeneration,
    PreviewArtifact,
    PromptRequest,
    RuntimeStore,
    get_user_pro_mode,
    set_user_pro_mode,
)
from core.states import PromptEditorStates
from core.telegram import callback_user_id, message_user_id
from core.ui import (
    MAX_REFERENCE_IMAGES,
    SIZES,
    paginated_keyboard,
    params_summary_for_mode,
)
from model_downloader import ModelDownloader
from smart_prompt import SmartPromptService

from .prompt_editor_lora import (
    checkpoint_base_model as _checkpoint_base_model,
    incompatible_loras as _incompatible_loras,
    lora_base_model as _lora_base_model,
    lora_compatibility as _lora_compatibility,
    lora_picker_items as _lora_picker_items,
    lora_trained_words as _lora_trained_words,
    merge_prompt_with_words as _merge_prompt_with_words,
    show_lora_menu as _show_lora_menu,
)
from .prompt_editor_references import (
    add_reference_from_message as _add_reference_from_message,
    collect_reference_images as _collect_reference_images,
    denoise_from_reference_strength as _denoise_from_reference_strength,
    download_reference_image as _download_reference_image,
    make_reference_image as _make_reference_image,
    show_reference_menu as _show_reference_menu,
)
from .prompt_editor_send import (
    deliver_generated_images as _deliver_generated_images,
    generation_result_keyboard as _generation_result_keyboard,
    preview_image_keyboard as _preview_image_keyboard,
)
from .prompt_editor_handlers_smart import (
    PromptEditorSmartHandlersDeps,
    register_prompt_editor_smart_handlers,
)
from .prompt_editor_handlers_lora import (
    PromptEditorLoraHandlersDeps,
    register_prompt_editor_lora_handlers,
)
from .prompt_editor_handlers_references import (
    PromptEditorReferenceHandlersDeps,
    register_prompt_editor_reference_handlers,
)
from .prompt_editor_handlers_send import (
    PromptEditorSendHandlersDeps,
    register_prompt_editor_send_handlers,
)
from .prompt_editor_handlers_flow import (
    PromptEditorFlowHandlersDeps,
    register_prompt_editor_flow_handlers,
)
from .prompt_editor_handlers_edit import (
    PromptEditorEditHandlersDeps,
    register_prompt_editor_edit_handlers,
)
from .prompt_editor_handlers_exchange import (
    PromptEditorExchangeHandlersDeps,
    register_prompt_editor_exchange_handlers,
)
from .prompt_editor_handlers_thematic import (
    PromptEditorThematicHandlersDeps,
    register_prompt_editor_thematic_handlers,
)
from .prompt_editor_smart import (
    merge_prompt_text as _merge_prompt_text,
    prompt_input_text as _prompt_input_text,
    prompt_preview as _prompt_preview,
    smart_prompt_result_keyboard as _smart_prompt_result_keyboard,
)
from .prompt_editor_ui import (
    back_keyboard as _back_keyboard,
    editor_keyboard as _editor_keyboard,
    operation_action_text as _operation_action_text,
)

logger = logging.getLogger(__name__)

SMART_PROMPT_INPUT_MAX_CHARS = 4000


@dataclass
class PromptEditorService:
    ensure_models: Callable[[Message], Awaitable[bool]]
    normalize_params: Callable[[GenerationParams], GenerationParams]
    show_prompt_editor: Callable[..., Awaitable[None]]


def register_prompt_editor_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    smart_prompt: SmartPromptService | None = None,
) -> PromptEditorService:
    def _remember_prompt_panel(req: PromptRequest, panel_msg: Message) -> None:
        if panel_msg.chat is None:
            return
        req.ui_chat_id = panel_msg.chat.id
        req.ui_message_id = panel_msg.message_id

        uid: int | None = None
        for user_id, active_req in runtime.active_prompt_requests.items():
            if active_req is req:
                uid = user_id
                break
        if uid is not None:
            runtime.user_ui_panels[uid] = {
                "chat_id": panel_msg.chat.id,
                "message_id": panel_msg.message_id,
            }

    async def _edit_prompt_panel_by_anchor(
        req: PromptRequest,
        source_message: Message,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message | None:
        if req.ui_chat_id is None or req.ui_message_id is None:
            return None

        bot = source_message.bot
        if bot is None:
            return None

        try:
            edited = await bot.edit_message_text(
                text=text,
                chat_id=req.ui_chat_id,
                message_id=req.ui_message_id,
                reply_markup=reply_markup,
            )
            if isinstance(edited, Message):
                _remember_prompt_panel(req, edited)
                return edited
        except Exception:
            return None
        return None

    async def _show_prompt_panel(
        message: Message,
        req: PromptRequest,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        prefer_edit: bool,
    ) -> Message:
        anchored = await _edit_prompt_panel_by_anchor(req, message, text, reply_markup)
        if anchored is not None:
            return anchored

        if prefer_edit:
            try:
                edited = await message.edit_text(text, reply_markup=reply_markup)
                if isinstance(edited, Message):
                    _remember_prompt_panel(req, edited)
                    return edited
            except Exception:
                pass

        sent = await message.answer(text, reply_markup=reply_markup)
        _remember_prompt_panel(req, sent)
        return sent

    async def _move_prompt_panel_to_bottom(
        message: Message,
        req: PromptRequest,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
    ) -> Message:
        prev_chat_id = req.ui_chat_id
        prev_message_id = req.ui_message_id

        sent = await message.answer(text, reply_markup=reply_markup)
        _remember_prompt_panel(req, sent)

        if (
            prev_chat_id is not None
            and prev_message_id is not None
            and (prev_chat_id != sent.chat.id or prev_message_id != sent.message_id)
            and message.bot is not None
        ):
            try:
                await message.bot.delete_message(
                    chat_id=prev_chat_id,
                    message_id=prev_message_id,
                )
            except Exception:
                pass

        return sent

    async def ensure_models(message: Message) -> bool:
        if not client.info.checkpoints:
            await message.answer(
                "\u23f3 \u0417\u0430\u0433\u0440\u0443\u0436\u0430\u044e \u0441\u043f\u0438\u0441\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439\u2026"
            )
            try:
                await client.refresh_info()
            except Exception as exc:
                await message.answer(
                    f"\u274c <b>\u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u044c\u0441\u044f \u043a ComfyUI</b>\n<code>{h(exc)}</code>"
                )
                return False
        if not client.info.checkpoints:
            await message.answer(
                "\u274c \u041d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0435 ComfyUI \u043d\u0435\u0442 \u043d\u0438 \u043e\u0434\u043d\u043e\u0433\u043e checkpoint."
            )
            return False
        return True

    def default_params() -> GenerationParams:
        return GenerationParams(
            width=cfg.default_width,
            height=cfg.default_height,
            steps=cfg.default_steps,
            cfg=cfg.default_cfg,
            sampler=cfg.default_sampler,
            scheduler=cfg.default_scheduler,
            denoise=cfg.default_denoise,
        )

    async def require_prompt_request_for_message(
        msg: Message,
        state: FSMContext,
    ) -> tuple[int, PromptRequest] | None:
        uid = message_user_id(msg)
        req = runtime.active_prompt_requests.get(uid)
        if req:
            return uid, req
        await state.clear()
        await msg.answer(
            "\u274c \u0410\u043a\u0442\u0438\u0432\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d. \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\u0442\u0435 /generate."
        )
        return None

    async def require_prompt_request_for_callback(
        cb: CallbackQuery,
    ) -> tuple[int, PromptRequest] | None:
        uid = callback_user_id(cb)
        req = runtime.active_prompt_requests.get(uid)
        if req:
            return uid, req
        await cb.answer(
            "\u274c \u041d\u0435\u0442 \u0430\u043a\u0442\u0438\u0432\u043d\u043e\u0433\u043e \u0437\u0430\u043f\u0440\u043e\u0441\u0430.",
            show_alert=True,
        )
        return None

    async def cleanup_user_message(message: Message) -> None:
        try:
            await message.delete()
        except Exception:
            return

    async def open_paginated_choice(
        cb: CallbackQuery,
        *,
        title: str,
        items: list[str],
        prefix: str,
        back_callback: str = "pe:back",
    ) -> None:
        kb = paginated_keyboard(
            items,
            0,
            prefix,
            extra=[
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data=back_callback,
                    )
                ]
            ],
        )
        await cb.message.edit_text(title, reply_markup=kb)
        await cb.answer()

    async def change_paginated_choice_page(
        cb: CallbackQuery,
        *,
        items: list[str],
        prefix: str,
        back_callback: str = "pe:back",
    ) -> None:
        page = int(cb.data.split(":", 1)[1])
        kb = paginated_keyboard(
            items,
            page,
            prefix,
            extra=[
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data=back_callback,
                    )
                ]
            ],
        )
        await cb.message.edit_reply_markup(reply_markup=kb)
        await cb.answer()

    async def set_prompt_param_from_callback(
        cb: CallbackQuery,
        state: FSMContext,
        *,
        field: str,
        value: Any,
        notice: str,
    ) -> None:
        payload = await require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        setattr(req.params, field, value)
        await show_prompt_editor(cb.message, state, uid, edit=True, notice=notice)
        await cb.answer()

    def normalize_params(params: GenerationParams) -> GenerationParams:
        if not params.checkpoint and client.info.checkpoints:
            params.checkpoint = client.info.checkpoints[0]
        if client.info.samplers and params.sampler not in client.info.samplers:
            params.sampler = client.info.samplers[0]
        if client.info.schedulers and params.scheduler not in client.info.schedulers:
            params.scheduler = client.info.schedulers[0]
        if (
            params.upscale_model
            and params.upscale_model not in client.info.upscale_models
        ):
            params.upscale_model = ""
        if client.info.loras:
            params.loras = [
                item for item in params.loras if item[0] in client.info.loras
            ]
        else:
            params.loras = []

        normalized_refs: list[dict[str, str]] = []
        for item in params.reference_images:
            if isinstance(item, dict):
                file_id = str(item.get("file_id", "")).strip()
                if not file_id:
                    continue
                ref_id = str(item.get("id") or uuid.uuid4().hex)
                normalized_refs.append({"id": ref_id, "file_id": file_id})
                continue

            if isinstance(item, str):
                file_id = item.strip()
                if file_id:
                    normalized_refs.append({"id": uuid.uuid4().hex, "file_id": file_id})

        params.reference_images = normalized_refs[:MAX_REFERENCE_IMAGES]
        try:
            params.reference_strength = float(params.reference_strength)
        except Exception:
            params.reference_strength = 0.8
        params.reference_strength = max(0.0, min(2.0, params.reference_strength))
        return params

    def operation_action_text(operation: str) -> str:
        return _operation_action_text(operation)

    def prompt_input_text(
        field_name: str,
        current_value: str,
        *,
        allow_clear: bool = False,
    ) -> str:
        return _prompt_input_text(
            field_name,
            current_value,
            allow_clear=allow_clear,
        )

    def back_keyboard(
        callback_data: str = "pe:back",
        text: str = "⬅️ Назад",
    ) -> InlineKeyboardMarkup:
        return _back_keyboard(callback_data=callback_data, text=text)

    def merge_prompt_text(base: str, addon: str) -> str:
        return _merge_prompt_text(base, addon)

    def prompt_preview(text: str, max_len: int = 1200) -> str:
        return _prompt_preview(text, max_len=max_len)

    def smart_prompt_result_keyboard() -> InlineKeyboardMarkup:
        return _smart_prompt_result_keyboard()

    def make_reference_image(file_id: str) -> dict[str, str]:
        return _make_reference_image(file_id)

    async def download_reference_image(message: Message, file_id: str) -> bytes:
        return await _download_reference_image(message, file_id)

    async def collect_reference_images(
        message: Message,
        refs: list[dict[str, str]],
    ) -> tuple[list[bytes], int]:
        return await _collect_reference_images(message, refs)

    def denoise_from_reference_strength(strength: float) -> float:
        return _denoise_from_reference_strength(strength)

    def smart_prompt_is_enabled() -> bool:
        return bool(smart_prompt and smart_prompt.is_enabled())

    def checkpoint_base_model(checkpoint_name: str) -> str:
        return _checkpoint_base_model(checkpoint_name, downloader)

    def lora_base_model(lora_name: str) -> str:
        return _lora_base_model(lora_name, downloader)

    def lora_trained_words(lora_name: str) -> list[str]:
        return _lora_trained_words(lora_name, downloader)

    def lora_compatibility(
        checkpoint_name: str,
        lora_name: str,
    ) -> tuple[str, str, str]:
        return _lora_compatibility(checkpoint_name, lora_name, downloader)

    def incompatible_loras(
        params: GenerationParams,
    ) -> list[tuple[str, str, str]]:
        return _incompatible_loras(params, downloader)

    def lora_picker_items(
        checkpoint_name: str,
    ) -> tuple[list[str], list[str]]:
        return _lora_picker_items(checkpoint_name, client, downloader)

    def merge_prompt_with_words(prompt: str, words: list[str]) -> str:
        return _merge_prompt_with_words(prompt, words)

    def changed_params_count(params: GenerationParams) -> int:
        current = normalize_params(GenerationParams(**asdict(params)))
        defaults = normalize_params(default_params())

        changed = 0
        if current.positive.strip() != defaults.positive.strip():
            changed += 1
        if current.negative.strip() != defaults.negative.strip():
            changed += 1
        if current.checkpoint != defaults.checkpoint:
            changed += 1
        if list(current.loras) != list(defaults.loras):
            changed += 1
        if current.upscale_model != defaults.upscale_model:
            changed += 1
        if current.width != defaults.width or current.height != defaults.height:
            changed += 1
        if current.steps != defaults.steps:
            changed += 1
        if abs(float(current.cfg) - float(defaults.cfg)) > 1e-6:
            changed += 1
        if current.sampler != defaults.sampler:
            changed += 1
        if current.scheduler != defaults.scheduler:
            changed += 1
        if abs(float(current.denoise) - float(defaults.denoise)) > 1e-6:
            changed += 1
        if current.seed != defaults.seed:
            changed += 1
        if current.batch_size != defaults.batch_size:
            changed += 1
        if (
            abs(float(current.reference_strength) - float(defaults.reference_strength))
            > 1e-6
        ):
            changed += 1
        if current.reference_images != defaults.reference_images:
            changed += 1
        if current.enable_hires_fix != defaults.enable_hires_fix:
            changed += 1
        if current.enable_freeu != defaults.enable_freeu:
            changed += 1
        if current.enable_pag != defaults.enable_pag:
            changed += 1

        return changed

    async def clear_smart_prompt_result_data(state: FSMContext) -> None:
        await state.update_data(
            pe_smart_prev_positive=None,
            pe_smart_prev_negative=None,
            pe_smart_generated_positive=None,
            pe_smart_generated_negative=None,
            pe_smart_notice=None,
        )

    async def show_smart_prompt_result_confirmation(
        message: Message,
        state: FSMContext,
        uid: int,
        *,
        edit: bool,
        notice: str = "",
    ) -> None:
        req = runtime.active_prompt_requests.get(uid)
        if not req:
            await state.clear()
            await message.answer("⚠️ Активный запрос не найден. Используйте /generate.")
            return

        data = await state.get_data()
        generated_positive = str(data.get("pe_smart_generated_positive") or "").strip()
        generated_negative = str(data.get("pe_smart_generated_negative") or "").strip()

        if not generated_positive and not generated_negative:
            await show_prompt_editor(
                message,
                state,
                uid,
                edit=edit,
                notice="⚠️ Результат Smart Prompt не найден.",
            )
            return

        lines: list[str] = ["🧠 <b>Smart Prompt (TIPO) готов</b>"]
        if notice:
            lines.extend(["", f"💬 <i>{h(notice)}</i>"])

        lines.extend(
            [
                "",
                "🟢 <b>Новый Positive:</b>",
                f"<code>{h(prompt_preview(generated_positive))}</code>",
                "",
                "🔴 <b>Новый Negative:</b>",
                f"<code>{h(prompt_preview(generated_negative))}</code>",
            ]
        )

        await state.set_state(PromptEditorStates.confirming_smart_prompt_result)
        text = "\n".join(lines)
        kb = smart_prompt_result_keyboard()
        await _show_prompt_panel(
            message,
            req,
            text,
            kb,
            prefer_edit=edit,
        )

    def add_reference_from_message(
        message: Message,
        refs: list[dict[str, str]],
    ) -> tuple[int, bool]:
        return _add_reference_from_message(message, refs)

    def editor_keyboard(
        req: PromptRequest,
        *,
        smart_prompt_enabled: bool,
        pro_mode: bool = False,
    ) -> InlineKeyboardMarkup:
        return _editor_keyboard(
            req,
            smart_prompt_enabled=smart_prompt_enabled,
            pro_mode=pro_mode,
        )

    async def show_prompt_editor(
        message: Message,
        state: FSMContext,
        uid: int,
        *,
        edit: bool = False,
        notice: str = "",
    ) -> None:
        req = runtime.active_prompt_requests.get(uid)
        if not req:
            await message.answer(
                "\u274c \u0410\u043a\u0442\u0438\u0432\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d. \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\u0442\u0435 /generate."
            )
            return

        req.params = normalize_params(req.params)
        await state.set_state(PromptEditorStates.editing)
        pro_mode = get_user_pro_mode(runtime, uid)

        lines: list[str] = []
        if notice:
            lines.append(f"\U0001f4ac <i>{h(notice)}</i>")
            lines.append("")
        lines.append(params_summary_for_mode(req.params, pro_mode=pro_mode))

        if smart_prompt_is_enabled():
            lines.append("\n🧠 <b>Smart Prompt (TIPO):</b> вкл.")

        ckpt_base = checkpoint_base_model(req.params.checkpoint)
        if ckpt_base:
            lines.append(f"\n🧬 <b>Checkpoint base:</b> <code>{h(ckpt_base)}</code>")

        bad_loras = incompatible_loras(req.params)
        if bad_loras:
            names = ", ".join(name for name, _, _ in bad_loras[:3])
            suffix = "" if len(bad_loras) <= 3 else f" и ещё {len(bad_loras) - 3}"
            lines.append(
                f"⚠️ <b>LoRA compatibility:</b> {len(bad_loras)} потенциально несовместимы ({h(names)}{h(suffix)})"
            )

        text = "\n".join(lines)
        kb = editor_keyboard(
            req,
            smart_prompt_enabled=smart_prompt_is_enabled(),
            pro_mode=pro_mode,
        )
        await _show_prompt_panel(
            message,
            req,
            text,
            kb,
            prefer_edit=edit,
        )

    async def open_prompt_request(
        message: Message,
        state: FSMContext,
        uid: int,
        params: GenerationParams,
        *,
        operation: str,
        notice: str = "",
    ) -> None:
        runtime.active_prompt_requests[uid] = PromptRequest(
            params=normalize_params(params),
            operation=operation,
        )
        await state.clear()
        await show_prompt_editor(message, state, uid, notice=notice)

    def generation_result_keyboard() -> InlineKeyboardMarkup:
        return _generation_result_keyboard()

    def preview_image_keyboard(artifact_id: str) -> InlineKeyboardMarkup:
        return _preview_image_keyboard(artifact_id)

    async def deliver_generated_images(
        message: Message,
        images: list[bytes],
        *,
        used_seed: int | str,
        mode: str,
        preview_keyboards: list[InlineKeyboardMarkup] | None = None,
    ) -> None:
        await _deliver_generated_images(
            message,
            images,
            used_seed=used_seed,
            mode=mode,
            preview_keyboards=preview_keyboards,
        )

    def _prune_preview_artifacts(uid: int, *, max_items: int = 40) -> None:
        owned = [
            item for item in runtime.preview_artifacts.values() if item.owner_uid == uid
        ]
        if len(owned) <= max_items:
            return
        owned.sort(key=lambda item: item.created_at)
        to_remove = owned[: len(owned) - max_items]
        for item in to_remove:
            runtime.preview_artifacts.pop(item.artifact_id, None)

    async def run_generate_operation(
        message: Message,
        state: FSMContext,
        uid: int,
    ) -> None:
        req = runtime.active_prompt_requests.get(uid)
        if not req:
            await message.answer(
                "\u274c \u0410\u043a\u0442\u0438\u0432\u043d\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d. \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0439\u0442\u0435 /generate."
            )
            return

        params = normalize_params(req.params)
        if not params.checkpoint:
            await show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice="❌ Выберите checkpoint в редакторе перед генерацией.",
            )
            return

        bad_loras = incompatible_loras(params)
        lora_warning = ""
        if bad_loras:
            listed = ", ".join(name for name, _, _ in bad_loras[:3])
            suffix = "" if len(bad_loras) <= 3 else f" и ещё {len(bad_loras) - 3}"
            lora_warning = (
                "⚠️ <b>Внимание:</b> потенциально несовместимые LoRA: "
                f"<code>{h(listed)}</code>{h(suffix)}"
            )

        import random as _random

        used_seed = params.seed if params.seed >= 0 else _random.randint(0, 2**63 - 1)
        generation_params = GenerationParams(**asdict(params))
        generation_params.seed = used_seed

        reference_images, failed_refs = await collect_reference_images(
            message,
            params.reference_images,
        )
        if params.reference_images and not reference_images:
            await show_prompt_editor(
                message,
                state,
                uid,
                edit=True,
                notice=(
                    "❌ Не удалось загрузить референс-картинки. "
                    "Загрузите их заново в редакторе."
                ),
            )
            return

        reference_mode = client.resolve_reference_mode(bool(reference_images))
        if reference_mode == "img2img" and reference_images:
            generation_params.denoise = min(
                generation_params.denoise,
                denoise_from_reference_strength(generation_params.reference_strength),
            )

        ckpt_short = h(truncate(params.checkpoint, 30))
        start_lines = [
            f"\u23f3 <b>\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f</b> | {ckpt_short} | {params.width}\u00d7{params.height}"
        ]
        if reference_images:
            ref_info = f"\U0001f5bc <b>Ref:</b> {len(reference_images)}"
            if failed_refs:
                ref_info += f" (\u274c {failed_refs})"
            if reference_mode == "ipadapter":
                ref_info += f" | IP-Adapter, str={generation_params.reference_strength}"
            else:
                ref_info += (
                    f" | img2img, str={generation_params.reference_strength}"
                    f", dn={generation_params.denoise}"
                )
            start_lines.append(ref_info)
        if lora_warning:
            start_lines.append(lora_warning)
        status_intro = "\n".join(start_lines)

        return_to_editor_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🔄 Вернуться в редактор",
                        callback_data="pe:gen:back",
                    )
                ]
            ]
        )

        generation_id = uuid.uuid4().hex
        cancel_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="\u274c \u041e\u0442\u043c\u0435\u043d\u0438\u0442\u044c",
                        callback_data=f"pe:gen:cancel:{generation_id}",
                    )
                ]
            ]
        )

        status_msg = await _show_prompt_panel(
            message,
            req,
            status_intro,
            cancel_kb,
            prefer_edit=True,
        )
        _remember_prompt_panel(req, status_msg)

        async def _generation_task(
            uid: int,
            generation_id: str,
            generation_params: GenerationParams,
            reference_images: list[bytes],
            status_msg: Message,
            status_intro: str,
            cancel_kb: InlineKeyboardMarkup,
        ) -> None:
            import time as _time

            last_edit_time = 0.0
            last_text = status_intro
            edit_interval = 0.8

            async def _progress(current: int, total: int, text: str) -> None:
                nonlocal last_edit_time, last_text
                try:
                    now = _time.monotonic()
                    is_final = total > 0 and current >= total

                    if total > 0:
                        pct = min(100, max(0, (current * 100) // total))
                        first_line = f"🔄 {h(text)} | {pct}%"
                    else:
                        first_line = f"⏳ {h(text)}"

                    lines = [first_line]
                    if total > 0:
                        lines.append(progress_bar(current, total))
                    lines.extend(["", status_intro])

                    next_text = "\n".join(lines)
                    if next_text == last_text:
                        return
                    if not is_final and now - last_edit_time < edit_interval:
                        return

                    last_text = next_text
                    last_edit_time = now

                    # Keep cancel button on intermediate updates
                    kb = None if is_final else cancel_kb
                    await status_msg.edit_text(next_text, reply_markup=kb)
                except Exception:
                    logger.warning("Progress update failed", exc_info=True)

            async def _prompt_id_cb(prompt_id: str) -> None:
                active = runtime.active_generations.get(generation_id)
                if active is not None:
                    active.prompt_id = prompt_id

            try:
                images = await client.generate(
                    generation_params,
                    reference_images=reference_images,
                    progress_cb=_progress,
                    prompt_id_cb=_prompt_id_cb,
                )
            except asyncio.CancelledError:
                try:
                    await status_msg.edit_text(
                        f"{status_intro}\n\n\u274c \u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f \u043e\u0442\u043c\u0435\u043d\u0435\u043d\u0430."
                    )
                except Exception:
                    pass
                raise
            except TimeoutError:
                await status_msg.edit_text(
                    f"{status_intro}\n\n⏰ Время ожидания истекло.",
                    reply_markup=return_to_editor_kb,
                )
                return
            except Exception as exc:
                await status_msg.edit_text(
                    f"{status_intro}\n\n❌ Ошибка: <code>{h(exc)}</code>",
                    reply_markup=return_to_editor_kb,
                )
                return
            finally:
                runtime.active_generations.pop(generation_id, None)

            if not images:
                await status_msg.edit_text(
                    f"{status_intro}\n\n❌ Изображения не найдены.",
                    reply_markup=return_to_editor_kb,
                )
                return

            runtime.last_params[uid] = GenerationParams(**asdict(params))
            runtime.last_seeds[uid] = used_seed
            runtime.persist()
            preview_keyboards: list[InlineKeyboardMarkup] = []
            for image_bytes in images:
                artifact_id = uuid.uuid4().hex
                artifact_params = GenerationParams(**asdict(params))
                try:
                    width, height = image_dimensions(image_bytes)
                    artifact_params.width = width
                    artifact_params.height = height
                except Exception:
                    pass
                artifact = PreviewArtifact(
                    artifact_id=artifact_id,
                    owner_uid=uid,
                    image_bytes=image_bytes,
                    params=artifact_params,
                    used_seed=int(used_seed),
                )
                runtime.preview_artifacts[artifact_id] = artifact
                preview_keyboards.append(preview_image_keyboard(artifact_id))
            _prune_preview_artifacts(uid)

            preview_notice = "🖼 Превью отправлено как Telegram-фото."
            try:
                await deliver_generated_images(
                    status_msg,
                    images,
                    used_seed=used_seed,
                    mode="photo",
                    preview_keyboards=preview_keyboards,
                )
            except Exception:
                logger.exception("Failed to deliver generated preview")
                preview_notice = (
                    "⚠️ Не удалось отправить превью как фото. "
                    "Можно скачать PNG без сжатия."
                )

            done_text = (
                f"✅ <b>Готово!</b> {len(images)} изобр. | Seed: <code>{used_seed}</code>\n"
                f"\n{preview_notice}\n"
                "Для каждой превью доступны: отправка PNG и меню улучшений."
            )
            try:
                await _move_prompt_panel_to_bottom(
                    status_msg,
                    req,
                    done_text,
                    generation_result_keyboard(),
                )
            except Exception:
                fallback = await status_msg.answer(
                    done_text,
                    reply_markup=generation_result_keyboard(),
                )
                _remember_prompt_panel(req, fallback)

        # Create task
        task = asyncio.create_task(
            _generation_task(
                uid,
                generation_id,
                generation_params,
                reference_images,
                status_msg,
                status_intro,
                cancel_kb,
            )
        )
        runtime.active_generations[generation_id] = ActiveGeneration(
            owner_uid=uid,
            generation_id=generation_id,
            task=task,
            status_msg=status_msg,
        )

    async def show_lora_menu(
        message: Message,
        state: FSMContext,
        uid: int,
        *,
        edit: bool = True,
        notice: str = "",
    ) -> None:
        await _show_lora_menu(
            message,
            state,
            uid,
            runtime=runtime,
            client=client,
            downloader=downloader,
            edit=edit,
            notice=notice,
        )

    async def show_reference_menu(
        message: Message,
        state: FSMContext,
        uid: int,
        *,
        edit: bool = True,
        notice: str = "",
    ) -> None:
        await _show_reference_menu(
            message,
            state,
            uid,
            runtime=runtime,
            client=client,
            edit=edit,
            notice=notice,
        )

    register_prompt_editor_flow_handlers(
        router,
        PromptEditorFlowHandlersDeps(
            runtime=runtime,
            client=client,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            ensure_models=ensure_models,
            default_params=default_params,
            open_prompt_request=open_prompt_request,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            show_prompt_editor=show_prompt_editor,
            changed_params_count=changed_params_count,
            run_generate_operation=run_generate_operation,
        ),
    )

    register_prompt_editor_smart_handlers(
        router,
        PromptEditorSmartHandlersDeps(
            smart_prompt=smart_prompt,
            smart_prompt_input_max_chars=SMART_PROMPT_INPUT_MAX_CHARS,
            max_reference_images=MAX_REFERENCE_IMAGES,
            logger=logger,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            clear_smart_prompt_result_data=clear_smart_prompt_result_data,
            show_smart_prompt_result_confirmation=show_smart_prompt_result_confirmation,
            show_prompt_editor=show_prompt_editor,
            smart_prompt_is_enabled=smart_prompt_is_enabled,
            add_reference_from_message=add_reference_from_message,
            merge_prompt_text=merge_prompt_text,
            prompt_input_text=prompt_input_text,
            back_keyboard=back_keyboard,
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_edit_handlers(
        router,
        PromptEditorEditHandlersDeps(
            runtime=runtime,
            client=client,
            sizes=SIZES,
            callback_user_id=callback_user_id,
            get_user_pro_mode=get_user_pro_mode,
            set_user_pro_mode=set_user_pro_mode,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            open_paginated_choice=open_paginated_choice,
            change_paginated_choice_page=change_paginated_choice_page,
            set_prompt_param_from_callback=set_prompt_param_from_callback,
            show_prompt_editor=show_prompt_editor,
            prompt_input_text=prompt_input_text,
            back_keyboard=back_keyboard,
            incompatible_loras=incompatible_loras,
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_exchange_handlers(
        router,
        PromptEditorExchangeHandlersDeps(
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            show_prompt_editor=show_prompt_editor,
            normalize_params=normalize_params,
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_thematic_handlers(
        router,
        PromptEditorThematicHandlersDeps(
            max_reference_images=MAX_REFERENCE_IMAGES,
            is_freeu_supported=lambda: client.info.freeu_supported,
            is_pag_supported=lambda: client.info.pag_supported,
            is_tiled_diffusion_supported=lambda: client.info.tiled_diffusion_supported,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            show_prompt_editor=show_prompt_editor,
            back_keyboard=back_keyboard,
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_lora_handlers(
        router,
        PromptEditorLoraHandlersDeps(
            runtime=runtime,
            callback_user_id=callback_user_id,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            show_lora_menu=show_lora_menu,
            lora_picker_items=lora_picker_items,
            lora_compatibility=lora_compatibility,
            lora_trained_words=lora_trained_words,
            merge_prompt_with_words=merge_prompt_with_words,
            open_paginated_choice=open_paginated_choice,
            change_paginated_choice_page=change_paginated_choice_page,
            back_keyboard=back_keyboard,
            list_available_loras=lambda: list(client.info.loras),
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_reference_handlers(
        router,
        PromptEditorReferenceHandlersDeps(
            runtime=runtime,
            callback_user_id=callback_user_id,
            max_reference_images=MAX_REFERENCE_IMAGES,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            require_prompt_request_for_message=require_prompt_request_for_message,
            show_reference_menu=show_reference_menu,
            make_reference_image=make_reference_image,
            cleanup_user_message=cleanup_user_message,
        ),
    )

    register_prompt_editor_send_handlers(
        router,
        PromptEditorSendHandlersDeps(
            logger=logger,
            runtime=runtime,
            client=client,
            require_prompt_request_for_callback=require_prompt_request_for_callback,
            show_prompt_editor=show_prompt_editor,
            deliver_generated_images=deliver_generated_images,
            generation_result_keyboard=generation_result_keyboard,
            preview_image_keyboard=preview_image_keyboard,
        ),
    )

    return PromptEditorService(
        ensure_models=ensure_models,
        normalize_params=normalize_params,
        show_prompt_editor=show_prompt_editor,
    )
