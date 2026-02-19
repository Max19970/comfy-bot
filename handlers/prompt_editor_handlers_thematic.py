from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from comfyui_client import GenerationParams
from core.html_utils import h, truncate
from core.runtime import PromptRequest
from core.states import PromptEditorStates
from core.ui import custom_btn


@dataclass
class PromptEditorThematicHandlersDeps:
    max_reference_images: int
    is_freeu_supported: Callable[[], bool]
    is_pag_supported: Callable[[], bool]
    is_tiled_diffusion_supported: Callable[[], bool]
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    require_prompt_request_for_message: Callable[
        [Message, FSMContext], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    back_keyboard: Callable[..., InlineKeyboardMarkup]
    cleanup_user_message: Callable[[Message], Awaitable[None]]


def register_prompt_editor_thematic_handlers(
    router: Router,
    deps: PromptEditorThematicHandlersDeps,
) -> None:
    async def _show_from_request_anchor(
        message: Message,
        req: PromptRequest,
        text: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        if req.ui_chat_id is not None and req.ui_message_id is not None and message.bot:
            try:
                edited = await message.bot.edit_message_text(
                    text=text,
                    chat_id=req.ui_chat_id,
                    message_id=req.ui_message_id,
                    reply_markup=reply_markup,
                )
                if isinstance(edited, Message):
                    req.ui_chat_id = edited.chat.id
                    req.ui_message_id = edited.message_id
                    return
            except Exception:
                pass

        sent = await message.answer(text, reply_markup=reply_markup)
        req.ui_chat_id = sent.chat.id
        req.ui_message_id = sent.message_id

    @router.callback_query(F.data == "pe:sub:more")
    async def pe_submenu_more(cb: CallbackQuery):
        """'More settings' hub for simple mode — links to thematic submenus."""
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        params = req.params

        enh_count = sum(
            [
                params.enable_hires_fix,
                params.enable_freeu,
                params.enable_pag,
                bool(params.upscale_model),
                params.enable_tiled_diffusion,
            ]
        )
        enh_label = (
            f"\u2728 \u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f ({enh_count})"
            if enh_count
            else "\u2728 \u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f"
        )

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="\U0001f4d0 \u0421\u044d\u043c\u043f\u043b\u0438\u043d\u0433",
                        callback_data="pe:sub:sampling",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\U0001f5bc \u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435",
                        callback_data="pe:sub:image",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=enh_label,
                        callback_data="pe:sub:enhancements",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\U0001f4be \u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c",
                        callback_data="pe:save",
                    ),
                    InlineKeyboardButton(
                        text="\U0001f4da \u041f\u0440\u0435\u0441\u0435\u0442\u044b",
                        callback_data="pe:presets",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="📋 Копировать/вставить",
                        callback_data="pe:exchange",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\u2699\ufe0f <b>\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438</b>\n"
            "\n"
            "\u0417\u0434\u0435\u0441\u044c \u0441\u043e\u0431\u0440\u0430\u043d\u044b \u0432\u0441\u0435 \u043f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438, "
            "\u0440\u0430\u0437\u0431\u0438\u0442\u044b\u0435 \u043f\u043e \u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u044f\u043c.\n"
            "\n"
            "\U0001f4d0 <b>\u0421\u044d\u043c\u043f\u043b\u0438\u043d\u0433</b> \u2014 \u0448\u0430\u0433\u0438, CFG, \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438\n"
            "\U0001f5bc <b>\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435</b> \u2014 \u0440\u0430\u0437\u043c\u0435\u0440, \u0431\u0430\u0442\u0447, \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u044b\n"
            "\u2728 <b>\u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f</b> \u2014 Hi-res, FreeU, PAG, \u0430\u043f\u0441\u043a\u0435\u0439\u043b",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:sub:sampling")
    async def pe_submenu_sampling(cb: CallbackQuery):
        """Sampling thematic submenu: Steps, CFG, Sampler, Scheduler, Denoise, Seed."""
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        params = req.params
        seed_text = str(params.seed) if params.seed >= 0 else "\U0001f3b2"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"\U0001f522 Steps {params.steps}",
                        callback_data="pe:edit:steps",
                    ),
                    InlineKeyboardButton(
                        text=f"CFG {params.cfg}",
                        callback_data="pe:edit:cfg",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\u2699\ufe0f {truncate(params.sampler, 14)}",
                        callback_data="pe:edit:sampler",
                    ),
                    InlineKeyboardButton(
                        text=f"\U0001f4c8 {truncate(params.scheduler, 14)}",
                        callback_data="pe:edit:scheduler",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"Denoise {params.denoise}",
                        callback_data="pe:edit:denoise",
                    ),
                    InlineKeyboardButton(
                        text=f"Seed {seed_text}",
                        callback_data="pe:edit:seed",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\U0001f4d0 <b>\u0421\u044d\u043c\u043f\u043b\u0438\u043d\u0433</b>\n"
            "\n"
            "\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u043f\u0440\u043e\u0446\u0435\u0441\u0441\u0430 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f.\n"
            "\n"
            "\u2022 <b>Steps</b> \u2014 \u043a\u043e\u043b-\u0432\u043e \u0448\u0430\u0433\u043e\u0432 \u0434\u0438\u0444\u0444\u0443\u0437\u0438\u0438. "
            "\u0411\u043e\u043b\u044c\u0448\u0435 = \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u0435\u0435, \u043d\u043e \u043c\u0435\u0434\u043b\u0435\u043d\u043d\u0435\u0435 (20\u201335 \u043e\u043f\u0442\u0438\u043c\u0430\u043b\u044c\u043d\u043e)\n"
            "\u2022 <b>CFG</b> \u2014 \u043d\u0430\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0442\u0440\u043e\u0433\u043e \u043c\u043e\u0434\u0435\u043b\u044c \u0441\u043b\u0435\u0434\u0443\u0435\u0442 \u043f\u0440\u043e\u043c\u043f\u0442\u0443. "
            "\u041d\u0438\u0436\u0435 = \u0441\u0432\u043e\u0431\u043e\u0434\u043d\u0435\u0435, \u0432\u044b\u0448\u0435 = \u0442\u043e\u0447\u043d\u0435\u0435 (5\u20138 \u043e\u0431\u044b\u0447\u043d\u043e)\n"
            "\u2022 <b>Denoise</b> \u2014 \u0441\u0438\u043b\u0430 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0439 (1.0 = \u043f\u043e\u043b\u043d\u0430\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f)\n"
            "\u2022 <b>Sampler/Scheduler</b> \u2014 \u0430\u043b\u0433\u043e\u0440\u0438\u0442\u043c \u0438 \u0440\u0430\u0441\u043f\u0438\u0441\u0430\u043d\u0438\u0435 \u0448\u0443\u043c\u043e\u0434\u0430\u0432\u0430\n"
            "\u2022 <b>Seed</b> \u2014 \u0437\u0435\u0440\u043d\u043e \u0441\u043b\u0443\u0447\u0430\u0439\u043d\u043e\u0441\u0442\u0438 (\u0444\u0438\u043a\u0441. \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 = \u043f\u043e\u0432\u0442\u043e\u0440\u044f\u0435\u043c\u044b\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442)\n"
            "\n"
            f"<b>Steps:</b> <code>{params.steps}</code>  "
            f"<b>CFG:</b> <code>{params.cfg}</code>  "
            f"<b>Denoise:</b> <code>{params.denoise}</code>\n"
            f"<b>Sampler:</b> <code>{h(params.sampler)}</code>  "
            f"<b>Sched:</b> <code>{h(params.scheduler)}</code>\n"
            f"<b>Seed:</b> {seed_text}",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:sub:image")
    async def pe_submenu_image(cb: CallbackQuery):
        """Image thematic submenu: Size, Batch, References, Ref Strength."""
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        params = req.params
        ref_count = len(params.reference_images)

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"\U0001f4cf {params.width}\u00d7{params.height}",
                        callback_data="pe:edit:size",
                    ),
                    InlineKeyboardButton(
                        text=f"\U0001f4e6 Batch {params.batch_size}",
                        callback_data="pe:edit:batch",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\U0001f5bc Ref ({ref_count})",
                        callback_data="pe:edit:refs",
                    ),
                    InlineKeyboardButton(
                        text=f"Ref Str {params.reference_strength}",
                        callback_data="pe:edit:ref_strength",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\U0001f5bc <b>\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435</b>\n"
            "\n"
            "\u041f\u0430\u0440\u0430\u043c\u0435\u0442\u0440\u044b \u0432\u044b\u0445\u043e\u0434\u043d\u043e\u0433\u043e \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f.\n"
            "\n"
            "\u2022 <b>\u0420\u0430\u0437\u043c\u0435\u0440</b> \u2014 \u0448\u0438\u0440\u0438\u043d\u0430\u00d7\u0432\u044b\u0441\u043e\u0442\u0430 \u0432 \u043f\u0438\u043a\u0441\u0435\u043b\u044f\u0445. "
            "\u0411\u043e\u043b\u044c\u0448\u0438\u0439 \u0440\u0430\u0437\u043c\u0435\u0440 \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u0431\u043e\u043b\u044c\u0448\u0435 VRAM\n"
            "\u2022 <b>Batch</b> \u2014 \u043a\u043e\u043b-\u0432\u043e \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0439 \u0437\u0430 \u0440\u0430\u0437. "
            "\u0411\u043e\u043b\u044c\u0448\u0435 = \u0434\u043e\u043b\u044c\u0448\u0435, \u043d\u043e \u0431\u043e\u043b\u044c\u0448\u0435 \u0432\u0430\u0440\u0438\u0430\u043d\u0442\u043e\u0432\n"
            "\u2022 <b>Ref</b> \u2014 \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u043d\u044b\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f (IP-Adapter). "
            "\u041c\u043e\u0434\u0435\u043b\u044c \u043f\u0435\u0440\u0435\u043d\u0438\u043c\u0430\u0435\u0442 \u0441\u0442\u0438\u043b\u044c/\u043a\u043e\u043c\u043f\u043e\u0437\u0438\u0446\u0438\u044e\n"
            "\u2022 <b>Ref Strength</b> \u2014 \u0441\u0438\u043b\u0430 \u0432\u043b\u0438\u044f\u043d\u0438\u044f \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u0430 (0.0\u20141.0)\n"
            "\n"
            f"<b>\u0420\u0430\u0437\u043c\u0435\u0440:</b> <code>{params.width}\u00d7{params.height}</code>  "
            f"<b>Batch:</b> <code>{params.batch_size}</code>\n"
            f"<b>Ref:</b> {ref_count}/{deps.max_reference_images}  "
            f"<b>Ref str:</b> <code>{params.reference_strength}</code>",
            reply_markup=kb,
        )
        await cb.answer()

    # ------------------------------------------------------------------
    # Enhancements hub + per-enhancement submenus
    # ------------------------------------------------------------------

    @router.callback_query(F.data == "pe:sub:enhancements")
    async def pe_submenu_enhancements(cb: CallbackQuery):
        """Enhancements hub: links to Hi-res Fix, FreeU, PAG, Upscaler submenus."""
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        params = req.params

        hires_status = "\u2705" if params.enable_hires_fix else "\u274c"
        freeu_status = "\u2705" if params.enable_freeu else "\u274c"
        pag_status = "\u2705" if params.enable_pag else "\u274c"
        upsc_status = "\u2705" if params.upscale_model else "\u274c"
        tiled_status = "\u2705" if params.enable_tiled_diffusion else "\u274c"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"\U0001f527 Hi-res Fix {hires_status}",
                        callback_data="pe:enh:hires",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\u26a1 FreeU {freeu_status}",
                        callback_data="pe:enh:freeu",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\U0001f3af PAG {pag_status}",
                        callback_data="pe:enh:pag",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\U0001f50d Upscaler {upsc_status}",
                        callback_data="pe:enh:upscaler",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=f"\U0001f9e9 Tiled Diffusion {tiled_status}",
                        callback_data="pe:enh:tiled",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:back",
                    )
                ],
            ]
        )

        # Status text
        enh_lines: list[str] = []
        if params.enable_hires_fix:
            enh_lines.append(
                f"\U0001f527 Hi-res Fix — \u00d7{params.hires_scale}, denoise {params.hires_denoise}"
            )
        if params.enable_freeu:
            enh_lines.append("\u26a1 FreeU V2 — \u0432\u043a\u043b")
        if params.enable_pag:
            enh_lines.append(f"\U0001f3af PAG — scale {params.pag_scale}")
        if params.upscale_model:
            enh_lines.append(
                f"\U0001f50d Upscaler \u2014 {h(truncate(params.upscale_model, 30))}"
            )
        if params.enable_tiled_diffusion:
            enh_lines.append(
                f"\U0001f9e9 HyperTile \u2014 tile {params.tile_size}, "
                f"VAE {params.vae_tile_size}, overlap {params.tile_overlap}"
            )

        enh_text = (
            "\n".join(enh_lines)
            if enh_lines
            else "<i>\u041d\u0435\u0442 \u0430\u043a\u0442\u0438\u0432\u043d\u044b\u0445 \u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u0439</i>"
        )

        await cb.message.edit_text(
            "\u2728 <b>\u0423\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f</b>\n"
            "\n"
            "\u0414\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u044b\u0435 \u043c\u043e\u0434\u0443\u043b\u0438, \u043a\u043e\u0442\u043e\u0440\u044b\u0435 \u0443\u043b\u0443\u0447\u0448\u0430\u044e\u0442 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u043e "
            "\u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438, \u043d\u043e \u043c\u043e\u0433\u0443\u0442 \u0443\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0442\u044c \u0432\u0440\u0435\u043c\u044f. "
            "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u043c\u043e\u0434\u0443\u043b\u044c \u0434\u043b\u044f \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438.\n"
            "\n"
            f"{enh_text}",
            reply_markup=kb,
        )
        await cb.answer()

    # --- Hi-res Fix submenu ---

    def _hires_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            "\u274c \u0412\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
            if params.enable_hires_fix
            else "\u2705 \u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        )
        rows = [
            [
                InlineKeyboardButton(
                    text=toggle_text,
                    callback_data="pe:toggle:hires",
                ),
            ],
        ]
        if params.enable_hires_fix:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Scale: \u00d7{params.hires_scale}",
                        callback_data="pe:hires:scale",
                    ),
                    InlineKeyboardButton(
                        text=f"Denoise: {params.hires_denoise}",
                        callback_data="pe:hires:denoise",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _hires_submenu_text(params: GenerationParams) -> str:
        status = (
            "\u2705 \u0412\u041a\u041b"
            if params.enable_hires_fix
            else "\u274c \u0412\u042b\u041a\u041b"
        )
        return (
            f"\U0001f527 <b>Hi-res Fix</b>\n"
            "\n"
            "\u0413\u0435\u043d\u0435\u0440\u0438\u0440\u0443\u0435\u0442 \u043a\u0430\u0440\u0442\u0438\u043d\u043a\u0443 \u0432 \u043c\u0430\u043b\u043e\u043c \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0438, "
            "\u0437\u0430\u0442\u0435\u043c \u0443\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0435\u0442 \u0438 \u0434\u043e\u0440\u0438\u0441\u043e\u0432\u044b\u0432\u0430\u0435\u0442 \u0434\u0435\u0442\u0430\u043b\u0438 "
            "\u0432\u0442\u043e\u0440\u044b\u043c \u043f\u0440\u043e\u0445\u043e\u0434\u043e\u043c. \u042d\u0442\u043e \u043f\u043e\u0437\u0432\u043e\u043b\u044f\u0435\u0442 \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c "
            "\u0431\u043e\u043b\u044c\u0448\u0438\u0435 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f \u0431\u0435\u0437 \u0445\u0430\u0440\u0430\u043a\u0442\u0435\u0440\u043d\u044b\u0445 \u0430\u0440\u0442\u0435\u0444\u0430\u043a\u0442\u043e\u0432.\n"
            "\n"
            "\u2022 <b>Scale</b> \u2014 \u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c \u0443\u0432\u0435\u043b\u0438\u0447\u0435\u043d\u0438\u044f (\u00d71.5 \u0434\u043b\u044f \u0431\u0430\u043b\u0430\u043d\u0441\u0430, \u00d72.0 \u0434\u043b\u044f \u043c\u0430\u043a\u0441\u0438\u043c\u0443\u043c\u0430)\n"
            "\u2022 <b>Denoise</b> \u2014 \u0441\u0442\u0435\u043f\u0435\u043d\u044c \u043f\u0435\u0440\u0435\u0440\u0438\u0441\u043e\u0432\u043a\u0438 (0.3\u20140.5 \u2014 \u0441\u043e\u0445\u0440\u0430\u043d\u044f\u0435\u0442 \u043a\u043e\u043c\u043f\u043e\u0437\u0438\u0446\u0438\u044e, "
            "0.6\u20140.7 \u2014 \u0431\u043e\u043b\u044c\u0448\u0435 \u043d\u043e\u0432\u044b\u0445 \u0434\u0435\u0442\u0430\u043b\u0435\u0439)\n"
            "\u26a0\ufe0f \u0423\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0435\u0442 \u0432\u0440\u0435\u043c\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438 \u043f\u0440\u0438\u043c\u0435\u0440\u043d\u043e \u0432\u0434\u0432\u043e\u0435.\n"
            "\n"
            f"<b>\u0421\u0442\u0430\u0442\u0443\u0441:</b> {status}\n"
            f"<b>Scale:</b> \u00d7{params.hires_scale}  <b>Denoise:</b> {params.hires_denoise}"
        )

    @router.callback_query(F.data == "pe:enh:hires")
    async def pe_enh_hires(cb: CallbackQuery):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        await cb.message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:hires")
    async def pe_toggle_hires(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        req.params.enable_hires_fix = not req.params.enable_hires_fix
        await cb.message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        status = (
            "\u0432\u043a\u043b\u044e\u0447\u0435\u043d"
            if req.params.enable_hires_fix
            else "\u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d"
        )
        await cb.answer(f"\U0001f527 Hi-res Fix {status}")

    @router.callback_query(F.data == "pe:hires:scale")
    async def pe_hires_scale(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="1.25", callback_data="pe_hrs:1.25"),
                    InlineKeyboardButton(text="1.5", callback_data="pe_hrs:1.5"),
                    InlineKeyboardButton(text="2.0", callback_data="pe_hrs:2.0"),
                ],
                custom_btn("pe_hrs:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:hires",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "Hi-res Fix scale (\u043c\u043d\u043e\u0436\u0438\u0442\u0435\u043b\u044c, 1.0\u20143.0):",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_hrs:"))
    async def pe_hires_scale_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_hires_scale)
            await cb.message.edit_text(
                "Hi-res scale (1.0\u20143.0):",
                reply_markup=deps.back_keyboard("pe:enh:hires"),
            )
            await cb.answer()
            return
        req.params.hires_scale = float(value)
        await cb.message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        await cb.answer(f"\u2705 Scale: \u00d7{req.params.hires_scale}")

    @router.message(PromptEditorStates.entering_custom_hires_scale, F.text)
    async def pe_custom_hires_scale(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            assert 1.0 <= value <= 3.0
        except Exception:
            await msg.answer("\u0427\u0438\u0441\u043b\u043e 1.0\u20143.0:")
            return
        req.params.hires_scale = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _hires_submenu_text(req.params),
            _hires_submenu_kb(req.params),
        )

    @router.callback_query(F.data == "pe:hires:denoise")
    async def pe_hires_denoise(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="0.3", callback_data="pe_hrd:0.3"),
                    InlineKeyboardButton(text="0.4", callback_data="pe_hrd:0.4"),
                    InlineKeyboardButton(text="0.5", callback_data="pe_hrd:0.5"),
                ],
                [
                    InlineKeyboardButton(text="0.6", callback_data="pe_hrd:0.6"),
                    InlineKeyboardButton(text="0.7", callback_data="pe_hrd:0.7"),
                ],
                custom_btn("pe_hrd:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:hires",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "Hi-res Fix denoise (0.0\u20141.0, \u043c\u0435\u043d\u044c\u0448\u0435 = \u0431\u043b\u0438\u0436\u0435 \u043a \u043e\u0440\u0438\u0433\u0438\u043d\u0430\u043b\u0443):",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_hrd:"))
    async def pe_hires_denoise_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_hires_denoise)
            await cb.message.edit_text(
                "Hi-res denoise (0.0\u20141.0):",
                reply_markup=deps.back_keyboard("pe:enh:hires"),
            )
            await cb.answer()
            return
        req.params.hires_denoise = float(value)
        await cb.message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        await cb.answer(f"\u2705 Hi-res denoise: {req.params.hires_denoise}")

    @router.message(PromptEditorStates.entering_custom_hires_denoise, F.text)
    async def pe_custom_hires_denoise(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            assert 0.0 <= value <= 1.0
        except Exception:
            await msg.answer("\u0427\u0438\u0441\u043b\u043e 0.0\u20141.0:")
            return
        req.params.hires_denoise = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _hires_submenu_text(req.params),
            _hires_submenu_kb(req.params),
        )

    # --- FreeU submenu ---

    @router.callback_query(F.data == "pe:enh:freeu")
    async def pe_enh_freeu(cb: CallbackQuery):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        status = (
            "\u2705 \u0412\u041a\u041b"
            if req.params.enable_freeu
            else "\u274c \u0412\u042b\u041a\u041b"
        )
        toggle_text = (
            "\u274c \u0412\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
            if req.params.enable_freeu
            else "\u2705 \u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        )

        if not deps.is_freeu_supported():
            toggle_text = "\u26a0\ufe0f \u041d\u0435\u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=toggle_text,
                        callback_data="pe:toggle:freeu",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:sub:enhancements",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\u26a1 <b>FreeU V2</b>\n"
            "\n"
            "\u0411\u0435\u0441\u043f\u043b\u0430\u0442\u043d\u043e\u0435 \u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u0435 \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0430 \u0431\u0435\u0437 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0439 "
            "\u043d\u0430\u0433\u0440\u0443\u0437\u043a\u0438. FreeU \u043f\u0430\u0442\u0447\u0438\u0442 U-Net \u043c\u043e\u0434\u0435\u043b\u044c, "
            "\u0443\u0441\u0438\u043b\u0438\u0432\u0430\u044f \u043d\u0438\u0437\u043a\u043e\u0447\u0430\u0441\u0442\u043e\u0442\u043d\u044b\u0435 \u0434\u0435\u0442\u0430\u043b\u0438 "
            "\u0438 \u043f\u043e\u0434\u0430\u0432\u043b\u044f\u044f \u0432\u044b\u0441\u043e\u043a\u043e\u0447\u0430\u0441\u0442\u043e\u0442\u043d\u044b\u0439 \u0448\u0443\u043c. "
            "\u0418\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f \u0441\u0442\u0430\u043d\u043e\u0432\u044f\u0442\u0441\u044f \u0447\u0451\u0442\u0447\u0435 \u0438 \u0434\u0435\u0442\u0430\u043b\u0438\u0437\u0438\u0440\u043e\u0432\u0430\u043d\u043d\u0435\u0435.\n"
            "\n"
            "\U0001f4a1 \u041d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442 \u0434\u043e\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c\u043d\u043e\u0433\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0438 "
            "\u0438 \u043d\u0435 \u0438\u043c\u0435\u0435\u0442 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043a \u2014 \u043f\u0440\u043e\u0441\u0442\u043e \u0432\u043a\u043b/\u0432\u044b\u043a\u043b.\n"
            "\n"
            f"<b>\u0421\u0442\u0430\u0442\u0443\u0441:</b> {status}",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:freeu")
    async def pe_toggle_freeu(cb: CallbackQuery, state: FSMContext):
        if not deps.is_freeu_supported():
            await cb.answer(
                "\u26a0\ufe0f FreeU_V2 \u043d\u043e\u0434\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430 \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0435 ComfyUI.",
                show_alert=True,
            )
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        req.params.enable_freeu = not req.params.enable_freeu
        status_word = (
            "\u0432\u043a\u043b\u044e\u0447\u0435\u043d"
            if req.params.enable_freeu
            else "\u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d"
        )
        # Re-render FreeU submenu
        status = (
            "\u2705 \u0412\u041a\u041b"
            if req.params.enable_freeu
            else "\u274c \u0412\u042b\u041a\u041b"
        )
        toggle_text = (
            "\u274c \u0412\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
            if req.params.enable_freeu
            else "\u2705 \u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        )
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=toggle_text,
                        callback_data="pe:toggle:freeu",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:sub:enhancements",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\u26a1 <b>FreeU V2</b>\n"
            "\n"
            "\u041f\u0430\u0442\u0447\u0438\u0442 \u043c\u043e\u0434\u0435\u043b\u044c \u0434\u043b\u044f \u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f \u0434\u0435\u0442\u0430\u043b\u0435\u0439.\n"
            "\n"
            f"<b>\u0421\u0442\u0430\u0442\u0443\u0441:</b> {status}",
            reply_markup=kb,
        )
        await cb.answer(f"\u26a1 FreeU {status_word}")

    # --- PAG submenu ---

    def _pag_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            "\u274c \u0412\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
            if params.enable_pag
            else "\u2705 \u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        )
        rows = [
            [
                InlineKeyboardButton(
                    text=toggle_text,
                    callback_data="pe:toggle:pag",
                ),
            ],
        ]
        if params.enable_pag:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Scale: {params.pag_scale}",
                        callback_data="pe:pag:scale",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _pag_submenu_text(params: GenerationParams) -> str:
        status = (
            "\u2705 \u0412\u041a\u041b"
            if params.enable_pag
            else "\u274c \u0412\u042b\u041a\u041b"
        )
        return (
            "\U0001f3af <b>PAG (Perturbed-Attention Guidance)</b>\n"
            "\n"
            "\u0423\u043b\u0443\u0447\u0448\u0430\u0435\u0442 \u0441\u043b\u0435\u0434\u043e\u0432\u0430\u043d\u0438\u0435 \u043f\u0440\u043e\u043c\u043f\u0442\u0443 "
            "\u0431\u0435\u0437 \u043f\u043e\u0432\u044b\u0448\u0435\u043d\u0438\u044f CFG, \u0437\u0430 \u0441\u0447\u0451\u0442 \u043f\u0435\u0440\u0442\u0443\u0440\u0431\u0430\u0446\u0438\u0438 "
            "\u043a\u0430\u0440\u0442 \u0432\u043d\u0438\u043c\u0430\u043d\u0438\u044f (self-attention). \u041f\u043e\u043c\u043e\u0433\u0430\u0435\u0442 \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c "
            "\u0431\u043e\u043b\u0435\u0435 \u0442\u043e\u0447\u043d\u0443\u044e \u043a\u043e\u043c\u043f\u043e\u0437\u0438\u0446\u0438\u044e \u0438 \u0434\u0435\u0442\u0430\u043b\u0438.\n"
            "\n"
            "\u2022 <b>Scale</b> \u2014 \u0441\u0438\u043b\u0430 \u044d\u0444\u0444\u0435\u043a\u0442\u0430 (1\u20143 \u043c\u044f\u0433\u043a\u043e, "
            "3\u20145 \u0443\u043c\u0435\u0440\u0435\u043d\u043d\u043e, 5+ \u0430\u0433\u0440\u0435\u0441\u0441\u0438\u0432\u043d\u043e)\n"
            "\u26a0\ufe0f \u041d\u0435\u0437\u043d\u0430\u0447\u0438\u0442\u0435\u043b\u044c\u043d\u043e \u0443\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0435\u0442 \u0432\u0440\u0435\u043c\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438.\n"
            "\n"
            f"<b>\u0421\u0442\u0430\u0442\u0443\u0441:</b> {status}\n"
            f"<b>Scale:</b> {params.pag_scale}"
        )

    @router.callback_query(F.data == "pe:enh:pag")
    async def pe_enh_pag(cb: CallbackQuery):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload

        if not deps.is_pag_supported():
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                            callback_data="pe:sub:enhancements",
                        )
                    ]
                ]
            )
            await cb.message.edit_text(
                "\U0001f3af <b>PAG</b>\n\n"
                "\u26a0\ufe0f PerturbedAttentionGuidance \u043d\u043e\u0434\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430 \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0435.",
                reply_markup=kb,
            )
            await cb.answer()
            return

        await cb.message.edit_text(
            _pag_submenu_text(req.params),
            reply_markup=_pag_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:pag")
    async def pe_toggle_pag(cb: CallbackQuery, state: FSMContext):
        if not deps.is_pag_supported():
            await cb.answer(
                "\u26a0\ufe0f PerturbedAttentionGuidance \u043d\u043e\u0434\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430 \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0435.",
                show_alert=True,
            )
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        req.params.enable_pag = not req.params.enable_pag
        status_word = (
            "\u0432\u043a\u043b\u044e\u0447\u0435\u043d"
            if req.params.enable_pag
            else "\u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d"
        )
        await cb.message.edit_text(
            _pag_submenu_text(req.params),
            reply_markup=_pag_submenu_kb(req.params),
        )
        await cb.answer(f"\U0001f3af PAG {status_word}")

    @router.callback_query(F.data == "pe:pag:scale")
    async def pe_pag_scale(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="1.0", callback_data="pe_pag:1.0"),
                    InlineKeyboardButton(text="2.0", callback_data="pe_pag:2.0"),
                    InlineKeyboardButton(text="3.0", callback_data="pe_pag:3.0"),
                ],
                [
                    InlineKeyboardButton(text="4.0", callback_data="pe_pag:4.0"),
                    InlineKeyboardButton(text="5.0", callback_data="pe_pag:5.0"),
                ],
                custom_btn("pe_pag:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:pag",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "PAG Scale (0.5\u201410.0):",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_pag:"))
    async def pe_pag_scale_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_pag_scale)
            await cb.message.edit_text(
                "PAG Scale (0.5\u201410.0):",
                reply_markup=deps.back_keyboard("pe:enh:pag"),
            )
            await cb.answer()
            return
        req.params.pag_scale = float(value)
        await cb.message.edit_text(
            _pag_submenu_text(req.params),
            reply_markup=_pag_submenu_kb(req.params),
        )
        await cb.answer(f"\u2705 PAG scale: {req.params.pag_scale}")

    @router.message(PromptEditorStates.entering_custom_pag_scale, F.text)
    async def pe_custom_pag_scale(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        uid, req = payload
        try:
            value = float((msg.text or "").strip())
            assert 0.5 <= value <= 10.0
        except Exception:
            await msg.answer("\u0427\u0438\u0441\u043b\u043e 0.5\u201410.0:")
            return
        req.params.pag_scale = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _pag_submenu_text(req.params),
            _pag_submenu_kb(req.params),
        )

    # --- Upscaler submenu ---

    @router.callback_query(F.data == "pe:enh:upscaler")
    async def pe_enh_upscaler(cb: CallbackQuery):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        model = req.params.upscale_model
        status = (
            f"\u2705 {h(truncate(model, 30))}"
            if model
            else "\u274c \u0412\u042b\u041a\u041b"
        )

        rows = [
            [
                InlineKeyboardButton(
                    text="\U0001f50d \u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043c\u043e\u0434\u0435\u043b\u044c",
                    callback_data="pe:edit:upscaler",
                ),
            ],
        ]
        if model:
            rows.append(
                [
                    InlineKeyboardButton(
                        text="\u274c \u041e\u0442\u043a\u043b\u044e\u0447\u0438\u0442\u044c",
                        callback_data="pe:upsc:off",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                    callback_data="pe:sub:enhancements",
                )
            ]
        )

        kb = InlineKeyboardMarkup(inline_keyboard=rows)
        await cb.message.edit_text(
            "\U0001f50d <b>Upscaler</b>\n"
            "\n"
            "\u0423\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0435\u0442 \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u0433\u043e\u0442\u043e\u0432\u043e\u0439 \u043a\u0430\u0440\u0442\u0438\u043d\u043a\u0438 "
            "\u0441 \u043f\u043e\u043c\u043e\u0449\u044c\u044e \u043d\u0435\u0439\u0440\u043e\u0441\u0435\u0442\u0438. \u0412 \u043e\u0442\u043b\u0438\u0447\u0438\u0435 \u043e\u0442 Hi-res Fix, "
            "\u043d\u0435 \u0434\u043e\u0431\u0430\u0432\u043b\u044f\u0435\u0442 \u043d\u043e\u0432\u044b\u0445 \u0434\u0435\u0442\u0430\u043b\u0435\u0439, \u0430 \u0438\u043d\u0442\u0435\u043b\u043b\u0435\u043a\u0442\u0443\u0430\u043b\u044c\u043d\u043e "
            "\u043c\u0430\u0441\u0448\u0442\u0430\u0431\u0438\u0440\u0443\u0435\u0442 \u0441\u0443\u0449\u0435\u0441\u0442\u0432\u0443\u044e\u0449\u0438\u0435.\n"
            "\n"
            "\U0001f4a1 \u041f\u043e\u0434\u0445\u043e\u0434\u0438\u0442 \u0434\u043b\u044f \u0431\u044b\u0441\u0442\u0440\u043e\u0433\u043e \u0443\u0432\u0435\u043b\u0438\u0447\u0435\u043d\u0438\u044f \u0440\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u0438\u044f, "
            "\u043a\u043e\u0433\u0434\u0430 \u043d\u0435 \u043d\u0443\u0436\u043d\u0430 \u0434\u043e\u0440\u0438\u0441\u043e\u0432\u043a\u0430.\n"
            "\u26a0\ufe0f \u0423\u0432\u0435\u043b\u0438\u0447\u0438\u0432\u0430\u0435\u0442 \u0432\u0440\u0435\u043c\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u0438.\n"
            "\n"
            f"<b>\u041c\u043e\u0434\u0435\u043b\u044c:</b> {status}",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:upsc:off")
    async def pe_upsc_off(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        uid, req = payload
        req.params.upscale_model = ""
        await deps.show_prompt_editor(
            cb.message,
            state,
            uid,
            edit=True,
            notice="\u2705 Upscaler \u043e\u0442\u043a\u043b\u044e\u0447\u0435\u043d.",
        )
        await cb.answer()

    # --- Tiled Diffusion submenu (HyperTile + VAE Tiled — built-in) ---

    def _tiled_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            "\u274c \u0412\u044b\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
            if params.enable_tiled_diffusion
            else "\u2705 \u0412\u043a\u043b\u044e\u0447\u0438\u0442\u044c"
        )

        rows = [
            [
                InlineKeyboardButton(
                    text=toggle_text,
                    callback_data="pe:toggle:tiled",
                ),
            ],
        ]
        if params.enable_tiled_diffusion:
            rows.extend(
                [
                    [
                        InlineKeyboardButton(
                            text=f"\U0001f9e9 Tile Size: {params.tile_size}",
                            callback_data="pe:tiled:tile_size",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text=f"\U0001f5bc VAE Tile: {params.vae_tile_size}",
                            callback_data="pe:tiled:vae_tile",
                        ),
                        InlineKeyboardButton(
                            text=f"\U0001f300 Overlap: {params.tile_overlap}",
                            callback_data="pe:tiled:overlap",
                        ),
                    ],
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _tiled_submenu_text(params: GenerationParams) -> str:
        status = (
            "\u2705 \u0412\u041a\u041b"
            if params.enable_tiled_diffusion
            else "\u274c \u0412\u042b\u041a\u041b"
        )
        lines = [
            "\U0001f9e9 <b>HyperTile + Tiled VAE</b>\n",
            "\u0420\u0430\u0437\u0431\u0438\u0432\u0430\u0435\u0442 \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044e \u043d\u0430 \u0442\u0430\u0439\u043b\u044b, "
            "\u043f\u043e\u0437\u0432\u043e\u043b\u044f\u044f \u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d\u043e \u0433\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c "
            "\u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f \u0431\u043e\u043b\u044c\u0448\u043e\u0433\u043e \u0440\u0430\u0437\u043c\u0435\u0440\u0430 "
            "\u0441 \u043c\u0435\u043d\u044c\u0448\u0438\u043c \u0440\u0430\u0441\u0445\u043e\u0434\u043e\u043c VRAM.\n",
            "\n",
        ]
        if params.enable_tiled_diffusion:
            lines.append(
                f"<b>Tile Size:</b> <code>{params.tile_size}</code>  "
                f"<b>VAE Tile:</b> <code>{params.vae_tile_size}</code>  "
                f"<b>Overlap:</b> <code>{params.tile_overlap}</code>\n"
            )
        lines.append(
            "\n"
            "\U0001f4a1 \u0418\u0441\u043f\u043e\u043b\u044c\u0437\u0443\u0435\u0442 \u0432\u0441\u0442\u0440\u043e\u0435\u043d\u043d\u044b\u0435 \u043d\u043e\u0434\u044b "
            "HyperTile + VAE Tiled \u2014 \u043d\u0438\u043a\u0430\u043a\u0438\u0445 "
            "\u0440\u0430\u0441\u0448\u0438\u0440\u0435\u043d\u0438\u0439 \u043d\u0435 \u0442\u0440\u0435\u0431\u0443\u0435\u0442\u0441\u044f.\n"
            "\n"
            f"<b>\u0421\u0442\u0430\u0442\u0443\u0441:</b> {status}"
        )
        return "".join(lines)

    @router.callback_query(F.data == "pe:enh:tiled")
    async def pe_enh_tiled(cb: CallbackQuery):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        await cb.message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:tiled")
    async def pe_toggle_tiled(cb: CallbackQuery, state: FSMContext):
        if not deps.is_tiled_diffusion_supported():
            await cb.answer(
                "\u26a0\ufe0f HyperTile \u043d\u043e\u0434\u0430 \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u0430 \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0435 ComfyUI.",
                show_alert=True,
            )
            return
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        req.params.enable_tiled_diffusion = not req.params.enable_tiled_diffusion
        status_word = (
            "\u0432\u043a\u043b\u044e\u0447\u0435\u043d"
            if req.params.enable_tiled_diffusion
            else "\u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d"
        )
        await cb.message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer(f"\U0001f9e9 HyperTile {status_word}")

    @router.callback_query(F.data == "pe:tiled:tile_size")
    async def pe_tiled_tile_size(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_ts:{v}")
                    for v in (128, 192, 256)
                ],
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_ts:{v}")
                    for v in (384, 512, 768)
                ],
                custom_btn("pe_ts:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:tiled",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\U0001f9e9 <b>Tile Size</b>\n\n"
            "\u0420\u0430\u0437\u043c\u0435\u0440 \u0442\u0430\u0439\u043b\u0430 \u0434\u043b\u044f HyperTile. "
            "\u041c\u0435\u043d\u044c\u0448\u0435 = \u0431\u044b\u0441\u0442\u0440\u0435\u0435, "
            "\u0431\u043e\u043b\u044c\u0448\u0435 = \u043a\u0430\u0447\u0435\u0441\u0442\u0432\u0435\u043d\u043d\u0435\u0435.\n\n"
            "\U0001f4a1 \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f: 256.",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_ts:"))
    async def pe_ts_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_tile_size)
            await cb.message.edit_text(
                "Tile Size (64\u20131024):",
                reply_markup=deps.back_keyboard("pe:enh:tiled"),
            )
            await cb.answer()
            return
        req.params.tile_size = int(value)
        await cb.message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer(f"Tile Size: {value}")

    @router.message(PromptEditorStates.entering_custom_tile_size, F.text)
    async def pe_custom_tile_size(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        _, req = payload
        try:
            value = int((msg.text or "").strip())
            assert 64 <= value <= 1024
        except Exception:
            await msg.answer(
                "\u0426\u0435\u043b\u043e\u0435 \u0447\u0438\u0441\u043b\u043e 64\u20131024:"
            )
            return
        req.params.tile_size = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _tiled_submenu_text(req.params),
            _tiled_submenu_kb(req.params),
        )

    @router.callback_query(F.data == "pe:tiled:vae_tile")
    async def pe_tiled_vae_tile(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_vt:{v}")
                    for v in (256, 384, 512)
                ],
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_vt:{v}")
                    for v in (768, 1024, 2048)
                ],
                custom_btn("pe_vt:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:tiled",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\U0001f5bc <b>VAE Tile Size</b>\n\n"
            "\u0420\u0430\u0437\u043c\u0435\u0440 \u0442\u0430\u0439\u043b\u0430 \u0434\u043b\u044f VAE "
            "encode/decode.\n\n"
            "\U0001f4a1 \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f: 512.",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_vt:"))
    async def pe_vt_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_vae_tile_size)
            await cb.message.edit_text(
                "VAE Tile Size (128\u20134096):",
                reply_markup=deps.back_keyboard("pe:enh:tiled"),
            )
            await cb.answer()
            return
        req.params.vae_tile_size = int(value)
        await cb.message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer(f"VAE Tile: {value}")

    @router.message(PromptEditorStates.entering_custom_vae_tile_size, F.text)
    async def pe_custom_vae_tile(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        _, req = payload
        try:
            value = int((msg.text or "").strip())
            assert 128 <= value <= 4096
        except Exception:
            await msg.answer(
                "\u0426\u0435\u043b\u043e\u0435 \u0447\u0438\u0441\u043b\u043e 128\u20134096:"
            )
            return
        req.params.vae_tile_size = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _tiled_submenu_text(req.params),
            _tiled_submenu_kb(req.params),
        )

    @router.callback_query(F.data == "pe:tiled:overlap")
    async def pe_tiled_overlap(cb: CallbackQuery):
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_tovlp:{v}")
                    for v in (32, 48, 64)
                ],
                [
                    InlineKeyboardButton(text=str(v), callback_data=f"pe_tovlp:{v}")
                    for v in (96, 128, 256)
                ],
                custom_btn("pe_tovlp:custom"),
                [
                    InlineKeyboardButton(
                        text="\u2b05\ufe0f \u041d\u0430\u0437\u0430\u0434",
                        callback_data="pe:enh:tiled",
                    )
                ],
            ]
        )
        await cb.message.edit_text(
            "\U0001f300 <b>Overlap</b>\n\n"
            "\u041f\u0435\u0440\u0435\u043a\u0440\u044b\u0442\u0438\u0435 \u043c\u0435\u0436\u0434\u0443 \u0442\u0430\u0439\u043b\u0430\u043c\u0438 VAE. "
            "\u0411\u043e\u043b\u044c\u0448\u0435 = \u043c\u0435\u043d\u044c\u0448\u0435 \u0448\u0432\u043e\u0432, "
            "\u043d\u043e \u043c\u0435\u0434\u043b\u0435\u043d\u043d\u0435\u0435.\n\n"
            "\U0001f4a1 \u0420\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u0443\u0435\u0442\u0441\u044f: 64.",
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("pe_tovlp:"))
    async def pe_tovlp_chosen(cb: CallbackQuery, state: FSMContext):
        payload = await deps.require_prompt_request_for_callback(cb)
        if not payload:
            return
        _, req = payload
        value = cb.data.split(":")[1]
        if value == "custom":
            await state.set_state(PromptEditorStates.entering_custom_tile_overlap)
            await cb.message.edit_text(
                "Tile Overlap (0\u20132048):",
                reply_markup=deps.back_keyboard("pe:enh:tiled"),
            )
            await cb.answer()
            return
        req.params.tile_overlap = int(value)
        await cb.message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer(f"Overlap: {value}")

    @router.message(PromptEditorStates.entering_custom_tile_overlap, F.text)
    async def pe_custom_tile_overlap(msg: Message, state: FSMContext):
        payload = await deps.require_prompt_request_for_message(msg, state)
        if not payload:
            return
        _, req = payload
        try:
            value = int((msg.text or "").strip())
            assert 0 <= value <= 2048
        except Exception:
            await msg.answer(
                "\u0426\u0435\u043b\u043e\u0435 \u0447\u0438\u0441\u043b\u043e 0\u20132048:"
            )
            return
        req.params.tile_overlap = value
        await state.set_state(PromptEditorStates.editing)
        await deps.cleanup_user_message(msg)
        await _show_from_request_anchor(
            msg,
            req,
            _tiled_submenu_text(req.params),
            _tiled_submenu_kb(req.params),
        )
