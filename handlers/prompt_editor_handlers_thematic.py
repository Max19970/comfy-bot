from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from core.html_utils import h, truncate
from core.interaction import require_callback_message
from core.models import GenerationParams
from core.prompt_enhancements import (
    numeric_control_range_text,
    numeric_control_rows,
    numeric_enhancement_control,
)
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService

from .prompt_editor_enhancements import enhancements_menu_label
from .prompt_editor_handler_guards import require_message_and_request
from .prompt_editor_scalar_utils import parse_scalar_value
from .prompt_editor_selection_utils import parse_value_selection, scalar_choice_keyboard
from .prompt_editor_session import show_prompt_panel


@dataclass
class PromptEditorThematicHandlersDeps:
    runtime: RuntimeStore
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
    localization: LocalizationService | None = None
    resolve_user_locale: Callable[..., str] | None = None


@dataclass
class ThematicScalarConfig:
    open_callback: str
    prefix: str
    custom_state: Any
    menu_text: str
    custom_prompt: str
    invalid_input_text: str
    values_rows: list[list[str]]
    back_callback: str
    set_value: Callable[[GenerationParams, int | float], None]
    parse_value: Callable[[str], int | float]
    validate_value: Callable[[int | float], bool]
    render_text: Callable[[GenerationParams], str]
    render_keyboard: Callable[[GenerationParams], InlineKeyboardMarkup]
    ack_text: Callable[[GenerationParams], str]


def register_prompt_editor_thematic_handlers(
    router: Router,
    deps: PromptEditorThematicHandlersDeps,
) -> None:
    def _resolved_locale(uid: int, *, telegram_locale: str | None) -> str | None:
        if deps.localization is None:
            return None
        prefs = deps.runtime.user_preferences.get(uid, {})
        selected_locale = read_user_locale(
            prefs,
            default_locale=deps.localization.default_locale(),
        )
        if deps.resolve_user_locale is None:
            return selected_locale
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
        if deps.localization is None:
            text = default
        else:
            locale = _resolved_locale(uid, telegram_locale=telegram_locale)
            text = deps.localization.t(key, locale=locale, params=params, default=default)
        if params:
            try:
                return text.format(**params)
            except Exception:
                return text
        return text

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

    async def _callback_context(
        cb: CallbackQuery,
    ) -> tuple[Message, int, PromptRequest] | None:
        return await require_message_and_request(
            cb,
            require_prompt_request_for_callback=deps.require_prompt_request_for_callback,
        )

    async def _show_from_request_anchor(
        message: Message,
        req: PromptRequest,
        text: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        await show_prompt_panel(
            deps.runtime,
            message,
            req,
            text,
            reply_markup,
            prefer_edit=False,
        )

    def _control_bounds(field: str) -> tuple[float, float]:
        control = numeric_enhancement_control(field)
        if control is None:
            raise ValueError(f"Unknown enhancement control: {field}")
        return control.min_value, control.max_value

    def _register_scalar_param(config: ThematicScalarConfig) -> None:
        @router.callback_query(F.data == config.open_callback)
        async def _open(cb: CallbackQuery):
            message = await require_callback_message(cb)
            if message is None:
                return
            await message.edit_text(
                config.menu_text,
                reply_markup=scalar_choice_keyboard(
                    prefix=config.prefix,
                    values_rows=config.values_rows,
                    back_callback=config.back_callback,
                ),
            )
            await cb.answer()

        @router.callback_query(F.data.startswith(f"{config.prefix}:"))
        async def _selected(cb: CallbackQuery, state: FSMContext):
            context = await _callback_context(cb)
            if context is None:
                return
            message, _, req = context
            value = await parse_value_selection(cb, prefix=config.prefix)
            if value is None:
                return
            if value == "custom":
                await state.set_state(cast(Any, config.custom_state))
                await message.edit_text(
                    config.custom_prompt,
                    reply_markup=deps.back_keyboard(config.back_callback),
                )
                await cb.answer()
                return

            try:
                parsed = parse_scalar_value(
                    value,
                    parse_value=config.parse_value,
                    validate_value=config.validate_value,
                )
                if parsed is None:
                    raise ValueError
            except ValueError:
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.thematic.error.invalid_value",
                        "❌ Некорректное значение.",
                    ),
                    show_alert=True,
                )
                return

            config.set_value(req.params, parsed)
            await message.edit_text(
                config.render_text(req.params),
                reply_markup=config.render_keyboard(req.params),
            )
            await cb.answer(config.ack_text(req.params))

        @router.message(cast(Any, config.custom_state), F.text)
        async def _custom(msg: Message, state: FSMContext):
            payload = await deps.require_prompt_request_for_message(msg, state)
            if not payload:
                return
            _, req = payload
            parsed = parse_scalar_value(
                (msg.text or ""),
                parse_value=config.parse_value,
                validate_value=config.validate_value,
            )
            if parsed is None:
                await msg.answer(config.invalid_input_text)
                return

            config.set_value(req.params, parsed)
            await state.set_state(PromptEditorStates.editing)
            await deps.cleanup_user_message(msg)
            await _show_from_request_anchor(
                msg,
                req,
                config.render_text(req.params),
                config.render_keyboard(req.params),
            )

    @router.callback_query(F.data == "pe:sub:more")
    async def pe_submenu_more(cb: CallbackQuery):
        """'More settings' hub for simple mode — links to thematic submenus."""
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        params = req.params

        enh_label = enhancements_menu_label(params)

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.more.button.sampling",
                            "📐 Сэмплинг",
                        ),
                        callback_data="pe:sub:sampling",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.more.button.image",
                            "🖼 Изображение",
                        ),
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
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.more.button.save",
                            "💾 Сохранить",
                        ),
                        callback_data="pe:save",
                    ),
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.more.button.presets",
                            "📚 Пресеты",
                        ),
                        callback_data="pe:presets",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.more.button.exchange",
                            "📋 Копировать/вставить",
                        ),
                        callback_data="pe:exchange",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.more.text",
                "⚙️ <b>Дополнительные настройки</b>\n"
                "\n"
                "Здесь собраны все параметры генерации, разбитые по категориям.\n"
                "\n"
                "📐 <b>Сэмплинг</b> — шаги, CFG, алгоритм генерации\n"
                "🖼 <b>Изображение</b> — размер, батч, референсы\n"
                "✨ <b>Улучшения</b> — Hi-res, FreeU, PAG, апскейл",
            ),
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:sub:sampling")
    async def pe_submenu_sampling(cb: CallbackQuery):
        """Sampling thematic submenu: Steps, CFG, Sampler, Scheduler, Denoise, Seed."""
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        params = req.params
        seed_text = str(params.seed) if params.seed >= 0 else "\U0001f3b2"

        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.sampling.button.steps",
                            "🔢 Steps {value}",
                            params={"value": params.steps},
                        ),
                        callback_data="pe:edit:steps",
                    ),
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.sampling.button.cfg",
                            "CFG {value}",
                            params={"value": params.cfg},
                        ),
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
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.sampling.button.denoise",
                            "Denoise {value}",
                            params={"value": params.denoise},
                        ),
                        callback_data="pe:edit:denoise",
                    ),
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.sampling.button.seed",
                            "Seed {value}",
                            params={"value": seed_text},
                        ),
                        callback_data="pe:edit:seed",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.sampling.text",
                "📐 <b>Сэмплинг</b>\n"
                "\n"
                "Настройки процесса генерации изображения.\n"
                "\n"
                "• <b>Steps</b> — кол-во шагов диффузии. Больше = качественнее, но медленнее (20–35 оптимально)\n"
                "• <b>CFG</b> — насколько строго модель следует промпту. Ниже = свободнее, выше = точнее (5–8 обычно)\n"
                "• <b>Denoise</b> — сила изменений (1.0 = полная генерация)\n"
                "• <b>Sampler/Scheduler</b> — алгоритм и расписание шумодава\n"
                "• <b>Seed</b> — зерно случайности (фикс. значение = повторяемый результат)\n"
                "\n",
            )
            + _t_cb(
                cb,
                "prompt_editor.thematic.sampling.summary",
                "<b>Steps:</b> <code>{steps}</code>  <b>CFG:</b> <code>{cfg}</code>  <b>Denoise:</b> <code>{denoise}</code>\n"
                "<b>Sampler:</b> <code>{sampler}</code>  <b>Sched:</b> <code>{scheduler}</code>\n"
                "<b>Seed:</b> {seed}",
                params={
                    "steps": params.steps,
                    "cfg": params.cfg,
                    "denoise": params.denoise,
                    "sampler": h(params.sampler),
                    "scheduler": h(params.scheduler),
                    "seed": seed_text,
                },
            ),
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:sub:image")
    async def pe_submenu_image(cb: CallbackQuery):
        """Image thematic submenu: Size, Batch, References, Ref Strength."""
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
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
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.image.button.batch",
                            "📦 Batch {value}",
                            params={"value": params.batch_size},
                        ),
                        callback_data="pe:edit:batch",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.image.button.references",
                            "🖼 Ref ({count})",
                            params={"count": ref_count},
                        ),
                        callback_data="pe:edit:refs",
                    ),
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.image.button.reference_strength",
                            "Ref Str {value}",
                            params={"value": params.reference_strength},
                        ),
                        callback_data="pe:edit:ref_strength",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            _t_cb(cb, "prompt_editor.thematic.image.value.vae_auto", "🧬 VAE: auto")
                            if not params.vae_name
                            else _t_cb(
                                cb,
                                "prompt_editor.thematic.image.value.vae_name",
                                "🧬 VAE: {name}",
                                params={"name": truncate(params.vae_name, 14)},
                            )
                        ),
                        callback_data="pe:edit:vae",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            _t_cb(
                                cb,
                                "prompt_editor.thematic.image.value.controlnet_off",
                                "🧷 ControlNet: off",
                            )
                            if not params.controlnet_name
                            else _t_cb(
                                cb,
                                "prompt_editor.thematic.image.value.controlnet_name",
                                "🧷 {name}",
                                params={"name": truncate(params.controlnet_name, 14)},
                            )
                        ),
                        callback_data="pe:edit:controlnet",
                    ),
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.image.button.controlnet_strength",
                            "Str {value}",
                            params={"value": params.controlnet_strength},
                        ),
                        callback_data="pe:edit:controlnet_strength",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            _t_cb(
                                cb,
                                "prompt_editor.thematic.image.value.embedding_off",
                                "🔤 Embedding: off",
                            )
                            if not params.embedding_name
                            else _t_cb(
                                cb,
                                "prompt_editor.thematic.image.value.embedding_name",
                                "🔤 {name}",
                                params={"name": truncate(params.embedding_name, 20)},
                            )
                        ),
                        callback_data="pe:edit:embedding",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:back",
                    )
                ],
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.image.text",
                "🖼 <b>Изображение</b>\n"
                "\n"
                "Параметры выходного изображения.\n"
                "\n"
                "• <b>Размер</b> — ширина×высота в пикселях. Больший размер требует больше VRAM\n"
                "• <b>Batch</b> — кол-во изображений за раз. Больше = дольше, но больше вариантов\n"
                "• <b>Ref</b> — референсные изображения (IP-Adapter). Модель перенимает стиль/композицию\n"
                "• <b>Ref Strength</b> — сила влияния референса (0.0–1.0)\n"
                "• <b>VAE</b> — переопределение VAE декодера\n"
                "• <b>ControlNet</b> — структурный контроль по референсу\n"
                "• <b>Embedding</b> — quick negative token (embedding:name)\n"
                "\n",
            )
            + _t_cb(
                cb,
                "prompt_editor.thematic.image.summary",
                "<b>Размер:</b> <code>{width}×{height}</code>  <b>Batch:</b> <code>{batch}</code>\n"
                "<b>Ref:</b> {ref_count}/{ref_limit}  <b>Ref str:</b> <code>{ref_strength}</code>\n"
                "<b>VAE:</b> <code>{vae}</code>  <b>ControlNet:</b> <code>{controlnet}</code>\n"
                "<b>CN str:</b> <code>{controlnet_strength}</code>  <b>Embedding:</b> <code>{embedding}</code>",
                params={
                    "width": params.width,
                    "height": params.height,
                    "batch": params.batch_size,
                    "ref_count": ref_count,
                    "ref_limit": deps.max_reference_images,
                    "ref_strength": params.reference_strength,
                    "vae": h(params.vae_name) if params.vae_name else "auto",
                    "controlnet": h(params.controlnet_name) if params.controlnet_name else "off",
                    "controlnet_strength": params.controlnet_strength,
                    "embedding": h(params.embedding_name) if params.embedding_name else "off",
                },
            ),
            reply_markup=kb,
        )
        await cb.answer()

    # ------------------------------------------------------------------
    # Enhancements hub + per-enhancement submenus
    # ------------------------------------------------------------------

    @router.callback_query(F.data == "pe:sub:enhancements")
    async def pe_submenu_enhancements(cb: CallbackQuery):
        """Enhancements hub: links to Hi-res Fix, FreeU, PAG, Upscaler submenus."""
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
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
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.enhancements.button.hires",
                            "🔧 Hi-res Fix {status}",
                            params={"status": hires_status},
                        ),
                        callback_data="pe:enh:hires",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.enhancements.button.freeu",
                            "⚡ FreeU {status}",
                            params={"status": freeu_status},
                        ),
                        callback_data="pe:enh:freeu",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.enhancements.button.pag",
                            "🎯 PAG {status}",
                            params={"status": pag_status},
                        ),
                        callback_data="pe:enh:pag",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.enhancements.button.upscaler",
                            "🔍 Upscaler {status}",
                            params={"status": upsc_status},
                        ),
                        callback_data="pe:enh:upscaler",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(
                            cb,
                            "prompt_editor.thematic.enhancements.button.tiled",
                            "🧩 Tiled Diffusion {status}",
                            params={"status": tiled_status},
                        ),
                        callback_data="pe:enh:tiled",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:back",
                    )
                ],
            ]
        )

        # Status text
        enh_lines: list[str] = []
        if params.enable_hires_fix:
            enh_lines.append(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.enhancements.status.hires",
                    "🔧 Hi-res Fix — ×{scale}, denoise {denoise}",
                    params={"scale": params.hires_scale, "denoise": params.hires_denoise},
                )
            )
        if params.enable_freeu:
            enh_lines.append(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.enhancements.status.freeu",
                    "⚡ FreeU V2 — вкл",
                )
            )
        if params.enable_pag:
            enh_lines.append(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.enhancements.status.pag",
                    "🎯 PAG — scale {scale}",
                    params={"scale": params.pag_scale},
                )
            )
        if params.upscale_model:
            enh_lines.append(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.enhancements.status.upscaler",
                    "🔍 Upscaler — {model}",
                    params={"model": h(truncate(params.upscale_model, 30))},
                )
            )
        if params.enable_tiled_diffusion:
            enh_lines.append(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.enhancements.status.tiled",
                    "🧩 HyperTile — tile {tile_size}, VAE {vae_tile}, overlap {overlap}",
                    params={
                        "tile_size": params.tile_size,
                        "vae_tile": params.vae_tile_size,
                        "overlap": params.tile_overlap,
                    },
                )
            )

        enh_text = (
            "\n".join(enh_lines)
            if enh_lines
            else _t_cb(
                cb,
                "prompt_editor.thematic.enhancements.empty",
                "<i>Нет активных улучшений</i>",
            )
        )

        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.enhancements.text",
                "✨ <b>Улучшения</b>\n"
                "\n"
                "Дополнительные модули, которые улучшают качество генерации, но могут увеличивать время. Выберите модуль для настройки.\n"
                "\n"
                "{details}",
                params={"details": enh_text},
            ),
            reply_markup=kb,
        )
        await cb.answer()

    # --- Hi-res Fix submenu ---

    def _hires_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            _t(0, "common.action.disable", "❌ Выключить", telegram_locale=None)
            if params.enable_hires_fix
            else _t(0, "common.action.enable", "✅ Включить", telegram_locale=None)
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
                        text=_t(
                            0,
                            "prompt_editor.thematic.hires.button.scale",
                            "Scale: ×{value}",
                            telegram_locale=None,
                            params={"value": params.hires_scale},
                        ),
                        callback_data="pe:hires:scale",
                    ),
                    InlineKeyboardButton(
                        text=_t(
                            0,
                            "prompt_editor.thematic.hires.button.denoise",
                            "Denoise: {value}",
                            telegram_locale=None,
                            params={"value": params.hires_denoise},
                        ),
                        callback_data="pe:hires:denoise",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text=_t(0, "common.action.back", "⬅️ Назад", telegram_locale=None),
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _hires_submenu_text(params: GenerationParams) -> str:
        status = (
            _t(0, "common.status.enabled_short", "✅ ВКЛ", telegram_locale=None)
            if params.enable_hires_fix
            else _t(0, "common.status.disabled_short", "❌ ВЫКЛ", telegram_locale=None)
        )
        return _t(
            0,
            "prompt_editor.thematic.hires.text",
            "🔧 <b>Hi-res Fix</b>\n"
            "\n"
            "Генерирует картинку в малом разрешении, затем увеличивает и дорисовывает детали вторым проходом. Это позволяет получить большие изображения без характерных артефактов.\n"
            "\n"
            "• <b>Scale</b> — множитель увеличения (×1.5 для баланса, ×2.0 для максимума)\n"
            "• <b>Denoise</b> — степень перерисовки (0.3–0.5 — сохраняет композицию, 0.6–0.7 — больше новых деталей)\n"
            "⚠️ Увеличивает время генерации примерно вдвое.\n"
            "\n"
            "<b>Статус:</b> {status}\n"
            "<b>Scale:</b> ×{scale}  <b>Denoise:</b> {denoise}",
            telegram_locale=None,
            params={
                "status": status,
                "scale": params.hires_scale,
                "denoise": params.hires_denoise,
            },
        )

    @router.callback_query(F.data == "pe:enh:hires")
    async def pe_enh_hires(cb: CallbackQuery):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        await message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:hires")
    async def pe_toggle_hires(cb: CallbackQuery, state: FSMContext):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        req.params.enable_hires_fix = not req.params.enable_hires_fix
        await message.edit_text(
            _hires_submenu_text(req.params),
            reply_markup=_hires_submenu_kb(req.params),
        )
        status = (
            _t_cb(cb, "common.value.enabled", "включен")
            if req.params.enable_hires_fix
            else _t_cb(cb, "common.value.disabled", "выключен")
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.thematic.hires.toggled",
                "🔧 Hi-res Fix {status}",
                params={"status": status},
            )
        )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:hires:scale",
            prefix="pe_hrs",
            custom_state=PromptEditorStates.entering_custom_hires_scale,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.scale.menu",
                    f"Hi-res Fix scale (множитель, {numeric_control_range_text('hires_scale')}):",
                    telegram_locale=None,
                )
            ),
            custom_prompt=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.scale.custom_prompt",
                    f"Hi-res scale ({numeric_control_range_text('hires_scale')}):",
                    telegram_locale=None,
                )
            ),
            invalid_input_text=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.scale.invalid",
                    f"Число {numeric_control_range_text('hires_scale')}:",
                    telegram_locale=None,
                )
            ),
            values_rows=numeric_control_rows("hires_scale"),
            back_callback="pe:enh:hires",
            set_value=lambda params, value: setattr(params, "hires_scale", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: _control_bounds("hires_scale")[0]
            <= float(value)
            <= _control_bounds("hires_scale")[1],
            render_text=_hires_submenu_text,
            render_keyboard=_hires_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.hires.scale.ack",
                "✅ Scale: x{value}",
                telegram_locale=None,
                params={"value": params.hires_scale},
            ),
        )
    )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:hires:denoise",
            prefix="pe_hrd",
            custom_state=PromptEditorStates.entering_custom_hires_denoise,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.denoise.menu",
                    "Hi-res Fix denoise ({range_text}, меньше = ближе к оригиналу):",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("hires_denoise")},
                )
            ),
            custom_prompt=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.denoise.custom_prompt",
                    "Hi-res denoise ({range_text}):",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("hires_denoise")},
                )
            ),
            invalid_input_text=(
                _t(
                    0,
                    "prompt_editor.thematic.hires.denoise.invalid",
                    "Число {range_text}:",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("hires_denoise")},
                )
            ),
            values_rows=numeric_control_rows("hires_denoise"),
            back_callback="pe:enh:hires",
            set_value=lambda params, value: setattr(params, "hires_denoise", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: _control_bounds("hires_denoise")[0]
            <= float(value)
            <= _control_bounds("hires_denoise")[1],
            render_text=_hires_submenu_text,
            render_keyboard=_hires_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.hires.denoise.ack",
                "✅ Hi-res denoise: {value}",
                telegram_locale=None,
                params={"value": params.hires_denoise},
            ),
        )
    )

    # --- FreeU submenu ---

    @router.callback_query(F.data == "pe:enh:freeu")
    async def pe_enh_freeu(cb: CallbackQuery):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        status = (
            _t_cb(cb, "common.status.enabled_short", "✅ ВКЛ")
            if req.params.enable_freeu
            else _t_cb(cb, "common.status.disabled_short", "❌ ВЫКЛ")
        )
        toggle_text = (
            _t_cb(cb, "common.action.disable", "❌ Выключить")
            if req.params.enable_freeu
            else _t_cb(cb, "common.action.enable", "✅ Включить")
        )

        if not deps.is_freeu_supported():
            toggle_text = _t_cb(cb, "common.status.unavailable", "⚠️ Недоступно")

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
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:sub:enhancements",
                    )
                ],
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.freeu.text",
                "⚡ <b>FreeU V2</b>\n"
                "\n"
                "Бесплатное улучшение качества без дополнительной нагрузки. FreeU патчит U-Net модель, усиливая низкочастотные детали и подавляя высокочастотный шум. Изображения становятся чётче и детализированнее.\n"
                "\n"
                "💡 Не требует дополнительного времени и не имеет настроек — просто вкл/выкл.\n"
                "\n"
                "<b>Статус:</b> {status}",
                params={"status": status},
            ),
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:freeu")
    async def pe_toggle_freeu(cb: CallbackQuery, state: FSMContext):
        if not deps.is_freeu_supported():
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.freeu.error.not_supported",
                    "⚠️ FreeU_V2 нода не найдена на сервере ComfyUI.",
                ),
                show_alert=True,
            )
            return
        context = await _callback_context(cb)
        if context is None:
            return
        message, uid, req = context
        req.params.enable_freeu = not req.params.enable_freeu
        status_word = (
            _t_cb(cb, "common.value.enabled", "включен")
            if req.params.enable_freeu
            else _t_cb(cb, "common.value.disabled", "выключен")
        )
        # Re-render FreeU submenu
        status = (
            _t_cb(cb, "common.status.enabled_short", "✅ ВКЛ")
            if req.params.enable_freeu
            else _t_cb(cb, "common.status.disabled_short", "❌ ВЫКЛ")
        )
        toggle_text = (
            _t_cb(cb, "common.action.disable", "❌ Выключить")
            if req.params.enable_freeu
            else _t_cb(cb, "common.action.enable", "✅ Включить")
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
                        text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                        callback_data="pe:sub:enhancements",
                    )
                ],
            ]
        )
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.freeu.short_text",
                "⚡ <b>FreeU V2</b>\n\nПатчит модель для улучшения деталей.\n\n<b>Статус:</b> {status}",
                params={"status": status},
            ),
            reply_markup=kb,
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.thematic.freeu.toggled",
                "⚡ FreeU {status}",
                params={"status": status_word},
            )
        )

    # --- PAG submenu ---

    def _pag_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            _t(0, "common.action.disable", "❌ Выключить", telegram_locale=None)
            if params.enable_pag
            else _t(0, "common.action.enable", "✅ Включить", telegram_locale=None)
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
                        text=_t(
                            0,
                            "prompt_editor.thematic.pag.button.scale",
                            "Scale: {value}",
                            telegram_locale=None,
                            params={"value": params.pag_scale},
                        ),
                        callback_data="pe:pag:scale",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text=_t(0, "common.action.back", "⬅️ Назад", telegram_locale=None),
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _pag_submenu_text(params: GenerationParams) -> str:
        status = (
            _t(0, "common.status.enabled_short", "✅ ВКЛ", telegram_locale=None)
            if params.enable_pag
            else _t(0, "common.status.disabled_short", "❌ ВЫКЛ", telegram_locale=None)
        )
        return _t(
            0,
            "prompt_editor.thematic.pag.text",
            "🎯 <b>PAG (Perturbed-Attention Guidance)</b>\n"
            "\n"
            "Улучшает следование промпту без повышения CFG, за счёт пертурбации карт внимания (self-attention). Помогает получить более точную композицию и детали.\n"
            "\n"
            "• <b>Scale</b> — сила эффекта (1–3 мягко, 3–5 умеренно, 5+ агрессивно)\n"
            "⚠️ Незначительно увеличивает время генерации.\n"
            "\n"
            "<b>Статус:</b> {status}\n"
            "<b>Scale:</b> {scale}",
            telegram_locale=None,
            params={"status": status, "scale": params.pag_scale},
        )

    @router.callback_query(F.data == "pe:enh:pag")
    async def pe_enh_pag(cb: CallbackQuery):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context

        if not deps.is_pag_supported():
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                            callback_data="pe:sub:enhancements",
                        )
                    ]
                ]
            )
            await message.edit_text(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.pag.not_supported.text",
                    "🎯 <b>PAG</b>\n\n⚠️ PerturbedAttentionGuidance нода не найдена на сервере.",
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return

        await message.edit_text(
            _pag_submenu_text(req.params),
            reply_markup=_pag_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:pag")
    async def pe_toggle_pag(cb: CallbackQuery, state: FSMContext):
        if not deps.is_pag_supported():
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.pag.error.not_supported",
                    "⚠️ PerturbedAttentionGuidance нода не найдена на сервере.",
                ),
                show_alert=True,
            )
            return
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        req.params.enable_pag = not req.params.enable_pag
        status_word = (
            _t_cb(cb, "common.value.enabled", "включен")
            if req.params.enable_pag
            else _t_cb(cb, "common.value.disabled", "выключен")
        )
        await message.edit_text(
            _pag_submenu_text(req.params),
            reply_markup=_pag_submenu_kb(req.params),
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.thematic.pag.toggled",
                "🎯 PAG {status}",
                params={"status": status_word},
            )
        )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:pag:scale",
            prefix="pe_pag",
            custom_state=PromptEditorStates.entering_custom_pag_scale,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.pag.scale.menu",
                    "PAG Scale ({range_text}):",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("pag_scale")},
                )
            ),
            custom_prompt=(
                _t(
                    0,
                    "prompt_editor.thematic.pag.scale.custom_prompt",
                    "PAG Scale ({range_text}):",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("pag_scale")},
                )
            ),
            invalid_input_text=(
                _t(
                    0,
                    "prompt_editor.thematic.pag.scale.invalid",
                    "Число {range_text}:",
                    telegram_locale=None,
                    params={"range_text": numeric_control_range_text("pag_scale")},
                )
            ),
            values_rows=numeric_control_rows("pag_scale"),
            back_callback="pe:enh:pag",
            set_value=lambda params, value: setattr(params, "pag_scale", float(value)),
            parse_value=lambda raw: float(raw),
            validate_value=lambda value: _control_bounds("pag_scale")[0]
            <= float(value)
            <= _control_bounds("pag_scale")[1],
            render_text=_pag_submenu_text,
            render_keyboard=_pag_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.pag.scale.ack",
                "✅ PAG scale: {value}",
                telegram_locale=None,
                params={"value": params.pag_scale},
            ),
        )
    )

    # --- Upscaler submenu ---

    @router.callback_query(F.data == "pe:enh:upscaler")
    async def pe_enh_upscaler(cb: CallbackQuery):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        model = req.params.upscale_model
        status = (
            f"✅ {h(truncate(model, 30))}"
            if model
            else _t_cb(cb, "common.status.disabled_short", "❌ ВЫКЛ")
        )

        rows = [
            [
                InlineKeyboardButton(
                    text=_t_cb(
                        cb,
                        "prompt_editor.thematic.upscaler.button.select",
                        "🔍 Выбрать модель",
                    ),
                    callback_data="pe:edit:upscaler",
                ),
            ],
        ]
        if model:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_t_cb(cb, "common.action.disable", "❌ Отключить"),
                        callback_data="pe:upsc:off",
                    ),
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text=_t_cb(cb, "common.action.back", "⬅️ Назад"),
                    callback_data="pe:sub:enhancements",
                )
            ]
        )

        kb = InlineKeyboardMarkup(inline_keyboard=rows)
        await message.edit_text(
            _t_cb(
                cb,
                "prompt_editor.thematic.upscaler.text",
                "🔍 <b>Upscaler</b>\n"
                "\n"
                "Увеличивает разрешение готовой картинки с помощью нейросети. В отличие от Hi-res Fix, не добавляет новых деталей, а интеллектуально масштабирует существующие.\n"
                "\n"
                "💡 Подходит для быстрого увеличения разрешения, когда не нужна дорисовка.\n"
                "⚠️ Увеличивает время генерации.\n"
                "\n"
                "<b>Модель:</b> {status}",
                params={"status": status},
            ),
            reply_markup=kb,
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:upsc:off")
    async def pe_upsc_off(cb: CallbackQuery, state: FSMContext):
        context = await _callback_context(cb)
        if context is None:
            return
        message, uid, req = context
        req.params.upscale_model = ""
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=_t_cb(
                cb,
                "prompt_editor.thematic.upscaler.notice.disabled",
                "✅ Upscaler отключен.",
            ),
        )
        await cb.answer()

    # --- Tiled Diffusion submenu (HyperTile + VAE Tiled — built-in) ---

    def _tiled_submenu_kb(params: GenerationParams) -> InlineKeyboardMarkup:
        toggle_text = (
            _t(0, "common.action.disable", "❌ Выключить", telegram_locale=None)
            if params.enable_tiled_diffusion
            else _t(0, "common.action.enable", "✅ Включить", telegram_locale=None)
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
                            text=_t(
                                0,
                                "prompt_editor.thematic.tiled.button.tile_size",
                                "🧩 Tile Size: {value}",
                                telegram_locale=None,
                                params={"value": params.tile_size},
                            ),
                            callback_data="pe:tiled:tile_size",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            text=_t(
                                0,
                                "prompt_editor.thematic.tiled.button.vae_tile",
                                "🖼 VAE Tile: {value}",
                                telegram_locale=None,
                                params={"value": params.vae_tile_size},
                            ),
                            callback_data="pe:tiled:vae_tile",
                        ),
                        InlineKeyboardButton(
                            text=_t(
                                0,
                                "prompt_editor.thematic.tiled.button.overlap",
                                "🌀 Overlap: {value}",
                                telegram_locale=None,
                                params={"value": params.tile_overlap},
                            ),
                            callback_data="pe:tiled:overlap",
                        ),
                    ],
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text=_t(0, "common.action.back", "⬅️ Назад", telegram_locale=None),
                    callback_data="pe:sub:enhancements",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _tiled_submenu_text(params: GenerationParams) -> str:
        status = (
            _t(0, "common.status.enabled_short", "✅ ВКЛ", telegram_locale=None)
            if params.enable_tiled_diffusion
            else _t(0, "common.status.disabled_short", "❌ ВЫКЛ", telegram_locale=None)
        )
        details = ""
        if params.enable_tiled_diffusion:
            details = _t(
                0,
                "prompt_editor.thematic.tiled.details",
                "<b>Tile Size:</b> <code>{tile_size}</code>  <b>VAE Tile:</b> <code>{vae_tile}</code>  <b>Overlap:</b> <code>{overlap}</code>\n",
                telegram_locale=None,
                params={
                    "tile_size": params.tile_size,
                    "vae_tile": params.vae_tile_size,
                    "overlap": params.tile_overlap,
                },
            )
        return _t(
            0,
            "prompt_editor.thematic.tiled.text",
            "🧩 <b>HyperTile + Tiled VAE</b>\n"
            "Разбивает генерацию на тайлы, позволяя эффективно генерировать изображения большого размера с меньшим расходом VRAM.\n"
            "\n"
            "{details}"
            "\n"
            "💡 Использует встроенные ноды HyperTile + VAE Tiled — никаких расширений не требуется.\n"
            "\n"
            "<b>Статус:</b> {status}",
            telegram_locale=None,
            params={"details": details, "status": status},
        )

    @router.callback_query(F.data == "pe:enh:tiled")
    async def pe_enh_tiled(cb: CallbackQuery):
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        await message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer()

    @router.callback_query(F.data == "pe:toggle:tiled")
    async def pe_toggle_tiled(cb: CallbackQuery, state: FSMContext):
        if not deps.is_tiled_diffusion_supported():
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.thematic.tiled.error.not_supported",
                    "⚠️ HyperTile нода не найдена на сервере ComfyUI.",
                ),
                show_alert=True,
            )
            return
        context = await _callback_context(cb)
        if context is None:
            return
        message, _, req = context
        req.params.enable_tiled_diffusion = not req.params.enable_tiled_diffusion
        status_word = (
            _t_cb(cb, "common.value.enabled", "включен")
            if req.params.enable_tiled_diffusion
            else _t_cb(cb, "common.value.disabled", "выключен")
        )
        await message.edit_text(
            _tiled_submenu_text(req.params),
            reply_markup=_tiled_submenu_kb(req.params),
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.thematic.tiled.toggled",
                "🧩 HyperTile {status}",
                params={"status": status_word},
            )
        )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:tiled:tile_size",
            prefix="pe_ts",
            custom_state=PromptEditorStates.entering_custom_tile_size,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.tiled.tile_size.menu",
                    "🧩 <b>Tile Size</b>\n\nРазмер тайла для HyperTile. Меньше = быстрее, больше = качественнее.\n\n💡 Рекомендуется: 256.",
                    telegram_locale=None,
                )
            ),
            custom_prompt=_t(
                0,
                "prompt_editor.thematic.tiled.tile_size.custom_prompt",
                "Tile Size (64-1024):",
                telegram_locale=None,
            ),
            invalid_input_text=_t(
                0,
                "prompt_editor.thematic.tiled.tile_size.invalid",
                "Целое число 64-1024:",
                telegram_locale=None,
            ),
            values_rows=[["128", "192", "256"], ["384", "512", "768"]],
            back_callback="pe:enh:tiled",
            set_value=lambda params, value: setattr(params, "tile_size", int(value)),
            parse_value=lambda raw: int(raw),
            validate_value=lambda value: 64 <= int(value) <= 1024,
            render_text=_tiled_submenu_text,
            render_keyboard=_tiled_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.tiled.tile_size.ack",
                "Tile Size: {value}",
                telegram_locale=None,
                params={"value": params.tile_size},
            ),
        )
    )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:tiled:vae_tile",
            prefix="pe_vt",
            custom_state=PromptEditorStates.entering_custom_vae_tile_size,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.tiled.vae_tile.menu",
                    "🖼 <b>VAE Tile Size</b>\n\nРазмер тайла для VAE encode/decode.\n\n💡 Рекомендуется: 512.",
                    telegram_locale=None,
                )
            ),
            custom_prompt=_t(
                0,
                "prompt_editor.thematic.tiled.vae_tile.custom_prompt",
                "VAE Tile Size (128-4096):",
                telegram_locale=None,
            ),
            invalid_input_text=_t(
                0,
                "prompt_editor.thematic.tiled.vae_tile.invalid",
                "Целое число 128-4096:",
                telegram_locale=None,
            ),
            values_rows=[["256", "384", "512"], ["768", "1024", "2048"]],
            back_callback="pe:enh:tiled",
            set_value=lambda params, value: setattr(params, "vae_tile_size", int(value)),
            parse_value=lambda raw: int(raw),
            validate_value=lambda value: 128 <= int(value) <= 4096,
            render_text=_tiled_submenu_text,
            render_keyboard=_tiled_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.tiled.vae_tile.ack",
                "VAE Tile: {value}",
                telegram_locale=None,
                params={"value": params.vae_tile_size},
            ),
        )
    )

    _register_scalar_param(
        ThematicScalarConfig(
            open_callback="pe:tiled:overlap",
            prefix="pe_tovlp",
            custom_state=PromptEditorStates.entering_custom_tile_overlap,
            menu_text=(
                _t(
                    0,
                    "prompt_editor.thematic.tiled.overlap.menu",
                    "🌀 <b>Overlap</b>\n\nПерекрытие между тайлами VAE. Больше = меньше швов, но медленнее.\n\n💡 Рекомендуется: 64.",
                    telegram_locale=None,
                )
            ),
            custom_prompt=_t(
                0,
                "prompt_editor.thematic.tiled.overlap.custom_prompt",
                "Tile Overlap (0-2048):",
                telegram_locale=None,
            ),
            invalid_input_text=_t(
                0,
                "prompt_editor.thematic.tiled.overlap.invalid",
                "Целое число 0-2048:",
                telegram_locale=None,
            ),
            values_rows=[["32", "48", "64"], ["96", "128", "256"]],
            back_callback="pe:enh:tiled",
            set_value=lambda params, value: setattr(params, "tile_overlap", int(value)),
            parse_value=lambda raw: int(raw),
            validate_value=lambda value: 0 <= int(value) <= 2048,
            render_text=_tiled_submenu_text,
            render_keyboard=_tiled_submenu_kb,
            ack_text=lambda params: _t(
                0,
                "prompt_editor.thematic.tiled.overlap.ack",
                "Overlap: {value}",
                telegram_locale=None,
                params={"value": params.tile_overlap},
            ),
        )
    )
