from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
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
from core.prompt_enhancements import (
    numeric_control_range_text,
)
from core.runtime import PreviewArtifact, PromptRequest, RuntimeStore
from core.ui_copy import main_menu_keyboard, start_text
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService
from domain.ui_text import UITextService
from infrastructure.comfyui_client import ComfyUIClient

from .prompt_editor_enhancement import (
    PromptEditorEnhancementDeps,
    start_image_enhancement,
)
from .prompt_editor_send_menu_utils import (
    apply_field_value,
    custom_field_meta,
    enhancement_preset_values,
    paginated_pick_keyboard,
    parse_shrink_size,
    shrink_value_keyboard,
    simple_value_keyboard,
    submenu_back_callback,
    submenu_for_field,
    toggle_hires_fix,
)


@dataclass
class PromptEditorSendHandlersDeps:
    logger: logging.Logger
    runtime: RuntimeStore
    client: ComfyUIClient
    require_prompt_request_for_callback: Callable[
        [CallbackQuery], Awaitable[tuple[int, PromptRequest] | None]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    generation_result_keyboard: Callable[[], InlineKeyboardMarkup]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]
    localization: LocalizationService
    ui_text: UITextService
    resolve_user_locale: Callable[..., str]


def register_prompt_editor_send_handlers(
    router: Router,
    deps: PromptEditorSendHandlersDeps,
) -> None:
    def _user_artifact(uid: int, artifact_id: str) -> PreviewArtifact | None:
        artifact = deps.runtime.preview_artifacts.get(artifact_id)
        if artifact and artifact.owner_uid == uid:
            return artifact
        return None

    def _has_pending_input(msg: Message) -> bool:
        uid = msg.from_user.id if msg.from_user else 0
        return uid > 0 and uid in deps.runtime.pending_image_inputs

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
        return _t(
            cb.from_user.id,
            key,
            default,
            telegram_locale=cb.from_user.language_code,
            params=params,
        )

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

    def _menu_translate(key: str, locale: str | None, default: str) -> str:
        return deps.localization.t(key, locale=locale, default=default)

    async def _callback_value(
        cb: CallbackQuery,
        *,
        prefix: str,
        invalid_key: str = "common.alert.invalid_request",
        invalid_default: str = "⚠️ Некорректный запрос.",
    ) -> str | None:
        parsed = ValueSelectionCallback.parse(cb.data or "", prefix=prefix)
        if parsed is None:
            await cb.answer(_t_cb(cb, invalid_key, invalid_default), show_alert=True)
            return None
        return parsed.value

    async def _artifact_from_callback(
        cb: CallbackQuery,
        *,
        prefix: str,
        missing_key: str = "prompt_editor.send.alert.image_not_found",
        missing_default: str = "⚠️ Картинка не найдена.",
    ) -> tuple[str, PreviewArtifact] | None:
        artifact_id = await _callback_value(cb, prefix=prefix)
        if artifact_id is None:
            return None
        artifact = _user_artifact(cb.from_user.id, artifact_id)
        if artifact is None:
            await cb.answer(_t_cb(cb, missing_key, missing_default), show_alert=True)
            return None
        return artifact_id, artifact

    async def _move_main_panel_to_bottom(
        uid: int,
        source_message: Message,
        text: str,
    ) -> None:
        req = deps.runtime.active_prompt_requests.get(uid)
        prev_chat_id = req.ui_chat_id if req else None
        prev_message_id = req.ui_message_id if req else None

        sent = await source_message.answer(
            text,
            reply_markup=deps.generation_result_keyboard(),
        )

        if req:
            req.ui_chat_id = sent.chat.id
            req.ui_message_id = sent.message_id
        deps.runtime.user_ui_panels[uid] = {
            "chat_id": sent.chat.id,
            "message_id": sent.message_id,
        }

        if (
            prev_chat_id is not None
            and prev_message_id is not None
            and (prev_chat_id != sent.chat.id or prev_message_id != sent.message_id)
            and source_message.bot is not None
        ):
            try:
                await source_message.bot.delete_message(
                    chat_id=prev_chat_id,
                    message_id=prev_message_id,
                )
            except TelegramBadRequest:
                pass

    def _artifact_hub_keyboard(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.hub.sampling",
                            "📐 Сэмплинг",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:sub:smp:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.hub.enhancements",
                            "✨ Улучшения",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:sub:enh:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.hub.size",
                            "🖼 Размер",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:sub:size:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.hub.to_editor",
                            "🧬 В редактор",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:to_editor:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.hub.run",
                            "▶️ Запустить улучшения",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:run:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "common.action.back",
                            "⬅️ Назад",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:back:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "common.menu.back_to_menu",
                            "⬅️ В меню",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data="menu:root",
                    )
                ],
            ]
        )

    def _artifact_sampling_keyboard(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> InlineKeyboardMarkup:
        params = artifact.params
        sampler_status = "✅" if artifact.enable_sampler_pass else "❌"
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.sampling.sampler_pass', '🔁 Сэмплер-проход', telegram_locale=telegram_locale)} "
                            f"{sampler_status}"
                        ),
                        callback_data=f"img:tgl:smp:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.sampling.steps', '🔢 Steps', telegram_locale=telegram_locale)} "
                            f"{params.steps}"
                        ),
                        callback_data=f"img:menu:steps:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.sampling.cfg', 'CFG', telegram_locale=telegram_locale)} "
                            f"{params.cfg}"
                        ),
                        callback_data=f"img:menu:cfg:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.sampling.denoise', 'Denoise', telegram_locale=telegram_locale)} "
                            f"{params.denoise}"
                        ),
                        callback_data=f"img:menu:den:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.sampling.sampler",
                            "⚙️ Sampler",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:menu:sampler:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "prompt_editor.send.menu.sampling.scheduler",
                            "📈 Scheduler",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:menu:scheduler:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "common.action.back",
                            "⬅️ Назад",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:open:{artifact.artifact_id}",
                    )
                ],
            ]
        )

    def _artifact_enhancements_keyboard(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> InlineKeyboardMarkup:
        params = artifact.params
        hires_status = "✅" if params.enable_hires_fix else "❌"
        freeu_status = "✅" if params.enable_freeu else "❌"
        pag_status = "✅" if params.enable_pag else "❌"
        upsc_status = "✅" if params.upscale_model else "❌"
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.hires', '🛠 Hi-res', telegram_locale=telegram_locale)} "
                            f"{hires_status}"
                        ),
                        callback_data=f"img:tgl:hi:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.freeu', '⚡ FreeU', telegram_locale=telegram_locale)} "
                            f"{freeu_status}"
                        ),
                        callback_data=f"img:tgl:fu:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.pag', '🎯 PAG', telegram_locale=telegram_locale)} "
                            f"{pag_status}"
                        ),
                        callback_data=f"img:tgl:pag:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.upscaler', '🔍 Upscaler', telegram_locale=telegram_locale)} "
                            f"{upsc_status}"
                        ),
                        callback_data=f"img:menu:up:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.hires_scale', 'Hi-res scale', telegram_locale=telegram_locale)} "
                            f"×{params.hires_scale}"
                        ),
                        callback_data=f"img:menu:hrs:{artifact.artifact_id}",
                    ),
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.hires_denoise', 'Hi-res den', telegram_locale=telegram_locale)} "
                            f"{params.hires_denoise}"
                        ),
                        callback_data=f"img:menu:hrd:{artifact.artifact_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.enh.pag_scale', 'PAG scale', telegram_locale=telegram_locale)} "
                            f"{params.pag_scale}"
                        ),
                        callback_data=f"img:menu:pags:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "common.action.back",
                            "⬅️ Назад",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:open:{artifact.artifact_id}",
                    )
                ],
            ]
        )

    def _artifact_size_keyboard(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> InlineKeyboardMarkup:
        shrink_label = _t(
            uid,
            "common.value.off",
            "off",
            telegram_locale=telegram_locale,
        )
        if artifact.shrink_width and artifact.shrink_height:
            shrink_label = f"{artifact.shrink_width}x{artifact.shrink_height}"
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.size.compression', '🗜 Сжатие', telegram_locale=telegram_locale)} "
                            f"{artifact.compression_percent}%"
                        ),
                        callback_data=f"img:menu:cmp:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=(
                            f"{_t(uid, 'prompt_editor.send.menu.size.shrink', '📦 Shrink', telegram_locale=telegram_locale)} "
                            f"{shrink_label}"
                        ),
                        callback_data=f"img:menu:shk:{artifact.artifact_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text=_t(
                            uid,
                            "common.action.back",
                            "⬅️ Назад",
                            telegram_locale=telegram_locale,
                        ),
                        callback_data=f"img:open:{artifact.artifact_id}",
                    )
                ],
            ]
        )

    def _artifact_hub_caption(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> str:
        params = artifact.params
        lines = [
            _t(
                uid,
                "prompt_editor.send.caption.hub.title",
                "✨ <b>Улучшения для этой картинки</b>",
                telegram_locale=telegram_locale,
            ),
            _t(
                uid,
                "prompt_editor.send.caption.hub.chain_step",
                "Шаг цепочки: <code>{step}</code>",
                telegram_locale=telegram_locale,
                params={"step": artifact.generation_step},
            ),
            _t(
                uid,
                "prompt_editor.send.caption.hub.seed",
                "Seed: <code>{seed}</code>",
                telegram_locale=telegram_locale,
                params={"seed": artifact.used_seed},
            ),
            _t(
                uid,
                "prompt_editor.send.caption.hub.size",
                "Размер: <code>{width}x{height}</code>",
                telegram_locale=telegram_locale,
                params={"width": params.width, "height": params.height},
            ),
        ]
        modes: list[str] = []
        if artifact.enable_sampler_pass:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.sampler",
                    "🔁 sampler",
                    telegram_locale=telegram_locale,
                )
            )
        if params.enable_hires_fix:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.hires",
                    "🛠 hi-res",
                    telegram_locale=telegram_locale,
                )
            )
        if params.enable_freeu:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.freeu",
                    "⚡ freeu",
                    telegram_locale=telegram_locale,
                )
            )
        if params.enable_pag:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.pag",
                    "🎯 pag",
                    telegram_locale=telegram_locale,
                )
            )
        if params.upscale_model:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.upscale",
                    "🔍 upscale",
                    telegram_locale=telegram_locale,
                )
            )
        if artifact.compression_percent < 100:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.compression",
                    "🗜 сжатие {percent}%",
                    telegram_locale=telegram_locale,
                    params={"percent": artifact.compression_percent},
                )
            )
        if artifact.shrink_width and artifact.shrink_height:
            modes.append(
                _t(
                    uid,
                    "prompt_editor.send.caption.hub.mode.shrink",
                    "📦 shrink {width}x{height}",
                    telegram_locale=telegram_locale,
                    params={
                        "width": artifact.shrink_width,
                        "height": artifact.shrink_height,
                    },
                )
            )
        lines.append(
            _t(
                uid,
                "prompt_editor.send.caption.hub.modes",
                "Режимы: {modes}",
                telegram_locale=telegram_locale,
                params={
                    "modes": ", ".join(modes)
                    if modes
                    else _t(
                        uid,
                        "prompt_editor.send.caption.hub.none",
                        "<i>не выбраны</i>",
                        telegram_locale=telegram_locale,
                    )
                },
            )
        )
        lines.append(
            _t(
                uid,
                "prompt_editor.send.caption.hub.help",
                "Откройте нужный тематический раздел для настройки.",
                telegram_locale=telegram_locale,
            )
        )
        return "\n".join(lines)

    def _artifact_sampling_caption(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> str:
        params = artifact.params
        sampler_pass = _t(
            uid,
            (
                "prompt_editor.send.caption.sampling.enabled"
                if artifact.enable_sampler_pass
                else "prompt_editor.send.caption.sampling.disabled"
            ),
            "✅ включен" if artifact.enable_sampler_pass else "❌ выключен",
            telegram_locale=telegram_locale,
        )
        return (
            _t(
                uid,
                "prompt_editor.send.caption.sampling.title",
                "📐 <b>Сэмплинг</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            "\n"
            + _t(
                uid,
                "prompt_editor.send.caption.sampling.sampler_pass",
                "<b>Сэмплер-проход:</b> {status}",
                telegram_locale=telegram_locale,
                params={"status": sampler_pass},
            )
            + "\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.sampling.label.steps', 'Steps', telegram_locale=telegram_locale)}:</b> <code>{params.steps}</code>  "
            f"<b>{_t(uid, 'prompt_editor.send.caption.sampling.label.cfg', 'CFG', telegram_locale=telegram_locale)}:</b> <code>{params.cfg}</code>\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.sampling.label.denoise', 'Denoise', telegram_locale=telegram_locale)}:</b> <code>{params.denoise}</code>\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.sampling.label.sampler', 'Sampler', telegram_locale=telegram_locale)}:</b> <code>{h(params.sampler)}</code>\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.sampling.label.scheduler', 'Scheduler', telegram_locale=telegram_locale)}:</b> <code>{h(params.scheduler)}</code>"
        )

    def _artifact_enhancements_caption(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> str:
        params = artifact.params
        upscaler = (
            h(params.upscale_model)
            if params.upscale_model
            else _t(uid, "common.value.off", "off", telegram_locale=telegram_locale)
        )
        return (
            _t(
                uid,
                "prompt_editor.send.caption.enh.title",
                "✨ <b>Улучшения</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            "\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.hires', 'Hi-res', telegram_locale=telegram_locale)}:</b> {'✅' if params.enable_hires_fix else '❌'}  "
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.scale', 'Scale', telegram_locale=telegram_locale)}:</b> <code>{params.hires_scale}</code>  "
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.denoise', 'Denoise', telegram_locale=telegram_locale)}:</b> <code>{params.hires_denoise}</code>\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.freeu', 'FreeU', telegram_locale=telegram_locale)}:</b> {'✅' if params.enable_freeu else '❌'}\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.pag', 'PAG', telegram_locale=telegram_locale)}:</b> {'✅' if params.enable_pag else '❌'}  "
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.scale', 'Scale', telegram_locale=telegram_locale)}:</b> <code>{params.pag_scale}</code>\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.enh.label.upscaler', 'Upscaler', telegram_locale=telegram_locale)}:</b> <code>{upscaler}</code>"
        )

    def _artifact_size_caption(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
    ) -> str:
        params = artifact.params
        shrink_line = _t(uid, "common.value.off", "off", telegram_locale=telegram_locale)
        if artifact.shrink_width and artifact.shrink_height:
            shrink_line = f"{artifact.shrink_width}x{artifact.shrink_height}"
        return (
            _t(
                uid,
                "prompt_editor.send.caption.size.title",
                "🖼 <b>Размер</b>",
                telegram_locale=telegram_locale,
            )
            + "\n"
            "\n"
            + _t(
                uid,
                "prompt_editor.send.caption.size.current",
                "<b>Текущий размер:</b> <code>{width}x{height}</code>",
                telegram_locale=telegram_locale,
                params={"width": params.width, "height": params.height},
            )
            + "\n"
            + _t(
                uid,
                "prompt_editor.send.caption.size.compression",
                "<b>Сжатие:</b> <code>{percent}%</code>",
                telegram_locale=telegram_locale,
                params={"percent": artifact.compression_percent},
            )
            + "\n"
            f"<b>{_t(uid, 'prompt_editor.send.caption.size.shrink_label', 'Shrink', telegram_locale=telegram_locale)}:</b> <code>{shrink_line}</code>\n"
            "\n"
            + _t(
                uid,
                "prompt_editor.send.caption.size.note1",
                "Сжатие применяется в самом конце улучшения, уже после sampler/hi-res/upscaler, с сохранением пропорций.",
                telegram_locale=telegram_locale,
            )
            + "\n"
            + _t(
                uid,
                "prompt_editor.send.caption.size.note2",
                "Shrink (XxY) ограничивает итог по рамке без апскейла.",
                telegram_locale=telegram_locale,
            )
        )

    def _artifact_menu_caption(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
        menu: str = "hub",
    ) -> str:
        if menu == "smp":
            return _artifact_sampling_caption(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        if menu == "enh":
            return _artifact_enhancements_caption(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        if menu == "size":
            return _artifact_size_caption(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        return _artifact_hub_caption(
            artifact,
            uid=uid,
            telegram_locale=telegram_locale,
        )

    def _artifact_menu_keyboard(
        artifact: PreviewArtifact,
        *,
        uid: int,
        telegram_locale: str | None,
        menu: str = "hub",
    ) -> InlineKeyboardMarkup:
        if menu == "smp":
            return _artifact_sampling_keyboard(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        if menu == "enh":
            return _artifact_enhancements_keyboard(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        if menu == "size":
            return _artifact_size_keyboard(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
            )
        return _artifact_hub_keyboard(
            artifact,
            uid=uid,
            telegram_locale=telegram_locale,
        )

    async def _edit_preview_message(
        cb: CallbackQuery,
        *,
        caption: str,
        reply_markup: InlineKeyboardMarkup,
    ) -> None:
        message = await require_callback_message(cb)
        if message is None:
            return
        try:
            await message.edit_caption(caption=caption, reply_markup=reply_markup)
            return
        except TelegramBadRequest:
            pass
        try:
            await message.edit_text(caption, reply_markup=reply_markup)
        except TelegramBadRequest:
            await message.answer(caption, reply_markup=reply_markup)

    async def _render_artifact_menu(
        cb: CallbackQuery,
        artifact: PreviewArtifact,
        *,
        menu: str = "hub",
    ) -> None:
        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        await _edit_preview_message(
            cb,
            caption=_artifact_menu_caption(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
                menu=menu,
            ),
            reply_markup=_artifact_menu_keyboard(
                artifact,
                uid=uid,
                telegram_locale=telegram_locale,
                menu=menu,
            ),
        )

    enhancement_deps = PromptEditorEnhancementDeps(
        logger=deps.logger,
        runtime=deps.runtime,
        client=deps.client,
        deliver_generated_images=deps.deliver_generated_images,
        preview_image_keyboard=deps.preview_image_keyboard,
        move_main_panel_to_bottom=_move_main_panel_to_bottom,
    )

    @router.callback_query(F.data.startswith("send:"))
    async def send_images(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        mode = await _callback_value(
            cb,
            prefix="send",
            invalid_key="prompt_editor.send.alert.invalid_mode",
            invalid_default="❌ Некорректный режим.",
        )
        if mode is None:
            return

        if mode == "new":
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
                    "prompt_editor.send.notice.return_to_editor_new",
                    "🔄 Возвращаемся к настройкам для новой генерации.",
                    telegram_locale=cb.from_user.language_code,
                ),
            )
            await cb.answer()
            return

        if mode == "cancel":
            await state.clear()
            uid = cb.from_user.id
            locale = _resolved_locale(
                uid,
                telegram_locale=cb.from_user.language_code,
            )
            start_panel_text = start_text(
                deps.localization,
                locale=locale,
                text_service=deps.ui_text,
            )
            root_keyboard = main_menu_keyboard(
                deps.localization,
                locale=locale,
                text_service=deps.ui_text,
            )
            try:
                await message.edit_text(
                    start_panel_text,
                    reply_markup=root_keyboard,
                )
            except TelegramBadRequest:
                await message.answer(
                    start_panel_text,
                    reply_markup=root_keyboard,
                )
            await cb.answer()
            return

        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.send.alert.unknown_send_mode",
                "❌ Неизвестный режим отправки.",
            ),
            show_alert=True,
        )

    @router.callback_query(F.data.startswith("img:png:"))
    async def image_send_png(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return

        artifact_payload = await _artifact_from_callback(
            cb,
            prefix="img:png",
            missing_key="prompt_editor.send.alert.image_not_found_or_unavailable",
            missing_default="⚠️ Картинка не найдена или недоступна.",
        )
        if artifact_payload is None:
            return
        _, artifact = artifact_payload

        image_bytes = deps.runtime.artifact_bytes(artifact)
        if not image_bytes:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_source_data_missing",
                    "⚠️ Исходные данные картинки не найдены.",
                ),
                show_alert=True,
            )
            return

        await deps.deliver_generated_images(
            message,
            [image_bytes],
            used_seed=artifact.used_seed,
            mode="file",
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.send.alert.png_sent",
                "📄 PNG отправлен",
            )
        )

    @router.callback_query(F.data.startswith("img:back:"))
    async def image_back(cb: CallbackQuery):
        artifact_payload = await _artifact_from_callback(cb, prefix="img:back")
        if artifact_payload is None:
            return
        _, artifact = artifact_payload
        uid = cb.from_user.id
        await _edit_preview_message(
            cb,
            caption=_t(
                uid,
                "prompt_editor.send.caption.preview_actions",
                "🖼 Шаг {step} | Seed: {seed}\nВыберите действие для этой картинки.",
                telegram_locale=cb.from_user.language_code,
                params={
                    "step": artifact.generation_step,
                    "seed": artifact.used_seed,
                },
            ),
            reply_markup=deps.preview_image_keyboard(
                artifact.artifact_id,
                artifact.parent_artifact_id,
            ),
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("img:goto_parent:"))
    async def image_goto_parent(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return

        artifact_payload = await _artifact_from_callback(cb, prefix="img:goto_parent")
        if artifact_payload is None:
            return
        _, artifact = artifact_payload
        if not artifact.parent_artifact_id:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.parent_not_found",
                    "⚠️ Исходник не найден.",
                ),
                show_alert=True,
            )
            return
        parent = _user_artifact(cb.from_user.id, artifact.parent_artifact_id)
        if not parent or parent.preview_message_id is None or parent.preview_chat_id is None:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.parent_link_unavailable",
                    "⚠️ Ссылка на исходник недоступна.",
                ),
                show_alert=True,
            )
            return
        if parent.preview_chat_id != message.chat.id:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.parent_other_chat",
                    "⚠️ Исходник в другом чате.",
                ),
                show_alert=True,
            )
            return
        await message.answer(
            _t_cb(
                cb,
                "prompt_editor.send.message.parent_image",
                "↩️ Исходная картинка",
            ),
            reply_to_message_id=parent.preview_message_id,
        )
        await cb.answer(_t_cb(cb, "prompt_editor.send.alert.done", "Готово"))

    @router.callback_query(F.data.startswith("img:open:"))
    async def image_open_enhancements(cb: CallbackQuery):
        artifact_payload = await _artifact_from_callback(cb, prefix="img:open")
        if artifact_payload is None:
            return
        _, artifact = artifact_payload

        await _render_artifact_menu(cb, artifact, menu="hub")
        await cb.answer()

    @router.callback_query(F.data.startswith("img:sub:"))
    async def image_open_submenu(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return

        menu_key = parts[2]
        artifact_id = parts[3]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        if menu_key not in {"smp", "enh", "size"}:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.unknown_section",
                    "⚠️ Неизвестный раздел.",
                ),
                show_alert=True,
            )
            return

        await _render_artifact_menu(cb, artifact, menu=menu_key)
        await cb.answer()

    @router.callback_query(F.data.startswith("img:tgl:"))
    async def image_toggle(cb: CallbackQuery):
        uid = cb.from_user.id
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        toggle_key = parts[2]
        artifact_id = parts[3]

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        if toggle_key == "smp":
            artifact.enable_sampler_pass = not artifact.enable_sampler_pass
            submenu = "smp"
        elif toggle_key == "hi":
            toggle_hires_fix(artifact)
            submenu = "enh"
        elif toggle_key == "fu":
            if not deps.client.info.freeu_supported:
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.freeu_not_supported",
                        "⚠️ FreeU не поддерживается сервером.",
                    ),
                    show_alert=True,
                )
                return
            artifact.params.enable_freeu = not artifact.params.enable_freeu
            if artifact.params.enable_freeu:
                artifact.enable_sampler_pass = True
            submenu = "enh"
        elif toggle_key == "pag":
            if not deps.client.info.pag_supported:
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.pag_not_supported",
                        "⚠️ PAG не поддерживается сервером.",
                    ),
                    show_alert=True,
                )
                return
            artifact.params.enable_pag = not artifact.params.enable_pag
            if artifact.params.enable_pag:
                artifact.enable_sampler_pass = True
            submenu = "enh"
        else:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.unknown_toggle",
                    "⚠️ Неизвестный переключатель.",
                ),
                show_alert=True,
            )
            return

        await _render_artifact_menu(cb, artifact, menu=submenu)
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.send.alert.updated",
                "✅ Обновлено",
            )
        )

    @router.callback_query(F.data.startswith("img:menu:"))
    async def image_menu_open(cb: CallbackQuery):
        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 4:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        menu_key = parts[2]
        artifact_id = parts[3]
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        back_callback = submenu_back_callback(menu_key, artifact_id)

        if menu_key == "steps":
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="steps",
                values=["10", "15", "20", "25", "30", "40"],
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.steps",
                    "Выберите Steps:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "cfg":
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="cfg",
                values=["4.0", "5.0", "6.0", "7.0", "8.0", "10.0"],
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.cfg",
                    "Выберите CFG:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "den":
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="denoise",
                values=["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.denoise",
                    "Выберите Denoise:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "hrs":
            hires_scale_values = enhancement_preset_values("hires_scale")
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="hires_scale",
                values=hires_scale_values,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.hires_scale",
                    "Выберите Hi-res scale ({range}):",
                    telegram_locale=telegram_locale,
                    params={"range": numeric_control_range_text("hires_scale")},
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "hrd":
            hires_denoise_values = enhancement_preset_values("hires_denoise")
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="hires_denoise",
                values=hires_denoise_values,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.hires_denoise",
                    "Выберите Hi-res denoise ({range}):",
                    telegram_locale=telegram_locale,
                    params={"range": numeric_control_range_text("hires_denoise")},
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "pags":
            pag_scale_values = enhancement_preset_values("pag_scale")
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="pag_scale",
                values=pag_scale_values,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.pag_scale",
                    "Выберите PAG scale ({range}):",
                    telegram_locale=telegram_locale,
                    params={"range": numeric_control_range_text("pag_scale")},
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "sampler":
            samplers = deps.client.info.samplers or ["euler"]
            kb = paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="sampler",
                items=samplers,
                page=0,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.sampler",
                    "Выберите sampler:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "scheduler":
            schedulers = deps.client.info.schedulers or ["normal"]
            kb = paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="scheduler",
                items=schedulers,
                page=0,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.scheduler",
                    "Выберите scheduler:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "up":
            no_upscale_label = _t(
                uid,
                "prompt_editor.send.value.no_upscale",
                "(без апскейла)",
                telegram_locale=telegram_locale,
            )
            upscalers = [no_upscale_label] + deps.client.info.upscale_models
            kb = paginated_pick_keyboard(
                artifact_id=artifact_id,
                menu="up",
                items=upscalers,
                page=0,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.upscaler",
                    "Выберите upscaler:",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "cmp":
            kb = simple_value_keyboard(
                artifact_id=artifact_id,
                key="compression_percent",
                values=["100", "90", "80", "70", "60", "50"],
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.compression",
                    "Выберите сжатие (%):",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return
        if menu_key == "shk":
            kb = shrink_value_keyboard(
                artifact_id=artifact_id,
                back_callback=back_callback,
                translate=_menu_translate,
                locale=locale,
            )
            await _edit_preview_message(
                cb,
                caption=_t(
                    uid,
                    "prompt_editor.send.menu.select.shrink",
                    "Выберите shrink-лимит (XxY).\nМетод shrink уменьшает картинку до рамки без апскейла и сохраняет пропорции.",
                    telegram_locale=telegram_locale,
                ),
                reply_markup=kb,
            )
            await cb.answer()
            return

        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.send.alert.unknown_menu",
                "⚠️ Неизвестное меню.",
            ),
            show_alert=True,
        )

    @router.callback_query(F.data.startswith("img:page:"))
    async def image_menu_page(cb: CallbackQuery):
        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        locale = _resolved_locale(uid, telegram_locale=telegram_locale)
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        menu = parts[2]
        artifact_id = parts[3]
        page = int(parts[4])
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        if menu == "sampler":
            items = deps.client.info.samplers or ["euler"]
            caption = _t(
                uid,
                "prompt_editor.send.menu.select.sampler",
                "Выберите sampler:",
                telegram_locale=telegram_locale,
            )
        elif menu == "scheduler":
            items = deps.client.info.schedulers or ["normal"]
            caption = _t(
                uid,
                "prompt_editor.send.menu.select.scheduler",
                "Выберите scheduler:",
                telegram_locale=telegram_locale,
            )
        elif menu == "up":
            items = [
                _t(
                    uid,
                    "prompt_editor.send.value.no_upscale",
                    "(без апскейла)",
                    telegram_locale=telegram_locale,
                )
            ] + deps.client.info.upscale_models
            caption = _t(
                uid,
                "prompt_editor.send.menu.select.upscaler",
                "Выберите upscaler:",
                telegram_locale=telegram_locale,
            )
        else:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.unknown_page",
                    "⚠️ Неизвестная страница.",
                ),
                show_alert=True,
            )
            return

        kb = paginated_pick_keyboard(
            artifact_id=artifact_id,
            menu=menu,
            items=items,
            page=page,
            back_callback=submenu_back_callback(menu, artifact_id),
            translate=_menu_translate,
            locale=locale,
        )
        await _edit_preview_message(cb, caption=caption, reply_markup=kb)
        await cb.answer()

    @router.callback_query(F.data.startswith("img:pick:"))
    async def image_menu_pick(cb: CallbackQuery):
        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        menu = parts[2]
        artifact_id = parts[3]
        idx = int(parts[4])

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        if menu == "sampler":
            items = deps.client.info.samplers or ["euler"]
            if idx < 0 or idx >= len(items):
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.invalid_choice",
                        "❌ Неверный выбор.",
                    ),
                    show_alert=True,
                )
                return
            artifact.params.sampler = items[idx]
            artifact.enable_sampler_pass = True
        elif menu == "scheduler":
            items = deps.client.info.schedulers or ["normal"]
            if idx < 0 or idx >= len(items):
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.invalid_choice",
                        "❌ Неверный выбор.",
                    ),
                    show_alert=True,
                )
                return
            artifact.params.scheduler = items[idx]
            artifact.enable_sampler_pass = True
        elif menu == "up":
            no_upscale_label = _t(
                uid,
                "prompt_editor.send.value.no_upscale",
                "(без апскейла)",
                telegram_locale=telegram_locale,
            )
            items = [no_upscale_label] + deps.client.info.upscale_models
            if idx < 0 or idx >= len(items):
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.invalid_choice",
                        "❌ Неверный выбор.",
                    ),
                    show_alert=True,
                )
                return
            chosen = items[idx]
            artifact.params.upscale_model = "" if chosen == no_upscale_label else chosen
        else:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.unknown_choice",
                    "⚠️ Неизвестный выбор.",
                ),
                show_alert=True,
            )
            return

        submenu = "smp" if menu in {"sampler", "scheduler"} else "enh"
        await _render_artifact_menu(cb, artifact, menu=submenu)
        await cb.answer(_t_cb(cb, "prompt_editor.send.alert.updated", "✅ Обновлено"))

    @router.callback_query(F.data.startswith("img:set:"))
    async def image_set_value(cb: CallbackQuery):
        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        data_value = cb.data or ""
        parts = data_value.split(":")
        if len(parts) != 5:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        field = parts[2]
        artifact_id = parts[3]
        raw_value = parts[4]

        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        try:
            parsed_value: float | int
            if field == "shrink_size":
                if raw_value == "off":
                    artifact.shrink_width = None
                    artifact.shrink_height = None
                    await _render_artifact_menu(cb, artifact, menu="size")
                    await cb.answer(
                        _t_cb(
                            cb,
                            "prompt_editor.send.alert.shrink_disabled",
                            "✅ Shrink выключен",
                        )
                    )
                    return
                parsed = parse_shrink_size(raw_value)
                if not parsed:
                    await cb.answer(
                        _t_cb(
                            cb,
                            "prompt_editor.send.alert.shrink_format_xy",
                            "⚠️ Формат shrink должен быть XxY.",
                        ),
                        show_alert=True,
                    )
                    return
                artifact.shrink_width, artifact.shrink_height = parsed
                await _render_artifact_menu(cb, artifact, menu="size")
                await cb.answer(
                    _t_cb(
                        cb,
                        "prompt_editor.send.alert.shrink_updated",
                        "✅ Shrink обновлен",
                    )
                )
                return
            if field in {"steps", "compression_percent"}:
                parsed_value = int(raw_value)
            else:
                parsed_value = float(raw_value)
            if not apply_field_value(artifact, field=field, value=parsed_value):
                await cb.answer(
                    _t_cb(cb, "common.alert.unknown_param", "⚠️ Неизвестный параметр."),
                    show_alert=True,
                )
                return
        except ValueError:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.value_apply_failed",
                    "⚠️ Не удалось применить значение.",
                ),
                show_alert=True,
            )
            return

        await _render_artifact_menu(cb, artifact, menu=submenu_for_field(field))
        await cb.answer(
            _t(
                uid,
                "prompt_editor.send.alert.param_updated",
                "✅ Параметр обновлен",
                telegram_locale=telegram_locale,
            )
        )

    @router.callback_query(F.data.startswith("img:custom:"))
    async def image_custom_start(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return

        value = await _callback_value(cb, prefix="img:custom")
        if value is None:
            return
        parts = value.split(":", 1)
        if len(parts) != 2:
            await cb.answer(
                _t_cb(cb, "common.alert.invalid_request", "⚠️ Некорректный запрос."),
                show_alert=True,
            )
            return
        field, artifact_id = parts

        uid = cb.from_user.id
        telegram_locale = cb.from_user.language_code
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            await cb.answer(
                _t_cb(
                    cb,
                    "prompt_editor.send.alert.image_not_found",
                    "⚠️ Картинка не найдена.",
                ),
                show_alert=True,
            )
            return

        if field == "shrink_size":
            deps.runtime.pending_image_inputs[uid] = {
                "artifact_id": artifact_id,
                "field": field,
                "submenu": "size",
            }
            await message.answer(
                _t(
                    uid,
                    "prompt_editor.send.prompt.custom_shrink",
                    "✏️ Введите shrink-лимит в формате XxY (например 1280x720).\nДопустимый диапазон каждой стороны: 1..4096.",
                    telegram_locale=telegram_locale,
                )
            )
            await cb.answer()
            return

        try:
            label, min_val, max_val = custom_field_meta(
                field,
                translate=_menu_translate,
                locale=_resolved_locale(uid, telegram_locale=telegram_locale),
            )
        except ValueError:
            await cb.answer(
                _t_cb(cb, "common.alert.unknown_param", "⚠️ Неизвестный параметр."),
                show_alert=True,
            )
            return

        deps.runtime.pending_image_inputs[uid] = {
            "artifact_id": artifact_id,
            "field": field,
            "submenu": submenu_for_field(field),
        }
        await message.answer(
            _t(
                uid,
                "prompt_editor.send.prompt.custom_value",
                "✏️ Введите {label} ({min}..{max}).\nМожно использовать точку или запятую.",
                telegram_locale=telegram_locale,
                params={"label": label, "min": min_val, "max": max_val},
            )
        )
        await cb.answer()

    @router.callback_query(F.data.startswith("img:to_editor:"))
    async def image_to_editor(cb: CallbackQuery, state: FSMContext):
        message = await require_callback_message(cb)
        if message is None:
            return

        artifact_payload = await _artifact_from_callback(cb, prefix="img:to_editor")
        if artifact_payload is None:
            return
        _, artifact = artifact_payload
        uid = cb.from_user.id

        deps.runtime.active_prompt_requests[uid] = PromptRequest(
            params=GenerationParams(**asdict(artifact.params)),
            operation="generate",
        )
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=False,
            notice=_t(
                uid,
                "prompt_editor.send.notice.to_editor",
                "🧬 Параметры картинки перенесены в главный редактор.",
                telegram_locale=cb.from_user.language_code,
            ),
        )
        await cb.answer()

    @router.message(_has_pending_input, F.text)
    async def image_custom_value_input(msg: Message):
        uid = msg.from_user.id if msg.from_user else 0
        if uid <= 0:
            return
        telegram_locale = msg.from_user.language_code if msg.from_user else None

        pending = deps.runtime.pending_image_inputs.get(uid)
        if not pending:
            return

        raw = (msg.text or "").strip()
        if raw.lower() in {"cancel", "/cancel", "отмена"}:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.send.input.cancelled",
                    "↩️ Ввод параметра отменен.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        artifact_id = str(pending.get("artifact_id") or "")
        field = str(pending.get("field") or "")
        artifact = _user_artifact(uid, artifact_id)
        if not artifact:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.send.input.image_unavailable",
                    "⚠️ Картинка уже недоступна.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        if field == "shrink_size":
            parsed = parse_shrink_size(raw)
            if not parsed:
                await msg.answer(
                    _t(
                        uid,
                        "prompt_editor.send.input.shrink_format",
                        "⚠️ Формат shrink: XxY, например 1280x720.",
                        telegram_locale=telegram_locale,
                    )
                )
                return
            artifact.shrink_width, artifact.shrink_height = parsed
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.send.input.shrink_updated",
                    "✅ Shrink обновлен. Вернитесь к превью-картинке и нажмите «✨ Улучшить».",
                    telegram_locale=telegram_locale,
                )
            )
            return

        try:
            label, min_val, max_val = custom_field_meta(
                field,
                translate=_menu_translate,
                locale=_resolved_locale(uid, telegram_locale=telegram_locale),
            )
        except ValueError:
            deps.runtime.pending_image_inputs.pop(uid, None)
            await msg.answer(
                _t(
                    uid,
                    "common.alert.unknown_param",
                    "⚠️ Неизвестный параметр.",
                    telegram_locale=telegram_locale,
                )
            )
            return

        value_raw = raw.replace(",", ".")
        try:
            value: float | int
            if field in {"steps", "compression_percent"}:
                value = int(float(value_raw))
            else:
                value = float(value_raw)
            if value < min_val or value > max_val:
                raise ValueError("out of range")
        except ValueError:
            await msg.answer(
                _t(
                    uid,
                    "prompt_editor.send.input.invalid_range",
                    "⚠️ Введите {label} в диапазоне {min}..{max}.",
                    telegram_locale=telegram_locale,
                    params={"label": label, "min": min_val, "max": max_val},
                )
            )
            return

        apply_field_value(artifact, field=field, value=value)

        deps.runtime.pending_image_inputs.pop(uid, None)
        await msg.answer(
            _t(
                uid,
                "prompt_editor.send.input.value_updated",
                "✅ {label} обновлен. Вернитесь к превью-картинке и нажмите «✨ Улучшить».",
                telegram_locale=telegram_locale,
                params={"label": label},
            )
        )

    @router.callback_query(F.data.startswith("img:run:"))
    async def image_run_enhancements(cb: CallbackQuery):
        message = await require_callback_message(cb)
        if message is None:
            return

        uid = cb.from_user.id
        artifact_payload = await _artifact_from_callback(cb, prefix="img:run")
        if artifact_payload is None:
            return
        _, artifact = artifact_payload

        await start_image_enhancement(
            message,
            uid=uid,
            artifact=artifact,
            deps=enhancement_deps,
        )
        await cb.answer(
            _t_cb(
                cb,
                "prompt_editor.send.alert.enhancement_started",
                "🚀 Улучшение запущено",
            )
        )
