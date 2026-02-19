from __future__ import annotations

import asyncio
import logging
import re

from aiogram import F, Router
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from comfyui_client import ComfyUIClient
from config import Config
from core.html_utils import h, truncate
from core.panels import remember_user_panel, render_user_panel
from core.runtime import RuntimeStore, get_user_pro_mode
from core.telegram import message_user_id

logger = logging.getLogger(__name__)


def _user_generations(runtime: RuntimeStore, uid: int):
    items = [
        item for item in runtime.active_generations.values() if item.owner_uid == uid
    ]
    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


TRAINING_TEXT = (
    "\U0001f393 <b>\u041e\u0431\u0443\u0447\u0435\u043d\u0438\u0435 ComfyBot (\u0434\u043b\u044f \u043d\u043e\u0432\u0438\u0447\u043a\u043e\u0432)</b>\n"
    "\n"
    "1) /models -> \u043f\u0440\u043e\u0432\u0435\u0440\u044c\u0442\u0435 \u0441\u043f\u0438\u0441\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439.\n"
    "2) /generate -> \u043e\u0442\u043a\u0440\u043e\u0439\u0442\u0435 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440.\n"
    "3) \u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 Checkpoint, \u0437\u0430\u0434\u0430\u0439\u0442\u0435 Positive/Negative.\n"
    "4) \u041f\u0440\u0438 \u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e\u0441\u0442\u0438 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u0442\u0435 Steps, CFG, Seed, \u0440\u0430\u0437\u043c\u0435\u0440.\n"
    "5) \u041d\u0430\u0436\u043c\u0438\u0442\u0435 \u00ab\u0413\u0435\u043d\u0435\u0440\u0438\u0440\u043e\u0432\u0430\u0442\u044c\u00bb.\n"
    "\n"
    "\u041f\u043e\u043b\u0435\u0437\u043d\u043e: /repeat, /presets, /download, /cancel.\n"
    "\u0415\u0449\u0451: \U0001f9e0 Smart (TIPO), \U0001f4ce LoRA, \U0001f5bc \u0440\u0435\u0444\u0435\u0440\u0435\u043d\u0441\u044b, \u2728 \u0443\u043b\u0443\u0447\u0448\u0435\u043d\u0438\u044f.\n"
    "\n"
    "\u041f\u043e\u043b\u043d\u044b\u0439 \u0433\u0430\u0439\u0434: <code>docs/COMMANDS.md</code>"
)


def register_common_handlers(
    router: Router,
    cfg: Config,
    client: ComfyUIClient,
    runtime: RuntimeStore,
) -> None:
    @router.message.outer_middleware()
    async def wl_msg(handler, event: Message, data: dict):
        if (
            cfg.allowed_users
            and event.from_user
            and event.from_user.id not in cfg.allowed_users
        ):
            await event.answer(
                "\u26d4 <b>\u0414\u043e\u0441\u0442\u0443\u043f \u0437\u0430\u043f\u0440\u0435\u0449\u0451\u043d</b>\n\u0412\u0430\u0448 ID \u043d\u0435 \u0432 \u0441\u043f\u0438\u0441\u043a\u0435 \u0440\u0430\u0437\u0440\u0435\u0448\u0451\u043d\u043d\u044b\u0445."
            )
            return
        try:
            return await handler(event, data)
        finally:
            runtime.persist()

    @router.callback_query.outer_middleware()
    async def wl_cb(handler, event: CallbackQuery, data: dict):
        if (
            cfg.allowed_users
            and event.from_user
            and event.from_user.id not in cfg.allowed_users
        ):
            await event.answer(
                "\u26d4 \u0414\u043e\u0441\u0442\u0443\u043f \u0437\u0430\u043f\u0440\u0435\u0449\u0451\u043d.",
                show_alert=True,
            )
            return
        try:
            return await handler(event, data)
        finally:
            runtime.persist()

    @router.message(CommandStart())
    async def cmd_start(message: Message):
        uid = message_user_id(message)
        await render_user_panel(
            message,
            runtime,
            uid,
            "\U0001f3a8 <b>ComfyUI Generator Bot</b>\n"
            "\n"
            "\U0001f680 <b>\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f</b>\n"
            "/generate \u2014 \u043d\u043e\u0432\u0430\u044f \u0433\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f\n"
            "/repeat \u2014 \u043f\u043e\u0432\u0442\u043e\u0440\u0438\u0442\u044c \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u044e\u044e\n"
            "/presets \u2014 \u0431\u0438\u0431\u043b\u0438\u043e\u0442\u0435\u043a\u0430 \u043f\u0440\u0435\u0441\u0435\u0442\u043e\u0432\n"
            "\n"
            "\U0001f4e6 <b>\u041c\u043e\u0434\u0435\u043b\u0438</b>\n"
            "/download \u2014 \u043f\u043e\u0438\u0441\u043a \u0438 \u0441\u043a\u0430\u0447\u0438\u0432\u0430\u043d\u0438\u0435 \u043a\u043e\u043c\u043f\u043e\u043d\u0435\u043d\u0442\u043e\u0432 (CivitAI / HuggingFace)\n"
            "/models \u2014 \u043e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u0441\u043f\u0438\u0441\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439\n"
            "\n"
            "\u2699\ufe0f <b>\u041f\u0440\u043e\u0447\u0435\u0435</b>\n"
            "/queue \u2014 \u043e\u0447\u0435\u0440\u0435\u0434\u044c ComfyUI\n"
            "/settings \u2014 \u043d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e\n"
            "/training \u2014 \u043f\u043e\u0448\u0430\u0433\u043e\u0432\u043e\u0435 \u043e\u0431\u0443\u0447\u0435\u043d\u0438\u0435\n"
            "/cancel \u2014 \u043e\u0442\u043c\u0435\u043d\u0438\u0442\u044c \u0442\u0435\u043a\u0443\u0449\u0443\u044e \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u044e\n"
            "\n"
            "\U0001f4a1 <i>\u0412 \u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440\u0435 \u0435\u0441\u0442\u044c \u043a\u043d\u043e\u043f\u043a\u0430</i> \U0001f527 \u041f\u0440\u043e <i>\u0434\u043b\u044f \u0440\u0430\u0441\u0448\u0438\u0440\u0435\u043d\u043d\u044b\u0445 \u043d\u0430\u0441\u0442\u0440\u043e\u0435\u043a.</i>\n"
            "\U0001f4cb <i>\u0422\u0430\u043c \u0436\u0435 \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e \u00ab\u041a\u043e\u043f\u0438\u0440\u043e\u0432\u0430\u0442\u044c/\u0432\u0441\u0442\u0430\u0432\u0438\u0442\u044c\u00bb \u0434\u043b\u044f \u043e\u0431\u043c\u0435\u043d\u0430 \u043f\u0440\u043e\u043c\u043f\u0442\u0430\u043c\u0438.</i>",
            prefer_edit=False,
        )

    @router.message(Command("help"))
    async def cmd_help(msg: Message):
        await cmd_start(msg)

    @router.message(Command("training"))
    @router.message(Command("learn"))
    async def cmd_training(msg: Message):
        uid = message_user_id(msg)
        await render_user_panel(msg, runtime, uid, TRAINING_TEXT)

    @router.message(Command("cancel"))
    async def cmd_cancel(msg: Message, state: FSMContext):
        uid = message_user_id(msg)
        state_name = await state.get_state()
        had_state = state_name is not None
        prompt_req = runtime.active_prompt_requests.get(uid)
        had_prompt = prompt_req is not None

        user_gens = _user_generations(runtime, uid)
        had_generation = bool(user_gens)
        gen_progress = ""
        if user_gens and user_gens[0].status_msg:
            status_text = (
                user_gens[0].status_msg.text or user_gens[0].status_msg.caption or ""
            ).strip()
            match = re.search(r"(\d{1,3})%", status_text)
            if match:
                pct = min(100, max(0, int(match.group(1))))
                gen_progress = f" ({pct}% готово)"

        dl_task = runtime.active_downloads.get(uid)
        had_download = bool(dl_task and not dl_task.done())

        cancelled_items: list[str] = []
        if prompt_req:
            pos = prompt_req.params.positive.strip() or "—"
            cancelled_items.append(
                f"Редактор промптов (Positive: «{h(truncate(pos, 36))}»)"
            )
        if had_generation:
            count = len(user_gens)
            suffix = "" if count == 1 else f" ({count} задач)"
            cancelled_items.append(
                f"Генерация{gen_progress or ' (в процессе)'}{suffix}"
            )
        if had_download:
            cancelled_items.append("Скачивание модели")
        if had_state and not cancelled_items:
            state_short = (
                state_name.split(":")[-1] if isinstance(state_name, str) else ""
            )
            cancelled_items.append(f"Текущее состояние ({h(state_short) or 'активно'})")

        if had_state:
            await state.clear()
        runtime.active_prompt_requests.pop(uid, None)

        if dl_task and not dl_task.done():
            dl_task.cancel()

        for gen in user_gens:
            if not gen.task.done():
                gen.task.cancel()
            if gen.prompt_id:
                # Attempt to cancel on ComfyUI side
                asyncio.create_task(client.cancel_prompt(gen.prompt_id))

        if cancelled_items:
            await render_user_panel(
                msg,
                runtime,
                uid,
                "❌ <b>Отменено:</b>\n"
                + "\n".join(f"• {item}" for item in cancelled_items),
            )
        elif had_state or had_prompt or had_generation or had_download:
            await render_user_panel(
                msg,
                runtime,
                uid,
                "❌ Операция отменена.",
            )
        else:
            await render_user_panel(
                msg,
                runtime,
                uid,
                "\u041d\u0435\u0447\u0435\u0433\u043e \u043e\u0442\u043c\u0435\u043d\u044f\u0442\u044c.",
            )

    @router.message(Command("models"))
    async def cmd_models(message: Message):
        uid = message_user_id(message)
        status_msg = await render_user_panel(
            message,
            runtime,
            uid,
            "\u23f3 \u041e\u0431\u043d\u043e\u0432\u043b\u044f\u044e \u0441\u043f\u0438\u0441\u043e\u043a \u043c\u043e\u0434\u0435\u043b\u0435\u0439\u2026",
        )
        try:
            info = await client.refresh_info()

            def _section(
                title: str, emoji: str, items: list[str], limit: int = 15
            ) -> str:
                count = len(items)
                header = f"{emoji} <b>{title}</b> ({count})"
                if not items:
                    return header
                shown = [f"  <code>{h(item)}</code>" for item in items[:limit]]
                if count > limit:
                    shown.append(
                        f"  <i>\u2026\u0438 \u0435\u0449\u0451 {count - limit}</i>"
                    )
                return header + "\n" + "\n".join(shown)

            parts = [
                _section("Checkpoints", "\U0001f9ea", info.checkpoints),
                _section("LoRA", "\U0001f9f2", info.loras),
                _section("Upscale", "\U0001f50d", info.upscale_models, limit=10),
                _section("VAE", "\U0001f9ec", info.vaes, limit=10),
                _section("ControlNet", "\U0001f9f7", info.controlnets, limit=10),
                _section("CLIP Vision", "\U0001f441", info.clip_vision_models, limit=5),
                _section("IP-Adapter", "\U0001f5bc", info.ipadapter_models, limit=5),
            ]

            ipa_icon = "\u2705" if info.ipadapter_supported else "\u274c"
            parts.append(
                f"\nIP-Adapter: {ipa_icon} {'доступен' if info.ipadapter_supported else 'недоступен'}"
            )

            await status_msg.edit_text(
                "\U0001f4e6 <b>\u041c\u043e\u0434\u0435\u043b\u0438 ComfyUI</b>\n\n"
                + "\n\n".join(parts)
            )
            remember_user_panel(runtime, uid, status_msg)
        except Exception as exc:
            await status_msg.edit_text(
                f"\u274c <b>\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u044f</b>\n<code>{h(exc)}</code>"
            )
            remember_user_panel(runtime, uid, status_msg)

    @router.message(Command("queue"))
    async def cmd_queue(msg: Message):
        uid = message_user_id(msg)
        try:
            queue = await client.get_queue_status()
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
            if running == 0 and pending == 0:
                status = "\U0001f7e2 \u041e\u0447\u0435\u0440\u0435\u0434\u044c \u043f\u0443\u0441\u0442\u0430"
            else:
                status = (
                    f"\u25b6\ufe0f <b>\u0412\u044b\u043f\u043e\u043b\u043d\u044f\u0435\u0442\u0441\u044f:</b> {running}\n"
                    f"\u23f3 <b>\u041e\u0436\u0438\u0434\u0430\u0435\u0442:</b> {pending}"
                )
            await render_user_panel(
                msg,
                runtime,
                uid,
                f"\U0001f4ca <b>\u041e\u0447\u0435\u0440\u0435\u0434\u044c ComfyUI</b>\n{status}",
            )
        except Exception as exc:
            await render_user_panel(
                msg,
                runtime,
                uid,
                f"\u274c \u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c \u0441\u0442\u0430\u0442\u0443\u0441: <code>{h(exc)}</code>",
            )

    @router.message(Command("settings"))
    async def cmd_settings(msg: Message):
        uid = message_user_id(msg)
        pro_mode = get_user_pro_mode(runtime, uid)
        mode_label = (
            "\U0001f527 \u041f\u0440\u043e"
            if pro_mode
            else "\U0001f7e2 \u041f\u0440\u043e\u0441\u0442\u043e\u0439"
        )

        smart_prompt_status = "\u274c \u0432\u044b\u043a\u043b\u044e\u0447\u0435\u043d"
        if cfg.smart_prompt_enabled:
            model = (
                cfg.smart_prompt_model
                or "(\u043c\u043e\u0434\u0435\u043b\u044c \u043d\u0435 \u0437\u0430\u0434\u0430\u043d\u0430)"
            )
            smart_prompt_status = f"\u2705 TIPO | <code>{h(model)}</code>"

        await render_user_panel(
            msg,
            runtime,
            uid,
            "\u2699\ufe0f <b>\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438 \u043f\u043e \u0443\u043c\u043e\u043b\u0447\u0430\u043d\u0438\u044e</b>\n"
            "\n"
            f"<b>\u0420\u0435\u0436\u0438\u043c:</b> {mode_label}\n"
            f"<b>\u0420\u0430\u0437\u043c\u0435\u0440:</b> <code>{cfg.default_width}\u00d7{cfg.default_height}</code>\n"
            f"<b>Steps:</b> <code>{cfg.default_steps}</code>  "
            f"<b>CFG:</b> <code>{cfg.default_cfg}</code>  "
            f"<b>Denoise:</b> <code>{cfg.default_denoise}</code>\n"
            f"<b>Sampler:</b> <code>{h(cfg.default_sampler)}</code>  "
            f"<b>Sched:</b> <code>{h(cfg.default_scheduler)}</code>\n"
            f"<b>Models:</b> <code>{h(cfg.comfyui_models_path)}</code>\n"
            f"<b>Smart Prompt:</b> {smart_prompt_status}",
        )

    @router.callback_query(F.data == "noop")
    async def noop(cb: CallbackQuery):
        await cb.answer()

    @router.message(F.text, ~F.text.startswith("/"), StateFilter(None))
    async def fallback_text(msg: Message):
        uid = message_user_id(msg)
        await render_user_panel(
            msg,
            runtime,
            uid,
            "Я не ожидаю текст. Используйте /generate, /help или /training.",
        )
