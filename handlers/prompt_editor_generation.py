from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import aiohttp
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from core.html_utils import h, progress_bar, truncate
from core.image_utils import image_dimensions
from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore


@dataclass
class PromptEditorGenerationDeps:
    runtime: RuntimeStore
    logger: logging.Logger
    client: _ClientLike
    normalize_params: Callable[[GenerationParams], GenerationParams]
    incompatible_loras: Callable[[GenerationParams], list[tuple[str, str, str]]]
    collect_reference_images: Callable[
        [Message, list[dict[str, str]]], Awaitable[tuple[list[bytes], int]]
    ]
    denoise_from_reference_strength: Callable[[float], float]
    show_prompt_editor: Callable[..., Awaitable[None]]
    show_prompt_panel: Callable[..., Awaitable[Message]]
    move_prompt_panel_to_bottom: Callable[..., Awaitable[Message]]
    remember_prompt_panel: Callable[[PromptRequest, Message], None]
    generation_result_keyboard: Callable[[], InlineKeyboardMarkup]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    prune_preview_artifacts: Callable[[int], None]


async def run_generate_operation(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    deps: PromptEditorGenerationDeps,
) -> None:
    req = deps.runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer("❌ Активный запрос не найден. Используйте /generate.")
        return

    params = deps.normalize_params(req.params)
    if not params.checkpoint:
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="❌ Выберите checkpoint в редакторе перед генерацией.",
        )
        return

    bad_loras = deps.incompatible_loras(params)
    lora_warning = ""
    if bad_loras:
        listed = ", ".join(name for name, _, _ in bad_loras[:3])
        suffix = "" if len(bad_loras) <= 3 else f" и ещё {len(bad_loras) - 3}"
        lora_warning = (
            "⚠️ <b>Внимание:</b> потенциально несовместимые LoRA: "
            f"<code>{h(listed)}</code>{h(suffix)}"
        )

    used_seed = params.seed if params.seed >= 0 else random.randint(0, 2**63 - 1)
    generation_params = GenerationParams(**asdict(params))
    generation_params.seed = used_seed

    reference_images, failed_refs = await deps.collect_reference_images(
        message,
        params.reference_images,
    )
    if params.controlnet_name and not reference_images:
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="❌ Для ControlNet нужно добавить хотя бы один референс.",
        )
        return
    if params.reference_images and not reference_images:
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="❌ Не удалось загрузить референс-картинки. Загрузите их заново в редакторе.",
        )
        return

    if params.controlnet_name and reference_images:
        reference_mode = "ipadapter" if deps.client.supports_ipadapter() else "none"
    else:
        reference_mode = deps.client.resolve_reference_mode(bool(reference_images))
    if reference_mode == "img2img" and reference_images:
        generation_params.denoise = min(
            generation_params.denoise,
            deps.denoise_from_reference_strength(generation_params.reference_strength),
        )

    ckpt_short = h(truncate(params.checkpoint, 30))
    start_lines = [f"⏳ <b>Генерация</b> | {ckpt_short} | {params.width}×{params.height}"]
    if reference_images:
        ref_info = f"🖼 <b>Ref:</b> {len(reference_images)}"
        if failed_refs:
            ref_info += f" (❌ {failed_refs})"
        if reference_mode == "ipadapter":
            ref_info += f" | IP-Adapter, str={generation_params.reference_strength}"
        elif reference_mode == "none":
            ref_info += " | референс загружен"
        else:
            ref_info += (
                f" | img2img, str={generation_params.reference_strength}"
                f", dn={generation_params.denoise}"
            )
        start_lines.append(ref_info)
    if generation_params.vae_name:
        start_lines.append(f"🧬 <b>VAE:</b> <code>{h(generation_params.vae_name)}</code>")
    if generation_params.controlnet_name:
        start_lines.append(
            "🧷 <b>ControlNet:</b> "
            f"<code>{h(generation_params.controlnet_name)}</code> "
            f"(str={generation_params.controlnet_strength})"
        )
    if generation_params.embedding_name:
        start_lines.append(
            f"🔤 <b>Embedding:</b> <code>{h(generation_params.embedding_name)}</code>"
        )
    if lora_warning:
        start_lines.append(lora_warning)
    status_intro = "\n".join(start_lines)

    return_to_editor_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Вернуться в редактор", callback_data="pe:gen:back")]
        ]
    )

    generation_id = uuid.uuid4().hex
    cancel_kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ Отменить", callback_data=f"pe:gen:cancel:{generation_id}"
                )
            ]
        ]
    )

    status_msg = await deps.show_prompt_panel(
        message,
        req,
        status_intro,
        cancel_kb,
        prefer_edit=True,
    )
    deps.remember_prompt_panel(req, status_msg)

    async def _generation_task() -> None:
        last_edit_time = 0.0
        last_text = status_intro
        edit_interval = 0.8
        expected_previews = max(1, generation_params.batch_size)
        ready_previews = 0
        sent_previews = 0
        preview_send_failures = 0

        async def _progress(current: int, total: int, text: str) -> None:
            nonlocal last_edit_time, last_text
            try:
                now = time.monotonic()
                is_final = total > 0 and current >= total

                if total > 0:
                    pct = min(100, max(0, (current * 100) // total))
                    first_line = f"🔄 {h(text)} | {pct}%"
                else:
                    first_line = f"⏳ {h(text)}"

                lines = [first_line]
                if total > 0:
                    lines.append(progress_bar(current, total))
                lines.append(f"🖼 Получено: <code>{sent_previews}/{expected_previews}</code>")
                lines.extend(["", status_intro])

                next_text = "\n".join(lines)
                if next_text == last_text:
                    return
                if not is_final and now - last_edit_time < edit_interval:
                    return

                last_text = next_text
                last_edit_time = now

                kb = None if is_final else cancel_kb
                await status_msg.edit_text(next_text, reply_markup=kb)
            except TelegramBadRequest:
                deps.logger.warning("Progress update failed", exc_info=True)

        async def _prompt_id_cb(prompt_id: str) -> None:
            active = deps.runtime.active_generations.get(generation_id)
            if active is not None:
                active.prompt_id = prompt_id
                deps.runtime.persist()

        async def _deliver_preview_image(image_bytes: bytes) -> None:
            nonlocal ready_previews, sent_previews, preview_send_failures
            ready_previews += 1
            artifact_id = uuid.uuid4().hex
            artifact_params = GenerationParams(**asdict(params))
            try:
                width, height = image_dimensions(image_bytes)
                artifact_params.width = width
                artifact_params.height = height
            except (OSError, ValueError):
                pass

            artifact = PreviewArtifact(
                artifact_id=artifact_id,
                owner_uid=uid,
                image_bytes=image_bytes,
                params=artifact_params,
                used_seed=int(used_seed),
            )
            deps.runtime.register_preview_artifact(artifact)
            deps.prune_preview_artifacts(uid)

            try:
                sent_preview_messages = await deps.deliver_generated_images(
                    status_msg,
                    [image_bytes],
                    used_seed=used_seed,
                    mode="photo",
                    preview_keyboards=[deps.preview_image_keyboard(artifact_id, None)],
                    index_offset=ready_previews - 1,
                    total_count=expected_previews,
                )
                if sent_preview_messages:
                    artifact.preview_chat_id = sent_preview_messages[0].chat.id
                    artifact.preview_message_id = sent_preview_messages[0].message_id
                    sent_previews += 1
                else:
                    preview_send_failures += 1
            except (TelegramBadRequest, RuntimeError):
                preview_send_failures += 1
                deps.logger.exception("Failed to deliver generated preview")

        try:
            images = await deps.client.generate(
                generation_params,
                reference_images=reference_images,
                progress_cb=_progress,
                prompt_id_cb=_prompt_id_cb,
                image_cb=_deliver_preview_image,
            )
            if images:
                for image_bytes in images:
                    await _deliver_preview_image(image_bytes)
        except asyncio.CancelledError:
            try:
                await status_msg.edit_text(f"{status_intro}\n\n❌ Генерация отменена.")
            except TelegramBadRequest:
                pass
            raise
        except TimeoutError:
            await status_msg.edit_text(
                f"{status_intro}\n\n⏰ Время ожидания истекло.",
                reply_markup=return_to_editor_kb,
            )
            return
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            RuntimeError,
            ValueError,
            OSError,
        ) as exc:
            await status_msg.edit_text(
                f"{status_intro}\n\n❌ Ошибка: <code>{h(exc)}</code>",
                reply_markup=return_to_editor_kb,
            )
            return
        finally:
            deps.runtime.active_generations.pop(generation_id, None)
            deps.runtime.persist()

        if ready_previews <= 0:
            await status_msg.edit_text(
                f"{status_intro}\n\n❌ Изображения не найдены.",
                reply_markup=return_to_editor_kb,
            )
            return

        deps.runtime.last_params[uid] = GenerationParams(**asdict(params))
        deps.runtime.last_seeds[uid] = used_seed
        deps.runtime.persist()

        preview_notice = "🖼 Превью отправлялись по мере готовности."
        if preview_send_failures > 0:
            preview_notice = (
                "⚠️ Часть превью не удалось отправить как фото. " "Можно скачать PNG без сжатия."
            )

        done_text = (
            f"✅ <b>Готово!</b> {sent_previews} изобр. | Seed: <code>{used_seed}</code>\n"
            f"\n{preview_notice}\n"
            "Для каждой превью доступны: отправка PNG и меню улучшений."
        )
        try:
            await deps.move_prompt_panel_to_bottom(
                status_msg,
                req,
                done_text,
                deps.generation_result_keyboard(),
            )
        except TelegramBadRequest:
            fallback = await status_msg.answer(
                done_text,
                reply_markup=deps.generation_result_keyboard(),
            )
            deps.remember_prompt_panel(req, fallback)

    task = asyncio.create_task(_generation_task())
    deps.runtime.active_generations[generation_id] = ActiveGeneration(
        owner_uid=uid,
        generation_id=generation_id,
        task=task,
        kind="generate",
        title="Генерация",
        status_msg=status_msg,
        status_chat_id=status_msg.chat.id,
        status_message_id=status_msg.message_id,
    )
    deps.runtime.persist()


class _ClientLike(Protocol):
    def supports_ipadapter(self) -> bool: ...

    def resolve_reference_mode(self, has_reference_images: bool) -> str: ...

    async def generate(
        self,
        params: GenerationParams,
        *,
        reference_images: list[bytes] | None = None,
        progress_cb: Callable[[int, int, str], Awaitable[None]] | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> list[bytes]: ...
