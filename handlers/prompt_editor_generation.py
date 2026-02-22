from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import aiohttp
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message

from application.prompt_generation_use_case import PromptGenerationUseCase
from core.html_utils import h, progress_bar, truncate
from core.image_utils import image_dimensions
from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, PromptRequest, RuntimeStore


@dataclass
class PromptEditorGenerationDeps:
    runtime: RuntimeStore
    logger: logging.Logger
    client: _ClientLike
    generation_use_case: PromptGenerationUseCase
    collect_reference_images: Callable[
        [Message, list[dict[str, str]]], Awaitable[tuple[list[bytes], int]]
    ]
    show_prompt_editor: Callable[..., Awaitable[None]]
    show_prompt_panel: Callable[..., Awaitable[Message]]
    move_prompt_panel_to_bottom: Callable[..., Awaitable[Message]]
    remember_prompt_panel: Callable[[PromptRequest, Message], None]
    generation_result_keyboard: Callable[[], InlineKeyboardMarkup]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    prune_preview_artifacts: Callable[[int], None]


def _return_to_editor_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Вернуться в редактор", callback_data="pe:gen:back")]
        ]
    )


def _cancel_generation_keyboard(generation_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="❌ Отменить", callback_data=f"pe:gen:cancel:{generation_id}"
                )
            ]
        ]
    )


def _build_generation_status_intro(
    *,
    params: GenerationParams,
    generation_params: GenerationParams,
    reference_images: list[bytes],
    failed_refs: int,
    reference_mode: str,
    lora_warning: str,
) -> str:
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
    return "\n".join(start_lines)


def _preview_notice_text(*, sent_previews: int, ready_previews: int, failures: int) -> str:
    if failures <= 0:
        return "🖼 Превью отправлялись по мере готовности."
    return (
        "⚠️ Часть превью не удалось отправить как фото "
        f"({sent_previews}/{ready_previews}). Можно скачать PNG без сжатия."
    )


def _generation_done_text(*, ready_previews: int, used_seed: int, preview_notice: str) -> str:
    return (
        f"✅ <b>Готово!</b> {ready_previews} изобр. | Seed: <code>{used_seed}</code>\n"
        f"\n{preview_notice}\n"
        "Для каждой превью доступны: отправка PNG и меню улучшений."
    )


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

    prepared = deps.generation_use_case.prepare(req.params)
    params = prepared.params
    if not params.checkpoint:
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice="❌ Выберите checkpoint в редакторе перед генерацией.",
        )
        return

    bad_loras = prepared.incompatible_loras
    lora_warning = ""
    warning_payload = deps.generation_use_case.lora_warning_payload(bad_loras)
    if warning_payload is not None:
        listed, suffix = warning_payload
        lora_warning = (
            "⚠️ <b>Внимание:</b> потенциально несовместимые LoRA: "
            f"<code>{h(listed)}</code>{h(suffix)}"
        )

    used_seed = prepared.used_seed
    generation_params = prepared.generation_params

    reference_images, failed_refs = await deps.collect_reference_images(
        message,
        params.reference_images,
    )
    ref_notice = deps.generation_use_case.reference_validation_notice(
        params,
        reference_image_count=len(reference_images),
    )
    if ref_notice:
        await deps.show_prompt_editor(
            message,
            state,
            uid,
            edit=True,
            notice=ref_notice,
        )
        return

    reference_mode = deps.generation_use_case.resolve_mode(
        params,
        has_reference_images=bool(reference_images),
    )
    deps.generation_use_case.apply_reference_adjustments(
        generation_params,
        reference_mode=reference_mode,
        has_reference_images=bool(reference_images),
    )

    status_intro = _build_generation_status_intro(
        params=params,
        generation_params=generation_params,
        reference_images=reference_images,
        failed_refs=failed_refs,
        reference_mode=reference_mode,
        lora_warning=lora_warning,
    )

    return_to_editor_kb = _return_to_editor_keyboard()

    generation_id = uuid.uuid4().hex
    cancel_kb = _cancel_generation_keyboard(generation_id)

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
        base_seed = int(used_seed)
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
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
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
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                preview_send_failures += 1
                deps.logger.exception("Failed to deliver generated preview")

        async def _single_progress(
            batch_index: int,
            current: int,
            total: int,
            text: str,
        ) -> None:
            if total > 0:
                overall_current = batch_index * total + current
                overall_total = expected_previews * total
                await _progress(
                    overall_current,
                    overall_total,
                    f"Батч {batch_index + 1}/{expected_previews}: {text}",
                )
                return
            await _progress(
                0,
                0,
                f"Батч {batch_index + 1}/{expected_previews}: {text}",
            )

        def _progress_for_batch(
            batch_index: int,
        ) -> Callable[[int, int, str], Awaitable[None]]:
            async def _inner(current: int, total: int, text: str) -> None:
                await _single_progress(batch_index, current, total, text)

            return _inner

        try:
            for batch_index in range(expected_previews):
                single_params = GenerationParams(**asdict(generation_params))
                single_params.batch_size = 1
                single_params.seed = base_seed + batch_index

                images = await deps.client.generate(
                    single_params,
                    reference_images=reference_images,
                    progress_cb=_progress_for_batch(batch_index),
                    prompt_id_cb=_prompt_id_cb,
                )
                if images:
                    for image_bytes in images:
                        await _deliver_preview_image(image_bytes)
        except asyncio.CancelledError:
            try:
                await status_msg.edit_text(f"{status_intro}\n\n❌ Генерация отменена.")
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                pass
            raise
        except TimeoutError:
            try:
                await status_msg.edit_text(
                    f"{status_intro}\n\n⏰ Время ожидания истекло.",
                    reply_markup=return_to_editor_kb,
                )
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                deps.logger.warning("Failed to send timeout status", exc_info=True)
            return
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            RuntimeError,
            ValueError,
            OSError,
            TelegramNetworkError,
        ) as exc:
            try:
                await status_msg.edit_text(
                    f"{status_intro}\n\n❌ Ошибка: <code>{h(exc)}</code>",
                    reply_markup=return_to_editor_kb,
                )
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                deps.logger.warning("Failed to send error status", exc_info=True)
            return
        except Exception as exc:
            deps.logger.exception("Unexpected generation task failure", exc_info=exc)
            try:
                await status_msg.edit_text(
                    f"{status_intro}\n\n❌ Неожиданная ошибка: <code>{h(exc)}</code>",
                    reply_markup=return_to_editor_kb,
                )
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                deps.logger.warning("Failed to send unexpected error status", exc_info=True)
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

        preview_notice = _preview_notice_text(
            sent_previews=sent_previews,
            ready_previews=ready_previews,
            failures=preview_send_failures,
        )
        done_text = _generation_done_text(
            ready_previews=ready_previews,
            used_seed=used_seed,
            preview_notice=preview_notice,
        )
        try:
            await deps.move_prompt_panel_to_bottom(
                status_msg,
                req,
                done_text,
                deps.generation_result_keyboard(),
            )
        except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
            try:
                fallback = await status_msg.answer(
                    done_text,
                    reply_markup=deps.generation_result_keyboard(),
                )
                deps.remember_prompt_panel(req, fallback)
            except (TelegramBadRequest, TelegramNetworkError, RuntimeError):
                deps.logger.warning("Failed to post generation completion status", exc_info=True)

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
    async def generate(
        self,
        params: GenerationParams,
        *,
        reference_images: list[bytes] | None = None,
        progress_cb: Callable[[int, int, str], Awaitable[None]] | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
        image_cb: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> list[bytes]: ...
