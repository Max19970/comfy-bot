from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import aiohttp
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.types import InlineKeyboardMarkup, Message

from core.html_utils import h
from core.image_utils import image_dimensions, resize_image_by_percent, shrink_image_to_box
from core.models import GenerationParams
from core.runtime import ActiveGeneration, PreviewArtifact, RuntimeStore
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService


class _ClientLike(Protocol):
    async def generate_from_image(
        self,
        params: GenerationParams,
        *,
        image_bytes: bytes,
        progress_cb: Callable[[int, int, str], Awaitable[None]] | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[bytes]: ...

    async def upscale_image_only(
        self,
        *,
        image_bytes: bytes,
        upscale_model: str,
        progress_cb: Callable[[int, int, str], Awaitable[None]] | None = None,
        prompt_id_cb: Callable[[str], Awaitable[None]] | None = None,
    ) -> list[bytes]: ...


@dataclass
class PromptEditorEnhancementDeps:
    logger: logging.Logger
    runtime: RuntimeStore
    client: _ClientLike
    deliver_generated_images: Callable[..., Awaitable[list[Message]]]
    preview_image_keyboard: Callable[[str, str | None], InlineKeyboardMarkup]
    move_main_panel_to_bottom: Callable[[int, Message, str], Awaitable[None]]
    localization: LocalizationService | None = None
    resolve_user_locale: Callable[..., str] | None = None


def _resolved_locale(
    deps: PromptEditorEnhancementDeps,
    uid: int,
    *,
    telegram_locale: str | None,
) -> str | None:
    if deps.localization is None:
        return None
    default_locale = deps.localization.default_locale()
    prefs = deps.runtime.user_preferences.get(uid, {})
    selected_locale = read_user_locale(prefs, default_locale=default_locale)
    if deps.resolve_user_locale is None:
        return selected_locale
    return deps.resolve_user_locale(
        user_locale=selected_locale,
        telegram_locale=telegram_locale,
    )


def _t(
    deps: PromptEditorEnhancementDeps,
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
        locale = _resolved_locale(deps, uid, telegram_locale=telegram_locale)
        text = deps.localization.t(key, locale=locale, params=params, default=default)
    if params:
        try:
            return text.format(**params)
        except Exception:
            return text
    return text


def _requires_image_generation(
    artifact: PreviewArtifact,
    params: GenerationParams,
) -> bool:
    return (
        artifact.enable_sampler_pass
        or params.enable_hires_fix
        or params.enable_freeu
        or params.enable_pag
    )


async def start_image_enhancement(
    message: Message,
    *,
    uid: int,
    artifact: PreviewArtifact,
    deps: PromptEditorEnhancementDeps,
) -> None:
    telegram_locale = message.from_user.language_code if message.from_user else None
    status_msg = await message.answer(
        _t(
            deps,
            uid,
            "prompt_editor.enhancement.status.starting",
            "⏳ Запускаю улучшение...",
            telegram_locale=telegram_locale,
        )
    )
    generation_id = f"enh_{uuid.uuid4().hex}"
    temp_job_id = f"job_{time.time_ns()}"

    try:
        await status_msg.edit_reply_markup(
            reply_markup=_enhancement_cancel_keyboard(
                generation_id,
                cancel_text=_t(
                    deps,
                    uid,
                    "prompt_editor.enhancement.button.cancel",
                    "❌ Отменить улучшение",
                    telegram_locale=telegram_locale,
                ),
            )
        )
    except (TelegramBadRequest, TelegramNetworkError):
        pass

    async def _progress(current: int, total: int, text: str) -> None:
        line = f"⏳ {h(text)}"
        if total > 0:
            line = f"⏳ {h(text)} ({current}/{total})"
        try:
            await status_msg.edit_text(
                line,
                reply_markup=_enhancement_cancel_keyboard(
                    generation_id,
                    cancel_text=_t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.button.cancel",
                        "❌ Отменить улучшение",
                        telegram_locale=telegram_locale,
                    ),
                ),
            )
        except (TelegramBadRequest, TelegramNetworkError):
            deps.logger.debug("Image enhancement progress update failed", exc_info=True)

    async def _safe_edit_status(text: str) -> None:
        try:
            await status_msg.edit_text(text, reply_markup=None)
        except (TelegramBadRequest, TelegramNetworkError):
            deps.logger.debug("Image enhancement status update failed", exc_info=True)

    async def _run() -> None:
        try:
            source_bytes = deps.runtime.artifact_bytes(artifact)
            if not source_bytes:
                await _safe_edit_status(
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.error.source_missing",
                        "❌ Не найдены данные исходной картинки.",
                        telegram_locale=telegram_locale,
                    )
                )
                return

            run_params = GenerationParams(**asdict(artifact.params))
            run_params.batch_size = 1
            run_params.reference_images = []
            run_params.reference_strength = 0.8
            run_params.enable_tiled_diffusion = False
            requires_image_generation = _requires_image_generation(artifact, run_params)
            if requires_image_generation and run_params.seed < 0:
                run_params.seed = random.randint(0, 2**63 - 1)
            result_seed = int(run_params.seed) if run_params.seed >= 0 else artifact.used_seed

            async def _prompt_id_cb(prompt_id: str) -> None:
                active = deps.runtime.active_generations.get(generation_id)
                if active is not None:
                    active.prompt_id = prompt_id
                    deps.runtime.persist()

            if requires_image_generation:
                skip_base_sampler_pass = (
                    run_params.enable_hires_fix and not artifact.enable_sampler_pass
                )
                if skip_base_sampler_pass:
                    setattr(run_params, "_skip_base_sampler_pass", True)
                images = await deps.client.generate_from_image(
                    run_params,
                    image_bytes=source_bytes,
                    progress_cb=_progress,
                    prompt_id_cb=_prompt_id_cb,
                )
                result_seed = int(run_params.seed)
            elif run_params.upscale_model:
                images = await deps.client.upscale_image_only(
                    image_bytes=source_bytes,
                    upscale_model=run_params.upscale_model,
                    progress_cb=_progress,
                    prompt_id_cb=_prompt_id_cb,
                )
            else:
                await _progress(
                    0,
                    0,
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.progress.comfy_disabled",
                        "Без ComfyUI: только сжатие",
                        telegram_locale=telegram_locale,
                    ),
                )
                images = [source_bytes]

            if not images:
                await _safe_edit_status(
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.error.empty_result",
                        "❌ ComfyUI не вернул изображение.",
                        telegram_locale=telegram_locale,
                    )
                )
                return

            result_image = images[0]
            if artifact.compression_percent < 100:
                await _progress(
                    0,
                    0,
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.progress.compress",
                        "Сжимаю до {percent}%",
                        telegram_locale=telegram_locale,
                        params={"percent": artifact.compression_percent},
                    ),
                )
                result_image = resize_image_by_percent(result_image, artifact.compression_percent)
            if artifact.shrink_width and artifact.shrink_height:
                await _progress(
                    0,
                    0,
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.progress.shrink",
                        "Shrink до {width}x{height}",
                        telegram_locale=telegram_locale,
                        params={
                            "width": artifact.shrink_width,
                            "height": artifact.shrink_height,
                        },
                    ),
                )
                result_image = shrink_image_to_box(
                    result_image,
                    artifact.shrink_width,
                    artifact.shrink_height,
                )

            next_params = GenerationParams(**asdict(run_params))
            try:
                next_w, next_h = image_dimensions(result_image)
                next_params.width = next_w
                next_params.height = next_h
            except (OSError, ValueError):
                pass

            next_artifact_id = uuid.uuid4().hex
            next_artifact = PreviewArtifact(
                artifact_id=next_artifact_id,
                owner_uid=artifact.owner_uid,
                image_bytes=result_image,
                params=next_params,
                used_seed=result_seed,
                parent_artifact_id=artifact.artifact_id,
                generation_step=artifact.generation_step + 1,
                enable_sampler_pass=artifact.enable_sampler_pass,
                compression_percent=artifact.compression_percent,
                shrink_width=artifact.shrink_width,
                shrink_height=artifact.shrink_height,
            )
            deps.runtime.register_preview_artifact(next_artifact)
            deps.runtime.prune_preview_artifacts(artifact.owner_uid)

            deps.runtime.last_params[artifact.owner_uid] = GenerationParams(**asdict(next_params))
            deps.runtime.last_seeds[artifact.owner_uid] = result_seed
            deps.runtime.persist()

            sent_previews = await deps.deliver_generated_images(
                status_msg,
                [result_image],
                used_seed=result_seed,
                mode="photo",
                preview_keyboards=[
                    deps.preview_image_keyboard(next_artifact_id, artifact.artifact_id)
                ],
            )
            if sent_previews:
                next_artifact.preview_chat_id = sent_previews[0].chat.id
                next_artifact.preview_message_id = sent_previews[0].message_id

            extra_lines: list[str] = []
            if artifact.compression_percent < 100:
                extra_lines.append(
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.notice.compression_applied",
                        "🗜 Сжатие применено: {percent}% от результата.",
                        telegram_locale=telegram_locale,
                        params={"percent": artifact.compression_percent},
                    )
                )
            if artifact.shrink_width and artifact.shrink_height:
                extra_lines.append(
                    _t(
                        deps,
                        uid,
                        "prompt_editor.enhancement.notice.shrink_applied",
                        "📦 Shrink применен: {width}x{height}.",
                        telegram_locale=telegram_locale,
                        params={
                            "width": artifact.shrink_width,
                            "height": artifact.shrink_height,
                        },
                    )
                )
            detail_block = "\n".join(extra_lines)
            if detail_block:
                detail_block += "\n"
            await deps.move_main_panel_to_bottom(
                artifact.owner_uid,
                status_msg,
                _t(
                    deps,
                    uid,
                    "prompt_editor.enhancement.notice.completed",
                    "✅ Улучшение завершено. Отправил новую превью.\n"
                    "{details}"
                    "Для каждой превью доступны: отправка PNG и меню улучшений.",
                    telegram_locale=telegram_locale,
                    params={"details": detail_block},
                ),
            )
        except asyncio.CancelledError:
            await _safe_edit_status(
                _t(
                    deps,
                    uid,
                    "prompt_editor.enhancement.error.cancelled",
                    "❌ Улучшение отменено.",
                    telegram_locale=telegram_locale,
                )
            )
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            TelegramNetworkError,
        ) as exc:
            deps.logger.exception("Image enhancement failed")
            await _safe_edit_status(
                _t(
                    deps,
                    uid,
                    "prompt_editor.enhancement.error.failed",
                    "❌ Ошибка улучшения: <code>{error}</code>",
                    telegram_locale=telegram_locale,
                    params={"error": h(exc)},
                )
            )
        finally:
            deps.runtime.active_image_jobs.pop(temp_job_id, None)
            deps.runtime.active_generations.pop(generation_id, None)
            deps.runtime.persist()

    task = asyncio.create_task(_run())
    deps.runtime.active_image_jobs[temp_job_id] = task
    deps.runtime.active_generations[generation_id] = ActiveGeneration(
        owner_uid=uid,
        generation_id=generation_id,
        task=task,
        kind="enhancement",
        title=_t(
            deps,
            uid,
            "prompt_editor.enhancement.title",
            "Улучшение",
            telegram_locale=telegram_locale,
        ),
        status_msg=status_msg,
        status_chat_id=status_msg.chat.id,
        status_message_id=status_msg.message_id,
    )
    deps.runtime.persist()


def _enhancement_cancel_keyboard(generation_id: str, *, cancel_text: str) -> InlineKeyboardMarkup:
    from core.ui_kit import build_keyboard
    from core.ui_kit.buttons import button

    return build_keyboard([[button(cancel_text, f"pe:gen:cancel:{generation_id}")]])
