from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from application.lora_catalog_service import LoraCatalogService
from core.html_utils import h
from core.models import GenerationParams
from core.runtime import PromptRequest, RuntimeStore
from core.states import PromptEditorStates
from core.ui import loras_text
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService
from domain.loras import EditorLoraSelection
from infrastructure.comfyui_client import ComfyUIClient

from .prompt_editor_session import show_prompt_panel


def checkpoint_base_model(checkpoint_name: str, catalog: LoraCatalogService) -> str:
    return catalog.checkpoint_base_model(checkpoint_name)


def lora_base_model(lora_name: str, catalog: LoraCatalogService) -> str:
    return catalog.lora_base_model(lora_name)


def lora_trained_words(lora_name: str, catalog: LoraCatalogService) -> list[str]:
    return catalog.lora_trigger_words(lora_name, limit=12)


def lora_compatibility(
    checkpoint_name: str,
    lora_name: str,
    catalog: LoraCatalogService,
) -> tuple[str, str, str]:
    return catalog.lora_compatibility(checkpoint_name, lora_name)


def incompatible_loras(
    params: GenerationParams,
    catalog: LoraCatalogService,
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for selection in params.lora_selections():
        status, ckpt_base, lora_base = lora_compatibility(
            params.checkpoint,
            selection.name,
            catalog,
        )
        if status == "incompatible":
            result.append((selection.name, ckpt_base, lora_base))
    return result


def lora_picker_items(
    checkpoint_name: str,
    client: ComfyUIClient,
    catalog: LoraCatalogService,
) -> tuple[list[str], list[str]]:
    entries: list[tuple[int, str, str, str]] = []
    for name in client.info.loras:
        status, _, lora_base = lora_compatibility(checkpoint_name, name, catalog)
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


def editor_lora_chain(req: PromptRequest) -> list[EditorLoraSelection]:
    return list(req.editor_loras)


def lora_chain_pairs(req: PromptRequest) -> list[tuple[str, float]]:
    return [selection.to_legacy_pair() for selection in editor_lora_chain(req)]


def add_editor_lora(
    req: PromptRequest,
    lora_name: str,
    strength: float,
    catalog: LoraCatalogService,
) -> EditorLoraSelection:
    selection = catalog.editor_selection(lora_name, strength=strength)
    req.add_editor_lora(selection)
    return selection


def remove_last_editor_lora(req: PromptRequest) -> bool:
    return req.remove_last_editor_lora()


def clear_editor_loras(req: PromptRequest) -> int:
    return req.clear_editor_loras()


async def show_lora_menu(
    message: Message,
    state: FSMContext,
    uid: int,
    *,
    runtime: RuntimeStore,
    client: ComfyUIClient,
    catalog: LoraCatalogService,
    edit: bool = True,
    notice: str = "",
    localization: LocalizationService | None = None,
    resolve_user_locale: Callable[..., str] | None = None,
    telegram_locale: str | None = None,
) -> None:
    def _t(
        key: str,
        default: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> str:
        if localization is None or resolve_user_locale is None:
            if params:
                try:
                    return default.format(**params)
                except (KeyError, ValueError):
                    return default
            return default
        prefs = runtime.user_preferences.get(uid, {})
        selected_locale = read_user_locale(
            prefs,
            default_locale=localization.default_locale(),
        )
        locale = resolve_user_locale(
            user_locale=selected_locale,
            telegram_locale=telegram_locale,
        )
        return localization.t(key, locale=locale, params=params, default=default)

    req = runtime.active_prompt_requests.get(uid)
    if not req:
        await message.answer(
            _t(
                "prompt_editor.lora.error.active_request_not_found",
                "Active request not found. Use /generate.",
            )
        )
        return

    lines: list[str] = []
    if notice:
        lines.append(f"💬 <i>{h(notice)}</i>")
        lines.append("")

    lines.append(_t("prompt_editor.lora.menu.title", "🧲 <b>LoRA chain</b>"))
    lines.append(
        _t(
            "prompt_editor.lora.menu.current_chain",
            "Current chain: {chain}",
            params={"chain": loras_text(lora_chain_pairs(req))},
        )
    )

    ckpt_base = checkpoint_base_model(req.params.checkpoint, catalog)
    if ckpt_base:
        lines.append(
            _t(
                "prompt_editor.lora.menu.checkpoint_base",
                "Checkpoint base: <code>{base}</code>",
                params={"base": h(ckpt_base)},
            )
        )

    bad = incompatible_loras(req.params, catalog)
    if bad:
        bad_names = ", ".join(name for name, _, _ in bad[:3])
        suffix = (
            ""
            if len(bad) <= 3
            else _t(
                "prompt_editor.lora.menu.more_suffix",
                " and {count} more",
                params={"count": len(bad) - 3},
            )
        )
        lines.append(
            _t(
                "prompt_editor.lora.menu.incompatibility",
                "⚠️ Potential LoRA incompatibility: {names}{suffix}",
                params={"names": h(bad_names), "suffix": h(suffix)},
            )
        )

    text = "\n".join(lines)
    kb = build_keyboard(
        [
            [button(_t("prompt_editor.lora.button.add", "➕ Add"), "pe:lora:add")],
            [
                button(
                    _t("prompt_editor.lora.button.remove_last", "➖ Remove last"),
                    "pe:lora:remove_last",
                )
            ],
            [button(_t("prompt_editor.lora.button.clear", "🗑 Clear all"), "pe:lora:clear")],
            [back_button("pe:back", text=_t("common.action.back", "⬅️ Back"))],
        ]
    )
    await show_prompt_panel(runtime, message, req, text, kb, prefer_edit=edit)
    await state.set_state(PromptEditorStates.editing)
