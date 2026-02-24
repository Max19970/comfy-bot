from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from aiogram import Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from application.model_downloader import (
    ModelDownloader,
    SearchResult,
    SearchVersionOption,
    apply_version_option,
)
from application.user_locale_resolver import DefaultUserLocaleResolver
from core.download_filters import (
    DOWNLOAD_BASE_CODE_TO_API,
    DOWNLOAD_FILTER_PROFILES,
    DOWNLOAD_PERIOD_CODE_TO_API,
    DOWNLOAD_SORT_CODE_TO_API,
    download_base_label,
    download_period_label,
    download_profile_label,
    download_sort_label,
    download_source_label,
)
from core.formatting import human_size, short_number
from core.html_utils import h
from core.panels import render_user_panel
from core.runtime import RuntimeStore
from core.states import DownloadStates
from core.telegram import callback_user_id, message_user_id
from core.ui_kit import back_button, build_keyboard
from core.ui_kit.buttons import button, cancel_button, noop_button
from core.user_preferences import read_user_locale
from domain.localization import LocalizationService
from infrastructure.comfyui_client import ComfyUIClient

from .download_flow_handlers import DownloadFlowDeps, register_download_flow_handlers

MODEL_TYPE_LABELS = {
    "checkpoint": "Checkpoint",
    "lora": "LoRA",
    "upscaler": "Upscaler",
    "embedding": "Embedding",
    "controlnet": "ControlNet",
    "vae": "VAE",
}
MODEL_TYPE_I18N_KEYS = {
    "checkpoint": "download.type.checkpoint",
    "lora": "download.type.lora",
    "upscaler": "download.type.upscaler",
    "embedding": "download.type.embedding",
    "controlnet": "download.type.controlnet",
    "vae": "download.type.vae",
}

TranslateText = Callable[[str, str | None, str, Mapping[str, object] | None], str]


def _tx(
    translate: TranslateText | None,
    key: str,
    locale: str | None,
    default: str,
    *,
    params: Mapping[str, object] | None = None,
) -> str:
    if translate is None:
        if not params:
            return default
        try:
            return default.format_map({str(k): v for k, v in params.items()})
        except (KeyError, ValueError, TypeError):
            return default
    return translate(key, locale, default, params)


def _label_translate_adapter(
    translate: TranslateText | None,
) -> Callable[[str, str | None, str], str] | None:
    if translate is None:
        return None

    def _inner(key: str, locale: str | None, default: str) -> str:
        return _tx(translate, key, locale, default)

    return _inner


def _mark(active: bool, label: str) -> str:
    return f"✅ {label}" if active else label


def _type_title(
    model_type: str,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    default = MODEL_TYPE_LABELS.get(model_type, model_type)
    key = MODEL_TYPE_I18N_KEYS.get(model_type, "download.type.unknown")
    return _tx(translate, key, locale, default)


def _source_title(
    source: str,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    return download_source_label(
        source,
        translate=_label_translate_adapter(translate),
        locale=locale,
    )


def _hydrate_result(data: dict[str, Any]) -> SearchResult:
    result = SearchResult(**data)
    hydrated: list[SearchVersionOption] = []
    for item in result.available_versions:
        if isinstance(item, SearchVersionOption):
            hydrated.append(item)
        elif isinstance(item, dict):
            hydrated.append(SearchVersionOption(**item))
    result.available_versions = hydrated
    return result


def _search_filters_summary(
    *,
    model_type: str,
    sort_code: str,
    period_code: str,
    base_code: str,
    include_nsfw: bool,
    source: str,
    author_nick: str,
    page_size: int,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    label_translate = _label_translate_adapter(translate)
    sort_label = download_sort_label(sort_code, translate=label_translate, locale=locale)
    parts = [
        _tx(
            translate,
            "download.filter.summary.sort",
            locale,
            "Сортировка: {value}",
            params={"value": sort_label},
        )
    ]
    if _supports_period_filter(source):
        period_label = download_period_label(period_code, translate=label_translate, locale=locale)
        parts.append(
            _tx(
                translate,
                "download.filter.summary.period",
                locale,
                "Период: {value}",
                params={"value": period_label},
            )
        )
    if _supports_base_filter(model_type=model_type, source=source):
        base_label = download_base_label(base_code, translate=label_translate, locale=locale)
        parts.append(
            _tx(
                translate,
                "download.filter.summary.base",
                locale,
                "Base: {value}",
                params={"value": base_label},
            )
        )
    if _supports_nsfw_filter(source):
        nsfw_state = _tx(
            translate,
            "download.filter.value.on" if include_nsfw else "download.filter.value.off",
            locale,
            "вкл" if include_nsfw else "выкл",
        )
        parts.append(
            _tx(
                translate,
                "download.filter.summary.nsfw",
                locale,
                "NSFW: {value}",
                params={"value": nsfw_state},
            )
        )
    if source in {"civitai", "all"} and author_nick.strip():
        parts.append(
            _tx(
                translate,
                "download.filter.summary.authors",
                locale,
                "Автор(ы): {authors}",
                params={"authors": author_nick.strip().lstrip("@")},
            )
        )
    parts.append(
        _tx(
            translate,
            "download.filter.summary.page_size",
            locale,
            "На страницу: {page_size}",
            params={"page_size": max(1, min(page_size, 10))},
        )
    )
    return " | ".join(parts)


def _supports_period_filter(source: str) -> bool:
    return source in {"civitai", "all"}


def _supports_nsfw_filter(source: str) -> bool:
    return source in {"civitai", "all"}


def _supports_base_filter(*, model_type: str, source: str) -> bool:
    if source not in {"civitai", "all"}:
        return False
    return model_type in {"checkpoint", "lora", "controlnet"}


def _result_meta_line(
    result: SearchResult,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    parts: list[str] = []
    if result.creator:
        parts.append(
            _tx(
                translate,
                "download.results.meta.creator",
                locale,
                "by {creator}",
                params={"creator": result.creator},
            )
        )
    if result.base_model:
        parts.append(result.base_model)
    if result.version_name:
        parts.append(result.version_name)
    if result.size_bytes > 0:
        parts.append(human_size(result.size_bytes))
    if result.rating > 0:
        parts.append(f"⭐ {result.rating:.2f}")
    if result.download_count > 0:
        parts.append(f"📥 {short_number(result.download_count)}")
    if not parts and result.description:
        return result.description
    return " | ".join(parts)


def _result_button_label(result: SearchResult) -> str:
    src_tag = "C" if result.source == "civitai" else "HF"
    label = result.name if len(result.name) <= 28 else result.name[:25] + "..."
    return f"[{src_tag}] {label}"


def _build_type_keyboard(
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    rows = [
        [
            button(
                _tx(translate, "download.type.checkpoint", locale, "Checkpoint"),
                "dltype:checkpoint",
            )
        ],
        [button(_tx(translate, "download.type.lora", locale, "LoRA"), "dltype:lora")],
        [
            button(
                _tx(translate, "download.type.upscaler", locale, "Upscaler"),
                "dltype:upscaler",
            )
        ],
        [
            button(
                _tx(translate, "download.type.embedding", locale, "Embedding"),
                "dltype:embedding",
            )
        ],
        [
            button(
                _tx(translate, "download.type.controlnet", locale, "ControlNet"),
                "dltype:controlnet",
            )
        ],
        [button(_tx(translate, "download.type.vae", locale, "VAE"), "dltype:vae")],
        [
            cancel_button(
                "dltype:cancel",
                text=_tx(translate, "common.action.cancel", locale, "❌ Отмена"),
            )
        ],
    ]
    return build_keyboard(rows)


def _build_source_keyboard(
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    return build_keyboard(
        [
            [
                button(
                    download_source_label(
                        "civitai",
                        translate=_label_translate_adapter(translate),
                        locale=locale,
                    ),
                    "dlsrc:civitai",
                )
            ],
            [
                button(
                    download_source_label(
                        "huggingface",
                        translate=_label_translate_adapter(translate),
                        locale=locale,
                    ),
                    "dlsrc:huggingface",
                )
            ],
            [
                button(
                    download_source_label(
                        "all",
                        translate=_label_translate_adapter(translate),
                        locale=locale,
                    ),
                    "dlsrc:all",
                )
            ],
            [
                back_button(
                    "dlsrc:back",
                    text=_tx(
                        translate,
                        "download.source.back_to_type",
                        locale,
                        "⬅️ Назад к типу",
                    ),
                )
            ],
        ]
    )


def _build_filter_keyboard(
    *,
    model_type: str,
    source: str,
    sort_code: str,
    period_code: str,
    base_code: str,
    include_nsfw: bool,
    author_nick: str,
    page_size: int,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    label_translate = _label_translate_adapter(translate)
    rows: list[list[InlineKeyboardButton]] = [
        [
            button(
                _mark(
                    sort_code == "downloads",
                    _tx(
                        translate,
                        "download.filter.button.sort.downloads",
                        locale,
                        "📥 Скачивания",
                    ),
                ),
                "dlflt:sort:downloads",
            ),
            button(
                _mark(
                    sort_code == "rating",
                    _tx(
                        translate,
                        "download.filter.button.sort.rating",
                        locale,
                        "⭐ Рейтинг",
                    ),
                ),
                "dlflt:sort:rating",
            ),
            button(
                _mark(
                    sort_code == "newest",
                    _tx(
                        translate,
                        "download.filter.button.sort.newest",
                        locale,
                        "🆕 Новые",
                    ),
                ),
                "dlflt:sort:newest",
            ),
        ]
    ]

    if _supports_period_filter(source):
        rows.append(
            [
                button(
                    _mark(
                        period_code == "all",
                        _tx(
                            translate,
                            "download.filter.button.period.all",
                            locale,
                            "🕒 Всё время",
                        ),
                    ),
                    "dlflt:period:all",
                ),
                button(
                    _mark(
                        period_code == "month",
                        _tx(
                            translate,
                            "download.filter.button.period.month",
                            locale,
                            "📆 Месяц",
                        ),
                    ),
                    "dlflt:period:month",
                ),
                button(
                    _mark(
                        period_code == "week",
                        _tx(
                            translate,
                            "download.filter.button.period.week",
                            locale,
                            "🗓 Неделя",
                        ),
                    ),
                    "dlflt:period:week",
                ),
            ]
        )

    show_base_filters = _supports_base_filter(model_type=model_type, source=source)
    if show_base_filters:
        rows.append(
            [
                button(
                    _tx(
                        translate,
                        "download.filter.button.base_menu",
                        locale,
                        "🧬 Базовые модели: {base}",
                        params={
                            "base": download_base_label(
                                base_code,
                                translate=label_translate,
                                locale=locale,
                            )
                        },
                    ),
                    "dlflt:base_menu",
                )
            ]
        )

    if _supports_nsfw_filter(source):
        rows.append(
            [
                button(
                    _tx(
                        translate,
                        "download.filter.button.nsfw.on"
                        if include_nsfw
                        else "download.filter.button.nsfw.off",
                        locale,
                        "🔞 NSFW: вкл" if include_nsfw else "🛡 NSFW: выкл",
                    ),
                    "dlflt:nsfw:toggle",
                )
            ]
        )

    if source in {"civitai", "all"}:
        author_label = author_nick.strip().lstrip("@") or _tx(
            translate,
            "download.filter.value.any_author",
            locale,
            "любой",
        )
        rows.append(
            [
                button(
                    _tx(
                        translate,
                        "download.filter.button.author",
                        locale,
                        "👤 Автор: {author}",
                        params={"author": author_label},
                    ),
                    "dlflt:author",
                )
            ]
        )

    rows.append(
        [
            button(
                download_profile_label("popular", translate=label_translate, locale=locale),
                "dlflt:profile:popular",
            ),
            button(
                download_profile_label("fresh", translate=label_translate, locale=locale),
                "dlflt:profile:fresh",
            ),
            button(
                download_profile_label("quality", translate=label_translate, locale=locale),
                "dlflt:profile:quality",
            ),
        ]
    )
    rows.append(
        [
            button(
                download_profile_label("anime", translate=label_translate, locale=locale),
                "dlflt:profile:anime",
            )
        ]
    )
    rows.append(
        [
            button(
                _mark(
                    page_size == 5,
                    _tx(
                        translate,
                        "download.filter.button.page_size",
                        locale,
                        "{size}/стр",
                        params={"size": 5},
                    ),
                ),
                "dlflt:pagesize:5",
            ),
            button(
                _mark(
                    page_size == 8,
                    _tx(
                        translate,
                        "download.filter.button.page_size",
                        locale,
                        "{size}/стр",
                        params={"size": 8},
                    ),
                ),
                "dlflt:pagesize:8",
            ),
            button(
                _mark(
                    page_size == 10,
                    _tx(
                        translate,
                        "download.filter.button.page_size",
                        locale,
                        "{size}/стр",
                        params={"size": 10},
                    ),
                ),
                "dlflt:pagesize:10",
            ),
        ]
    )

    rows.extend(
        [
            [
                button(
                    _tx(
                        translate,
                        "download.filter.button.enter_query",
                        locale,
                        "➡️ Ввести запрос",
                    ),
                    "dlflt:go",
                )
            ],
            [
                back_button(
                    "dlflt:back",
                    text=_tx(
                        translate,
                        "download.filter.back_to_sources",
                        locale,
                        "⬅️ Источники",
                    ),
                )
            ],
        ]
    )
    return build_keyboard(rows)


def _build_base_filter_keyboard(
    *,
    base_code: str,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> InlineKeyboardMarkup:
    label_translate = _label_translate_adapter(translate)
    rows: list[list[InlineKeyboardButton]] = [
        [
            button(
                _mark(
                    base_code == "all",
                    download_base_label("all", translate=label_translate, locale=locale),
                ),
                "dlflt:base:all",
            )
        ],
        [
            button(
                _mark(
                    base_code == "sd15",
                    download_base_label("sd15", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sd15",
            ),
            button(
                _mark(
                    base_code == "sd2",
                    download_base_label("sd2", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sd2",
            ),
        ],
        [
            button(
                _mark(
                    base_code == "sdxl09",
                    download_base_label("sdxl09", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sdxl09",
            ),
            button(
                _mark(
                    base_code == "sdxl",
                    download_base_label("sdxl", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sdxl",
            ),
        ],
        [
            button(
                _mark(
                    base_code == "sd3",
                    download_base_label("sd3", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sd3",
            ),
            button(
                _mark(
                    base_code == "sd35",
                    download_base_label("sd35", translate=label_translate, locale=locale),
                ),
                "dlflt:base:sd35",
            ),
        ],
        [
            button(
                _mark(
                    base_code == "pony",
                    download_base_label("pony", translate=label_translate, locale=locale),
                ),
                "dlflt:base:pony",
            ),
            button(
                _mark(
                    base_code == "flux",
                    download_base_label("flux", translate=label_translate, locale=locale),
                ),
                "dlflt:base:flux",
            ),
        ],
        [
            button(
                _mark(
                    base_code == "illustrious",
                    download_base_label(
                        "illustrious",
                        translate=label_translate,
                        locale=locale,
                    ),
                ),
                "dlflt:base:illustrious",
            ),
            button(
                _mark(
                    base_code == "noobai",
                    download_base_label("noobai", translate=label_translate, locale=locale),
                ),
                "dlflt:base:noobai",
            ),
        ],
        [
            back_button(
                "dlbase:back",
                text=_tx(
                    translate,
                    "download.base.back_to_filters",
                    locale,
                    "⬅️ К фильтрам",
                ),
            )
        ],
    ]
    return build_keyboard(rows)


def _confirmation_text(
    result: SearchResult,
    *,
    translate: TranslateText | None = None,
    locale: str | None = None,
) -> str:
    size_info = human_size(result.size_bytes)
    extras = [item for item in (result.file_fp, result.file_format) if item]
    if extras:
        size_info = f"{size_info} ({', '.join(extras)})"

    lines = [
        _tx(translate, "download.confirm.title", locale, "📦 <b>Скачать модель?</b>"),
        "",
        _tx(
            translate,
            "download.confirm.field.name",
            locale,
            "<b>Название:</b> {value}",
            params={"value": h(result.name)},
        ),
    ]
    if result.version_name:
        lines.append(
            _tx(
                translate,
                "download.confirm.field.version",
                locale,
                "<b>Версия:</b> {value}",
                params={"value": h(result.version_name)},
            )
        )
    if result.creator:
        lines.append(
            _tx(
                translate,
                "download.confirm.field.author",
                locale,
                "<b>Автор:</b> {value}",
                params={"value": h(result.creator)},
            )
        )
    if result.base_model:
        lines.append(
            _tx(
                translate,
                "download.confirm.field.base_model",
                locale,
                "<b>Базовая модель:</b> {value}",
                params={"value": h(result.base_model)},
            )
        )

    if result.rating > 0 or result.download_count > 0:
        stat_parts = []
        if result.rating > 0:
            stat_parts.append(f"⭐ {result.rating:.2f}")
        if result.download_count > 0:
            stat_parts.append(f"📥 {short_number(result.download_count)}")
        lines.append(
            _tx(
                translate,
                "download.confirm.field.popularity",
                locale,
                "<b>Популярность:</b> {value}",
                params={"value": " | ".join(stat_parts)},
            )
        )

    lines.extend(
        [
            _tx(
                translate,
                "download.confirm.field.file",
                locale,
                "<b>Файл:</b> <code>{value}</code>",
                params={"value": h(result.filename)},
            ),
            _tx(
                translate,
                "download.confirm.field.size",
                locale,
                "<b>Размер:</b> {value}",
                params={"value": h(size_info)},
            ),
            _tx(
                translate,
                "download.confirm.field.source",
                locale,
                "<b>Источник:</b> {value}",
                params={
                    "value": h(
                        _source_title(
                            result.source,
                            translate=translate,
                            locale=locale,
                        )
                    )
                },
            ),
            _tx(
                translate,
                "download.confirm.field.type",
                locale,
                "<b>Тип:</b> {value}",
                params={
                    "value": h(
                        _type_title(
                            result.model_type,
                            translate=translate,
                            locale=locale,
                        )
                    )
                },
            ),
        ]
    )

    if result.trained_words:
        words = ", ".join(result.trained_words[:8])
        lines.append(
            _tx(
                translate,
                "download.confirm.field.trigger_words",
                locale,
                "<b>Триггер-слова:</b> <code>{value}</code>",
                params={"value": h(words)},
            )
        )

    if result.tags:
        lines.append(
            _tx(
                translate,
                "download.confirm.field.tags",
                locale,
                "<b>Теги:</b> {value}",
                params={"value": h(", ".join(result.tags[:8]))},
            )
        )

    return "\n".join(lines)


def register_download_handlers(
    router: Router,
    client: ComfyUIClient,
    downloader: ModelDownloader,
    runtime: RuntimeStore,
    localization: LocalizationService,
) -> None:
    locale_resolver = DefaultUserLocaleResolver(localization)

    def _resolve_locale(uid: int, *, telegram_locale: str | None) -> str:
        prefs = runtime.user_preferences.get(uid, {})
        user_locale = read_user_locale(
            prefs,
            default_locale=localization.default_locale(),
        )
        return locale_resolver.resolve(
            user_locale=user_locale,
            telegram_locale=telegram_locale,
        )

    def _translate(
        key: str,
        locale: str | None,
        default: str,
        params: Mapping[str, object] | None = None,
    ) -> str:
        return localization.t(key, locale=locale, params=params, default=default)

    def _t(
        uid: int,
        key: str,
        default: str,
        *,
        telegram_locale: str | None,
        params: Mapping[str, object] | None = None,
    ) -> str:
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        return _translate(key, locale, default, params)

    async def _state_context(message: Message, state: FSMContext) -> tuple[int, str]:
        data = await state.get_data()
        raw_uid = data.get("dl_uid")
        uid = raw_uid if isinstance(raw_uid, int) else message_user_id(message)
        telegram_locale = message.from_user.language_code if message.from_user else None
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        return uid, locale

    async def _render_download_panel(
        message: Message,
        state: FSMContext,
        text: str,
        reply_markup: InlineKeyboardMarkup | None,
        *,
        prefer_edit: bool,
    ) -> Message:
        data = await state.get_data()
        uid = data.get("dl_uid")
        if isinstance(uid, int):
            return await render_user_panel(
                message,
                runtime,
                uid,
                text,
                reply_markup=reply_markup,
                prefer_edit=prefer_edit,
            )

        if prefer_edit:
            try:
                edited = await message.edit_text(text, reply_markup=reply_markup)
                if isinstance(edited, Message):
                    return edited
            except TelegramBadRequest:
                pass

        return await message.answer(text, reply_markup=reply_markup)

    async def show_filter_menu(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        _, locale = await _state_context(message, state)
        data = await state.get_data()
        model_type = data.get("dl_type", "checkpoint")
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()
        page_size = int(data.get("dl_page_size", 8) or 8)

        text = (
            _tx(_translate, "download.filter.panel.title", locale, "🧩 <b>Фильтры поиска</b>")
            + "\n"
            + _tx(
                _translate,
                "download.filter.panel.type",
                locale,
                "<b>Тип:</b> {value}",
                params={
                    "value": h(
                        _type_title(
                            model_type,
                            translate=_translate,
                            locale=locale,
                        )
                    )
                },
            )
            + "\n"
            + _tx(
                _translate,
                "download.filter.panel.source",
                locale,
                "<b>Источник:</b> {value}",
                params={
                    "value": h(
                        _source_title(
                            source,
                            translate=_translate,
                            locale=locale,
                        )
                    )
                },
            )
            + "\n"
            + f"<i>{h(_search_filters_summary(model_type=model_type, sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source, author_nick=author_nick, page_size=page_size, translate=_translate, locale=locale))}</i>"
        )
        kb = _build_filter_keyboard(
            model_type=model_type,
            source=source,
            sort_code=sort_code,
            period_code=period_code,
            base_code=base_code,
            include_nsfw=include_nsfw,
            author_nick=author_nick,
            page_size=page_size,
            translate=_translate,
            locale=locale,
        )
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_query_prompt(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        _, locale = await _state_context(message, state)
        kb = build_keyboard(
            [
                [
                    back_button(
                        "dlqry:back",
                        text=_tx(
                            _translate,
                            "download.query.back_to_filters",
                            locale,
                            "⬅️ Назад к фильтрам",
                        ),
                    )
                ]
            ]
        )
        text = _tx(
            _translate,
            "download.query.prompt",
            locale,
            "✏️ Введите поисковый запрос или прямую ссылку на модель CivitAI/HuggingFace:",
        )
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_author_prompt(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        _, locale = await _state_context(message, state)
        data = await state.get_data()
        current = str(data.get("dl_author", "")).strip().lstrip("@")
        kb = build_keyboard(
            [
                [
                    back_button(
                        "dlauth:back",
                        text=_tx(
                            _translate,
                            "download.author.back_to_filters",
                            locale,
                            "⬅️ Назад к фильтрам",
                        ),
                    )
                ]
            ]
        )
        current_value = current or _tx(
            _translate,
            "download.filter.value.any_author",
            locale,
            "любой",
        )
        text = (
            _tx(
                _translate,
                "download.author.title",
                locale,
                "👤 <b>Фильтр по автору CivitAI</b>",
            )
            + "\n"
            + _tx(
                _translate,
                "download.author.instructions",
                locale,
                "Введите один или несколько ников через запятую (без @).",
            )
            + "\n"
            + _tx(
                _translate,
                "download.author.clear_hint",
                locale,
                "Отправьте <code>-</code>, чтобы убрать фильтр.",
            )
            + "\n"
            + _tx(
                _translate,
                "download.author.current",
                locale,
                "<b>Текущее значение:</b> <code>{value}</code>",
                params={"value": h(current_value)},
            )
        )
        await _render_download_panel(message, state, text, kb, prefer_edit=edit)

    async def show_base_filter_menu(
        message: Message,
        state: FSMContext,
        *,
        edit: bool,
    ) -> None:
        _, locale = await _state_context(message, state)
        data = await state.get_data()
        base_code = str(data.get("dl_base", "all"))
        kb = _build_base_filter_keyboard(base_code=base_code, translate=_translate, locale=locale)
        text = (
            _tx(_translate, "download.base.title", locale, "🧬 <b>Базовые модели</b>")
            + "\n"
            + _tx(
                _translate,
                "download.base.description",
                locale,
                "Выберите семейство моделей для фильтрации поиска.",
            )
            + "\n"
            + _tx(
                _translate,
                "download.base.current",
                locale,
                "<b>Текущий выбор:</b> {value}",
                params={
                    "value": h(
                        download_base_label(
                            base_code,
                            translate=_label_translate_adapter(_translate),
                            locale=locale,
                        )
                    )
                },
            )
        )
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_results_menu(
        message: Message,
        state: FSMContext,
        results: list[SearchResult],
        *,
        edit: bool,
        page: int = 0,
        page_size: int = 8,
        can_continue: bool = False,
        notice: str = "",
    ) -> None:
        _, locale = await _state_context(message, state)
        data = await state.get_data()
        source = data.get("dl_source", "all")
        sort_code = data.get("dl_sort", "downloads")
        period_code = data.get("dl_period", "all")
        base_code = data.get("dl_base", "all")
        include_nsfw = bool(data.get("dl_nsfw", False))
        author_nick = str(data.get("dl_author", "")).strip()

        total = len(results)
        page_size = max(1, min(page_size, 10))
        total_pages = max(1, (total + page_size - 1) // page_size)
        current_page = max(0, min(page, total_pages - 1))
        start = current_page * page_size
        page_items = results[start : start + page_size]

        rows: list[list[InlineKeyboardButton]] = []
        for offset, result in enumerate(page_items):
            absolute_index = start + offset
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_result_button_label(result),
                        callback_data=f"dlpick:{absolute_index}",
                    )
                ]
            )

        nav: list[InlineKeyboardButton] = []
        if current_page > 0:
            nav.append(
                button(
                    _tx(_translate, "download.results.nav.first", locale, "⏮ В начало"),
                    "dlpick:first",
                )
            )
        if current_page > 0:
            nav.append(
                button(
                    _tx(_translate, "download.results.nav.prev", locale, "◀️ Назад"),
                    "dlpick:prev",
                )
            )
        nav.append(noop_button(f"· {current_page + 1}/{total_pages} ·"))
        if current_page < total_pages - 1:
            nav.append(
                button(
                    _tx(_translate, "download.results.nav.next", locale, "▶️ Далее"),
                    "dlpick:next",
                )
            )
        if current_page < total_pages - 1:
            nav.append(
                button(
                    _tx(_translate, "download.results.nav.last", locale, "⏭ В конец"),
                    "dlpick:last",
                )
            )
        if nav:
            rows.append(nav)

        if can_continue:
            rows.append(
                [
                    button(
                        _tx(
                            _translate,
                            "download.results.button.more",
                            locale,
                            "🔎 Продолжить поиск",
                        ),
                        "dlpick:more",
                    )
                ]
            )
        rows.append(
            [
                back_button(
                    "dlpick:new",
                    text=_tx(
                        _translate,
                        "download.results.button.new_search",
                        locale,
                        "⬅️ Новый поиск",
                    ),
                )
            ]
        )
        rows.append(
            [
                cancel_button(
                    "dlpick:cancel",
                    text=_tx(_translate, "common.action.cancel", locale, "❌ Отмена"),
                )
            ]
        )

        lines = [
            _tx(_translate, "download.results.title", locale, "📦 <b>Результаты поиска</b>"),
            f"<i>{h(_search_filters_summary(model_type=data.get('dl_type', 'checkpoint'), sort_code=sort_code, period_code=period_code, base_code=base_code, include_nsfw=include_nsfw, source=source, author_nick=author_nick, page_size=page_size, translate=_translate, locale=locale))}</i>",
            "",
        ]
        notice_text = notice.strip()
        if notice_text:
            lines.append(f"💬 <i>{h(notice_text)}</i>")
            lines.append("")
        max_text_len = 3800
        for offset, result in enumerate(page_items):
            index = start + offset
            meta = _result_meta_line(result, translate=_translate, locale=locale)
            name_value = result.name if len(result.name) <= 140 else result.name[:137] + "..."
            meta_value = meta if len(meta) <= 220 else meta[:217] + "..."
            item_lines = [
                f"{index + 1}. <b>{h(name_value)}</b>",
                f"   <i>{h(meta_value)}</i>",
            ]
            if result.model_type == "lora" and result.trained_words:
                words = ", ".join(result.trained_words[:4])
                words_value = words if len(words) <= 180 else words[:177] + "..."
                item_lines.append(f"   🔑 <code>{h(words_value)}</code>")

            projected = "\n".join(
                lines
                + item_lines
                + [
                    "",
                    _tx(
                        _translate,
                        "download.results.choose_model",
                        locale,
                        "Выберите модель:",
                    ),
                ]
            )
            if len(projected) > max_text_len:
                lines.append(
                    _tx(
                        _translate,
                        "download.results.truncated",
                        locale,
                        "… список сокращён, уточните запрос для более точного результата.",
                    )
                )
                break
            lines.extend(item_lines)
        lines.append(
            "\n"
            + _tx(
                _translate,
                "download.results.choose_model",
                locale,
                "Выберите модель:",
            )
        )

        text = "\n".join(lines)
        kb = build_keyboard(rows)
        await _render_download_panel(
            message,
            state,
            text,
            kb,
            prefer_edit=edit,
        )

    async def show_download_confirmation(
        message: Message,
        state: FSMContext,
        result: SearchResult,
    ) -> None:
        _, locale = await _state_context(message, state)
        rows: list[list[InlineKeyboardButton]] = []
        if result.preview_url:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_tx(
                            _translate, "download.confirm.button.preview", locale, "👀 Превью"
                        ),
                        url=result.preview_url,
                    )
                ]
            )
        if result.model_url:
            rows.append(
                [
                    InlineKeyboardButton(
                        text=_tx(
                            _translate,
                            "download.confirm.button.model_page",
                            locale,
                            "🔗 Страница модели",
                        ),
                        url=result.model_url,
                    )
                ]
            )
        rows.append(
            [
                button(
                    _tx(_translate, "download.confirm.button.download", locale, "⬇️ Скачать"),
                    "dlconfirm:yes",
                ),
                cancel_button(
                    "dlconfirm:no",
                    text=_tx(_translate, "common.action.cancel", locale, "❌ Отмена"),
                ),
            ]
        )
        rows.append(
            [
                back_button(
                    "dlconfirm:back",
                    text=_tx(
                        _translate,
                        "download.confirm.back_to_results",
                        locale,
                        "⬅️ К результатам",
                    ),
                )
            ]
        )
        kb = build_keyboard(rows)
        await _render_download_panel(
            message,
            state,
            _confirmation_text(result, translate=_translate, locale=locale),
            kb,
            prefer_edit=True,
        )

    async def _start_download(
        message: Message,
        state: FSMContext,
        uid: int,
    ) -> None:
        telegram_locale = message.from_user.language_code if message.from_user else None
        locale = _resolve_locale(uid, telegram_locale=telegram_locale)
        await state.update_data(dl_uid=uid)
        await state.set_state(DownloadStates.choosing_type)
        await _render_download_panel(
            message,
            state,
            _t(
                uid,
                "download.start.choose_type",
                "📦 <b>Скачивание компонентов</b>\nВыберите тип:",
                telegram_locale=telegram_locale,
            ),
            _build_type_keyboard(translate=_translate, locale=locale),
            prefer_edit=True,
        )

    register_download_flow_handlers(
        DownloadFlowDeps(
            router=router,
            runtime=runtime,
            downloader=downloader,
            client=client,
            callback_user_id=callback_user_id,
            message_user_id=message_user_id,
            start_download=_start_download,
            render_download_panel=_render_download_panel,
            build_type_keyboard=(
                lambda locale: _build_type_keyboard(translate=_translate, locale=locale)
            ),
            build_source_keyboard=(
                lambda locale: _build_source_keyboard(translate=_translate, locale=locale)
            ),
            type_title=(
                lambda model_type, locale: _type_title(
                    model_type,
                    translate=_translate,
                    locale=locale,
                )
            ),
            show_filter_menu=show_filter_menu,
            show_author_prompt=show_author_prompt,
            show_base_filter_menu=show_base_filter_menu,
            show_query_prompt=show_query_prompt,
            supports_base_filter=_supports_base_filter,
            supports_nsfw_filter=_supports_nsfw_filter,
            base_code_to_api=DOWNLOAD_BASE_CODE_TO_API,
            sort_code_to_api=DOWNLOAD_SORT_CODE_TO_API,
            period_code_to_api=DOWNLOAD_PERIOD_CODE_TO_API,
            show_results_menu=show_results_menu,
            hydrate_result=_hydrate_result,
            human_size=human_size,
            short_number=short_number,
            apply_version_option=apply_version_option,
            show_download_confirmation=show_download_confirmation,
            filter_profiles=DOWNLOAD_FILTER_PROFILES,
            translate=_translate,
            resolve_locale=_resolve_locale,
        )
    )
