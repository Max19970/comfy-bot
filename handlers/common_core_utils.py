from __future__ import annotations

from typing import Any

from domain.localization import LocalizationService


def _t(
    localization: LocalizationService | None,
    key: str,
    *,
    locale: str | None,
    default: str,
) -> str:
    if localization is None:
        return default
    return localization.t(key, locale=locale, default=default)


def ensure_user_preferences(runtime: Any, uid: int) -> dict[str, Any]:
    if uid not in runtime.user_preferences:
        runtime.user_preferences[uid] = {}
    return runtime.user_preferences[uid]


def set_pref(runtime: Any, uid: int, key: str, value: Any) -> None:
    prefs = ensure_user_preferences(runtime, uid)
    prefs[key] = value


def get_training_mode(runtime: Any, uid: int) -> str:
    raw = runtime.user_preferences.get(uid, {}).get("training_mode", "simple")
    mode = str(raw).strip().lower()
    return mode if mode in {"simple", "advanced"} else "simple"


def set_training_mode(runtime: Any, uid: int, mode: str) -> None:
    set_pref(runtime, uid, "training_mode", mode)


def get_training_page(runtime: Any, uid: int) -> int:
    raw = runtime.user_preferences.get(uid, {}).get("training_page", 0)
    if isinstance(raw, int):
        return max(0, raw)
    return 0


def set_training_page(runtime: Any, uid: int, page: int) -> None:
    set_pref(runtime, uid, "training_page", max(0, page))


def training_pages(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[tuple[str, str]]:
    return [
        (
            _t(
                localization,
                "common.training.page.1.title",
                locale=locale,
                default="Что делает бот",
            ),
            _t(
                localization,
                "common.training.page.1.simple",
                locale=locale,
                default=(
                    "ComfyBot - это интерфейс к ComfyUI.\n"
                    "Вы выбираете параметры и запускаете задачи кнопками."
                ),
            ),
        ),
        (
            _t(
                localization,
                "common.training.page.2.title",
                locale=locale,
                default="Первый запуск",
            ),
            _t(
                localization,
                "common.training.page.2.simple",
                locale=locale,
                default=(
                    "1) Обновите список моделей.\n"
                    "2) Откройте генерацию.\n"
                    "3) Выберите Checkpoint и заполните Positive.\n"
                    "4) Нажмите Генерировать."
                ),
            ),
        ),
        (
            _t(
                localization,
                "common.training.page.3.title",
                locale=locale,
                default="Главные параметры",
            ),
            _t(
                localization,
                "common.training.page.3.simple",
                locale=locale,
                default=(
                    "Steps - детализация и время.\n"
                    "CFG - строгость следования prompt.\n"
                    "Seed - повторяемость результата.\n"
                    "Размер - нагрузка на VRAM и скорость."
                ),
            ),
        ),
        (
            _t(
                localization,
                "common.training.page.4.title",
                locale=locale,
                default="Улучшение результата",
            ),
            _t(
                localization,
                "common.training.page.4.simple",
                locale=locale,
                default=(
                    "Откройте меню улучшений у превью.\n"
                    "Включите нужные режимы (sampler/upscale/hi-res).\n"
                    "Запустите улучшение и при необходимости отмените задачу."
                ),
            ),
        ),
        (
            _t(
                localization,
                "common.training.page.5.title",
                locale=locale,
                default="Диагностика",
            ),
            _t(
                localization,
                "common.training.page.5.simple",
                locale=locale,
                default=(
                    "Если что-то не работает: проверьте соединение с ComfyUI,\n"
                    "наличие checkpoint, очередь задач и статус ошибок в боте."
                ),
            ),
        ),
    ]


def training_advanced(
    *,
    localization: LocalizationService | None = None,
    locale: str | None = None,
) -> list[str]:
    return [
        _t(
            localization,
            "common.training.page.1.advanced",
            locale=locale,
            default=(
                "Бот не рисует сам: он собирает параметры, ставит задачу в очередь ComfyUI "
                "и отслеживает выполнение."
            ),
        ),
        _t(
            localization,
            "common.training.page.2.advanced",
            locale=locale,
            default=(
                "Минимум для стабильного старта: валидный checkpoint, положительный "
                "prompt и доступный ComfyUI URL."
            ),
        ),
        _t(
            localization,
            "common.training.page.3.advanced",
            locale=locale,
            default=(
                "Steps увеличивает количество итераций денойзинга; CFG управляет силой "
                "conditioning; seed фиксирует стохастику."
            ),
        ),
        _t(
            localization,
            "common.training.page.4.advanced",
            locale=locale,
            default=(
                "Для улучшений: sampler-pass полезен для перерендера, upscaler - для роста "
                "размера, hi-res - для детализации."
            ),
        ),
        _t(
            localization,
            "common.training.page.5.advanced",
            locale=locale,
            default=(
                "Если задача пропала из списка, обновите «Мои задачи», проверьте очередь "
                "и статус prompt_id в ComfyUI."
            ),
        ),
    ]
