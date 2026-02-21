from __future__ import annotations

from typing import Any


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


def training_pages() -> list[tuple[str, str]]:
    return [
        (
            "Что делает бот",
            "ComfyBot - это интерфейс к ComfyUI.\n"
            "Вы выбираете параметры и запускаете задачи кнопками.",
        ),
        (
            "Первый запуск",
            "1) Обновите список моделей.\n"
            "2) Откройте генерацию.\n"
            "3) Выберите Checkpoint и заполните Positive.\n"
            "4) Нажмите Генерировать.",
        ),
        (
            "Главные параметры",
            "Steps - детализация и время.\n"
            "CFG - строгость следования prompt.\n"
            "Seed - повторяемость результата.\n"
            "Размер - нагрузка на VRAM и скорость.",
        ),
        (
            "Улучшение результата",
            "Откройте меню улучшений у превью.\n"
            "Включите нужные режимы (sampler/upscale/hi-res).\n"
            "Запустите улучшение и при необходимости отмените задачу.",
        ),
        (
            "Диагностика",
            "Если что-то не работает: проверьте соединение с ComfyUI,\n"
            "наличие checkpoint, очередь задач и статус ошибок в боте.",
        ),
    ]


def training_advanced() -> list[str]:
    return [
        "Бот не рисует сам: он собирает параметры, ставит задачу в очередь ComfyUI и отслеживает выполнение.",
        "Минимум для стабильного старта: валидный checkpoint, положительный prompt и доступный ComfyUI URL.",
        "Steps увеличивает количество итераций денойзинга; CFG управляет силой conditioning; seed фиксирует стохастику.",
        "Для улучшений: sampler-pass полезен для перерендера, upscaler - для роста размера, hi-res - для детализации.",
        "Если задача пропала из списка, обновите «Мои задачи», проверьте очередь и статус prompt_id в ComfyUI.",
    ]
