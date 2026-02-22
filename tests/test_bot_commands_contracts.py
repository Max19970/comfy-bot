from __future__ import annotations

from bot import default_bot_commands


def test_default_bot_commands_contract_is_stable() -> None:
    commands = default_bot_commands()

    assert [item.command for item in commands] == [
        "start",
        "help",
        "generate",
        "repeat",
        "presets",
        "download",
        "models",
        "queue",
        "jobs",
        "settings",
        "training",
        "cancel",
    ]

    assert [item.description for item in commands] == [
        "Главное меню",
        "Помощь",
        "Новая генерация",
        "Повтор последней",
        "Мои пресеты",
        "Скачать модель",
        "Список моделей",
        "Очередь ComfyUI",
        "Активные задачи",
        "Настройки",
        "Обучение",
        "Отменить операцию",
    ]
