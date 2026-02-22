# Refactor Guardrails

## Назначение

Этот документ фиксирует обязательные правила безопасного рефакторинга.
Цель: каждый этап должен быть завершен без потери работоспособности бота в моменте.

## Протокол завершения этапа

Каждый этап считается завершенным только после всех шагов:

1. Выполнены автоматические проверки (см. Quality Gate).
2. Подготовлен отчет: что изменено, какие риски закрыты, что осталось.
3. Передан ручной чеклист проверки в Telegram.
4. Работа остановлена до подтверждения `ok` от ревьюера.
5. После `ok` создается один целостный commit этапа.

## Quality Gate (обязательный минимум)

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
python -m compileall -q .
```

## Архитектурные границы для новой разработки

Legacy-модули (`core/*`, `handlers/*`, `model_downloader.py`, `comfyui_client.py`,
`smart_prompt.py`) допускаются как зона пошаговой миграции.

Новый код добавляется в целевые слои:

- `domain/` — бизнес-сущности и value objects.
- `application/` — use-cases и orchestration.
- `infrastructure/` — внешние API, файловая система, адаптеры.
- `presentation/` — UI/transport адаптеры.

Правило зависимостей:

- `domain` не зависит от `application`, `infrastructure`, `presentation`, `handlers`, `core`.
- `application` не зависит от `presentation`, `handlers`.
- `infrastructure` не зависит от `presentation`, `handlers`.
- `core` не импортирует `handlers`.

Эти границы дополнительно проверяются тестом `tests/test_architecture_boundaries.py`.

## Контракты обратной совместимости

До финальной миграции запрещено ломать:

- callback payload-форматы в Telegram flow;
- ключи runtime/session, используемые для восстановления состояния;
- публичные команды (`/generate`, `/download`, `/models`, `/queue`, `/settings`, `/cancel`);
- сохраненные пресеты и данные в `sessions/runtime.json`.

Изменения этих контрактов допускаются только в отдельном этапе с мигратором
и явным ручным сценарием проверки.

## Ручной smoke-чек (минимум на этап)

- `/start` и переходы по главному меню.
- `/generate` -> открытие редактора -> возврат в меню.
- `/download` -> вход в flow -> возврат/отмена.
- `/models` и `/queue` открываются без ошибок.

Расширенный чеклист: `docs/UI_MIGRATION_PLAN.md` (раздел "Матрица ручной проверки").
