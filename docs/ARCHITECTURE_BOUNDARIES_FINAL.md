# Финальная карта архитектурных границ

## Цель

Документ фиксирует итоговые границы слоев после этапов 1-14 и служит
операционным ориентиром для дальнейших изменений без возврата к монолитным
обработчикам и скрытым cross-layer зависимостям.

## Слои и ответственность

- `domain/`
  - Чистые предметные правила и value object логика.
  - Не зависит от `core/`, `application/`, `infrastructure/`, `handlers/`, `presentation/`.
- `application/`
  - Use-case оркестрация и прикладные сервисы.
  - Не зависит от `handlers/` и `presentation/`.
- `infrastructure/`
  - Адаптеры внешних систем (ComfyUI transport/execution, model source clients, TIPO backend).
  - Не зависит от `handlers/` и `presentation/`.
- `core/`
  - Кросс-секционные контракты и shared utilities (callbacks, interaction, runtime, UI kit).
  - Не зависит от `handlers/`.
- `handlers/`
  - Сценарная Telegram-логика, wiring feature-веток, пользовательские flow.
  - Использует `application/`, `infrastructure/`, `core/`, `domain/` через явные deps.
- `presentation/`
  - Изолированный слой представления (при использовании).

## Ключевые композиционные точки

- `bootstrap/app_context.py`
  - Композиция runtime + сервисов + внешних адаптеров.
- `handlers/registry.py`
  - Централизованная регистрация handler-модулей.
- `handlers/prompt_editor.py`
  - Composition root для prompt editor подветок.

## Итоговая декомпозиция критичных flow

- Prompt editor
  - Subhandler wiring: `handlers/prompt_editor_subhandlers.py`
  - Генерация/доставка: `handlers/prompt_editor_generation.py`, `handlers/prompt_editor_send.py`
  - Enhancement-runner: `handlers/prompt_editor_enhancement.py`
  - Edit/Thematic/Lora/References cleanup: отдельные модули + shared guards/scalar helpers
    (`handlers/prompt_editor_handler_guards.py`, `handlers/prompt_editor_scalar_utils.py`).
- Download flow
  - State/view/util split: `handlers/download_flow_state.py`,
    `handlers/download_flow_version_view.py`, `handlers/download_flow_utils.py`.
- Presets
  - Storage/UI/flow split: `handlers/presets_storage.py`, `handlers/presets_ui.py`,
    `handlers/presets_flow.py`, orchestration в `handlers/presets.py`.
- Runtime persistence
  - Snapshot и persistence adapter: `core/runtime_snapshot.py`, `core/runtime_persistence.py`.

## Границы, которые считаются обязательными

- Не возвращать логические "god-modules" в `handlers/`.
- Не дублировать callback parsing и guard-паттерны в feature-коде.
- Не внедрять прямые cross-layer импорты, которые нарушают boundary-тесты.
- Любая новая UI-ветка должна использовать shared interaction/UI callbacks helpers.

## Контрольные механизмы

- Архитектурные boundary-тесты: `tests/test_architecture_boundaries.py`.
- Контрактные callback/runtime тесты:
  - `tests/test_callback_payload_contracts.py`
  - `tests/test_handler_callback_contracts.py`
  - `tests/test_runtime_session_contracts.py`
  - `tests/test_refactor_completion_contracts.py`

## Правило расширения после рефакторинга

1. Сначала определить слой изменения (domain/application/infrastructure/handlers).
2. Добавить/обновить узкий helper или adapter вместо расширения монолитного handler-файла.
3. Для новых callback payload использовать схему из `core/callbacks.py`.
4. Закрывать изменения тестами контракта и регрессии по затронутому сценарию.
