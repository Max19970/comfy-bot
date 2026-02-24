# План задачи: plugin_kernel_handler_pluginization

> [!IMPORTANT]
> План создан по шаблону `docs/dev/tasks/_templates/task_plan_template.md`.
>
> Перед началом выполнения:
> 1. План должен быть явно одобрен пользователем.
> 2. Этапы выполняются строго последовательно.
> 3. После каждого этапа требуется явное подтверждение пользователя.

---

## Navigation

- [Quick start checklist](#quick-start-checklist)
- [Plan metadata](#plan-metadata-optional-but-recommended)
- [Task context](#task-context)
- [Topic validation result](#topic-validation-result)
- [Stage-by-stage plan](#stage-by-stage-plan)
- [Stage acceptance criteria](#stage-acceptance-criteria)
- [Risk notes](#risk-notes)
- [Unplanned additional tasks](#unplanned-additional-tasks)
- [Stage approval log](#stage-approval-log)

## Section legend

| Marker | Meaning |
| --- | --- |
| `[M]` | Обязательная секция |
| `[O]` | Опциональная секция |

## Quick start checklist [M]

- [x] Выбран `short_task_name` и имя файла плана.
- [x] Заполнен раздел `Task context` с четкими границами scope.
- [x] Заполнен раздел `Topic validation result` с явным решением gate.
- [x] Построен модульный план этапов с конкретными подзадачами и проверками.
- [x] Заполнены критерии приемки и таблица рисков.
- [x] Инициализирован `Stage approval log`.

## Plan metadata (optional but recommended) [O]

| Field | Value |
| --- | --- |
| Task slug | `plugin_kernel_handler_pluginization` |
| Work type | `refactor` |
| Date | `2026-02-24` |
| Owner | `OpenCode agent` |
| Linked request | `Новая задача: реализация P1/P2 (Plugin Kernel + Handler Pluginization)` |

## Task context

### Request and outcome [M]

- Request summary: реализовать пакеты `P1/P2` из согласованного roadmap: `Plugin Kernel Foundation` и `Telegram Handler Decoupling`.
- Desired outcome:
  - В проекте появляется плагинное ядро с контрактами, discovery и валидацией совместимости.
  - Регистрация Telegram handlers переводится с статического wiring на plugin-модель без регрессии пользовательских flow.
  - Базовые плагины подключаются по конфигурации/реестру, а не через жестко зашитый список импортов.

### Scope boundaries [M]

- In scope:
  - Реализация `plugins/contracts.py` и `plugins/loader.py` (или эквивалентного plugin-kernel пакета).
  - Внедрение `HandlerPlugin` контракта и built-in plugin-адаптеров для текущих блоков (`common`, `prompt_editor`, `presets`, `download`).
  - Интеграция plugin registration path в `handlers/registry.py` и bootstrap wiring.
  - Добавление/обновление тестов на discovery, порядок регистрации, совместимость и отсутствие поведенческих регрессий.
- Out of scope [O]:
  - Реализация `P3+` (Model Source Providers, Runtime State Stabilization и т.д.).
  - Полный демонтаж legacy API вне границ `P1/P2`.

### Primary targets and constraints [M]

- Files/components: `handlers/registry.py`, `bootstrap/app_context.py`, `core/config.py`, новый пакет `plugins/`, возможно `handlers/plugins/` и профильные тесты в `tests/`.
- Systems/areas: backend architecture, dependency composition, handler orchestration.
- Constraints and assumptions [O]:
  - Поведение пользовательских команд `/start`, `/generate`, `/download`, `/jobs` должно остаться совместимым.
  - Обратная совместимость текущих публичных функций регистрации должна быть сохранена минимум на один миграционный цикл.

## Topic validation result

### Detected topics [M]

- Внедрение plugin kernel (контракты, discovery, совместимость).
- Декомпозиция и перевод handler-регистрации на plugin-модель.

### Validation decision [M]

| Check | Result |
| --- | --- |
| Is this one coherent topic? | `yes` |
| Split required? | `no` |
| Gate status | `Topic Validation Gate passed` |

- Decision notes:
  - Темы `P1` и `P2` имеют единый прямой результат: сделать расширение handler-слоя возможным без модификации ядра.

## Stage-by-stage plan

### Stage 1 - Plugin kernel foundation

#### Stage objective [M]

- Ввести базовые plugin-контракты и discovery-loader, не меняя runtime-поведение бота.

#### Sub-tasks [M]

- [ ] Добавить package `plugins/` с контрактами (`PluginDescriptor`, `PluginApiVersion`, `HandlerPlugin` протокол, capability flags).
- [ ] Реализовать loader/discovery с детерминированным порядком, проверкой duplicate plugin id и api-version совместимости.
- [ ] Расширить конфиг (например, `HANDLER_PLUGIN_PACKAGES`) с безопасным default на built-in плагины.
- [ ] Добавить unit-тесты plugin loader и контрактных ошибок.

#### Why this stage does not break overall project operability [M]

- Изменения аддитивные: plugin kernel добавляется рядом с текущим путём регистрации и пока не заменяет его.

#### Validation/check steps [M]

1. `python -m pytest -q tests/test_architecture_boundaries.py`
2. `python -m pytest -q tests/test_handler_registry_bridge.py` и новые тесты plugin loader

#### Expected artifacts [O]

- `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-1_plugin-kernel-foundation_report.md`

### Stage 2 - Built-in handler plugins and registry bridge

#### Stage objective [M]

- Обернуть текущие handler-блоки в built-in плагины и подготовить plugin-driven registration bridge.

#### Sub-tasks [M]

- [ ] Создать built-in plugin adapters для существующих блоков (`common`, `prompt_editor`, `presets`, `download`).
- [ ] Ввести dependency contract для plugin registration context (замена широких `Any`/неявного wiring).
- [ ] Интегрировать bridge-слой в `handlers/registry.py`, сохранив совместимость текущих публичных функций.
- [ ] Обновить тесты bridge/контрактов регистрации.

#### Why this stage does not break overall project operability [M]

- Используется compatibility bridge: existing entrypoints сохраняются и маршрутизируют в plugin path.

#### Validation/check steps [M]

1. `python -m pytest -q tests/test_handler_registry_bridge.py tests/test_handler_callback_contracts.py`
2. Smoke сценарий запуска бота и проверка переходов `/start` -> меню

#### Expected artifacts [O]

- `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-2_builtin-handler-plugins_report.md`

### Stage 3 - Plugin-first handler registration

#### Stage objective [M]

- Переключить регистрацию handlers на plugin-first путь с контролем порядка и guardrails.

#### Sub-tasks [M]

- [ ] Перевести `handlers/registry.py` на цикл по discovered plugins с явной сортировкой/приоритетом.
- [ ] Добавить startup diagnostics (список подключенных плагинов, версии, предупреждения несовместимости).
- [ ] Обновить архитектурные guardrails/тесты, чтобы запретить возврат к статическому hardcoded wiring.

#### Why this stage does not break overall project operability [M]

- Built-in плагины полностью покрывают текущий набор handlers; ordering фиксируется явно и тестируется.

#### Validation/check steps [M]

1. `python -m pytest -q tests/test_architecture_boundaries.py tests/test_handler_registry_bridge.py`
2. `python -m pytest -q tests/test_callback_payload_contracts.py tests/test_runtime_session_contracts.py`

#### Expected artifacts [O]

- `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-3_plugin-first-registration_report.md`

### Stage 4 - Finalize P1/P2 package and migration docs

#### Stage objective [M]

- Завершить P1/P2 deliverables, зафиксировать миграционные правила и финальную проверку совместимости.

#### Sub-tasks [M]

- [ ] Подготовить краткую документацию контракта плагинов и инструкцию добавления нового handler plugin.
- [ ] Обновить/добавить тесты для end-to-end регистрации handlers через plugin path.
- [ ] Выполнить финальный регрессионный прогон целевых тестов и зафиксировать результаты.

#### Why this stage does not break overall project operability [M]

- Документация и тесты усиливают стабильность, а runtime-поведение подтверждается regression-проверками.

#### Validation/check steps [M]

1. `python -m pytest -q tests/test_architecture_boundaries.py tests/test_handler_registry_bridge.py tests/test_callback_payload_contracts.py tests/test_handler_callback_contracts.py tests/test_runtime_session_contracts.py`
2. Ручная проверка пользовательских flow `/start`, `/generate`, `/download`, `/jobs`

#### Expected artifacts [O]

- `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-4_final-p1-p2-package_report.md`

## Stage acceptance criteria

| Stage | Acceptance criteria | Evidence |
| --- | --- | --- |
| Stage 1 | Plugin kernel контракты и loader реализованы, discovery детерминирован и валидируется | Тесты plugin loader + отчет Stage 1 |
| Stage 2 | Built-in handlers зарегистрированы через plugin adapters с сохранением совместимости | `tests/test_handler_registry_bridge.py` + отчет Stage 2 |
| Stage 3 | Статическое wiring заменено plugin-first registration path без регрессии контрактов | Контрактные тесты callbacks/runtime + отчет Stage 3 |
| Stage 4 | Подготовлен финальный P1/P2 пакет с документацией и подтвержденной совместимостью flow | Финальный тест-прогон + отчет Stage 4 |

## Risk notes

| Risk | Impact | Mitigation | Owner [O] |
| --- | --- | --- | --- |
| Нарушение порядка регистрации handlers | high: callbacks могут матчиться неверно | Явное поле приоритета в plugin descriptor + тест порядка | OpenCode agent |
| Незаметная несовместимость plugin API | high: частичное падение регистрации при старте | Version/capability checks + fail-fast diagnostics | OpenCode agent |
| Регрессия пользовательских flow при переключении на plugin-first | high: деградация UX | Поэтапный bridge + обязательные contract/smoke проверки на каждом этапе | OpenCode agent |
| Рост сложности bootstrap wiring | medium: усложнение сопровождения | Централизовать plugin loading в одном модуле composition root | OpenCode agent |

## Unplanned additional tasks

| Date | Task | Why needed | Resolution |
| --- | --- | --- | --- |

If none:

- None.

## Stage approval log

| Date | Event | Reference |
| --- | --- | --- |
| 2026-02-24 | Plan created; awaiting approval | `docs/dev/tasks/plugin_kernel_handler_pluginization/2026-02-24_plugin_kernel_handler_pluginization_plan.md` |
| 2026-02-24 | Stage 1 implemented; awaiting explicit user approval | `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-1_plugin-kernel-foundation_report.md` |
| 2026-02-24 | Stage 1 approved by user | User message: `Этап 1 одобрен` |
| 2026-02-24 | Stage 1 committed | `8cbc258` |
| 2026-02-24 | Stage 2 implemented; awaiting explicit user approval | `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-2_builtin-handler-plugins_report.md` |
| 2026-02-24 | Stage 2 approved by user | User message: `Этап 2 одобрен` |
