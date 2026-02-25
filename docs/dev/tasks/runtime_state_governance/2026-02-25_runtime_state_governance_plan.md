# План задачи: runtime_state_governance

> [!IMPORTANT]
> План создан по шаблону `docs/dev/tasks/_templates/task_plan_template.md`.
>
> Перед началом выполнения:
> 1. План должен быть явно одобрен пользователем.
> 2. Этапы выполняются строго последовательно.
> 3. После каждого этапа требуется явное подтверждение пользователя.

---

## Навигация

- [Быстрый чеклист](#быстрый-чеклист-m)
- [Метаданные плана](#метаданные-плана-optional-but-recommended-o)
- [Контекст задачи](#контекст-задачи)
- [Результат валидации темы](#результат-валидации-темы)
- [Поэтапный план](#поэтапный-план)
- [Критерии приемки этапов](#критерии-приемки-этапов)
- [Риски](#риски)
- [Незапланированные дополнительные задачи](#незапланированные-дополнительные-задачи)
- [Журнал одобрений этапов](#журнал-одобрений-этапов)

## Легенда секций

| Marker | Meaning |
| --- | --- |
| `[M]` | Обязательная секция |
| `[O]` | Опциональная секция |

## Быстрый чеклист [M]

- [x] Выбран `short_task_name` и имя файла плана.
- [x] Заполнен раздел `Контекст задачи` с четкими границами scope.
- [x] Заполнен раздел `Результат валидации темы` с явным решением gate.
- [x] Сформирован модульный план этапов с конкретными подзадачами и проверками.
- [x] Заполнены критерии приемки и таблица рисков.
- [x] Инициализирован `Журнал одобрений этапов`.

## Метаданные плана (optional but recommended) [O]

| Field | Value |
| --- | --- |
| Task slug | `runtime_state_governance` |
| Work type | `refactor` |
| Date | `2026-02-25` |
| Owner | `OpenCode agent` |
| Linked request | `Новая задача: реализация P5/P6` |

## Контекст задачи

### Запрос и ожидаемый результат [M]

- Запрос: запустить новую задачу на реализацию `P5/P6`.
- Ожидаемый результат:
  - Реализована стабилизация runtime state (`P5`): typed runtime gateways, снижение прямых мутаций глобальных dict и безопасная персистентность через write-behind очередь.
  - Реализован governance/deprecation слой (`P6`): совместимость контрактов расширений, матрица совместимости и управляемый lifecycle legacy API.
  - Сохранены пользовательские сценарии `/start`, `/generate`, `/download`, `/jobs` и восстановление состояния после рестарта.

### Границы scope [M]

- In scope:
  - Проектирование и внедрение typed runtime gateways поверх `RuntimeStore`.
  - Внедрение write-behind/debounce persistence очереди с flush-checkpoints на критических переходах.
  - Поэтапная миграция горячих handler-путей с прямых `runtime.*` мутаций на gateway API.
  - Внедрение governance/deprecation механик для plugin/provider/node contracts (версионирование, compatibility checks, deprecation warnings).
  - Тесты, документация и migration notes для безопасного rollout `P5/P6`.
- Out of scope [O]:
  - Реализация новых продуктовых фич, не относящихся к runtime/governance.
  - Полное удаление legacy компонентов в этом цикле (только управляемая deprecation-подготовка).

### Основные цели и ограничения [M]

- Files/components: `core/runtime.py`, `core/runtime_persistence.py`, `core/runtime_snapshot.py`, `handlers/*` (горячие runtime-пути), `bootstrap/app_context.py`, `plugins/*`, `application/model_source_provider_loader.py`, `infrastructure/comfy_nodes/registry.py`, профильные тесты в `tests/`, документация в `docs/`.
- Systems/areas: backend runtime state, persistence lifecycle, extension contracts governance.
- Constraints and assumptions [O]:
  - Изменения должны быть инкрементальными и обратимо безопасными (feature-safe rollout).
  - Важно не деградировать производительность и UX в длинных пользовательских flow.

## Результат валидации темы

### Обнаруженные темы [M]

- `P5`: Runtime State Stabilization (typed gateways + persistence queue).
- `P6`: Governance and Deprecation (контрактная совместимость и lifecycle legacy API).

### Решение по валидации [M]

| Check | Result |
| --- | --- |
| Is this one coherent topic? | `yes` |
| Split required? | `no` |
| Gate status | `Topic Validation Gate passed` |

- Decision notes:
  - `P5/P6` образуют единую цель: сделать runtime и extension-контракты устойчивыми, расширяемыми и предсказуемыми в сопровождении.

## Поэтапный план

### Этап 1 - Typed runtime gateways foundation

#### Цель этапа [M]

- Ввести typed gateways для ключевых runtime-доменов и подготовить совместимый foundation-слой без изменения пользовательского поведения.

#### Подзадачи [M]

- [x] Добавить gateway-контракты для prompt sessions, generation jobs, download sessions и user preferences.
- [x] Реализовать адаптеры поверх текущего `RuntimeStore` с backward-compatible API.
- [x] Подготовить минимальные migration adapters для существующих handler dependencies.
- [x] Добавить unit/contract тесты на корректность gateway операций.

#### Почему этап не ломает работоспособность проекта [M]

- Foundation-слой добавляется поверх существующей модели хранения; текущие flow сохраняют старый путь до завершения миграции.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_runtime_session_contracts.py`
2. `python -m pytest -q tests/test_architecture_boundaries.py`

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-1_runtime-gateways-foundation_report.md`

### Этап 2 - Persistence queue and flush checkpoints

#### Цель этапа [M]

- Стабилизировать персистентность runtime через write-behind/debounce queue и критические flush точки.

#### Подзадачи [M]

- [ ] Реализовать write-behind persistence queue с контролем частоты записи.
- [ ] Добавить flush-checkpoints на критических переходах (start/finish generation, state transitions, graceful shutdown).
- [ ] Интегрировать новый путь в lifecycle приложения и middleware.
- [ ] Добавить тесты устойчивости state после рестарта/ошибок.

#### Почему этап не ломает работоспособность проекта [M]

- Сохраняется совместимый persist fallback; новый queue-путь покрывается тестами и запускается безопасно.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_runtime_session_contracts.py tests/test_callback_payload_contracts.py`
2. Ручная проверка: рестарт во время активной генерации и восстановление `/jobs`.

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-2_persistence-queue_report.md`

### Этап 3 - Handler migration to gateway-only runtime access

#### Цель этапа [M]

- Снизить прямые мутации `runtime.*` в handler-слое и перевести ключевые потоки на gateway API.

#### Подзадачи [M]

- [ ] Мигрировать горячие handler-потоки (`generate/download/jobs`) на typed gateways.
- [ ] Добавить guardrail-тест против новых прямых runtime dict-мутаций в целевых модулях.
- [ ] Обновить bridge/contract тесты под новый runtime access path.

#### Почему этап не ломает работоспособность проекта [M]

- Миграция выполняется инкрементно с сохранением fallback-пути и проверкой пользовательских контрактов.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_handler_callback_contracts.py tests/test_runtime_session_contracts.py tests/test_architecture_boundaries.py`
2. Ручной smoke: `/start`, `/generate`, `/download`, `/jobs`.

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-3_handler-runtime-migration_report.md`

### Этап 4 - Governance/deprecation framework and final package

#### Цель этапа [M]

- Завершить `P6`: внедрить governance/deprecation практики и финализировать пакет P5/P6.

#### Подзадачи [M]

- [ ] Ввести совместимый governance-контур для plugin/provider/node contracts (version checks, compatibility matrix).
- [ ] Добавить deprecation lifecycle (warning policy, migration deadlines, release gate checklist).
- [ ] Обновить документацию и операционные инструкции по сопровождению контрактов.
- [ ] Выполнить финальный регрессионный прогон P5/P6.

#### Почему этап не ломает работоспособность проекта [M]

- Governance и deprecation добавляются как надстройка контроля и обратной совместимости без изменения пользовательского API.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_architecture_boundaries.py tests/test_runtime_session_contracts.py tests/test_handler_callback_contracts.py tests/test_model_source_provider_loader.py tests/test_comfy_node_registry.py`
2. Ручной smoke пользовательских flow и проверка предупреждений deprecation в логах.

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-4_final-p5-p6-package_report.md`

## Критерии приемки этапов

| Stage | Acceptance criteria | Evidence |
| --- | --- | --- |
| Stage 1 | Typed runtime gateways реализованы и контрактно покрыты | `tests/test_runtime_session_contracts.py` + отчет Stage 1 |
| Stage 2 | Persistence queue работает с checkpoint flush и без потери состояния | Тесты runtime/session + manual restart scenario + отчет Stage 2 |
| Stage 3 | Ключевые handler flow используют gateway access path, прямые мутации снижены | Guardrail тесты + flow smoke + отчет Stage 3 |
| Stage 4 | Governance/deprecation framework задокументирован и покрыт проверками | Финальный regression suite + отчет Stage 4 |

## Риски

| Risk | Impact | Mitigation | Owner [O] |
| --- | --- | --- | --- |
| Потеря состояния при write-behind queue в edge-сценариях | high: восстановление задач может ломаться | Flush-checkpoints на критических переходах + тесты рестарта/аварий | OpenCode agent |
| Частичная миграция handler-path оставит двойную модель runtime доступа | medium: техдолг и сложная диагностика | Четкий список приоритетных потоков + guardrail против новых прямых мутаций | OpenCode agent |
| Избыточно строгие governance checks заблокируют обратную совместимость | medium: нестабильный rollout расширений | Версионирование с compatibility window + явная deprecation policy | OpenCode agent |
| Рост latency из-за лишних flush/validation операций | medium: ухудшение UX | Debounce queue, batch persist, профилирование hot paths | OpenCode agent |

## Незапланированные дополнительные задачи

| Date | Task | Why needed | Resolution |
| --- | --- | --- | --- |
| 2026-02-25 | Добавлен динамический loader runtime gateway extensions (`RUNTIME_GATEWAY_EXTENSION_PACKAGES`) | Пользователь потребовал сохранить расширяемость без правок core для P5/P6 контура runtime gateways | Реализован `core/runtime_gateway_extension_loader.py`, подключение в `bootstrap/app_context.py`, добавлены тесты/документация. |

If none:

- None.

## Журнал одобрений этапов

| Date | Event | Reference |
| --- | --- | --- |
| 2026-02-25 | Plan created; awaiting approval | `docs/dev/tasks/runtime_state_governance/2026-02-25_runtime_state_governance_plan.md` |
| 2026-02-25 | Stage 1 implemented; awaiting explicit user approval | `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-1_runtime-gateways-foundation_report.md` |
| 2026-02-25 | Stage 1 updated with runtime gateway extension hook support; awaiting explicit user approval | `docs/dev/tasks/runtime_state_governance/reports/2026-02-25_stage-1_runtime-gateways-foundation_report.md` |
