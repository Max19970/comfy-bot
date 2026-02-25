# План задачи: provider_workflow_extensions

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
| Task slug | `provider_workflow_extensions` |
| Work type | `refactor` |
| Date | `2026-02-25` |
| Owner | `OpenCode agent` |
| Linked request | `Новая задача: закрыть P3/P4` |

## Контекст задачи

### Запрос и ожидаемый результат [M]

- Запрос: начать новую задачу, закрывающую `P3/P4` из ранее согласованного архитектурного roadmap.
- Ожидаемый результат:
  - Реализован `P3` (`ModelSourceProvider` слой) с подключаемыми provider-адаптерами без правки ядра `ModelDownloader` при добавлении новых источников.
  - Реализован `P4` (`Workflow Extension Contracts`) с конфигурируемыми пакетами comfy-нод и валидацией конфликтов/контрактов.
  - Сохранена обратная совместимость пользовательских сценариев `/download`, `/models`, `/generate`.

### Границы scope [M]

- In scope:
  - Введение порта `ModelSourceProvider`, provider-registry и маршрутизации источников для поиска/скачивания моделей.
  - Интеграция существующих источников (`civitai`, `huggingface`) через адаптеры provider-слоя.
  - Введение `COMFY_NODE_PACKAGES`/эквивалентного конфига для загрузки comfy-нод из нескольких пакетов.
  - Валидация конфликтов `node_id` и проверка совместимости stage-label/контрактов workflow extension.
  - Обновление тестов, документации и migration notes в границах `P3/P4`.
- Out of scope [O]:
  - Полная реализация `P5/P6` (runtime-state stabilization и governance/deprecation).
  - Изменение пользовательского UI/UX, не связанное с provider/workflow extension архитектурой.

### Основные цели и ограничения [M]

- Files/components: `application/model_downloader.py`, `application/download_search_use_case.py`, `infrastructure/model_source_clients.py`, `infrastructure/comfy_nodes/*`, `core/config.py`, `docs/CONFIG.md`, профильные тесты `tests/test_download_*`, `tests/test_comfy_node_system.py`.
- Systems/areas: backend architecture, application ports/adapters, workflow graph extension surface.
- Constraints and assumptions [O]:
  - Совместимость текущих callback/контрактов должна быть сохранена.
  - По умолчанию поведение должно оставаться прежним (`built-in` providers/nodes), а расширяемость включаться конфигурацией.

## Результат валидации темы

### Обнаруженные темы [M]

- `P3`: provider-абстракция и развязка `ModelDownloader` от конкретных источников.
- `P4`: расширяемость workflow-нод через динамическую загрузку и контрактные ограничения.

### Решение по валидации [M]

| Check | Result |
| --- | --- |
| Is this one coherent topic? | `yes` |
| Split required? | `no` |
| Gate status | `Topic Validation Gate passed` |

- Decision notes:
  - `P3/P4` объединены единой целью: сделать архитектуру источников моделей и workflow-расширений plugin-capable без модификации ядра.

## Поэтапный план

### Этап 1 - Контракты Model Source Providers и базовый registry

#### Цель этапа [M]

- Ввести application-контракты provider-слоя и реестр провайдеров без изменения внешнего поведения `/download`.

#### Подзадачи [M]

- [ ] Определить `ModelSourceProvider` протокол(ы) и единый контракт операций поиска/метаданных/скачивания.
- [ ] Реализовать provider-registry и source routing policy (`all/civitai/huggingface`).
- [ ] Подключить built-in adapters поверх текущих клиентов инфраструктуры.
- [ ] Добавить unit-тесты на registry, routing и совместимость default source.

#### Почему этап не ломает работоспособность проекта [M]

- Контракты и registry добавляются как совместимый слой-обертка, текущий API остается доступным.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_download_search_use_case.py`
2. `python -m pytest -q tests/test_model_downloader.py`

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-1_model-source-provider-foundation_report.md`

### Этап 2 - Интеграция ModelDownloader с provider-слоем

#### Цель этапа [M]

- Переключить `ModelDownloader` на provider-first обработку источников при сохранении backward compatibility.

#### Подзадачи [M]

- [ ] Перевести `search/download` orchestration в `ModelDownloader` на вызовы provider-registry.
- [ ] Сохранить совместимость существующих публичных методов/форматов ответов.
- [ ] Добавить диагностику выбранного provider и fallback-обработку ошибок источника.
- [ ] Обновить/добавить интеграционные тесты для multi-source сценариев.

#### Почему этап не ломает работоспособность проекта [M]

- Сохраняются те же пользовательские команды и форматы данных; меняется только внутренний routing.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_download_search_use_case.py tests/test_download_flow_utils.py`
2. Ручной smoke `/download` для источников `all`, `civitai`, `huggingface`.

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-2_model-downloader-provider-integration_report.md`

### Этап 3 - Workflow extension contracts и multi-package node discovery

#### Цель этапа [M]

- Реализовать расширяемую загрузку comfy-нод из конфигурируемых пакетов с детерминированным порядком и защитой от конфликтов.

#### Подзадачи [M]

- [ ] Добавить конфиг-параметр `COMFY_NODE_PACKAGES` (или эквивалент) с безопасным default.
- [ ] Реализовать загрузку нод из нескольких пакетов и дедупликацию package-списка.
- [ ] Добавить строгую валидацию конфликтов `node_id` и корректности `stage_labels` контрактов.
- [ ] Обновить тесты discovery и guardrails для расширяемости workflow.

#### Почему этап не ломает работоспособность проекта [M]

- Default пакет остается прежним; новые пакеты подключаются только через конфигурацию.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_comfy_node_system.py tests/test_comfy_workflow_builder.py`
2. `python -m pytest -q tests/test_architecture_boundaries.py`

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-3_workflow-extension-contracts_report.md`

### Этап 4 - Финализация P3/P4 (документация, e2e-проверки, миграционные заметки)

#### Цель этапа [M]

- Закрыть пакет P3/P4 с финальной документацией контрактов, e2e проверками и readiness-критериями.

#### Подзадачи [M]

- [ ] Обновить `docs/CONFIG.md` и профильные docs по provider/node extension contracts.
- [ ] Добавить e2e тесты для подключения нового provider и нового node package без правки ядра.
- [ ] Прогнать финальный регрессионный набор тестов по затронутым зонам.
- [ ] Подготовить финальный отчет Stage 4 и зафиксировать migration notes.

#### Почему этап не ломает работоспособность проекта [M]

- Изменения документации/тестов и финальные проверки подтверждают стабильность внедренных слоев.

#### Шаги проверки [M]

1. `python -m pytest -q tests/test_download_search_use_case.py tests/test_model_downloader.py tests/test_comfy_node_system.py tests/test_comfy_workflow_builder.py tests/test_architecture_boundaries.py`
2. Ручной smoke `/download` + `/generate` после включения дополнительных package/provider entries.

#### Ожидаемые артефакты [O]

- `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-4_final-p3-p4-package_report.md`

## Критерии приемки этапов

| Stage | Acceptance criteria | Evidence |
| --- | --- | --- |
| Stage 1 | Добавлен provider-контракт и registry без ломки текущего поведения | Тесты provider foundation + отчет Stage 1 |
| Stage 2 | `ModelDownloader` работает через provider-layer и сохраняет совместимость API | Интеграционные тесты + ручной smoke `/download` + отчет Stage 2 |
| Stage 3 | Workflow-ноды загружаются из конфигурируемых пакетов с валидацией конфликтов | Тесты comfy node system + guardrails + отчет Stage 3 |
| Stage 4 | Полный P3/P4 пакет завершен, документы и e2e проверки подтверждают расширяемость | Финальный regression pack + отчет Stage 4 |

## Риски

| Risk | Impact | Mitigation | Owner [O] |
| --- | --- | --- | --- |
| Регрессия поисковой выдачи при переносе на provider routing | high: деградация `/download` | Пошаговая миграция с сохранением совместимых API и интеграционными тестами | OpenCode agent |
| Конфликты `node_id` между пакетами comfy-нод | high: падение сборки workflow | Fail-fast валидация duplicate id + диагностические сообщения при старте | OpenCode agent |
| Некорректный fallback источников моделей | medium: частичная недоступность результатов | Явная fallback policy и покрытие сценариев `all/civitai/huggingface` | OpenCode agent |
| Рост времени старта из-за multi-package discovery | medium: ухудшение UX запуска | Дедупликация пакетов, deterministic loading, ограничение объема диагностики | OpenCode agent |

## Незапланированные дополнительные задачи

| Date | Task | Why needed | Resolution |
| --- | --- | --- | --- |
| 2026-02-25 | Добавить динамическую загрузку provider-слоя без модификации ядра | Follow-up запрос пользователя во время Stage 1 | done (реализовано в Stage 1 foundation) |

If none:

- N/A (выполнена одна дополнительная задача в рамках Stage 1).

## Журнал одобрений этапов

| Date | Event | Reference |
| --- | --- | --- |
| 2026-02-25 | Plan created; awaiting approval | `docs/dev/tasks/provider_workflow_extensions/2026-02-25_provider_workflow_extensions_plan.md` |
| 2026-02-25 | Stage 1 implemented; awaiting explicit user approval | `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-1_model-source-provider-foundation_report.md` |
| 2026-02-25 | Stage 1 дополнен dynamic-loading provider extension; ожидается явное одобрение | `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-1_model-source-provider-foundation_report.md` |
| 2026-02-25 | Stage 1 approved by user | User message: `Этап 1 одобрен` |
| 2026-02-25 | Stage 1 committed | `2832db9` |
| 2026-02-25 | Stage 2 implemented; awaiting explicit user approval | `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-2_model-downloader-provider-integration_report.md` |
| 2026-02-25 | Stage 2 approved by user | User message: `Этап 2 одобрен` |
| 2026-02-25 | Stage 2 committed | `e9dce96` |
| 2026-02-25 | Stage 3 implemented; awaiting explicit user approval | `docs/dev/tasks/provider_workflow_extensions/reports/2026-02-25_stage-3_workflow-extension-contracts_report.md` |
| 2026-02-25 | Stage 3 approved by user | User message: `Этап 3 одобрен` |
