# Отчет по завершению Stage 2: debt-catalog

> [!IMPORTANT]
> Отчет создан на основе `docs/dev/tasks/_templates/stage_completion_report_template.md`.
>
> Перед запросом одобрения:
> 1. Все обязательные секции заполнены.
> 2. В `Changed artifacts` указаны точные пути.
> 3. Все placeholder-значения заменены.

---

## Navigation

- [Quick report checklist](#quick-report-checklist)
- [Stage metadata](#stage-metadata)
- [What was implemented](#what-was-implemented)
- [Self-verification results](#self-verification-results)
- [Manual verification steps](#manual-verification-steps)
- [Changed artifacts](#changed-artifacts)
- [Approval request](#approval-request)

## Section legend

| Marker | Meaning |
| --- | --- |
| `[M]` | Mandatory section |
| `[O]` | Optional section |

## Quick report checklist [M]

- [x] Filled stage metadata and related plan path.
- [x] Listed concrete implemented changes.
- [x] Added self-verification outcomes with clear pass/fail status.
- [x] Added exact manual verification steps.
- [x] Listed changed artifact paths.
- [x] Requested explicit approval.

## Stage metadata

| Field | Value |
| --- | --- |
| Task | `core_architecture_cleanup` |
| Work type | `mixed-single-topic` |
| Stage number | `2` |
| Stage short name | `debt-catalog` |
| Date | `2026-02-24` |
| Related plan | `docs/dev/tasks/core_architecture_cleanup/2026-02-24_core_architecture_cleanup_plan.md` |

## What was implemented

- Проведен доказательный аудит архитектурных узких мест в слоях `handlers`, `core`, `application`, `infrastructure` с фокусом на поддерживаемость, расширяемость и стоимость изменений.
- Сформирован приоритизированный каталог архитектурного долга с полями: проблема, влияние, приоритет, затронутые модули, путь устранения.
- Для каждой находки добавлены конкретные точки в коде (модули и/или участки) и практический remediation-path для Stage 3 roadmap.
- По follow-up запросу добавлен отдельный анализ расширения без модификации ядра и текущих механизмов динамической загрузки.

### Расширение без модификации ядра и динамические загрузки

| Зона | Текущий статус | Доказательная привязка | Вывод для расширения |
| --- | --- | --- | --- |
| UI text modifiers pipeline | `готово` | `core/config.py:85`, `bootstrap/app_context.py:65`, `infrastructure/ui_text_pipeline.py:15` | Можно подключать новые модификаторы через `UI_TEXT_MODIFIER_FACTORIES` без правок core-кода. |
| Comfy workflow node discovery | `готово (внутри кодовой базы)` | `infrastructure/comfy_nodes/registry.py:29`, `infrastructure/comfy_nodes/builder.py:17`, `tests/test_comfy_node_system.py:13` | Новые workflow-ноды добавляются как отдельные модули с `register_nodes`, без изменения существующих node-файлов. |
| Локализация и UI copy profiles | `готово` | `infrastructure/localization_catalog.py:18`, `infrastructure/ui_text_resources.py:103` | Новые локали/профили подключаются через файловые ресурсы JSON без модификации ядра. |
| Override package для comfy nodes | `частично готово` | `infrastructure/comfy_nodes/builder.py:14`, `infrastructure/comfy_workflow_builder.py:60` | Технический seam есть (`package_name`), но не выведен в runtime-конфиг/плагин-контракт. |
| Smart Prompt backend loading | `частично готово` | `infrastructure/tipo_backend.py:178`, `application/smart_prompt_service.py:110` | Динамическая загрузка зависимостей есть, но provider-фасад фактически зафиксирован на TIPO. |
| Handler extension model | `не готово` | `handlers/registry.py:16`, `handlers/registry.py:39` | Регистрация handler-модулей статическая; plugin-discovery для сценариев Telegram отсутствует. |
| Model source extension model | `не готово` | `application/model_downloader.py:52`, `application/model_downloader.py:1306` | Источники моделей жестко зашиты (`civitai/huggingface/all`), добавление нового источника требует изменения ядра. |

#### Практический вывод Stage 2 по запросу пользователя

- Без модификации ядра сейчас реально расширяются: `UI text pipeline`, локализация/копирайт-профили, workflow-ноды в существующей discoverable-папке.
- Для полноценных внешних динамических расширений (plugins вне ядра) отсутствуют устойчивые контракты в двух критичных зонах: `handlers` и `model sources`.
- Приоритет Stage 3/4: перевести эти зоны на plugin-capable порты и runtime-discovery, чтобы новые функции подключались конфигурацией/entrypoint-ами, а не редактированием core-файлов.

### Каталог архитектурного долга (Stage 2)

| ID | Проблема | Приоритет | Влияние на сопровождение | Доказательная привязка | Путь устранения |
| --- | --- | --- | --- | --- | --- |
| A-01 | Монолитные handler-регистраторы и god-функции | `P0` | Резко растет риск регрессий, конфликтов при параллельной разработке, стоимость локальных изменений | `handlers/common_core_handlers.py:67`, `handlers/prompt_editor_handlers_send.py:66`, `handlers/download.py:883`; длина регистраторов: `2580/1924/600` строк | Декомпозировать по feature-controller модулям (menu/settings/jobs/download/send), ограничить размер файла/функции, выносить pure UI/state helpers в отдельные пакеты |
| A-02 | Глобальный mutable `RuntimeStore` с прямыми мутациями из handler-слоя | `P0` | Сильная связанность, сложная диагностика race/состояния, низкая тестируемость | `core/runtime.py:134`, прямые обращения по `handlers/*` (`89` вхождений по ключевым runtime-коллекциям), мутации в `handlers/prompt_editor_generation.py:559`, `handlers/common_jobs_handlers.py:193` | Ввести typed runtime gateways (`PromptSessionStore`, `GenerationJobStore`, `UserPreferencesStore`) и закрыть прямой доступ к внутренним dict |
| A-03 | Персист runtime на каждый message/callback без коалесинга | `P1` | Избыточный I/O и потенциальное ухудшение latency под нагрузкой | `handlers/common_middleware.py:71`, `handlers/common_middleware.py:94`, `core/runtime.py:197`, `core/storage.py:93` | Добавить debounce/write-behind persistence queue и явные transactional checkpoints на критических переходах |
| A-04 | Целевая слоистая архитектура заявлена, но миграция неполная (`presentation` фактически пуст) | `P1` | Новые изменения продолжают оседать в legacy-слоях, что тормозит архитектурное оздоровление | `presentation/__init__.py:1`, `docs/ARCHITECTURE.md:23`, `docs/REFACTOR_GUARDRAILS.md:29` | Ввести реальный `presentation` composition (controllers/view models), запретить новый feature-код в legacy-пакетах через guardrails |
| A-05 | `ComfyUIClient` совмещает transport/orchestration/progress/workflow fallback в одном классе | `P1` | Трудно локализовать сбои и безопасно расширять pipeline | `infrastructure/comfyui_client.py` (`1293` строки), `_stream_progress_via_websocket` (`593-877`), `wait_for_completion` (`942-1086`), fallback `ipadapter -> img2img` (`1256-1265`) | Разделить на `ComfyTransportGateway`, `ProgressTracker`, `HistoryPoller`, `GenerationFacade`; заменить broad fallback на capability-driven strategy |
| A-06 | `ModelDownloader` объединяет use-case, внешние API, файловую инфраструктуру и форматирование | `P1` | Затруднена модульная эволюция источников моделей и изоляция тестов | `application/model_downloader.py` (`1404` строки), методы `search_civitai` (`651-793`), `download_model` (`1145-1284`), прямые импорты `infrastructure/*` (`22-29`) | Выделить порты в application и адаптеры в infrastructure; оставить в use-case только orchestration/политику |
| A-07 | Callback-контракты частично типизированы, но в основном строковые литералы | `P2` | Хрупкость payload-протокола, высокая вероятность тихих ошибок при рефакторинге | `core/callbacks.py` покрывает только 3 схемы; в handlers выявлено `176` определений `callback_data` | Ввести типизированные callback schemas для bounded contexts (`download/editor/jobs`) и генерацию парсеров/констант |
| A-08 | Dependency contracts размыты из-за `Any` и перегруженных `*Deps` | `P2` | Слабые compile-time гарантии, рост скрытых wiring-ошибок | `handlers/download_flow_handlers.py:36`, `handlers/common_core_handlers.py:46`, `handlers/common_delete_handlers.py:22`, `handlers/common_jobs_handlers.py:19` | Заменить `Any` на Protocol-интерфейсы, сократить deps-пакеты до узких feature-контрактов |
| A-09 | Дублирование и рассинхрон helper-логики (пример: author filters и локализационные `_t`) | `P2` | Повышается вероятность несогласованного поведения и drift между модулями | `application/download_search_use_case.py:101` и `handlers/download_flow_utils.py:38`; `def _t(` встречается в `30` модулях | Централизовать helper-функции в shared services и оставить в feature-слоях только orchestration calls |
| A-10 | Legacy/compatibility слой сохранен без явного deprecation-контракта | `P3` | Долгосрочное удвоение поддержки и неясные границы публичного API | `bot_core.py:1`, отсутствие внутренних импортов (`grep` по репозиторию), wrapper-проверки в `tests/test_refactor_completion_contracts.py:30` | Зафиксировать deprecation-policy и жизненный цикл legacy API; ограничить новые зависимости на compatibility-фасад |

### Правило приоритизации

- Использована risk-модель: `Приоритет = (Влияние на стабильность и скорость изменений) x (Частота затрагивания flow) x (Стоимость безопасного изменения)`.
- Критичные для стартового рефакторинга зоны: `A-01`, `A-02`, `A-03`, `A-05`, `A-06`.

## Self-verification results

### Verification snapshot [M]

| Metric | Value |
| --- | --- |
| Total checks | `6` |
| Passed | `6` |
| Failed | `0` |
| Final status | `pass` |

### Check summary [M]

| Check | Command or method | Result | Notes |
| --- | --- | --- | --- |
| Guardrail: архитектурные границы | `python -m pytest -q tests/test_architecture_boundaries.py` | pass | Базовые import-границы не нарушены, что подтверждает корректность baseline перед каталогом долга. |
| Guardrail: callback payload contracts | `python -m pytest -q tests/test_callback_payload_contracts.py tests/test_handler_callback_contracts.py` | pass | Текущие callback-контракты стабильны и могут использоваться как миграционный safety net. |
| Guardrail: runtime session contracts | `python -m pytest -q tests/test_runtime_session_contracts.py` | pass | Контракт сериализации runtime подтвержден для планирования безопасной эволюции state-слоя. |
| Анализ распределения сложности по слоям | `python -c "..."` (агрегация строк/файлов по слоям) | pass | Подтверждена концентрация логики в `handlers` (`20080` строк). |
| Анализ крупных функций | `python -c "..."` (AST-анализ длины функций) | pass | Подтверждены сверхкрупные регистраторы и hotspot-функции в `handlers`/`infrastructure`. |
| Анализ связности runtime и callback-литералов | `python -c "..."` + `grep` по ключевым паттернам | pass | Подтверждены массовые прямые мутации runtime-state и фрагментация callback-схем. |

### Defects found and resolved [O]

- No defects found during this stage.

## Manual verification steps

1. Пройдите пользовательский сценарий генерации: `/generate` -> настройка checkpoint/LoRA/reference -> запуск -> просмотр превью/меню улучшений.
2. Пройдите сценарий загрузки моделей: `/download` -> выбор типа/источника -> настройка фильтров -> поиск -> выбор результата -> подтверждение скачивания.
3. Во время активной генерации перезапустите бота и проверьте сценарий восстановления задач через `/jobs` и повторный вход в рабочие меню.

Ожидаемый результат: все сценарии работают, но наблюдается сложный и разветвленный UX/flow-контур, что подтверждает корректность приоритизации архитектурного долга для Stage 3.

## Changed artifacts

| Artifact path | Change type | Purpose |
| --- | --- | --- |
| `docs/dev/tasks/core_architecture_cleanup/reports/2026-02-24_stage-2_debt-catalog_report.md` | added | Отчет Stage 2 с каталогом архитектурного долга и приоритизацией. |
| `docs/dev/tasks/core_architecture_cleanup/2026-02-24_core_architecture_cleanup_plan.md` | updated | Дополнен журнал одобрений (одобрение и commit Stage 1). |

## Unplanned additional tasks executed [O]

| Task | Why needed | Plan log reference | Outcome |
| --- | --- | --- | --- |
| Анализ расширения без модификации ядра и динамических загрузок | Пользовательский follow-up в рамках Stage 2 | `docs/dev/tasks/core_architecture_cleanup/2026-02-24_core_architecture_cleanup_plan.md` (`Unplanned additional tasks`) | Выполнено, добавлено в отчет Stage 2 |

## Known limitations or follow-up notes [O]

- Каталог Stage 2 фиксирует приоритеты и векторы исправления; детальная последовательность миграции и инкременты рефакторинга будут формализованы в Stage 3.

## Approval request

| Item | Value |
| --- | --- |
| Status | `Awaiting explicit user approval for Stage 2` |
| Next action after approval | commit Stage 2 и переход к Stage 3 (целевая архитектура и стратегия миграции) |
