# Отчет по завершению Stage 3: target-architecture-roadmap

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
| Stage number | `3` |
| Stage short name | `target-architecture-roadmap` |
| Date | `2026-02-24` |
| Related plan | `docs/dev/tasks/core_architecture_cleanup/2026-02-24_core_architecture_cleanup_plan.md` |

## What was implemented

- Спроектирована целевая архитектура с приоритетом расширения без модификации ядра: plugin contracts, runtime discovery и постепенная миграция legacy-слоев.
- Найденные проблемы Stage 2 сгруппированы в целевые workstreams с зависимостями, инкрементами и критериями безопасного внедрения.
- Подготовлен приоритизированный roadmap (quick wins -> среднесрочные -> долгосрочные шаги) с анти-регрессионными проверками.

### Целевое архитектурное состояние

| Слой/контур | Целевая модель | Ключевой результат |
| --- | --- | --- |
| Composition root | `bootstrap` создает registry плагинов и контейнер портов | Новые расширения подключаются конфигом (`plugin packages`) без правок ядра |
| Presentation | `presentation.telegram` как модульный набор plugin-controller пакетов | Handler-фичи подключаются через `HandlerPlugin` контракты, не через статический `handlers/registry.py` |
| Application | Use-case orchestration + порты (`ModelSourceProvider`, `SmartPromptProvider`, runtime stores) | Добавление нового источника/провайдера не требует изменения existing use cases |
| Infrastructure | Адаптеры внешних API + динамически подключаемые provider packages | Внешние интеграции эволюционируют независимо от бизнес-логики |
| Workflow pipeline | Comfy node packages с runtime-discovery и валидацией конфликтов | Новые workflow ноды добавляются отдельными пакетами/модулями |
| Runtime state | Typed facade поверх runtime state и отложенный persist | Снижение связанности handler-кода и I/O нагрузки |

### Workstreams и последовательность миграции

| WS | Охват Stage 2 проблем | Основные действия | Порядок внедрения |
| --- | --- | --- | --- |
| WS-1: Extension Kernel | `A-04`, `A-08`, `A-10` | Ввести `PluginDescriptor`/`PluginRegistry`, discovery через список пакетов, capability checks | `1` (foundation, без изменения поведения)
| WS-2: Handler Pluginization | `A-01`, `A-08` | Перевести регистрацию Telegram-фич на `HandlerPlugin` интерфейсы, выделить bounded handlers | `2` (после WS-1)
| WS-3: Model Source Providers | `A-06`, `A-09` | Вынести CivitAI/HF в `ModelSourceProvider`, добавить provider registry и source routing | `3` (после WS-1)
| WS-4: Workflow Extension Surface | `A-05`, `A-07` | Конфигурируемый список node packages, конфликт-валидация node_id, capability flags | `4` (параллельно WS-3, после WS-1)
| WS-5: Runtime Facade + Persistence Queue | `A-02`, `A-03` | Typed runtime gateways, write-behind/debounce persist, транзакционные checkpoints | `5` (после WS-2, перед финальной оптимизацией)
| WS-6: Contract Governance | `A-07`, `A-10` | Версионирование plugin API, contract tests, deprecation-policy | `сквозной` (старт в WS-1, завершение к release)

### Инкрементный план (без остановки разработки)

| Инкремент | Изменение | Anti-regression guard |
| --- | --- | --- |
| I-1 | Добавить `plugins/contracts.py` и `plugins/loader.py` с no-op встроенными плагинами | `tests/test_architecture_boundaries.py`, smoke `/start` |
| I-2 | Обернуть текущие `register_common/prompt_editor/presets/download` в built-in `HandlerPlugin` адаптеры | `tests/test_handler_registry_bridge.py`, callback-contract тесты |
| I-3 | Ввести `ModelSourceProvider` и адаптеры CivitAI/HF, оставить старый API `ModelDownloader.search()` как фасад-совместимость | `tests/test_download_search_use_case.py`, ручной `/download` flow |
| I-4 | Открыть runtime-конфиг для node packages (`COMFY_NODE_PACKAGES`) + валидацию дубликатов `node_id` | `tests/test_comfy_node_system.py`, генерация с обычным и upscale workflow |
| I-5 | Ввести `RuntimeSessionStore` facade и отложенный persist queue за feature flag | `tests/test_runtime_session_contracts.py`, сценарий перезапуска и восстановления `/jobs` |
| I-6 | Включить plugin governance (API version field, deprecation warnings, compatibility matrix) | contract tests + release checklist |

### Roadmap приоритизации

| Горизонт | Задачи | Результат |
| --- | --- | --- |
| Quick wins (1-2 спринта) | WS-1 + начало WS-2 (I-1, I-2) | Расширение handler-фич через plugin registration без прямой правки `handlers/registry.py` |
| Среднесрочно (2-4 спринта) | WS-3 + WS-4 (I-3, I-4) | Новые model sources и workflow nodes подключаются через provider/node packages |
| Долгосрочно (4+ спринтов) | WS-5 + WS-6 (I-5, I-6) | Устойчивая архитектура с контролируемой эволюцией runtime state и API-контрактов плагинов |

### Риски миграции и митигации

| Риск | Влияние | Митигация |
| --- | --- | --- |
| Дрейф callback payload при pluginization handlers | высокий | Контрактные тесты payload + backward-compat adapters в каждом инкременте |
| Несовместимые сторонние плагины | средний | Явный `api_version` и capability validation при загрузке |
| Регресс производительности из-за extra abstraction | средний | Профилирование на hot-path, lazy plugin loading, кэширование registries |
| Состояние runtime рассинхронизируется при write-behind | высокий | Flush on critical transitions + crash-safe fsync checkpoints |

## Self-verification results

### Verification snapshot [M]

| Metric | Value |
| --- | --- |
| Total checks | `4` |
| Passed | `4` |
| Failed | `0` |
| Final status | `pass` |

### Check summary [M]

| Check | Command or method | Result | Notes |
| --- | --- | --- | --- |
| Архитектурные границы слоев | `python -m pytest -q tests/test_architecture_boundaries.py` | pass | Базовые layer-ограничения сохранены для безопасной эволюции roadmap. |
| Контракты dynamic workflow discovery | `python -m pytest -q tests/test_comfy_node_system.py` | pass | Подтверждены deterministic discovery и поддержка пакетов node-пайплайна. |
| Контракты dynamic ui-text pipeline | `python -m pytest -q tests/test_ui_text_pipeline.py` | pass | Подтверждена работоспособность factory-based модификаторов как extension seam. |
| Актуализация плана/отчета | Review артефактов Stage 3 и журнала одобрений | pass | Зафиксированы commit Stage 2 и стратегия миграции Stage 3. |

### Defects found and resolved [O]

- No defects found during this stage.

## Manual verification steps

1. Проверьте пользовательский сценарий `/generate` и `/download` до любых рефакторинг-изменений как baseline для будущей миграции.
2. Подключите кастомный UI-text modifier через `UI_TEXT_MODIFIER_FACTORIES` и убедитесь, что поведение UI меняется без правок core-файлов.
3. Добавьте тестовую comfy node в discoverable package и убедитесь, что workflow собирается и генерация продолжает работать.

Ожидаемый результат: расширения в существующих seam-зонах работают конфигурационно, а зоны без seam (handlers/model sources/runtime) четко покрыты roadmap-инкрементами.

## Changed artifacts

| Artifact path | Change type | Purpose |
| --- | --- | --- |
| `docs/dev/tasks/core_architecture_cleanup/reports/2026-02-24_stage-3_target-architecture-roadmap_report.md` | added | Отчет Stage 3 с целевой архитектурой и стратегией миграции. |
| `docs/dev/tasks/core_architecture_cleanup/2026-02-24_core_architecture_cleanup_plan.md` | updated | Обновлен журнал одобрений (одобрение/commit Stage 2). |

## Unplanned additional tasks executed [O]

| Task | Why needed | Plan log reference | Outcome |
| --- | --- | --- | --- |

## Known limitations or follow-up notes [O]

- Stage 3 формирует целевую архитектурную траекторию; implementation-пакет конкретных изменений и метрик завершения будет зафиксирован в Stage 4.

## Approval request

| Item | Value |
| --- | --- |
| Status | `Awaiting explicit user approval for Stage 3` |
| Next action after approval | commit Stage 3 и переход к Stage 4 (финальный пакет рекомендаций, метрики и проверочные сценарии) |
