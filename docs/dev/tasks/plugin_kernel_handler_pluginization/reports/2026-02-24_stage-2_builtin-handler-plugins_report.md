# Отчет по завершению Stage 2: builtin-handler-plugins

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
| Task | `plugin_kernel_handler_pluginization` |
| Work type | `refactor` |
| Stage number | `2` |
| Stage short name | `builtin-handler-plugins` |
| Date | `2026-02-24` |
| Related plan | `docs/dev/tasks/plugin_kernel_handler_pluginization/2026-02-24_plugin_kernel_handler_pluginization_plan.md` |

## What was implemented

- Реализованы built-in handler plugins в `handlers/plugins/builtin.py` для текущих bounded блоков: `common`, `prompt_editor`, `presets`, `download`.
- Добавлен явный dependency contract `HandlerPluginDeps` (Protocol) для plugin registration context, чтобы исключить неявное wiring в plugin-адаптерах.
- Интегрирован plugin-driven bridge в `handlers/registry.py`:
  - загрузка плагинов из `HANDLER_PLUGIN_PACKAGES`;
  - фильтрация по capability `handlers.registration`;
  - сохранение совместимости через fallback на legacy registration path при пустом plugin-list.
- В `HandlerPluginContext.shared` введено соглашение по shared-объектам:
  - `handlers.localization.bridge` для единообразной localization-обвязки;
  - `handlers.prompt_editor.service` для межплагинной зависимости `presets -> prompt_editor`.
- Обновлены тесты bridge-контрактов (`tests/test_handler_registry_bridge.py`) под plugin path и добавлен сценарий fallback на legacy.

## Self-verification results

### Verification snapshot [M]

| Metric | Value |
| --- | --- |
| Total checks | `2` |
| Passed | `2` |
| Failed | `0` |
| Final status | `pass` |

### Check summary [M]

| Check | Command or method | Result | Notes |
| --- | --- | --- | --- |
| Stage 2 contract/regression checks | `python -m pytest -q tests/test_architecture_boundaries.py tests/test_handler_registry_bridge.py tests/test_handler_plugin_loader.py tests/test_handler_callback_contracts.py` | pass | Подтверждены plugin bridge контракты, архитектурные ограничения и callback совместимость. |
| Lint for Stage 2 files | `python -m ruff check plugins handlers/registry.py handlers/plugins/builtin.py tests/test_handler_registry_bridge.py` | pass | Стиль и импорт-правила соблюдены для измененных модулей Stage 2. |

### Defects found and resolved [O]

- No defects found during this stage.

## Manual verification steps

1. Запустите бота (`python bot.py`) с `HANDLER_PLUGIN_PACKAGES=handlers.plugins.builtin`.
2. В Telegram проверьте пользовательский путь `/start` -> `Генерация` -> `Пресеты` -> `Скачать модель`.
3. Убедитесь, что переходы работают без ошибок и локализация сообщений/кнопок отображается корректно.
4. (Опционально) Установите `HANDLER_PLUGIN_PACKAGES=` (пусто), перезапустите бота и повторите `/start`: поведение должно сохраниться за счет legacy fallback-path.

## Changed artifacts

| Artifact path | Change type | Purpose |
| --- | --- | --- |
| `handlers/plugins/builtin.py` | updated | Реализация built-in handler plugins и shared-контрактов между плагинами. |
| `handlers/registry.py` | updated | Plugin-driven registration bridge + legacy fallback. |
| `plugins/contracts.py` | updated | Уточнен контекст регистрации (убраны широкие `Any` в context fields). |
| `tests/test_handler_registry_bridge.py` | updated | Контрактные тесты bridge-пути и legacy fallback. |
| `docs/dev/tasks/plugin_kernel_handler_pluginization/reports/2026-02-24_stage-2_builtin-handler-plugins_report.md` | added | Отчет Stage 2. |
| `docs/dev/tasks/plugin_kernel_handler_pluginization/2026-02-24_plugin_kernel_handler_pluginization_plan.md` | updated | Журнал одобрений/событий Stage 1-2. |

## Unplanned additional tasks executed [O]

| Task | Why needed | Plan log reference | Outcome |
| --- | --- | --- | --- |

## Known limitations or follow-up notes [O]

- На Stage 2 plugin bridge внедрен, но plugin-first регистрация еще не обязательна архитектурно; Stage 3 закроет окончательное переключение и guardrails на отказ от hardcoded wiring.

## Approval request

| Item | Value |
| --- | --- |
| Status | `Awaiting explicit user approval for Stage 2` |
| Next action after approval | commit Stage 2 и переход к Stage 3 (plugin-first handler registration + guardrails) |
