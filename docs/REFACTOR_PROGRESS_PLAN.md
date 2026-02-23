# План итеративного рефакторинга и прогресс

Этот файл — единый источник статуса по нашему плану рефакторинга.

## Протокол работы

1. Работаем по этапам, каждый этап безопасно завершаемый.
2. После реализации этапа:
   - я даю отчет,
   - указываю, что проверить вручную,
   - жду ваше `ок`.
3. После `ок` этап считается закрытым и помечается в этом файле.
4. По вашему отдельному запросу делается commit этапа.

## Сводка статуса

- Всего этапов: **15**
- Закрыто: **14**
- В работе: **0**
- Ожидают выполнения: **1**

---

## Этапы 1-10 (выполнено)

### Этап 1 — Страховочная сетка рефакторинга (контрактные тесты)
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `c061680`
- Цель: зафиксировать поведенческие контракты перед глубокими изменениями.
- Сделано:
  - Контракты callback payload: `tests/test_callback_payload_contracts.py`
  - Контракты runtime/session schema + миграции: `tests/test_runtime_session_contracts.py`
  - Контракты критичных callback для flow: `tests/test_handler_callback_contracts.py`
- Что защищает: формат callback, ключи persisted runtime, базовые UI-flow контракты.

### Этап 2 — Composition Root и DIP в точке входа
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `2b30e3a`
- Цель: отделить bootstrap/wiring от runtime-логики, упростить DI.
- Сделано:
  - Вынесены app services/telegram stack: `app_context.py`
  - Введен deps-контейнер регистрации хендлеров: `handlers/registry.py`
  - Стабилизирован контракт команд бота: `bot.py`, `tests/test_bot_commands_contracts.py`
- Что улучшено: ясная композиция зависимостей, меньшая сцепка в entrypoint.

### Этап 3 — Декомпозиция transport-слоя ComfyUI
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `f8e2e9c`
- Цель: вынести HTTP/WS/session в изолированный адаптер.
- Сделано:
  - Новый transport adapter: `infrastructure/comfy_transport.py`
  - `ComfyUIClient` переведен на transport-контракт: `comfyui_client.py`
  - Контрактные тесты transport wiring: `tests/test_comfyui_transport_layer.py`
- Что улучшено: ComfyUIClient меньше зависит от raw networking.

### Этап 4 — Декомпозиция orchestration в ComfyUI
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `af53591`
- Цель: отделить run/wait/collect orchestration от клиента.
- Сделано:
  - Новый orchestrator: `infrastructure/comfy_execution_orchestrator.py`
  - Делегирование `_run_workflow_and_collect`: `comfyui_client.py`
  - Тесты realtime/polling/dedup: `tests/test_comfy_execution_orchestrator.py`
- Что улучшено: orchestration и transport разделены, проще сопровождение.

### Этап 5 — Рефакторинг ModelDownloader (provider clients)
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `36f2992`
- Цель: убрать «комбайн» сетевых вызовов из downloader.
- Сделано:
  - Выделены провайдеры/файл-даунлоадер: `infrastructure/model_source_clients.py`
  - `ModelDownloader` делегирует CivitAI/HF/transfer в адаптеры: `model_downloader.py`
  - Тесты адаптеров: `tests/test_model_source_clients.py`
- Что улучшено: SRP и DIP в download-подсистеме.

### Этап 6 — Рефакторинг Smart Prompt (application vs infrastructure)
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `9b9415d`
- Цель: отделить TIPO backend runtime от прикладной оркестрации.
- Сделано:
  - Выделен backend adapter: `infrastructure/tipo_backend.py`
  - `SmartPromptService` переведен на backend protocol/DI: `smart_prompt.py`
  - Тесты orchestration сервиса: `tests/test_smart_prompt_service.py`
- Что улучшено: SmartPromptService стал application-слоем, инфраструктура вынесена.

### Этап 7 — Декомпозиция runtime/persistence
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `641e58a`
- Цель: разделить state, snapshot/migration и persistence adapter.
- Сделано:
  - Snapshot/migration: `core/runtime_snapshot.py`
  - Persistence adapter: `core/runtime_persistence.py`
  - Совместимый фасад состояния: `core/runtime.py`
  - Тесты adapter: `tests/test_runtime_persistence_adapter.py`
- Что улучшено: runtime-модуль разгружен, схема и IO изолированы.

### Этап 8 — Core UI/utility cleanup
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `6d55da3`
- Цель: снизить смешение и дубли в UI summary и image utils.
- Сделано:
  - Вынесены summary builders: `core/ui_summary.py`
  - В `core/ui.py` оставлены совместимые обертки API
  - Упрощены image helper paths: `core/image_utils.py`
  - Тесты: `tests/test_ui_summary.py`, `tests/test_image_utils.py`
- Что улучшено: DRY/KISS в core presentation/helpers.

### Этап 9 — Refactor common handlers
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `01ebc2d`
- Цель: уменьшить монолитность common-слоя handlers.
- Сделано:
  - Delete/jobs вынесены в отдельные модули:
    - `handlers/common_delete_handlers.py`
    - `handlers/common_jobs_handlers.py`
  - Middleware доступа/auto-persist вынесен: `handlers/common_middleware.py`
  - Настройки вынесены в pure parsing logic: `handlers/common_settings_logic.py`
  - Совместимый фасад сохранен: `handlers/common_ops_handlers.py`
  - Тесты parsing logic: `tests/test_common_settings_logic.py`
- Что улучшено: разделение ответственности в common handlers.

### Этап 10 — Refactor download flow handlers
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `92a2757`
- Цель: декомпозировать state/view части download flow и убрать дубли.
- Сделано:
  - State helpers: `handlers/download_flow_state.py`
  - Version selection view builder: `handlers/download_flow_version_view.py`
  - `download_flow_handlers.py` переведен на новые helpers
  - Тесты: `tests/test_download_flow_state.py`, `tests/test_download_flow_version_view.py`
- Что улучшено: меньше дублирования в переходах результата/версий и пагинации.

---

## Этапы 11-15

### Этап 11 — Prompt editor wiring и ISP
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `844ee11`
- Цель: уменьшить «god-deps» и перегруженный wiring в prompt editor.
- Сделано:
  - `PromptEditorSubhandlersDeps` переведен с «плоского» god-deps на узкие feature-deps (`flow/smart/edit/exchange/thematic/lora/references/send`): `handlers/prompt_editor_subhandlers.py`
  - Сборка deps для каждой feature-группы вынесена в composition root prompt editor: `handlers/prompt_editor.py`
  - Добавлена проверка контрактного wiring-а subhandlers: `tests/test_prompt_editor_subhandlers.py`
  - Исправлена устойчивость send/generation к `TelegramNetworkError` (retry + безопасная обработка фона): `handlers/prompt_editor_send.py`, `handlers/prompt_editor_generation.py`
  - Добавлены регрессионные тесты retry/fallback доставки изображений: `tests/test_prompt_editor_send.py`
- Что улучшено: ниже сцепка и уже интерфейсы deps в prompt editor; фоновая генерация больше не падает с `Task exception was never retrieved` при кратковременных сетевых сбоях Telegram.

### Этап 12 — Prompt generation/send flows
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `75da05c`
- Цель: разделить orchestration генерации, прогресса и доставки артефактов.
- Сделано:
  - Вынесен enhancement-runner из `handlers/prompt_editor_handlers_send.py` в отдельный модуль `handlers/prompt_editor_enhancement.py`.
  - В `handlers/prompt_editor_generation.py` выделены helpers для intro/status/done-текста и cancel/back клавиатур.
  - Усилены guard-ветки обновления статуса в enhancement flow при Telegram API ошибках.
- Что улучшено: жизненный цикл генерации/улучшения стал предсказуемее, а send-handlers заметно уменьшились и лучше разделены по ответственности.

### Этап 13 — Prompt edit/thematic/lora/references/session cleanup
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `da947aa`
- Цель: разукрупнить крупные handler-модули редактора.
- Сделано:
  - Централизованы callback-guards в `handlers/prompt_editor_handler_guards.py` и подключены в edit/thematic/lora/references handlers.
  - Централизован scalar parsing/validation в `handlers/prompt_editor_scalar_utils.py` и повторно использован в edit/thematic scalar flows.
  - Убраны дубли `require_prompt_request_for_callback + require_callback_message` в `handlers/prompt_editor_handlers_edit.py` и `handlers/prompt_editor_handlers_thematic.py`.
  - Референсные и LoRA callback flows переведены на единый guard-паттерн: `handlers/prompt_editor_handlers_references.py`, `handlers/prompt_editor_handlers_lora.py`.
  - Нормализация runtime-параметров декомпозирована в `handlers/prompt_editor_session.py` (clamp helpers + reference normalization helpers).
  - Добавлены тесты на новые utility/guard/session-paths: `tests/test_prompt_editor_handler_guards.py`, `tests/test_prompt_editor_scalar_utils.py`, `tests/test_prompt_editor_session.py`.
- Ожидаемый результат: выше cohesion, ниже coupling, меньше дублирования.

### Этап 14 — Presets и финальная интеграционная стабилизация
- Статус: ✅ Закрыт, ✅ Закоммичен
- Commit: `<будет заполнено после коммита этапа 14>`
- Цель: упорядочить preset-flow и cross-module orchestration.
- Сделано:
  - Выделены preset storage adapters: `handlers/presets_storage.py` (`load/save/resolve/normalize` helpers).
  - Выделены preset UI builders: `handlers/presets_ui.py` (списки/кнопки/empty-state/overwrite UI).
  - Выделены overwrite-flow helpers: `handlers/presets_flow.py` (state payloads, parse callback data, cleanup temp messages).
  - `handlers/presets.py` упрощен: основной routing-flow опирается на split helpers вместо локальных монолитных utility-блоков.
  - Добавлены тесты на presets split-модули: `tests/test_presets_storage.py`, `tests/test_presets_ui.py`, `tests/test_presets_flow.py`.
- Ожидаемый результат: более надежная работа пресетов и меньше скрытых связей.

### Этап 15 — Финальный проход и закрытие техдолга
- Статус: ⏳ Ожидает выполнения
- Цель: закрыть остаточные низкоприоритетные пункты и зафиксировать финальное состояние.
- План:
  - Финальный cleanup по `docs/ARCH_ANALYSIS.md`
  - Обновить сводную документацию по границам архитектуры и итогу рефакторинга
  - Выполнить полный regression-check по сценариям R01–R13
- Ожидаемый результат: завершенный цикл рефакторинга с документированным итогом.

---

## Правило обновления этого файла

После каждого закрытого этапа я обновляю:

1. Статус этапа (ожидает -> закрыт).
2. Commit hash (если commit выполнен).
3. Краткую сводку «что сделано» и «что улучшено».
4. Общую сводку прогресса (сколько закрыто/осталось).
