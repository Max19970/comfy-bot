# Отчет регрессии R01-R13

Дата: 2026-02-23

## Что проверяли

Полный regression-check по матрице `R01-R13` из `docs/UI_MIGRATION_PLAN.md`.

Запущенная команда:

```bash
python -m pytest tests/test_bot_commands_contracts.py tests/test_handler_callback_contracts.py tests/test_prompt_editor_subhandlers.py tests/test_prompt_editor_selection_utils.py tests/test_prompt_editor_lora_helpers.py tests/test_prompt_editor_handler_guards.py tests/test_prompt_editor_scalar_utils.py tests/test_prompt_editor_session.py tests/test_prompt_editor_send.py tests/test_prompt_editor_send_menu_utils.py tests/test_prompt_editor_enhancements.py tests/test_prompt_generation_use_case.py tests/test_prompt_exchange.py tests/test_prompt_request_loras.py tests/test_smart_prompt_service.py tests/test_smart_prompt_text.py tests/test_presets_flow.py tests/test_presets_storage.py tests/test_presets_ui.py tests/test_storage_params.py tests/test_download_flow_state.py tests/test_download_flow_version_view.py tests/test_download_flow_utils.py tests/test_download_filters.py tests/test_download_search_use_case.py tests/test_model_source_clients.py tests/test_runtime_store.py tests/test_runtime_persistence_adapter.py tests/test_runtime_session_contracts.py tests/test_refactor_completion_contracts.py tests/test_queue_utils.py tests/test_common_settings_logic.py
```

Результат: `89 passed in 3.16s`.

## Матрица покрытия

- `R01` (`/start`, меню, базовые переходы)
  - `tests/test_bot_commands_contracts.py`
  - `tests/test_handler_callback_contracts.py`
- `R02` (`/generate`, editor open/back/cancel/menu)
  - `tests/test_handler_callback_contracts.py`
  - `tests/test_prompt_editor_subhandlers.py`
- `R03` (редактирование `Positive/Negative`, сохранение состояния)
  - `tests/test_prompt_generation_use_case.py`
  - `tests/test_prompt_exchange.py`
  - `tests/test_prompt_editor_session.py`
- `R04` (size/steps/cfg/seed/batch/sampler/scheduler)
  - `tests/test_prompt_editor_selection_utils.py`
  - `tests/test_prompt_editor_scalar_utils.py`
  - `tests/test_prompt_editor_session.py`
- `R05` (LoRA add/remove/clear/confirm)
  - `tests/test_prompt_editor_lora_helpers.py`
  - `tests/test_prompt_request_loras.py`
- `R06` (references add/preview/remove/clear)
  - `tests/test_prompt_editor_session.py`
  - `tests/test_prompt_editor_send_menu_utils.py`
- `R07` (Smart Prompt)
  - `tests/test_smart_prompt_service.py`
  - `tests/test_smart_prompt_text.py`
- `R08` (generation run/progress/cancel)
  - `tests/test_prompt_generation_use_case.py`
  - `tests/test_prompt_editor_send.py`
  - `tests/test_prompt_editor_enhancements.py`
- `R09` (result screen: new/cancel/menu)
  - `tests/test_handler_callback_contracts.py`
  - `tests/test_prompt_editor_send.py`
- `R10` (presets save/overwrite/load/delete)
  - `tests/test_presets_flow.py`
  - `tests/test_presets_storage.py`
  - `tests/test_presets_ui.py`
  - `tests/test_storage_params.py`
- `R11` (download-flow: type/source/filter/search/confirm/cancel)
  - `tests/test_download_flow_state.py`
  - `tests/test_download_flow_version_view.py`
  - `tests/test_download_flow_utils.py`
  - `tests/test_download_filters.py`
  - `tests/test_download_search_use_case.py`
  - `tests/test_model_source_clients.py`
- `R12` (`/models`, `/queue`, `/jobs`, `/settings`, `/cancel`)
  - `tests/test_bot_commands_contracts.py`
  - `tests/test_queue_utils.py`
  - `tests/test_common_settings_logic.py`
- `R13` (restart + runtime restore)
  - `tests/test_runtime_store.py`
  - `tests/test_runtime_persistence_adapter.py`
  - `tests/test_runtime_session_contracts.py`
  - `tests/test_refactor_completion_contracts.py`

## Итог

- Автоматизированная регрессия по матрице R01-R13 прошла без падений.
- Контрактные boundary/runtime/callback проверки остаются зелеными.
