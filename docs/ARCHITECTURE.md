# Архитектура проекта

## Общая схема

ComfyBot работает как Telegram-бот на `aiogram`, который оркестрирует генерацию через ComfyUI API.

Поток запроса выглядит так:

1. Пользователь запускает команду или нажимает кнопку в Telegram.
2. Handler-модули формируют состояние сценария и параметры генерации.
3. `ComfyUIClient` собирает workflow и отправляет задачу в ComfyUI (`/prompt`).
4. Бот отслеживает выполнение через WebSocket и polling (`/history`).
5. Готовые изображения загружаются через `/view` и отправляются пользователю.

## Слои и зоны ответственности

- `domain/`
  - Предметные сущности и политики (например, `base_model_policy`, типизированные LoRA-объекты).
- `application/`
  - Use-case логика и сервисная оркестрация (генерация, поиск моделей, локализация и т.д.).
- `infrastructure/`
  - Адаптеры внешних систем (ComfyUI transport/execution/workflow builder, файловые и API-репозитории).
- `presentation/`
  - Транспортные и UI-адаптеры (используются там, где требуется отдельный слой представления).
- `core/`
  - Общие runtime-контракты и UI-утилиты, которые переиспользуются в handler-слое.
- `handlers/`
  - Telegram-сценарии и пользовательские flow: команды, callback-ветки, экранные переходы.

## Ключевые модули

- `bot.py`
  - Точка входа.
  - Инициализирует приложение, настраивает команды Telegram и запускает polling.

- `app_context.py`
  - Композиция зависимостей приложения (`Bot`, `Dispatcher`, сервисы, runtime).
  - Единый lifecycle для закрытия внешних сессий.

- `config.py`
  - Загрузка и нормализация переменных окружения.
  - Формирование объекта `Config` и вычисляемых путей к каталогам моделей.

- `comfyui_client.py`
  - Взаимодействие с ComfyUI API.
  - Обновление доступных ресурсов (`/object_info`), выполнение workflow и получение результатов.

- `model_downloader.py`
  - Поиск и загрузка моделей из CivitAI/HuggingFace.
  - Подбор целевых файлов и синхронизация локального индекса метаданных.

- `smart_prompt.py`
  - Интеграция Smart Prompt на базе локальной модели TIPO.
  - Преобразование естественного языка в структуру positive/negative.

## Handler-карта

- `handlers/registry.py`
  - Централизованная регистрация всех handler-модулей.

- `handlers/common.py` + специализированные модули (`handlers/common_core_handlers.py`, `handlers/common_jobs_handlers.py`, `handlers/common_delete_handlers.py`, `handlers/common_middleware.py`)
  - Главное меню, сервисные команды, middleware доступа, общие пользовательские панели.

- `handlers/prompt_editor.py` + подмодули редактора (`handlers/prompt_editor_subhandlers.py`, `handlers/prompt_editor_generation.py`, `handlers/prompt_editor_send.py`, `handlers/prompt_editor_enhancement.py`)
  - Редактор генерации: ввод параметров, LoRA, референсы, Smart Prompt, запуск и выдача результата.

- `handlers/download.py` + модули download-flow (`handlers/download_flow_handlers.py`, `handlers/download_flow_state.py`, `handlers/download_flow_utils.py`, `handlers/download_flow_version_view.py`)
  - Пошаговый flow поиска и загрузки моделей с фильтрами, версиями и навигацией.

- `handlers/presets.py` + модули preset-flow (`handlers/presets_storage.py`, `handlers/presets_ui.py`, `handlers/presets_flow.py`)
  - CRUD пресетов и связанный UI/flow.

## Runtime-состояние

`RuntimeStore` хранит пользовательское состояние и рабочие артефакты:

- `active_prompt_requests` — активные сессии редактора.
- `last_params`, `last_seeds` — данные последней генерации для `/repeat`.
- `user_preferences` — настройки пользователя (режим UI, локаль и др.).
- `user_ui_panels` — привязка пользовательских панелей интерфейса.
- `preview_artifacts` — артефакты превью и параметры улучшения.
- `active_generations` — метаданные активных/восстановленных задач генерации.
- `active_downloads`, `active_image_jobs` — текущие асинхронные задачи процесса.

### Что восстанавливается после рестарта

Слой persistence (`core/runtime_snapshot.py`, `core/runtime_persistence.py`) восстанавливает:

- `active_prompt_requests`;
- `last_params` и `last_seeds`;
- `user_preferences` и `user_ui_panels`;
- `preview_artifacts`;
- `active_generations` как восстановленные записи (`restored=True`) без живых `asyncio.Task`.

Живые объекты задач (`active_downloads`, `active_image_jobs`, `task` внутри `active_generations`) не переносятся между процессами и создаются заново в новых сессиях.

## Персистентные данные

- `presets/<telegram_user_id>.json` — пользовательские пресеты.
- `sessions/runtime.json` — снимок runtime-состояния.
- `<COMFYUI_MODELS_PATH>/.comfybot_model_index.json` — локальный индекс загруженных моделей.

## Особенности генерации

- Референсы объединяются в единый входной артефакт перед отправкой в ComfyUI.
- При доступном IP-Adapter используется режим `ipadapter`, иначе применяется `img2img` fallback.
- LoRA-цепочка хранится и нормализуется через типизированные объекты (`EditorLoraSelection` и связанные модели).
- Улучшения (`Hi-res`, `FreeU`, `PAG`, upscaler-пути) управляются из editor flow и runtime preview-артефактов.

## Проверка архитектурных границ

- Границы импортов и layer-правила контролируются тестом `tests/test_architecture_boundaries.py`.
- Контрактные ограничения по callback/runtime покрываются тестами:
  - `tests/test_callback_payload_contracts.py`
  - `tests/test_handler_callback_contracts.py`
  - `tests/test_runtime_session_contracts.py`
  - `tests/test_refactor_completion_contracts.py`

## Где расширять проект

- Новый источник моделей: `model_downloader.py` и адаптеры в `infrastructure/`.
- Новый параметр генерации: доменная/прикладная модель + UI в handler-модулях редактора + сборка workflow в `comfyui_client.py`.
- Новые команды меню: `handlers/common.py` и профильные handler-модули.
- Новые локали интерфейса: `locales/<locale>/locale.meta.json` и `locales/<locale>/messages.json`.
