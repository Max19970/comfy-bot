# Архитектура проекта

## Общая схема

Проект состоит из Telegram-бота на `aiogram` и клиента ComfyUI API.

Поток выглядит так:

1. Пользователь отправляет команду/нажимает кнопку в Telegram.
2. Обработчики (`handlers/*`) меняют состояние FSM и параметры генерации.
3. `ComfyUIClient` собирает workflow под выбранные параметры.
4. Workflow отправляется в ComfyUI (`/prompt`), бот отслеживает прогресс.
5. После завершения бот скачивает изображения (`/view`) и отправляет пользователю.

## Модули и ответственность

- `bot.py`
  - Точка входа.
  - Собирает `AppContext` и запускает polling.
  - Проверяет доступность ComfyUI и управляет жизненным циклом приложения.

- `app_context.py`
  - Централизованная сборка зависимостей (`Bot`, `Dispatcher`, `Router`, сервисы).
  - Единая точка закрытия всех сетевых сессий (`close`).

- `config.py`
  - Загружает `.env`.
  - Хранит runtime-конфигурацию (`Config`).
  - Даёт вычисляемые пути к папкам моделей (`checkpoints`, `loras`, `upscale_models`).

- `core/*`
  - `core/states.py` — FSM-состояния.
  - `core/runtime.py` — runtime-хранилище (`RuntimeStore`) и структуры активных задач.
  - `core/storage.py` — сериализация параметров и пресеты.
  - `core/html_utils.py` — безопасное форматирование и компактные представления текста.
  - `core/ui.py` — совместимый UI helper-слой (summary/пагинация).
  - `core/ui_kit/*` — shared UI builders (кнопки, навигация, диалоги, пагинация).
  - `core/interaction.py` — единый callback/message interaction flow.
  - `core/callbacks.py` — типизированные callback-схемы.
  - `core/panels.py` — централизованный рендер пользовательских панелей.
  - `core/download_filters.py` — общие helper-правила фильтрации base-моделей.
  - `core/image_utils.py` — подготовка изображений под ограничения Telegram.
  - `core/telegram.py` — утилиты для извлечения user id из Message/Callback.

- `bot_core.py`
  - Совместимый facade для старых импортов (`from bot_core import ...`).
  - Делегирует реализацию в `core/*`.

- `comfyui_client.py`
  - HTTP/WebSocket взаимодействие с ComfyUI.
  - Кэш доступных моделей/самплеров/нод.
  - Сборка workflow графа под текущие параметры.
  - Очередь, прогресс, ожидание завершения, скачивание результатов.

- `model_downloader.py`
  - Поиск моделей в CivitAI/HuggingFace.
  - Выбор подходящих файлов под тип модели.
  - Скачивание в нужную папку ComfyUI.
  - Локальный индекс метаданных (`.comfybot_model_index.json`) в каталоге моделей ComfyUI.

- `smart_prompt.py`
  - Локальная интеграция TIPO для Smart Prompt.
  - Преобразование запроса на естественном языке в теговые `positive/negative`.
  - Ленивая загрузка модели и семплирование без внешних LLM API.

- `handlers/common.py`
  - Общие команды (`/start`, `/help`, `/models`, `/queue`, `/settings`, `/cancel`).
  - Whitelist middleware.

- `handlers/registry.py`
  - Единая точка регистрации всех handler-модулей.

- `handlers/prompt_editor.py`
  - Основной интерактивный редактор генерации.
  - Работа с параметрами, референсами, LoRA, Smart Prompt, подменю режимов.
  - Запуск генерации и доставка результата.

- `handlers/prompt_editor_*.py`
  - `prompt_editor_ui.py` — построение клавиатур и базовых UI-блоков.
  - `prompt_editor_smart.py` — helper-логика Smart Prompt.
  - `prompt_editor_lora.py` — helper-логика LoRA и рендер LoRA-меню.
  - `prompt_editor_references.py` — helper-логика референсов и рендер меню.
  - `prompt_editor_send.py` — отправка превью/PNG и клавиатура действий.
  - Используются `prompt_editor.py` как подмодули без изменения поведения сценариев.

- `handlers/presets.py`
  - CRUD пресетов пользователя.

- `handlers/download.py`
  - Пошаговый мастер скачивания моделей.

## Runtime-состояние пользователя

`RuntimeStore` хранит:

- `active_prompt_requests` — текущая сессия редактора и параметры.
- `active_generations` — активные задачи генерации.
- `active_downloads` — активные задачи скачивания моделей.
- `last_params`, `last_seeds` — данные последней генерации для `/repeat`.
- `user_preferences` — пользовательские настройки интерфейса (например, pro mode).

При каждом входящем апдейте runtime-состояние синхронизируется на диск.
После перезапуска автоматически восстанавливаются:

- `active_prompt_requests` (можно продолжить редактор с сохранёнными параметрами);
- `last_params` и `last_seeds` (команда `/repeat` доступна после рестарта);
- `user_preferences` (например, выбранный режим интерфейса).

`active_generations` и `active_downloads` не восстанавливаются (это живые async-задачи).

## Персистентные данные

- `presets/<telegram_user_id>.json` — пресеты пользователя.
- `sessions/runtime.json` — снимок пользовательских runtime-сессий между перезапусками.
- `<COMFYUI_MODELS_PATH>/.comfybot_model_index.json` — индекс скачанных моделей и метаданных.

## Особенности генерации

- Референсы автоматически компонуются в одно изображение для ComfyUI input.
- Если IP-Adapter доступен, используется он; иначе fallback в img2img.
- Поддержаны цепочки LoRA (`loras: list[(name, strength)]`).
- Дополнительно можно включать Hi-res Fix, FreeU, PAG и upscaler.
- Прогресс может идти через WebSocket + fallback на polling `/history`.

## Где расширять проект

- Новый источник моделей: `model_downloader.py`.
- Тонкая настройка Smart Prompt (TIPO): `smart_prompt.py`.
- Новые пользовательские команды: `handlers/common.py`.
- Новый параметр генерации: `GenerationParams` + UI в `handlers/prompt_editor.py` + сборка workflow в `comfyui_client.py`.

## Миграция UI-архитектуры

Для полной и бесповоротной замены интерфейса на расширяемую архитектуру принят отдельный
контракт миграции: `docs/UI_MIGRATION_PLAN.md`.

Коротко о целевом состоянии:

- UI собирается из shared компонентов (`core/ui/*`), а не вручную в каждом handler.
- Рендер панелей и callback-message fallback централизованы (`core/interaction.py`, `core/panels.py`).
- Callback-контракты описываются схемами, а не ручным парсингом строк.
- Каждый этап миграции выполняется инкрементально по протоколу: изменения -> уведомление ->
  ручная проверка -> commit.

Текущий статус: этапы 0-10 завершены, выполняется финализация (этап 11).
