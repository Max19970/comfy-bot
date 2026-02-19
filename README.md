# ComfyBot

Telegram-бот для генерации изображений через ComfyUI.

Проект даёт удобный интерактивный редактор промптов, работу с пресетами, скачивание моделей из CivitAI/HuggingFace и Smart Prompt на базе TIPO.

## Возможности

- Интерактивный `/generate` с полным редактором параметров (checkpoint, LoRA, sampler/scheduler, seed, size, batch, denoise и т.д.).
- Режимы интерфейса: простой и про.
- Референс-картинки (IP-Adapter, если доступен, или fallback в img2img).
- Улучшения пайплайна: Hi-res Fix, FreeU, PAG, Upscale.
- Smart Prompt (TIPO) с преобразованием естественного языка в теговые Positive/Negative.
- Обмен промптами между пользователями: экспорт/импорт полного набора параметров редактора (код обмена).
- Пресеты пользователя: сохранить, загрузить, перезаписать, удалить.
- Runtime-сессии между перезапусками бота (восстановление `/repeat`, активного редактора и режима UI).
- Поиск и скачивание моделей прямо из бота (`/download`) с фильтрами и прогрессом.
- Мониторинг очереди ComfyUI (`/queue`) и обновление списка моделей (`/models`).
- Whitelist-доступ по Telegram ID.

## Быстрый старт

1. Поднимите ComfyUI и убедитесь, что API доступен.
2. Установите Python 3.10+.
3. Установите зависимости:

```bash
pip install -r requirements.txt
```

4. Создайте `.env` из примера и заполните обязательные параметры:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Минимально нужно заполнить:
- `TELEGRAM_BOT_TOKEN`
- `COMFYUI_URL`
- `ALLOWED_USERS` (рекомендуется для приватного доступа)

5. Запустите бота:

```bash
python bot.py
```

6. В Telegram напишите боту `/start`.

Подробная установка: `docs/SETUP.md`

## Команды

- `/start`, `/help` — справка.
- `/generate` — новая генерация через редактор.
- `/repeat` — повтор последней генерации (с новым случайным seed).
- `/presets` — библиотека пресетов пользователя.
- `/download` — поиск/скачивание моделей.
- `/models` — обновить и показать модели, видимые ComfyUI.
- `/queue` — статус очереди ComfyUI.
- `/settings` — текущие дефолтные настройки.
- `/training` — текстовое обучение для новичков (пошаговый старт).
- `/cancel` — отменить текущую операцию (генерацию, скачивание, ввод).

Подробный гайд: `docs/COMMANDS.md`

## Документация

- `docs/SETUP.md` — установка и первый запуск.
- `docs/CONFIG.md` — все переменные окружения и примеры.
- `docs/COMMANDS.md` — сценарии работы в Telegram.
- `docs/TRAINING.md` — обучение для начинающих: от первого запуска до стабильного результата.
- `docs/ARCHITECTURE.md` — устройство проекта, поток данных, зоны ответственности.
- `docs/TROUBLESHOOTING.md` — типовые проблемы и решения.
- `SECURITY.md` — правила безопасной публикации и работы с секретами.
- `CONTRIBUTING.md` — рекомендации по развитию проекта.

## Структура проекта

```text
bot.py                   # Точка входа
app_context.py           # Сборка зависимостей и lifecycle приложения
config.py                # Конфиг из .env
comfyui_client.py        # Клиент ComfyUI API и сборка workflow
model_downloader.py      # Поиск/скачивание моделей и метаданные
smart_prompt.py          # Локальная интеграция TIPO для Smart Prompt
core/                    # Доменные модули: состояния, runtime, UI, storage, media
bot_core.py              # Совместимый facade для core/*
handlers/
  registry.py            # Централизованная регистрация обработчиков
  common.py              # /start, /help, /models, /queue, /settings, /cancel
  prompt_editor.py       # Оркестрация редактора генерации
  prompt_editor_ui.py    # Клавиатуры/базовые UI-блоки редактора
  prompt_editor_smart.py # Smart Prompt helper-функции
  prompt_editor_lora.py  # LoRA helper-функции и меню
  prompt_editor_references.py # Работа с референсами и меню
  prompt_editor_send.py  # Отправка результатов генерации
  presets.py             # Пресеты пользователя
  download.py            # Диалог скачивания моделей
presets/                 # Локальные пресеты пользователей (runtime)
sessions/                # Снимок runtime-сессий между перезапусками
```

## Безопасность и публикация

- Файл `.env` и пользовательские runtime-данные исключены через `.gitignore`.
- Не публикуйте токены Telegram/CivitAI/HuggingFace.
- Перед публичным релизом рекомендуется сделать ротацию ключей.

Подробно: `SECURITY.md`

## Лицензия

Лицензия в репозиторий пока не добавлена. Перед публичной публикацией выберите и добавьте подходящий файл `LICENSE`.
