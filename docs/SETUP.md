# Установка и запуск

## 1) Требования

- Python 3.10+
- Запущенный ComfyUI с доступным HTTP API
- Telegram-бот, созданный через `@BotFather`

Рекомендуется:
- отдельная виртуальная среда Python
- доступ к CivitAI/HuggingFace токенам (если нужен `/download` для приватных/gated моделей)

## 2) Подготовка окружения

```bash
python -m venv .venv
```

Активация окружения:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

## 3) Настройка `.env`

Скопируйте шаблон:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Минимум для старта:

- `TELEGRAM_BOT_TOKEN` — токен бота от BotFather
- `COMFYUI_URL` — URL ComfyUI API (например `http://127.0.0.1:8000`)
- `ALLOWED_USERS` — Telegram ID разрешённых пользователей через запятую

Если `ALLOWED_USERS` оставить пустым, бот будет отвечать всем.

Полное описание всех переменных: `docs/CONFIG.md`

## 4) Проверка ComfyUI

Перед запуском убедитесь, что ComfyUI доступен по сети с той машины, где запущен бот:

- API отвечает (`/system_stats`)
- в ComfyUI есть хотя бы один checkpoint
- при необходимости установлены узлы для IP-Adapter, FreeU, PAG

Бот сам покажет статус при старте в логах.

## 5) Запуск

```bash
python bot.py
```

После запуска:

1. Откройте диалог с ботом в Telegram.
2. Выполните `/start`.
3. Выполните `/training`, если нужен пошаговый текстовый курс для новичка.
4. Выполните `/models` и проверьте, что список моделей загружен.
5. Запустите первую генерацию через `/generate`.

## 6) Первая проверка функционала

Рекомендуемый минимальный smoke test:

1. `/models` — видны checkpoints/samplers/schedulers.
2. `/generate` — открылся редактор.
3. Выберите checkpoint, задайте простой positive prompt.
4. Нажмите «Генерировать».
5. Убедитесь, что пришло превью и доступна кнопка PNG.

## 7) Рекомендации перед публикацией на GitHub

- Убедитесь, что `.env` не попадает в коммит.
- Убедитесь, что пользовательские файлы в `presets/` не попадают в коммит.
- Проверьте, что в `.env.example` только безопасные placeholder-значения.
- Добавьте `LICENSE` под выбранную модель лицензирования.

Дополнительно: `SECURITY.md`
