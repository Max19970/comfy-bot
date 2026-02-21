# Конфигурация (`.env`)

Конфиг загружается через `python-dotenv` в `config.py`.

## Обязательные параметры

| Переменная | Обязательно | По умолчанию | Описание |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Да | — | Токен Telegram-бота от `@BotFather`. Без него запуск завершится ошибкой. |

## Базовые параметры

| Переменная | Обязательно | По умолчанию | Описание |
|---|---|---|---|
| `COMFYUI_URL` | Нет | `http://127.0.0.1:8000` | Базовый URL ComfyUI API. |
| `ALLOWED_USERS` | Нет | пусто | Список Telegram ID через запятую. Если пусто, бот не ограничивает доступ. |
| `COMFYUI_MODELS_PATH` | Нет | `C:\Programs\ComfyUI\models` | Путь к каталогу моделей ComfyUI (для скачивания моделей). |

Пример:

```env
ALLOWED_USERS=123456789,987654321
```

## Параметры генерации по умолчанию

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DEFAULT_WIDTH` | `512` | Ширина изображения по умолчанию. |
| `DEFAULT_HEIGHT` | `512` | Высота изображения по умолчанию. |
| `DEFAULT_STEPS` | `20` | Количество шагов sampler по умолчанию. |
| `DEFAULT_CFG` | `7.0` | CFG Scale по умолчанию. |
| `DEFAULT_SAMPLER` | `euler` | Sampler по умолчанию. |
| `DEFAULT_SCHEDULER` | `normal` | Scheduler по умолчанию. |
| `DEFAULT_DENOISE` | `1.0` | Denoise по умолчанию. |

## Токены для поиска и скачивания моделей

| Переменная | Обязательно | Описание |
|---|---|---|
| `CIVITAI_API_KEY` | Нет | Нужен для части моделей/лимитов CivitAI. |
| `HUGGINGFACE_TOKEN` | Нет | Нужен для gated/private моделей HuggingFace. |

Примечание: поддерживается алиас `HF_TOKEN` (используется, если `HUGGINGFACE_TOKEN` не задан).

Если токены не заданы, открытые модели всё равно можно искать/скачивать (в зависимости от источника и ограничений).

## Smart Prompt (TIPO)

Smart Prompt использует локальную модель TIPO (без внешних LLM API).

| Переменная | По умолчанию | Описание |
|---|---|---|
| `SMART_PROMPT_PROVIDER` | `tipo` | Режим работы: `tipo` или `disabled`. |
| `SMART_PROMPT_MODEL` | `KBlueLeaf/TIPO-500M-ft` | HuggingFace model id для TIPO. |
| `SMART_PROMPT_TIMEOUT` | `120` | Таймаут обработки в секундах. |
| `SMART_PROMPT_TEMPERATURE` | `0.35` | Температура семплирования TIPO. |
| `SMART_PROMPT_TOP_P` | `0.95` | Параметр nucleus sampling. |
| `SMART_PROMPT_MIN_P` | `0.05` | Порог min-p (если поддерживается версией `transformers`). |
| `SMART_PROMPT_TOP_K` | `80` | Параметр top-k sampling. |
| `SMART_PROMPT_DEVICE` | `auto` | Устройство: `auto`, `cpu`, `cuda`. |
| `SMART_PROMPT_SEED` | `-1` | Seed TIPO (`-1` = случайный). |
| `SMART_PROMPT_TAG_LENGTH` | `long` | Длина тегового вывода: `very_short`/`short`/`long`/`very_long`. |
| `SMART_PROMPT_NL_LENGTH` | `short` | Целевая длина NL-части для внутренних операций TIPO. |
| `SMART_PROMPT_BAN_TAGS` | `text, watermark, signature` | Теги, которые удаляются из positive и добавляются в negative. |
| `SMART_PROMPT_NEGATIVE_BASE` | см. `.env.example` | Базовый negative prompt, к которому добавляются ban-теги. |

### Минимальный пример

```env
SMART_PROMPT_PROVIDER=tipo
SMART_PROMPT_MODEL=KBlueLeaf/TIPO-500M-ft
SMART_PROMPT_DEVICE=auto
```

### Производительный пример (GPU)

```env
SMART_PROMPT_PROVIDER=tipo
SMART_PROMPT_MODEL=KBlueLeaf/TIPO-500M-ft
SMART_PROMPT_DEVICE=cuda
SMART_PROMPT_TEMPERATURE=0.35
SMART_PROMPT_TOP_P=0.95
SMART_PROMPT_TOP_K=80
```

## Типичные ошибки конфигурации

- Неверный `TELEGRAM_BOT_TOKEN` -> бот не стартует.
- Неверный `COMFYUI_URL` -> `/models` и генерация падают с ошибкой подключения.
- Не установлены зависимости `tipo-kgen/transformers/torch` -> Smart Prompt недоступен.
- Пустой `ALLOWED_USERS` в публичном окружении -> бот доступен всем.
