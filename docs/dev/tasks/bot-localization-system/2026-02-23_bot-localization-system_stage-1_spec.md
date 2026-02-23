# Спецификация Stage 1: архитектура и контракты локализации

## Цель

- Зафиксировать архитектурный контракт локализации до реализации ядра.
- Определить структуру ресурсных файлов (`metadata + messages`) так, чтобы добавление новой локали выполнялось без изменений кода ядра.
- Подготовить единый key-space для миграции пользовательских текстов из `core/*` и `handlers/*`.

## Принципы проектирования

- **SOLID:** зависимости UI-слоя идут в абстракцию сервиса локализации, а не в файловый загрузчик.
- **KISS:** хранение переводов в плоском JSON (`key -> value`) без избыточной вложенности.
- **DRY:** повторяемые подписи кнопок и стандартные сообщения выносятся в общие ключи.
- **GRASP / High cohesion:** каждая ответственность отделена: загрузка ресурсов, резолв локали, выдача перевода.
- **Low coupling:** `handlers/*` не знают о формате файлов локалей, используют только `LocalizationService`.

## Инвентаризация пользовательских текстов

### Категории текстов

1. Статические тексты экранов и разделов (help/start/training/settings/menu).
2. Подписи кнопок и навигации (back/cancel/menu root/actions).
3. Alert/ошибки callback-flow (invalid selection/session expired/not found).
4. Параметризованные уведомления (`{count}`, `{name}`, `{path}`, `{value}`, `{index}`).
5. Многострочные панели со сводками и списками.

### Зоны кода с пользовательскими текстами

- `core/*` (ключевые точки):
  - `core/ui_copy.py`
  - `core/ui_kit/buttons.py`
  - `core/ui_kit/dialogs.py`
  - `core/interaction.py`
  - `core/download_filters.py`
  - `core/prompt_exchange.py`
  - `core/ui_summary.py`
  - `core/runtime.py` (дефолтные заголовки задач)
  - `core/runtime_snapshot.py` (дефолтные заголовки задач)
- `handlers/common*.py`: главное меню, сервис, настройки, обучение, jobs/delete/access.
- `handlers/download*.py`: мастер скачивания, фильтры, версии, статусы и ошибки.
- `handlers/presets*.py`: CRUD пресетов, подтверждения, ошибки ввода.
- `handlers/prompt_editor*.py` и `handlers/prompt_editor_handlers_*.py`: основной объем UI редактора генерации.

Примечание: по результатам grep-инвентаризации присутствуют сотни строк пользовательского текста в `handlers/*`; в Stage 4 миграция выполняется пакетно по namespace-группам, а не хаотично по файлам.

## Единая схема ключей переводов

### Формат ключа

- Формат: `domain.section.item`.
- Разделитель: точка.
- Разрешены только `a-z`, `0-9`, `.`.
- Ключ должен описывать смысл, а не конкретный файл.

### Базовые namespace-группы

- `button.*` - базовые общие кнопки (`back`, `cancel`, `menu_root`, `custom_value`).
- `menu.root.*`, `menu.generation.*`, `menu.models.*`, `menu.service.*`.
- `common.*` - shared панели/сообщения common flow (`training`, `settings`, `jobs`, `delete_model`).
- `download.*` - фильтры, источники, выбор версии, статусы скачивания, ошибки.
- `presets.*` - библиотека пресетов, ввод имени, подтверждения.
- `prompt_editor.*` - редактор генерации, поля, подсказки, валидации, уведомления.
- `error.*` - общие ошибки и fallback-тексты.
- `status.*` - общие статусные маркеры (`enabled`, `disabled`, `in_queue`, `running`).

### Правила параметризации

- Плейсхолдеры в формате `str.format`: `{count}`, `{name}`, `{filename}`, `{path}`, `{index}`.
- Для каждого параметризованного ключа обязательны:
  - стабильные имена параметров;
  - одинаковый набор параметров во всех локалях.
- Значения без параметров не должны использовать форматирование.

## Формат файлов локали

## Директория ресурсов

```text
locales/
  ru/
    locale.meta.json
    messages.json
  en/
    locale.meta.json
    messages.json
```

### `locale.meta.json` (обязательно для каждой локали)

Минимальная структура:

```json
{
  "schema_version": 1,
  "locale": "ru",
  "name": "Russian",
  "native_name": "Русский",
  "messages_file": "messages.json",
  "is_default": true,
  "enabled": true,
  "fallback_locale": null
}
```

Требования валидации:

- `schema_version`: целое число, поддерживаемая версия схемы.
- `locale`: строка, код локали (`ru`, `en`, опционально с регионом).
- `name`: display name для внутренних списков/логов.
- `native_name`: самоназвание языка для UI-переключателя.
- `messages_file`: относительный путь к JSON с переводами.
- `is_default`: `true` только у одной локали в каталоге.
- `enabled`: локаль доступна для использования.
- `fallback_locale`: `null` или код локали; если указан, должен существовать.

### `messages.json` (переводы)

Формат:

```json
{
  "button.back": "⬅️ Назад",
  "menu.root.generation": "🎨 Генерация",
  "presets.empty": "📂 Нет пресетов. Создайте через /generate.",
  "common.jobs.active_count": "🧵 <b>Мои задачи</b>\nАктивно: <b>{count}</b>"
}
```

Требования:

- Объект JSON верхнего уровня: `dict[str, str]`.
- Ключи уникальны и соответствуют правилам key-space.
- Значение каждого ключа: строка (включая HTML-разметку, если используется в текущем UI).
- Для стартовых языков (`ru`, `en`) ключевое покрытие должно быть одинаковым.

## Доменные контракты локализации (предварительная спецификация)

Ниже фиксируется контракт, который будет реализован в Stage 2.

```python
from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True, slots=True)
class LocaleMetadata:
    schema_version: int
    locale: str
    name: str
    native_name: str
    messages_file: str
    is_default: bool
    enabled: bool
    fallback_locale: str | None


@dataclass(frozen=True, slots=True)
class TranslationBundle:
    metadata: LocaleMetadata
    messages: Mapping[str, str]


class TranslationCatalog(Protocol):
    def list_locales(self) -> list[str]: ...
    def default_locale(self) -> str: ...
    def get_bundle(self, locale: str) -> TranslationBundle | None: ...


class LocalizationService(Protocol):
    def t(
        self,
        key: str,
        *,
        locale: str | None = None,
        params: Mapping[str, object] | None = None,
        default: str | None = None,
    ) -> str: ...


class UserLocaleResolver(Protocol):
    def resolve(
        self,
        *,
        user_locale: str | None,
        telegram_locale: str | None,
    ) -> str: ...
```

Правила слоя:

- `infrastructure` отвечает за чтение и валидацию файлов локалей.
- `application` отвечает за резолв локали, fallback и форматирование.
- `presentation`/`handlers` используют только `LocalizationService`.
- Добавление новой локали не должно требовать изменений в `domain/application` коде.

## Критерии готовности Stage 1

- Покрыты категории текстов: статические, параметризованные, alert/reply/keyboard labels.
- Зафиксирован контракт, позволяющий добавлять локаль только новыми resource-файлами.
- Определены namespace-правила для миграции без дублирования ключей.
