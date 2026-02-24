# Локализация и UI Text System

## Цель

Локализация в проекте закреплена как часть обязательной текстовой системы интерфейса:

- пользовательский текст должен идти через `UITextService`;
- локализация подключается как модификатор pipeline;
- для handler-слоя используется bridge (`UITextLocalizationBridge`), чтобы даже прямые вызовы `localization.t(...)` проходили через text-system.

Это позволяет поддерживать текущие и будущие локали без изменения ядра и снижает риск появления "непереведенных" участков интерфейса.

## Базовая архитектура

### 1) Канонический слой UI-текстов

- `ui_text/registry.json` - реестр `text_id` и дефолтных значений.
- `ui_text/profiles/*.profile.json` - профили copy override.
- `application/ui_text_service.py` - резолв текста через pipeline модификаторов.

### 2) Pipeline модификаторов

- Модификаторы подключаются динамически через `UI_TEXT_MODIFIER_FACTORIES`.
- По умолчанию включены:
  - `infrastructure.ui_text_modifiers:create_localization_modifier`
  - `infrastructure.ui_text_modifiers:create_copy_profile_modifier`

### 3) Bridge для handler-слоя

- `application/ui_text_localization_bridge.py`.
- В `handlers/registry.py` все регистрации обработчиков получают bridge-локализацию.
- Это гарантирует прохождение через text-system даже в местах, где используется не UI Kit, а прямые Telegram-конструкторы.

## Структура локалей

```text
locales/
  ru/
    locale.meta.json
    messages.json
  en/
    locale.meta.json
    messages.json
```

Для каждой локали обязательны:

- `locale.meta.json` - метаданные локали;
- `messages.json` - словарь `key -> text`.

### Формат metadata

Минимальный пример `locales/ru/locale.meta.json`:

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

Правила:

- `is_default=true` должен быть ровно у одной локали;
- `fallback_locale` либо `null`, либо существующая локаль;
- `messages_file` указывает на валидный JSON-объект.

## Операционный workflow

### Добавить новый UI-текст

1. Добавьте `text_id` в `ui_text/registry.json`.
2. Добавьте перевод в `locales/<locale>/messages.json` (минимум в default locale).
3. Запустите синхронизацию ключей и аудит.

### Добавить новую локаль

1. Создайте `locales/<code>/`.
2. Добавьте `locale.meta.json` и `messages.json`.
3. Запустите `sync` для автодобавления отсутствующих ключей.
4. Заполните переводы и прогоните `audit`.

## Инструмент сопровождения локалей

Используйте `tools/i18n/locale_maintenance.py`.

### 1) Скан реально используемых ключей

```bash
python tools/i18n/locale_maintenance.py --project-root . scan-runtime
```

### 2) Аудит локалей

```bash
python tools/i18n/locale_maintenance.py --project-root . audit
```

### 3) Жесткая проверка паритета ключей (CI-safe)

```bash
python tools/i18n/locale_maintenance.py --project-root . audit --strict
```

`--strict` валит проверку на:

- missing keys;
- orphan keys;
- placeholder mismatches.

Дополнительно доступны флаги:

- `--fail-on-mixed-script`
- `--fail-on-untranslated`

### 4) Синхронизация отсутствующих ключей

Dry-run:

```bash
python tools/i18n/locale_maintenance.py --project-root . sync --missing-strategy source
```

Запись в файлы:

```bash
python tools/i18n/locale_maintenance.py --project-root . sync --missing-strategy source --write
```

## Обязательные проверки

Минимальный i18n/UI-текст набор:

```bash
python -m pytest -q tests/test_ui_text_localization_bridge.py tests/test_handler_registry_bridge.py
python -m pytest -q tests/test_ui_text_guardrails.py
python tools/i18n/locale_maintenance.py --project-root . audit --strict
python -m pytest -q tests/test_localization_catalog.py tests/test_localization_service.py tests/test_localization_resources.py
```

Полный quality gate:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest -q
```
