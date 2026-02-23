# Локализация

## Цель

Система локализации построена так, чтобы новые языки добавлялись только ресурсными файлами (`metadata + messages`) без изменений кода ядра.

## Структура ресурсов

```text
locales/
  ru/
    locale.meta.json
    messages.json
  en/
    locale.meta.json
    messages.json
```

Для каждой локали обязательны оба файла:

- `locale.meta.json` - метаданные локали;
- `messages.json` - словарь переводов (`key -> text`).

## Формат metadata

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

- `schema_version` - целое число, поддерживаемая версия схемы;
- `locale` - код локали (`ru`, `en`, `de`, `pt-br` и т.д.);
- `messages_file` - относительный путь к JSON со строками;
- `is_default=true` должен быть ровно у одной локали;
- `fallback_locale` может быть `null` или ссылаться на существующую локаль.

## Формат messages

`messages.json` должен быть объектом вида:

```json
{
  "ui.start": "...",
  "ui.menu.generation": "..."
}
```

Требования:

- ключи должны быть строками;
- значения должны быть строками;
- для RU/EN используется одинаковый набор ключей (проверяется тестом).

## Как добавить новый язык без правки ядра

1. Создайте директорию `locales/<code>`.
2. Добавьте `locale.meta.json` с корректным `locale` и `messages_file`.
3. Добавьте `messages.json` с нужными ключами переводов.
4. При необходимости укажите `fallback_locale` на существующую локаль.
5. Запустите проверки качества.

Если файлы валидны, язык будет автоматически обнаружен каталогом локализации при старте приложения.

## Проверки

Минимальный набор:

```bash
python -m pytest -q tests/test_localization_catalog.py tests/test_localization_service.py tests/test_localization_resources.py
```

Полный quality gate:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
```
