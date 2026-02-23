# Отчет о завершении этапа 3: user-locale-resolution-and-switch

## Навигация

- [Быстрый чек-лист отчета](#быстрый-чек-лист-отчета-m)
- [Метаданные этапа](#метаданные-этапа)
- [Что реализовано](#что-реализовано)
- [Результаты самопроверки](#результаты-самопроверки)
- [Шаги ручной проверки](#шаги-ручной-проверки)
- [Измененные артефакты](#измененные-артефакты)
- [Запрос одобрения](#запрос-одобрения)

## Легенда разделов

| Маркер | Значение |
| --- | --- |
| `[M]` | Обязательный раздел |
| `[O]` | Опциональный раздел |

## Быстрый чек-лист отчета [M]

- [x] Заполнены метаданные этапа и путь к связанному плану.
- [x] Перечислены конкретные реализованные изменения.
- [x] Добавлены результаты самопроверки со статусом pass/fail.
- [x] Добавлены точные шаги ручной проверки.
- [x] Перечислены пути измененных артефактов.
- [x] Запрошено явное одобрение этапа.

## Метаданные этапа

| Поле | Значение |
| --- | --- |
| Задача | `bot-localization-system` |
| Тип работ | `feature` |
| Номер этапа | `3` |
| Короткое имя этапа | `user-locale-resolution-and-switch` |
| Дата | `2026-02-23` |
| Связанный план | `docs/dev/tasks/bot-localization-system/2026-02-23_bot-localization-system_plan.md` |

## Что реализовано

- В `core/user_preferences.py` добавлена поддержка пользовательской локали:
  - `read_user_locale(...)` для безопасного чтения locale из prefs;
  - нормализация `locale` в `normalize_user_preferences(...)`.
- Реализован `DefaultUserLocaleResolver` в `application/user_locale_resolver.py` с приоритетом:
  1) явный выбор пользователя,
  2) locale из Telegram профиля,
  3) default locale каталога.
- Интегрирован resolver в common-flow:
  - `handlers/common.py` создает resolver и передает в deps;
  - `handlers/common_core_handlers.py` использует resolver для расчета текущего языка в настройках.
- Добавлен пользовательский сценарий переключения языка через сервисные настройки:
  - новая кнопка `🌐 Язык` в экране `⚙️ Настройки`;
  - экран выбора языка `menu:settings:locale`;
  - сохранение выбора через `menu:settings:set:locale:<code>` в `runtime.user_preferences[uid]["locale"]`.
- Сохранена обратная совместимость runtime snapshot:
  - поле `locale` проходит через текущую нормализацию пользовательских prefs и корректно сериализуется/десериализуется.
- Добавлены тесты на нормализацию locale, приоритет резолва и runtime-контракт пользовательских prefs.

## Результаты самопроверки

### Сводка проверок [M]

| Метрика | Значение |
| --- | --- |
| Всего проверок | `4` |
| Успешно | `4` |
| Провалено | `0` |
| Итоговый статус | `pass` |

### Список проверок [M]

| Проверка | Команда или метод | Результат | Примечание |
| --- | --- | --- | --- |
| Lint (ruff) | `python -m ruff check core/user_preferences.py application/__init__.py application/user_locale_resolver.py handlers/common.py handlers/common_core_handlers.py tests/test_user_preferences.py tests/test_user_locale_resolver.py tests/test_runtime_session_contracts.py` | pass | Нарушений стиля и импортов нет |
| Typing (mypy) | `python -m mypy core/user_preferences.py application/user_locale_resolver.py handlers/common.py handlers/common_core_handlers.py` | pass | Ошибок типизации не найдено |
| Таргетные тесты Stage 3 | `python -m pytest -q tests/test_user_preferences.py tests/test_user_locale_resolver.py tests/test_runtime_session_contracts.py` | pass | Все тесты по локали и persistence-контрактам прошли |
| Полная регрессия | `python -m pytest -q` | pass | Общий тестовый набор проекта прошел |

### Обнаруженные дефекты и исправления [O]

- На этом этапе дефекты не обнаружены.

## Шаги ручной проверки

1. Откройте в боте `⚙️ Настройки` и убедитесь, что отображается строка текущего языка интерфейса.
2. Нажмите `🌐 Язык`, выберите `English (en)`, вернитесь в `⚙️ Настройки` и проверьте, что выбранный язык отображается как текущий.
3. Перезапустите бота и снова откройте `⚙️ Настройки`; убедитесь, что выбранный язык сохранился в пользовательских настройках.

## Измененные артефакты

| Путь артефакта | Тип изменения | Назначение |
| --- | --- | --- |
| `core/user_preferences.py` | `updated` | Нормализация/чтение пользовательской локали |
| `application/user_locale_resolver.py` | `added` | Resolver с приоритетом `user -> telegram -> default` |
| `application/__init__.py` | `updated` | Экспорт `DefaultUserLocaleResolver` |
| `handlers/common.py` | `updated` | Инициализация resolver и передача в common deps |
| `handlers/common_core_handlers.py` | `updated` | UI-сценарий выбора языка и применение resolver в настройках |
| `tests/test_user_preferences.py` | `updated` | Тесты нормализации locale и `read_user_locale` |
| `tests/test_user_locale_resolver.py` | `added` | Тесты приоритетов выбора локали |
| `tests/test_runtime_session_contracts.py` | `updated` | Проверка сохранения/миграции locale в runtime payload |
| `docs/dev/tasks/bot-localization-system/2026-02-23_bot-localization-system_plan.md` | `updated` | Обновлен журнал этапов и фиксация commit Stage 2 |
| `docs/dev/tasks/bot-localization-system/reports/2026-02-23_stage-3_user-locale-resolution-and-switch_report.md` | `added` | Отчет о завершении этапа 3 |

## Незапланированные дополнительные задачи выполненные [O]

- None.

## Известные ограничения или заметки на следующий этап [O]

- На Stage 3 реализован выбор/сохранение/резолв языка пользователя, но массовый перевод текстов handlers/core на ключи локализации выполняется на Stage 4.

## Запрос одобрения

| Пункт | Значение |
| --- | --- |
| Статус | `Ожидается явное одобрение пользователя для этапа 3` |
| Следующее действие после одобрения | commit Stage 3 и переход к Stage 4 |
