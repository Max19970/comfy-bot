# Handler plugins

Этот документ описывает контракт и процесс подключения handler-плагинов без правки ядра регистрации.

## Зачем это нужно

- Убрать статическое wiring в `handlers/registry.py`.
- Добавлять новые handler-блоки через отдельные модули/пакеты.
- Сохранять совместимость и контролировать порядок регистрации.

## Как работает регистрация

1. `handlers/registry.py` читает `HANDLER_PLUGIN_PACKAGES`.
2. Loader ищет в пакетах модули с функцией `register_plugins(registry)`.
3. Каждый плагин регистрируется в `PluginRegistry`.
4. Плагины сортируются по `(order, plugin_id)`.
5. Регистрируются только плагины с capability `handlers.registration`.

Если список пакетов пустой или не дал плагинов, используется fallback-пакет `handlers.plugins.builtin`.

## Контракт плагина

Базовые типы находятся в `plugins/contracts.py`:

- `PluginApiVersion` — версия API плагина.
- `PluginDescriptor` — метаданные плагина.
- `HandlerPluginContext` — контекст регистрации (`router`, `deps`, `shared`).
- `HandlerPlugin` — протокол с методом `register(context)`.

Минимальный `descriptor`:

- `plugin_id`: уникальный идентификатор.
- `display_name`: человекочитаемое имя.
- `api_version`: совместимость с runtime (сейчас `1.0`).
- `order`: порядок запуска (меньше — раньше).
- `capabilities`: должен содержать `handlers.registration`.

## Точка входа пакета

Каждый пакет плагинов экспортирует функцию:

```python
def register_plugins(registry: PluginRegistry) -> None:
    registry.register(MyHandlerPlugin())
```

Loader вызывает ее автоматически при discovery.

## Пример плагина

```python
from __future__ import annotations

from plugins.contracts import HandlerPluginContext, PluginDescriptor, PluginRegistry


class MetricsHandlersPlugin:
    descriptor = PluginDescriptor(
        plugin_id="handlers.metrics",
        display_name="Metrics handlers",
        order=450,
    )

    def register(self, context: HandlerPluginContext) -> None:
        deps = context.deps
        router = context.router
        # register_metrics_handlers(router, deps.runtime, ...)


def register_plugins(registry: PluginRegistry) -> None:
    registry.register(MetricsHandlersPlugin())
```

## Рекомендации по зависимостям

- Для typed-доступа к `context.deps` используйте локальный `Protocol` в модуле плагина.
- Для межплагинных зависимостей используйте `context.shared` с явными ключами.
- Не импортируйте в `handlers/registry.py` конкретные handler-модули напрямую.

## Добавление нового handler-плагина

1. Создайте модуль плагина в package из `HANDLER_PLUGIN_PACKAGES`.
2. Реализуйте класс с `descriptor` и `register(context)`.
3. Добавьте регистрацию в `register_plugins(registry)`.
4. Выставьте корректный `order` относительно зависимых плагинов.
5. Добавьте/обновите тесты:
   - plugin loader;
   - bridge/порядок регистрации;
   - callback/runtime контракты, если затрагиваются.
6. Проверьте flow в Telegram (`/start`, целевой сценарий плагина).

## Диагностика проблем

- `Failed to load handler plugins` — ошибка импорта пакета/модуля или несовместимая API версия.
- `Duplicate plugin id` — два плагина с одинаковым `plugin_id`.
- `No handler plugins with registration capability` — у найденных плагинов нет `handlers.registration`.
- `Prompt editor service is not registered` — нарушен порядок плагинов (например, `presets` раньше `prompt_editor`).
