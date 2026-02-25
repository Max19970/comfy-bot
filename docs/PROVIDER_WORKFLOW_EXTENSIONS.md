# Provider & Workflow Extensions

Документ описывает, как расширять источники моделей и workflow-ноды без изменения ядра.

## Что уже поддерживается

- Динамическая загрузка источников моделей через `MODEL_SOURCE_PROVIDER_PACKAGES`.
- Динамическая загрузка workflow-нод через `COMFY_NODE_PACKAGES`.
- Fail-fast проверки конфликтов на старте (duplicate `source`, duplicate `node_id`, конфликтующие `stage_labels`).

## Model source providers

### Конфигурация

```env
MODEL_SOURCE_PROVIDER_PACKAGES=application.model_source_plugins.builtin,my_project.model_source_plugins
```

Каждый package может содержать модули с функцией:

```python
def register_providers(registry, context) -> None:
    ...
```

`context` содержит колбэки и зависимости для встроенных операций (поиск/fetch для CivitAI/HF и токены).

### Минимальный провайдер

```python
from application.model_downloader import SearchResult


class ExampleProvider:
    source = "example"

    async def search(self, request):
        return [
            SearchResult(
                name="Example model",
                source=self.source,
                model_id="example/model",
                version_id="v1",
                filename="example.safetensors",
                download_url="https://example.org/model.safetensors",
                model_type=request.model_type,
                download_count=0,
            )
        ]

    async def resolve_direct(self, request):
        return []

    def download_headers(self):
        return {}


def register_providers(registry, context):
    registry.register(ExampleProvider())
```

## Workflow node packages

### Конфигурация

```env
COMFY_NODE_PACKAGES=infrastructure.comfy_nodes.nodes,my_project.comfy_nodes
```

Каждый package может содержать модули с функцией:

```python
def register_nodes(registry) -> None:
    ...
```

### Минимальная нода

```python
from infrastructure.comfy_nodes.contracts import WorkflowStageLabel


class ExampleNode:
    node_id = "example_debug"
    phase = 90
    order = 10

    def apply(self, state):
        state.add_node("ExampleDebug", {"message": "ok"})

    def stage_labels(self):
        return {
            "ExampleDebug": WorkflowStageLabel(
                localization_key="workflow.stage.example_debug",
                default_text="Example debug node",
            )
        }


def register_nodes(registry):
    registry.register(ExampleNode())
```

## Диагностика

- `Duplicate model source provider: ...` — в пакетах зарегистрированы источники с одинаковым `source`.
- `Duplicate comfy node id: ...` — конфликт `node_id` между node packages.
- `Conflicting stage label for class type ...` — разные `WorkflowStageLabel` для одного `class_type`.
- `No model source providers discovered` — загрузчик не смог получить ни одного провайдера из заданных пакетов.

## Рекомендации

- Для каждого нового provider/node package добавляйте контрактный тест.
- Для rollout используйте поэтапное включение пакетов в `.env`.
- Не редактируйте ядро (`ModelDownloader`, базовые registry-модули) для каждого нового расширения.
