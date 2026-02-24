# Отчет о завершении этапа 2: toggle_pipeline_fix

> [!IMPORTANT]
> Отчет создан по шаблону `docs/dev/tasks/_templates/stage_completion_report_template.md`.
> Требуется явное одобрение пользователя перед переходом к следующему этапу.

---

## Навигация

- [Быстрый чек-лист отчета](#быстрый-чек-лист-отчета-m)
- [Метаданные этапа](#метаданные-этапа)
- [Что реализовано](#что-реализовано)
- [Результаты самопроверки](#результаты-самопроверки)
- [Шаги ручной проверки](#шаги-ручной-проверки)
- [Измененные артефакты](#измененные-артефакты)
- [Запрос одобрения](#запрос-одобрения)

## Легенда секций

| Маркер | Значение |
| --- | --- |
| `[M]` | Обязательная секция |
| `[O]` | Опциональная секция |

## Быстрый чек-лист отчета [M]

- [x] Заполнены метаданные этапа и путь к связанному плану.
- [x] Перечислены конкретные результаты этапа.
- [x] Добавлены результаты самопроверки со статусами pass/fail.
- [x] Добавлены точные шаги ручной проверки.
- [x] Указаны точные пути измененных артефактов.
- [x] Запрошено явное одобрение этапа.

## Метаданные этапа

| Поле | Значение |
| --- | --- |
| Task | `hires_fix_sampler_toggle_bug` |
| Тип работ | `bugfix` |
| Номер этапа | `2` |
| Короткое имя этапа | `toggle_pipeline_fix` |
| Дата | `2026-02-24` |
| Связанный план | `docs/dev/tasks/hires_fix_sampler_toggle_bug/2026-02-24_hires_fix_sampler_toggle_bug_plan.md` |

## Что реализовано

- В UI-toggle логике убрана связка Hi-Res -> sampler-pass: переключение Hi-Res больше не изменяет `enable_sampler_pass` (`handlers/prompt_editor_handlers_send.py`).
- Добавлен явный helper `toggle_hires_fix(...)`, чтобы поведение переключателя было изолированным и тестируемым (`handlers/prompt_editor_send_menu_utils.py`).
- В enhancement-пайплайне добавлено условие `_requires_image_generation(...)`: запуск `generate_from_image(...)` теперь происходит, если включен sampler-pass **или** активны sampling-enhancement флаги (`enable_hires_fix`, `enable_freeu`, `enable_pag`, `enable_tiled_diffusion`) (`handlers/prompt_editor_enhancement.py`).
- Логика генерации seed синхронизирована с новым условием запуска generation-пути: случайный seed назначается при любом запуске generation из enhancement, а не только при sampler-pass (`handlers/prompt_editor_enhancement.py`).
- Добавлены регрессионные тесты:
  - на независимость переключения Hi-Res от sampler-pass;
  - на матрицу условий запуска generation-пути при разных enhancement-флагах (`tests/test_prompt_editor_send_menu_utils.py`, `tests/test_prompt_editor_enhancement.py`).

## Результаты самопроверки

### Снимок проверок [M]

| Метрика | Значение |
| --- | --- |
| Всего проверок | `3` |
| Пройдено | `3` |
| Провалено | `0` |
| Финальный статус | `pass` |

### Сводка проверок [M]

| Проверка | Команда или метод | Результат | Примечание |
| --- | --- | --- | --- |
| Unit-тесты меню улучшений | `python -m pytest tests/test_prompt_editor_send_menu_utils.py -q` | pass | Все тесты файла прошли. |
| Unit-тесты enhancement-пайплайна | `python -m pytest tests/test_prompt_editor_enhancement.py -q` | pass | Все тесты файла прошли. |
| Линт измененных файлов | `python -m ruff check handlers/prompt_editor_enhancement.py handlers/prompt_editor_handlers_send.py handlers/prompt_editor_send_menu_utils.py tests/test_prompt_editor_enhancement.py tests/test_prompt_editor_send_menu_utils.py` | pass | `All checks passed!` |

### Найденные и закрытые дефекты [O]

| Дефект | Исправление | Результат перепроверки |
| --- | --- | --- |
| Включение Hi-Res автоматически включало sampler-pass | Удалена принудительная установка sampler-pass из ветки `img:tgl:hi` | pass: покрыто unit-тестом `test_toggle_hires_fix_does_not_change_sampler_pass` |
| При `enable_hires_fix=True` и `enable_sampler_pass=False` enhancement не применял Hi-Res | Добавлено расширенное условие запуска generation-пути `_requires_image_generation(...)` | pass: покрыто unit-тестом `test_requires_image_generation_for_hires_without_sampler_pass` |

## Шаги ручной проверки

1. Откройте улучшения для превью-изображения и включите `Hi-Res Fix`; проверьте, что переключатель `Сэмплер-проход` не меняет состояние автоматически.
2. При включенном `Hi-Res Fix` вручную выключите `Сэмплер-проход` и запустите `✨ Улучшить`.
3. Убедитесь, что улучшение выполняется через ComfyUI generation-путь (не fallback "Без ComfyUI: только сжатие") и Hi-Res применяется.

## Измененные артефакты

| Путь артефакта | Тип изменения | Назначение |
| --- | --- | --- |
| `handlers/prompt_editor_handlers_send.py` | updated | Убрана принудительная привязка Hi-Res к sampler-pass. |
| `handlers/prompt_editor_send_menu_utils.py` | updated | Добавлен helper переключения Hi-Res для изолированной логики. |
| `handlers/prompt_editor_enhancement.py` | updated | Исправлено условие запуска generation-пути и seed-логика. |
| `tests/test_prompt_editor_send_menu_utils.py` | updated | Добавлен тест на отсутствие автопереключения sampler-pass при Hi-Res toggle. |
| `tests/test_prompt_editor_enhancement.py` | added | Добавлены тесты матрицы условий `_requires_image_generation`. |
| `docs/dev/tasks/hires_fix_sampler_toggle_bug/2026-02-24_hires_fix_sampler_toggle_bug_plan.md` | updated | Отмечено выполнение подзадач этапа 2 и обновлен журнал согласований. |
| `docs/dev/tasks/hires_fix_sampler_toggle_bug/reports/2026-02-24_stage-2_toggle_pipeline_fix_report.md` | added | Зафиксированы результаты этапа 2. |

## Запрос одобрения

| Пункт | Значение |
| --- | --- |
| Статус | `Awaiting explicit user approval for Stage 2` |
| Следующее действие после одобрения | Закоммитить этап 2 и перейти к этапу 3 (финальная тестовая/регрессионная верификация). |
