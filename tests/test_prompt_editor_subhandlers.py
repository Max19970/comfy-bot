from __future__ import annotations

from aiogram import Router

import handlers.prompt_editor_subhandlers as subhandlers


def test_register_prompt_editor_subhandlers_delegates_grouped_deps(monkeypatch) -> None:
    calls: list[tuple[str, Router, object]] = []

    def _capture(name: str):
        def _inner(router: Router, deps: object) -> None:
            calls.append((name, router, deps))

        return _inner

    monkeypatch.setattr(subhandlers, "register_prompt_editor_flow_handlers", _capture("flow"))
    monkeypatch.setattr(subhandlers, "register_prompt_editor_smart_handlers", _capture("smart"))
    monkeypatch.setattr(subhandlers, "register_prompt_editor_edit_handlers", _capture("edit"))
    monkeypatch.setattr(
        subhandlers,
        "register_prompt_editor_exchange_handlers",
        _capture("exchange"),
    )
    monkeypatch.setattr(
        subhandlers,
        "register_prompt_editor_thematic_handlers",
        _capture("thematic"),
    )
    monkeypatch.setattr(subhandlers, "register_prompt_editor_lora_handlers", _capture("lora"))
    monkeypatch.setattr(
        subhandlers,
        "register_prompt_editor_reference_handlers",
        _capture("references"),
    )
    monkeypatch.setattr(subhandlers, "register_prompt_editor_send_handlers", _capture("send"))

    router = Router()
    flow = object()
    smart = object()
    edit = object()
    exchange = object()
    thematic = object()
    lora = object()
    references = object()
    send = object()

    deps = subhandlers.PromptEditorSubhandlersDeps(
        router=router,
        flow=flow,  # type: ignore[arg-type]
        smart=smart,  # type: ignore[arg-type]
        edit=edit,  # type: ignore[arg-type]
        exchange=exchange,  # type: ignore[arg-type]
        thematic=thematic,  # type: ignore[arg-type]
        lora=lora,  # type: ignore[arg-type]
        references=references,  # type: ignore[arg-type]
        send=send,  # type: ignore[arg-type]
    )

    subhandlers.register_prompt_editor_subhandlers(deps)

    assert calls == [
        ("flow", router, flow),
        ("smart", router, smart),
        ("edit", router, edit),
        ("exchange", router, exchange),
        ("thematic", router, thematic),
        ("lora", router, lora),
        ("references", router, references),
        ("send", router, send),
    ]
