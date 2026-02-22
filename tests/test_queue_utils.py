from __future__ import annotations

from core.queue_utils import queue_item_prompt_id


def test_queue_item_prompt_id_from_tuple_payload() -> None:
    assert queue_item_prompt_id((3, "prompt-123", {"meta": 1})) == "prompt-123"


def test_queue_item_prompt_id_from_dict_payload() -> None:
    assert queue_item_prompt_id({"prompt_id": "prompt-321"}) == "prompt-321"
    assert queue_item_prompt_id({"id": "prompt-654"}) == "prompt-654"


def test_queue_item_prompt_id_returns_empty_for_unknown_payload() -> None:
    assert queue_item_prompt_id((1, 2)) == ""
    assert queue_item_prompt_id({"prompt_id": 100}) == ""
    assert queue_item_prompt_id("prompt") == ""
