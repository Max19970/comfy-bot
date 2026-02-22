from __future__ import annotations

from core.callbacks import (
    IndexedSelectionCallback,
    PagedSelectionCallback,
    ValueSelectionCallback,
)


def test_indexed_selection_callback_contract_is_stable() -> None:
    packed = IndexedSelectionCallback(prefix="pe_ckpt", index=42).pack()
    assert packed == "pe_ckpt:42"

    assert IndexedSelectionCallback.parse(packed, prefix="pe_ckpt") == IndexedSelectionCallback(
        prefix="pe_ckpt",
        index=42,
    )
    assert IndexedSelectionCallback.parse("pe_ckpt:abc", prefix="pe_ckpt") is None
    assert IndexedSelectionCallback.parse("pe_ckpt", prefix="pe_ckpt") is None
    assert IndexedSelectionCallback.parse("other:42", prefix="pe_ckpt") is None


def test_paged_selection_callback_contract_is_stable() -> None:
    packed = PagedSelectionCallback(prefix="pe_ckpt", page=3).pack()
    assert packed == "pe_ckpt_page:3"

    assert PagedSelectionCallback.parse(packed, prefix="pe_ckpt") == PagedSelectionCallback(
        prefix="pe_ckpt",
        page=3,
    )
    assert PagedSelectionCallback.parse("pe_ckpt_page:abc", prefix="pe_ckpt") is None
    assert PagedSelectionCallback.parse("pe_ckpt:3", prefix="pe_ckpt") is None
    assert PagedSelectionCallback.parse("other_page:3", prefix="pe_ckpt") is None


def test_value_selection_callback_contract_is_stable() -> None:
    packed = ValueSelectionCallback(prefix="pe_seed", value="random").pack()
    assert packed == "pe_seed:random"

    assert ValueSelectionCallback.parse(packed, prefix="pe_seed") == ValueSelectionCallback(
        prefix="pe_seed",
        value="random",
    )
    assert ValueSelectionCallback.parse("other:random", prefix="pe_seed") is None


def test_value_selection_callback_preserves_colons_and_empty_values() -> None:
    with_colon = ValueSelectionCallback.parse("pe_seed:manual:123", prefix="pe_seed")
    assert with_colon == ValueSelectionCallback(prefix="pe_seed", value="manual:123")

    empty = ValueSelectionCallback.parse("pe_seed:", prefix="pe_seed")
    assert empty == ValueSelectionCallback(prefix="pe_seed", value="")
