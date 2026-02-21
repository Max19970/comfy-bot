from __future__ import annotations

from core.callbacks import (
    IndexedSelectionCallback,
    PagedSelectionCallback,
    ValueSelectionCallback,
)


def test_indexed_selection_pack_parse() -> None:
    callback = IndexedSelectionCallback(prefix="pe_ckpt", index=3)
    packed = callback.pack()
    assert packed == "pe_ckpt:3"

    parsed = IndexedSelectionCallback.parse(packed, prefix="pe_ckpt")
    assert parsed == callback


def test_paged_selection_pack_parse() -> None:
    callback = PagedSelectionCallback(prefix="pe_ckpt", page=2)
    packed = callback.pack()
    assert packed == "pe_ckpt_page:2"

    parsed = PagedSelectionCallback.parse(packed, prefix="pe_ckpt")
    assert parsed == callback


def test_value_selection_pack_parse() -> None:
    callback = ValueSelectionCallback(prefix="pe_seed", value="random")
    packed = callback.pack()
    assert packed == "pe_seed:random"

    parsed = ValueSelectionCallback.parse(packed, prefix="pe_seed")
    assert parsed == callback
