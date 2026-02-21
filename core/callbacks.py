from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexedSelectionCallback:
    prefix: str
    index: int

    def pack(self) -> str:
        return f"{self.prefix}:{self.index}"

    @classmethod
    def parse(cls, data: str, *, prefix: str) -> IndexedSelectionCallback | None:
        marker = f"{prefix}:"
        if not data.startswith(marker):
            return None
        raw_index = data[len(marker) :]
        try:
            index = int(raw_index)
        except ValueError:
            return None
        return cls(prefix=prefix, index=index)


@dataclass(frozen=True)
class PagedSelectionCallback:
    prefix: str
    page: int

    def pack(self) -> str:
        return f"{self.prefix}_page:{self.page}"

    @classmethod
    def parse(cls, data: str, *, prefix: str) -> PagedSelectionCallback | None:
        marker = f"{prefix}_page:"
        if not data.startswith(marker):
            return None
        raw_page = data[len(marker) :]
        try:
            page = int(raw_page)
        except ValueError:
            return None
        return cls(prefix=prefix, page=page)


@dataclass(frozen=True)
class ValueSelectionCallback:
    prefix: str
    value: str

    def pack(self) -> str:
        return f"{self.prefix}:{self.value}"

    @classmethod
    def parse(cls, data: str, *, prefix: str) -> ValueSelectionCallback | None:
        marker = f"{prefix}:"
        if not data.startswith(marker):
            return None
        return cls(prefix=prefix, value=data[len(marker) :])
