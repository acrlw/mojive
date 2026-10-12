"""Optional Inspector field presentation; document edits remain with InspectorPanel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class InspectorPresentation(Protocol):
    def begin_table(self, table_id: str) -> bool: ...

    def row(
        self,
        label: str,
        tooltip: str = "",
        *,
        unit: str = "",
        toggle_unit=None,
        toggle: bool = False,
        components: int = 1,
    ) -> None: ...

    def section(self, label: str, flags: int) -> bool: ...

    def joint_metadata(self, joint) -> None: ...

    def numeric(
        self,
        item_id: str,
        value: float,
        speed: float,
        lo: float,
        hi: float,
        fmt: str,
        flags: int,
        *,
        integer: bool = False,
    ) -> tuple[bool, float]: ...

    def numbers(
        self,
        item_id: str,
        values: Sequence[float],
        speed: float,
        lo: float,
        hi: float,
        fmt: str,
        flags: int,
    ) -> tuple[bool, Sequence[float]]: ...

    def toggle(self, item_id: str, value: bool) -> tuple[bool, bool]: ...

    def color(self, item_id: str, values: Sequence[float]) -> tuple[bool, Sequence[float]]: ...

    def value_rail(self, item_id: str, value: float, bounds, **options): ...

    def text(self, item_id: str, value: str, flags: int) -> tuple[bool, str]: ...

    def vectors(
        self, node, table_id: str, rows, editable, *, units: tuple[str, ...] = ()
    ) -> tuple: ...

    def vector_row(self, node, label: str, name: str, values, **options) -> tuple: ...
