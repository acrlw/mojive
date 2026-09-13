"""Immediate drawing protocol for ImGui-owned UI geometry."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Draw2D(Protocol):
    """Immediate-mode overlays in logical window coordinates, with float RGBA colors.

    Later calls paint over earlier ones. Geometry helpers own local paths;
    adapters own submission and antialiasing. See docs/how-to/ui-drawing.md.
    """

    def line(
        self, a, b, color, width: float, *, cap: str = "butt", smoothing: float | None = None
    ) -> None:
        """Stroke authored coordinates exactly, like a two-point polyline."""
        ...

    def arrow(
        self,
        a,
        b,
        color,
        width: float = 2.0,
        *,
        head_length: float = 7.0,
        head_width: float = 8.0,
        corner_radius: float = 1.0,
        join_radius: float | None = None,
        smoothing: float | None = None,
        round_tail: bool = False,
    ) -> None:
        """Draw one arrow outline; smoothing None uses the adapter's default.

        Radius zero is sharp; smoothing zero uses circular corners. Explicit
        dimensions use the same units as endpoints. The adapter controls AA.
        """
        ...

    def polyline(
        self,
        points,
        color,
        width: float,
        *,
        closed: bool = False,
        cap: str = "butt",
        smoothing: float | None = None,
    ) -> None: ...

    def convex_fill(self, points, color) -> None: ...

    def indexed_fill(
        self,
        points,
        indices,
        color,
        *,
        outline=(),
        hole=(),
        origin=None,
        direction=(1.0, 0.0),
        fringe_width=1.0,
    ) -> None:
        """Fill triangles and optional AA contours, optionally placing a local mesh.

        With origin set, direction is the unit local X axis in window coordinates.
        Placement is rigid; fringe_width is measured in logical pixels.
        """
        ...

    def triangle_fan_fill(self, points, color) -> None:
        """Fill a simple polygon visible from its first vertex, such as a sector."""
        ...

    def concave_fill(self, points, color) -> None: ...

    def fringed_concave_fill(self, points, color, *, origin=None) -> None:
        """Concave fill with a 1 px alpha-gradient fringe instead of builtin AA."""
        ...

    def circle(
        self, center, radius: float, color, width: float = 1.0, *, segments: int = 0
    ) -> None: ...

    def circle_filled(self, center, radius: float, color, *, segments: int = 0) -> None: ...

    def rect(
        self,
        lo,
        hi,
        color,
        width: float = 1.0,
        *,
        rounding: float = 0.0,
        smoothing: float | None = None,
    ) -> None: ...

    def rect_filled(
        self, lo, hi, color, *, rounding: float = 0.0, smoothing: float | None = None
    ) -> None: ...

    def text(self, pos, color, text: str, *, pixel_snap: bool = True) -> None: ...

    def text_size(self, text: str) -> tuple[float, float]: ...

    def text_ink_bounds(self, text: str) -> tuple[float, float, float, float] | None:
        """Visible glyph bounds relative to the text pen position."""
        ...

    def centered_label(self, text: str, center, color, max_width: float) -> None:
        """Text centered on its ink box at ``center``, shrinking to fit ``max_width``."""
        ...
