from __future__ import annotations

import math

import pytest
from design.tools.ui_icon_concepts import (
    ICON_BOUND_DIAMETER,
    ICON_FAMILIES,
    ICON_GRID,
    ICON_MIN_CLEARANCE,
    draw_concept_icon,
)

from mojive.curves2d import arrow_mesh


class _RecordingDraw:
    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.radial_extent = 0.0
        self.widths: list[float] = []
        self.lines: list[tuple] = []
        self.fills = 0

    def _add(self, points, pad: float = 0.0) -> None:
        for x, y in points:
            self.bounds[0] = min(self.bounds[0], float(x) - pad)
            self.bounds[1] = min(self.bounds[1], float(y) - pad)
            self.bounds[2] = max(self.bounds[2], float(x) + pad)
            self.bounds[3] = max(self.bounds[3], float(y) + pad)
            self.radial_extent = max(self.radial_extent, math.hypot(float(x), float(y)) + pad)

    def line(self, a, b, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        self.lines.append((a, b, float(width)))
        self._add((a, b), float(width) * 0.5)

    def arrow(
        self,
        a,
        b,
        _color,
        width,
        *,
        head_length,
        head_width,
        corner_radius,
        join_radius,
        **_kwargs,
    ) -> None:
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        _vertices, _indices, outline = arrow_mesh(
            length,
            float(width),
            head_length=float(head_length),
            head_width=float(head_width),
            corner_radius=float(corner_radius),
            join_radius=float(join_radius),
        )
        points = tuple((a[0] + x * ux - y * uy, a[1] + x * uy + y * ux) for x, y in outline)
        self.fills += 1
        self._add(points)

    def polyline(self, points, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        self._add(tuple(points), float(width) * 0.5)

    def fringed_concave_fill(self, points, _color, **_kwargs) -> None:
        self.fills += 1
        self._add(tuple(points))

    def circle(self, center, radius, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        pad = float(radius) + float(width) * 0.5
        self._add((center,), pad)

    def circle_filled(self, center, radius, _color, **_kwargs) -> None:
        self._add((center,), float(radius))

    def rect(self, lo, hi, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        self._add((lo, hi), float(width) * 0.5)

    def rect_filled(self, lo, hi, _color, **_kwargs) -> None:
        self._add((lo, hi))


def _icons():
    return tuple(name for _family, icons in ICON_FAMILIES for _label, name in icons)


def _render(name: str, size: float) -> _RecordingDraw:
    draw = _RecordingDraw()
    draw_concept_icon(draw, (0.0, 0.0), size, name, (1.0, 1.0, 1.0, 1.0))
    return draw


def test_concept_catalog_has_unique_named_members() -> None:
    names = _icons()

    assert len(ICON_FAMILIES) == 6
    assert len(names) == len(set(names))
    assert all(icons for _family, icons in ICON_FAMILIES)


@pytest.mark.parametrize("name", _icons())
def test_concept_icons_stay_inside_the_shared_canvas(name: str) -> None:
    size = 20.0
    draw = _render(name, size)

    assert draw.bounds[0] < draw.bounds[2]
    assert draw.bounds[1] < draw.bounds[3]
    assert min(draw.bounds) >= -size * 0.5 - 1e-6
    assert max(draw.bounds) <= size * 0.5 + 1e-6


@pytest.mark.parametrize("name", _icons())
def test_concept_icon_ink_stays_inside_circular_placement_bound(name: str) -> None:
    size = ICON_GRID
    draw = _render(name, size)
    guide_radius = ICON_BOUND_DIAMETER * 0.5

    assert draw.radial_extent <= guide_radius - ICON_MIN_CLEARANCE + 1e-6


@pytest.mark.parametrize("name", _icons())
def test_concept_icon_bounds_are_centered_in_placement_circle(name: str) -> None:
    draw = _render(name, ICON_GRID)
    center_x = (draw.bounds[0] + draw.bounds[2]) * 0.5
    center_y = (draw.bounds[1] + draw.bounds[3]) * 0.5

    assert center_x == pytest.approx(0.0, abs=0.05)
    assert center_y == pytest.approx(0.0, abs=0.05)


def test_body_cube_has_three_interior_edges() -> None:
    draw = _render("tool-body", ICON_GRID)

    assert len(draw.lines) == 3


@pytest.mark.parametrize("name", _icons())
def test_concept_icon_bounds_and_strokes_scale_as_one_master(name: str) -> None:
    captures = tuple(_render(name, size) for size in (14.0, 20.0, 32.0, 56.0))
    normalized_bounds = tuple(
        tuple(value / size for value in draw.bounds)
        for size, draw in zip((14.0, 20.0, 32.0, 56.0), captures, strict=True)
    )

    for bounds in normalized_bounds[1:]:
        assert bounds == pytest.approx(normalized_bounds[0], abs=1e-6)
    for size, draw in zip((14.0, 20.0, 32.0, 56.0), captures, strict=True):
        assert all(width / size <= 2.2 / ICON_GRID for width in draw.widths)
