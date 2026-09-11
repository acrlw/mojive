from __future__ import annotations

import math

import pytest
from design.tools.ui_icon_concepts import (
    ICON_BOUND_DIAMETER,
    ICON_FAMILIES,
    ICON_GRID,
    ICON_MIN_CLEARANCE,
    draw_concept_icon,
    icon_alignment_anchor,
    icon_metrics,
    minimum_enclosing_circle,
)

from mojive.curves2d import arrow_mesh


class _RecordingDraw:
    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.radial_extent = 0.0
        self.widths: list[float] = []
        self.lines: list[tuple] = []
        self.arrows: list[tuple] = []
        self.circles: list[tuple] = []
        self.filled_circles: list[tuple] = []
        self.indexed_fills: list[tuple] = []
        self.polylines: list[tuple] = []
        self.filled_paths: list[tuple] = []
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
        round_tail = bool(_kwargs.get("round_tail", False))
        self.arrows.append((a, b, float(width), round_tail))
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
            round_tail=round_tail,
        )
        points = tuple((a[0] + x * ux - y * uy, a[1] + x * uy + y * ux) for x, y in outline)
        self.fills += 1
        self._add(points)

    def polyline(self, points, _color, width, **_kwargs) -> None:
        points = tuple(points)
        self.polylines.append((points, float(width), _kwargs))
        self.widths.append(float(width))
        self._add(points, float(width) * 0.5)

    def fringed_concave_fill(self, points, _color, **_kwargs) -> None:
        self.filled_paths.append(tuple(points))
        self.fills += 1
        self._add(tuple(points))

    def indexed_fill(
        self,
        points,
        indices,
        _color,
        *,
        outline=(),
        hole=(),
        **_kwargs,
    ) -> None:
        points = tuple(points)
        record = (points, tuple(indices), tuple(outline), tuple(hole))
        self.indexed_fills.append(record)
        self.fills += 1
        self._add(record[2] or points)

    def concave_fill(self, points, color) -> None:
        self.fringed_concave_fill(points, color)

    def circle(self, center, radius, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        self.circles.append((center, float(radius), float(width)))
        pad = float(radius) + float(width) * 0.5
        self._add((center,), pad)

    def circle_filled(self, center, radius, _color, **_kwargs) -> None:
        self.filled_circles.append((center, float(radius)))
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


def test_minimum_enclosing_circle_tracks_the_radial_envelope() -> None:
    center, radius = minimum_enclosing_circle(((1.0, 0.0), (5.0, 0.0), (3.0, 2.0)))

    assert center == pytest.approx((3.0, 0.0))
    assert radius == pytest.approx(2.0)


@pytest.mark.parametrize("name", _icons())
def test_concept_icons_stay_inside_the_shared_canvas(name: str) -> None:
    size = 20.0
    draw = _render(name, size)

    assert draw.bounds[0] < draw.bounds[2]
    assert draw.bounds[1] < draw.bounds[3]
    assert min(draw.bounds) >= -size * 0.5 - 1e-6
    assert max(draw.bounds) <= size * 0.5 + 1e-6


@pytest.mark.parametrize("name", _icons())
def test_concept_icon_geometry_stays_inside_circular_placement_bound(name: str) -> None:
    size = ICON_GRID
    draw = _render(name, size)
    guide_radius = ICON_BOUND_DIAMETER * 0.5

    assert draw.radial_extent <= guide_radius - ICON_MIN_CLEARANCE + 1e-6


@pytest.mark.parametrize(
    "name",
    tuple(name for name in _icons() if name not in {"tool-scale", "tool-snap"}),
)
def test_concept_icon_bounds_are_centered_in_placement_circle(name: str) -> None:
    draw = _render(name, ICON_GRID)
    center_x = (draw.bounds[0] + draw.bounds[2]) * 0.5
    center_y = (draw.bounds[1] + draw.bounds[3]) * 0.5

    assert center_x == pytest.approx(0.0, abs=0.05)
    assert center_y == pytest.approx(0.0, abs=0.05)


def test_body_cube_has_three_interior_edges() -> None:
    draw = _render("tool-body", ICON_GRID)

    assert len(draw.lines) == 3
    assert any(
        options.get("closed") and len(points) > 6 for points, _width, options in draw.polylines
    )


@pytest.mark.parametrize(
    "name",
    ("tool-move", "tool-world", "tool-body"),
)
def test_symmetric_tool_icons_center_their_filled_area(name: str) -> None:
    assert icon_metrics(name).area_centroid == pytest.approx((0.0, 0.0), abs=0.05)


@pytest.mark.parametrize("name", ("tool-world", "tool-body"))
def test_sparse_frame_tools_use_the_reviewed_larger_envelope(name: str) -> None:
    metrics = icon_metrics(name)

    assert 2.0 <= metrics.radial_clearance <= 3.0


def test_camera_candidates_share_one_master_and_box_anchor() -> None:
    snapshot = _render("key-snapshot", ICON_GRID)
    helper = _render("helper-camera", ICON_GRID)

    assert snapshot.__dict__ == helper.__dict__
    assert icon_alignment_anchor("key-snapshot") == "box"
    assert icon_alignment_anchor("helper-camera") == "box"


@pytest.mark.parametrize(
    "name",
    ("tool-scale",),
)
def test_semantically_anchored_symmetric_icons_center_their_area(name: str) -> None:
    assert icon_metrics(name).area_centroid == pytest.approx((0.0, 0.0), abs=0.05)


@pytest.mark.parametrize("name", ("transport-play", "panel-right", "panel-down"))
def test_filled_triangles_use_rounded_g3_contours(name: str) -> None:
    draw = _render(name, ICON_GRID)

    assert draw.filled_paths
    assert all(len(path) > 3 for path in draw.filled_paths)


def test_rotate_uses_axis_rings_while_reset_uses_one_arrow() -> None:
    rotate = _render("tool-rotate", ICON_GRID)
    reset = _render("transport-reset", ICON_GRID)

    assert rotate.fills >= 3
    assert reset.fills == 1


def test_scale_uses_three_integrated_box_handles() -> None:
    draw = _render("tool-scale", ICON_GRID)

    assert draw.fills == 3
    assert not draw.lines
    assert all(len(path) > 8 for path in draw.filled_paths)
    assert len(draw.filled_circles) == 1
    dot_center, dot_radius = draw.filled_circles[0]
    assert dot_center == pytest.approx((0.0, 0.0), abs=1e-6)
    nearest_shaft = min(
        math.dist(dot_center, point) for path in draw.filled_paths for point in path
    )
    assert nearest_shaft > dot_radius


def test_snap_uses_production_g3_u_path_with_two_rounded_end_blocks() -> None:
    draw = _render("tool-snap", ICON_GRID)

    assert len(draw.polylines) == 1
    path = draw.polylines[0][0]
    assert len(path) > 16
    assert not draw.lines
    assert len(draw.filled_paths) == 2
    assert all(len(path) > 4 for path in draw.filled_paths)
    assert icon_alignment_anchor("tool-snap") == "arc"
    assert max(x for x, _y in path) == pytest.approx(max(y for _x, y in path), abs=1e-6)


def test_search_uses_one_hollow_g3_lens_and_handle_mesh() -> None:
    draw = _render("panel-search", ICON_GRID)

    assert len(draw.indexed_fills) == 1
    assert not draw.circles
    assert not draw.lines
    points, indices, outline, hole = draw.indexed_fills[0]
    assert len(points) > 200
    assert len(indices) > 300
    assert len(outline) > 100
    assert len(hole) > 100
    assert set(hole).issubset(set(points))
    edge_lengths = tuple(
        math.dist(current, following)
        for current, following in zip(outline, (*outline[1:], outline[0]), strict=True)
    )
    hole_edge_lengths = tuple(
        math.dist(current, following)
        for current, following in zip(hole, (*hole[1:], hole[0]), strict=True)
    )
    assert min(edge_lengths) > 0.02
    assert min(hole_edge_lengths) > 0.02


def test_sort_arrow_tail_and_tip_align_with_visible_bar_extents() -> None:
    draw = _render("panel-sort", ICON_GRID)

    assert len(draw.lines) == 3
    assert len(draw.arrows) == 1
    stroke_top = min(a[1] for a, _b, _width in draw.lines)
    bottom_line = max(draw.lines, key=lambda line: line[0][1])
    stroke_bottom = bottom_line[0][1]
    arrow_start, arrow_end, _arrow_width, round_tail = draw.arrows[0]
    assert arrow_start[1] == pytest.approx(stroke_top, abs=0.01)
    assert arrow_end[1] == pytest.approx(stroke_bottom + bottom_line[2] * 0.5, abs=0.01)
    assert round_tail


@pytest.mark.parametrize("name", _icons())
def test_minimum_bounding_circle_is_only_a_containment_diagnostic(name: str) -> None:
    metrics = icon_metrics(name)

    assert metrics.bounding_radius <= ICON_BOUND_DIAMETER * 0.5 - ICON_MIN_CLEARANCE + 1e-6


@pytest.mark.parametrize(
    "name",
    (*(name for name in _icons() if name not in {"tool-scale", "tool-snap"}),),
)
def test_box_anchored_icons_center_their_visible_bounds(name: str) -> None:
    assert icon_metrics(name).center_offset == pytest.approx((0.0, 0.0), abs=0.05)


@pytest.mark.parametrize(
    "name",
    tuple(
        name for name in _icons() if name.startswith("transport-") and name != "transport-record"
    ),
)
def test_transport_contours_do_not_use_raw_stroked_corners(name: str) -> None:
    draw = _render(name, ICON_GRID)

    assert not draw.lines
    assert not draw.polylines
    assert draw.filled_paths
    assert all(len(path) > 4 for path in draw.filled_paths)


def test_hidden_eye_uses_three_lashes_instead_of_a_slash() -> None:
    hidden = _render("panel-hidden", ICON_GRID)
    visible = _render("panel-visible", ICON_GRID)

    assert len(hidden.lines) == 3
    assert not visible.lines


@pytest.mark.parametrize("name", _icons())
def test_concept_icon_bounds_and_strokes_scale_as_one_master(name: str) -> None:
    sizes = (14.0, 24.0, 56.0, 112.0)
    captures = tuple(_render(name, size) for size in sizes)
    normalized_bounds = tuple(
        tuple(value / size for value in draw.bounds)
        for size, draw in zip(sizes, captures, strict=True)
    )

    for bounds in normalized_bounds[1:]:
        assert bounds == pytest.approx(normalized_bounds[0], abs=1e-6)
    for size, draw in zip(sizes, captures, strict=True):
        assert all(width / size <= 2.2 / ICON_GRID for width in draw.widths)
