from __future__ import annotations

import math

import pytest
from design.tools import ui_icon_concepts as icon_concepts
from design.tools.ui_icon_concepts import (
    BOX_CENTERED_ICONS,
    ICON_BOUND_DIAMETER,
    ICON_DEFAULT_PADDING,
    ICON_FAMILIES,
    ICON_GRID,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_MAX_PADDING,
    ICON_MIN_CLEARANCE,
    ICON_PADDING_BIAS,
    RESET_RING_CENTER,
    RING_CENTERED_ICONS,
    ROTATE_FRAME_PADDING,
    STATUS_MOUSE_DEFAULT_WIDTH,
    draw_concept_icon,
    icon_alignment_anchor,
    icon_alignment_center,
    icon_component_group,
    icon_metrics,
)

from mojive.curves2d import arrow_mesh
from mojive.ui.viewport_widgets import (
    CAPSULE_SMOOTHING,
    OVERLAY_GEOMETRY,
    mouse_button_geometry,
)


class _RecordingDraw:
    def __init__(self) -> None:
        self.bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        self.circular_extent = 0.0
        self.widths: list[float] = []
        self.lines: list[tuple] = []
        self.arrows: list[tuple] = []
        self.circles: list[tuple] = []
        self.filled_circles: list[tuple] = []
        self.solid_colors: list[tuple] = []
        self.indexed_fills: list[tuple] = []
        self.indexed_fringe_widths: list[float] = []
        self.polylines: list[tuple] = []
        self.filled_paths: list[tuple] = []
        self.convex_paths: list[tuple] = []
        self.fills = 0

    def _add(self, points, pad: float = 0.0) -> None:
        for x, y in points:
            self.bounds[0] = min(self.bounds[0], float(x) - pad)
            self.bounds[1] = min(self.bounds[1], float(y) - pad)
            self.bounds[2] = max(self.bounds[2], float(x) + pad)
            self.bounds[3] = max(self.bounds[3], float(y) + pad)
            self.circular_extent = max(self.circular_extent, math.hypot(float(x), float(y)) + pad)

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

    def convex_fill(self, points, color, **_kwargs) -> None:
        points = tuple(points)
        self.convex_paths.append(points)
        self.solid_colors.append(color)
        self.fills += 1
        self._add(points)

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
        self.indexed_fringe_widths.append(float(_kwargs.get("fringe_width", 1.0)))
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
        self.solid_colors.append(_color)
        self._add((center,), float(radius))

    def rect(self, lo, hi, _color, width, **_kwargs) -> None:
        self.widths.append(float(width))
        self._add((lo, hi), float(width) * 0.5)

    def rect_filled(self, lo, hi, _color, **_kwargs) -> None:
        self.solid_colors.append(_color)
        self._add((lo, hi))


def _icons():
    return tuple(name for _family, icons in ICON_FAMILIES for _label, name in icons)


def _render(
    name: str,
    size: float,
    *,
    padding: float | None = None,
    mouse_width: float = STATUS_MOUSE_DEFAULT_WIDTH,
) -> _RecordingDraw:
    draw = _RecordingDraw()
    draw_concept_icon(
        draw,
        (0.0, 0.0),
        size,
        name,
        (1.0, 1.0, 1.0, 1.0),
        padding=padding,
        mouse_width=mouse_width,
    )
    return draw


def test_concept_catalog_has_unique_named_members() -> None:
    names = _icons()

    assert len(ICON_FAMILIES) == 7
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
def test_concept_icon_geometry_stays_inside_circular_placement_bound(name: str) -> None:
    size = ICON_GRID
    draw = _render(name, size)
    guide_radius = ICON_BOUND_DIAMETER * 0.5

    required_clearance = ROTATE_FRAME_PADDING if name == "tool-rotate" else ICON_MIN_CLEARANCE
    assert draw.circular_extent <= guide_radius - required_clearance + 1e-3


@pytest.mark.parametrize("name", _icons())
def test_default_layout_fits_candidates_to_the_reference_family_padding(name: str) -> None:
    group = icon_component_group(name)
    expected = (
        ROTATE_FRAME_PADDING
        if name == "tool-rotate"
        else ICON_GROUP_LAYOUT_DEFAULTS[group] + ICON_PADDING_BIAS.get(name, 0.0)
    )
    assert icon_metrics(name).circular_clearance == pytest.approx(expected, abs=1e-5)


def test_component_groups_own_independent_candidate_names_and_layout_defaults() -> None:
    assert icon_component_group("playback-play") == "Viewport playback"
    assert icon_component_group("transport-play") == "Keyframe transport"
    assert ICON_GROUP_LAYOUT_DEFAULTS["Viewport tools"] == 0.5
    assert ICON_GROUP_LAYOUT_DEFAULTS["Viewport playback"] == ICON_DEFAULT_PADDING


def test_concept_icons_use_the_declared_geometric_anchor_groups() -> None:
    assert {
        "tool-snap",
        "playback-previous",
        "playback-next",
        "playback-more",
        "transport-first",
        "transport-previous",
        "transport-next",
        "transport-last",
        "transport-more",
    } == BOX_CENTERED_ICONS
    assert {"playback-reset", "transport-reset"} == RING_CENTERED_ICONS
    assert all(
        icon_alignment_anchor(name)
        == (
            "box"
            if name in BOX_CENTERED_ICONS
            else "ring"
            if name in RING_CENTERED_ICONS
            else "circle"
        )
        for name in _icons()
    )


def test_rotate_outer_frame_matches_the_placement_circle() -> None:
    metrics = icon_metrics("tool-rotate")

    assert metrics.enclosing_center == pytest.approx((0.0, 0.0), abs=1e-6)
    assert metrics.circular_clearance == pytest.approx(ROTATE_FRAME_PADDING, abs=1e-5)


def test_rotate_frame_ignores_candidate_padding_control() -> None:
    baseline = _render("tool-rotate", ICON_GRID)
    adjusted = _render("tool-rotate", ICON_GRID, padding=ICON_MAX_PADDING)

    assert adjusted.__dict__ == baseline.__dict__


@pytest.mark.parametrize("padding", (ICON_MIN_CLEARANCE, ICON_DEFAULT_PADDING, ICON_MAX_PADDING))
def test_icon_padding_control_sets_the_circular_clearance(padding: float) -> None:
    metrics = icon_metrics("panel-right", padding=padding)

    assert metrics.circular_clearance == pytest.approx(padding, abs=1e-5)


@pytest.mark.parametrize("name", ("status-info", "status-warning", "status-error"))
def test_reviewed_severity_geometry_ignores_candidate_layout_controls(name: str) -> None:
    baseline = _render(name, ICON_GRID)
    adjusted = _render(name, ICON_GRID, padding=ICON_MAX_PADDING)

    assert adjusted.__dict__ == baseline.__dict__


@pytest.mark.parametrize("padding", (ICON_MIN_CLEARANCE - 0.01, ICON_MAX_PADDING + 0.01))
def test_icon_padding_rejects_values_outside_the_review_range(padding: float) -> None:
    with pytest.raises(ValueError, match="icon padding"):
        _render("panel-right", ICON_GRID, padding=padding)


@pytest.mark.parametrize("name", _icons())
def test_concept_icons_center_their_declared_anchor(name: str) -> None:
    assert icon_alignment_center(name) == pytest.approx((0.0, 0.0), abs=0.01)


@pytest.mark.parametrize("name", tuple(sorted(BOX_CENTERED_ICONS)))
def test_box_anchored_icons_center_their_complete_visible_bounds(name: str) -> None:
    assert icon_metrics(name).center_offset == pytest.approx((0.0, 0.0), abs=0.01)


@pytest.mark.parametrize(
    "name", tuple(name for name in _icons() if icon_alignment_anchor(name) == "circle")
)
def test_circle_anchored_icons_center_their_minimum_enclosing_circle(name: str) -> None:
    assert icon_metrics(name).enclosing_center == pytest.approx((0.0, 0.0), abs=0.01)


def test_body_cube_has_three_interior_edges() -> None:
    draw = _render("tool-body", ICON_GRID)

    assert len(draw.lines) == 3
    assert any(
        options.get("closed") and len(points) > 6 for points, _width, options in draw.polylines
    )


def test_body_internal_strokes_stop_below_outer_shell_inner_edge() -> None:
    draw = _render("tool-body", ICON_GRID)
    shell, shell_width, _options = draw.polylines[0]
    shell_inner_edge = min(math.hypot(*point) for point in shell) - shell_width * 0.5

    assert all(
        math.hypot(*end) + width * 0.5 <= shell_inner_edge + 1e-6
        for _start, end, width in draw.lines
    )


def test_world_internal_strokes_stop_at_outer_ring_inner_edge() -> None:
    draw = _render("tool-world", ICON_GRID)
    _center, outer_radius, outer_width = draw.circles[0]
    inner_edge = outer_radius - outer_width * 0.5

    assert all(
        max(math.hypot(*point) for point in (start, end)) + width * 0.5 <= inner_edge + 1e-6
        for start, end, width in draw.lines
    )
    assert all(
        max(math.hypot(*point) for point in points) + width * 0.5 <= inner_edge + 1e-6
        for points, width, _options in draw.polylines
    )


@pytest.mark.parametrize("name", ("tool-world", "tool-body"))
def test_sparse_frame_tools_use_the_shared_near_boundary_envelope(name: str) -> None:
    metrics = icon_metrics(name)

    assert metrics.circular_clearance == pytest.approx(
        ICON_GROUP_LAYOUT_DEFAULTS["Viewport tools"], abs=1e-5
    )


def test_camera_candidates_share_one_master_and_circle_anchor() -> None:
    snapshot = _render("key-snapshot", ICON_GRID)
    helper = _render("helper-camera", ICON_GRID)

    assert snapshot.__dict__ == helper.__dict__
    assert icon_alignment_anchor("key-snapshot") == "circle"
    assert icon_alignment_anchor("helper-camera") == "circle"


@pytest.mark.parametrize("name", ("status-mouse-left", "status-mouse-right", "status-mouse-wheel"))
def test_mouse_candidates_apply_the_control_accent(name: str) -> None:
    draw = _RecordingDraw()
    accent = (0.2, 0.7, 0.3, 1.0)

    draw_concept_icon(
        draw,
        (0.0, 0.0),
        ICON_GRID,
        name,
        (1.0, 1.0, 1.0, 1.0),
        accent_color=accent,
    )

    assert draw.solid_colors == [accent]


@pytest.mark.parametrize("button", ("left", "right"))
def test_mouse_candidates_reuse_the_original_knockout_geometry(button: str) -> None:
    draw = _RecordingDraw()
    painter = icon_concepts._Painter(draw, (0.0, 0.0), ICON_GRID, (1.0, 1.0, 1.0, 1.0))
    icon_concepts._draw_status(
        painter,
        f"status-mouse-{button}",
        (0.2, 0.3, 0.4, 1.0),
        mouse_width=OVERLAY_GEOMETRY.hint_mouse_width,
    )
    expected = mouse_button_geometry(
        -OVERLAY_GEOMETRY.hint_mouse_width * 0.5,
        -OVERLAY_GEOMETRY.hint_control_height * 0.5,
        OVERLAY_GEOMETRY.hint_mouse_width,
        OVERLAY_GEOMETRY.hint_control_height,
        button,
        outline_width=OVERLAY_GEOMETRY.hint_mouse_stroke,
        geometry=OVERLAY_GEOMETRY,
        smoothing=CAPSULE_SMOOTHING,
    )

    assert expected is not None
    assert draw.polylines[0][0] == expected.visible_shell
    assert draw.convex_paths == [expected.fill]
    assert not draw.lines
    assert not draw.filled_circles


def test_scale_centers_its_minimum_enclosing_circle() -> None:
    metrics = icon_metrics("tool-scale")

    assert icon_alignment_anchor("tool-scale") == "circle"
    assert metrics.enclosing_center == pytest.approx((0.0, 0.0), abs=0.01)


@pytest.mark.parametrize("name", ("transport-play", "panel-right", "panel-down"))
def test_filled_triangles_use_rounded_g3_contours(name: str) -> None:
    draw = _render(name, ICON_GRID)

    assert draw.filled_paths
    assert all(len(path) > 3 for path in draw.filled_paths)


def test_rotate_uses_axis_rings_while_reset_uses_one_arrow() -> None:
    rotate = _render("tool-rotate", ICON_GRID)
    reset = _render("transport-reset", ICON_GRID)

    assert rotate.fills >= 3
    assert len(rotate.indexed_fills) == rotate.fills
    assert not rotate.circles
    assert len(set(rotate.indexed_fringe_widths)) == 1
    assert reset.fills == 1


def test_rotate_antialias_fringe_scales_until_crossing_gaps_can_hold_one_pixel() -> None:
    widths = []
    for size in (14.0, 24.0, 56.0, 112.0):
        draw = _render("tool-rotate", size)
        assert len(set(draw.indexed_fringe_widths)) == 1
        widths.append(draw.indexed_fringe_widths[0])

    assert widths[0] < widths[1] < widths[2] < widths[3]
    assert widths[3] == pytest.approx(1.0, abs=0.002)


def test_scale_uses_three_integrated_box_handles() -> None:
    draw = _render("tool-scale", ICON_GRID)

    assert draw.fills == 3
    assert not draw.lines
    assert all(len(path) > 8 for path in draw.filled_paths)
    assert len(draw.filled_circles) == 1
    dot_center, dot_radius = draw.filled_circles[0]
    nearest_shaft = min(
        math.dist(dot_center, point) for path in draw.filled_paths for point in path
    )
    assert nearest_shaft > dot_radius


def test_scale_shaft_matches_the_fitted_rotate_ring_weight() -> None:
    scale = _render("tool-scale", ICON_GRID)
    rotate = _render("tool-rotate", ICON_GRID)
    scale_shaft_width = math.dist(scale.filled_paths[0][0], scale.filled_paths[0][-1])
    _points, _indices, outer, inner = rotate.indexed_fills[0]
    rotate_ring_width = sum(math.hypot(*point) for point in outer) / len(outer) - sum(
        math.hypot(*point) for point in inner
    ) / len(inner)

    assert scale_shaft_width == pytest.approx(rotate_ring_width, rel=0.02)


def test_snap_uses_production_g3_u_path_with_two_rounded_end_blocks() -> None:
    draw = _render("tool-snap", ICON_GRID)

    assert len(draw.polylines) == 1
    path = draw.polylines[0][0]
    assert len(path) > 16
    assert not draw.lines
    assert len(draw.filled_paths) == 2
    assert all(len(path) > 4 for path in draw.filled_paths)
    assert icon_alignment_anchor("tool-snap") == "box"


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


def test_play_triangle_is_equilateral() -> None:
    points = icon_concepts._triangle_source(1.0)
    lengths = tuple(math.dist(a, b) for a, b in zip(points, points[1:] + points[:1], strict=True))

    assert lengths == pytest.approx((lengths[0],) * 3, abs=1e-6)


@pytest.mark.parametrize("direction", (-1.0, 1.0))
def test_transport_chevron_tip_is_a_right_angle(direction: float) -> None:
    chevron = icon_concepts._chevron_centerline(direction, 0.0)

    def vertex_angle(points, vertex_index: int) -> float:
        vertex = points[vertex_index]
        arms = tuple(
            (point[0] - vertex[0], point[1] - vertex[1])
            for index, point in enumerate(points)
            if index != vertex_index
        )
        cosine = sum(a * b for a, b in zip(*arms, strict=True)) / (
            math.hypot(*arms[0]) * math.hypot(*arms[1])
        )
        return math.degrees(math.acos(cosine))

    chevron_tip = 1
    assert vertex_angle(chevron, chevron_tip) == pytest.approx(90.0, abs=1e-6)


def test_playback_and_transport_more_share_one_completed_shape() -> None:
    assert (
        _render("playback-more", ICON_GRID).__dict__
        == _render("transport-more", ICON_GRID).__dict__
    )


@pytest.mark.parametrize("name", ("transport-first", "transport-last"))
def test_transport_end_bar_matches_its_triangle_height(name: str) -> None:
    draw = _render(name, ICON_GRID)
    triangle, bar = draw.filled_paths

    assert min(y for _x, y in bar) == pytest.approx(min(y for _x, y in triangle), abs=1e-6)
    assert max(y for _x, y in bar) == pytest.approx(max(y for _x, y in triangle), abs=1e-6)


@pytest.mark.parametrize(
    "name",
    tuple(
        name
        for name in _icons()
        if name.startswith(("transport-", "playback-")) and not name.endswith("record")
    ),
)
def test_transport_contours_do_not_use_raw_stroked_corners(name: str) -> None:
    draw = _render(name, ICON_GRID)

    assert not draw.lines
    assert not draw.polylines
    assert draw.filled_paths
    assert all(len(path) > 4 for path in draw.filled_paths)


def test_viewport_playback_chevrons_do_not_outgrow_play_or_pause() -> None:
    heights = {
        name: icon_metrics(name).bounds[3] - icon_metrics(name).bounds[1]
        for name in ("playback-previous", "playback-play", "playback-pause", "playback-next")
    }

    reference = max(heights["playback-play"], heights["playback-pause"])
    assert heights["playback-previous"] <= reference
    assert heights["playback-next"] <= reference


def test_keyframe_transport_chevrons_do_not_outgrow_play_or_pause() -> None:
    heights = {
        name: icon_metrics(name).bounds[3] - icon_metrics(name).bounds[1]
        for name in ("transport-previous", "transport-play", "transport-pause", "transport-next")
    }

    reference = max(heights["transport-play"], heights["transport-pause"])
    assert heights["transport-previous"] <= reference
    assert heights["transport-next"] <= reference


@pytest.mark.parametrize("prefix", ("playback", "transport"))
def test_record_is_slightly_smaller_than_stop(prefix: str) -> None:
    record = icon_metrics(f"{prefix}-record")
    stop = icon_metrics(f"{prefix}-stop")
    record_width = record.bounds[2] - record.bounds[0]
    stop_width = stop.bounds[2] - stop.bounds[0]

    assert 0.9 * stop_width < record_width < stop_width


def test_status_mouse_width_is_adjustable_without_moving_its_center() -> None:
    narrow = icon_metrics("status-mouse-left", mouse_width=9.5)
    wide = icon_metrics("status-mouse-left", mouse_width=14.5)

    assert wide.bounds[2] - wide.bounds[0] > narrow.bounds[2] - narrow.bounds[0]
    assert wide.enclosing_center == pytest.approx((0.0, 0.0), abs=0.01)


@pytest.mark.parametrize("name", tuple(sorted(RING_CENTERED_ICONS)))
def test_reset_icons_center_their_authored_ring(name: str) -> None:
    assert RESET_RING_CENTER == (0.0, 0.47)
    assert icon_alignment_anchor(name) == "ring"
    assert icon_alignment_center(name) == pytest.approx((0.0, 0.0), abs=1e-6)


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
    normalized_widths = tuple(
        tuple(width / size for width in draw.widths)
        for size, draw in zip(sizes, captures, strict=True)
    )
    for widths in normalized_widths[1:]:
        assert widths == pytest.approx(normalized_widths[0], abs=1e-6)
    assert all(width <= 2.5 / ICON_GRID + 1e-6 for width in normalized_widths[0])
