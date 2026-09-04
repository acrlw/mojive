import math
from dataclasses import replace
from itertools import pairwise

import pytest

from mojive.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
from mojive.ui.viewport_widgets import (
    _FRAME_ARROW_CORNER_RADIUS_PT,
    _MOVE_ARROW_BASE,
    _MOVE_ARROW_TIP,
    _MOVE_ARROW_WING,
    DEFAULT_VIEWPORT_LABELS,
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    HINT_CHROME_SCALE,
    OVERLAY_CLIP_PADDING,
    OVERLAY_GEOMETRY,
    PLAYBACK_CHROME_SCALE,
    TOOL_CHROME_SCALE,
    TOOL_GLYPH_SCALE,
    ToolHint,
    ToolHintRegistry,
    ViewportChromeRegistry,
    ViewportControl,
    ViewportLabels,
    _hint_groups,
    _polygon_area,
    _rotate_visible_ring_polygons,
    _status_performance_layout,
    _viewport_tooltip_padding,
    capsule_points,
    draw_hint,
    draw_mouse_hint_glyph,
    draw_playback_glyph,
    draw_projection_glyph,
    draw_projection_label,
    draw_status,
    draw_tool_glyph,
    fitting_tool_hints,
    format_simulation_metric,
    format_simulation_steps,
    format_simulation_time,
    hint_size,
    mouse_button_geometry,
    mouse_wheel_geometry,
    normalized_overlay_position,
    overlay_border_hit,
    playback_size,
    positioned_overlay_rect,
    tool_column_size,
    tool_hints_size,
    viewport_chrome_scale,
)


def test_normalized_overlay_position_clamps_and_round_trips() -> None:
    viewport = (100.0, 50.0, 400.0, 200.0)
    rect = positioned_overlay_rect(viewport, (120.0, 40.0), (0.5, 0.5), (0.0, 0.0))
    assert rect == pytest.approx((240.0, 130.0, 360.0, 170.0))
    assert normalized_overlay_position(viewport, rect) == pytest.approx((0.5, 0.5))

    clamped = positioned_overlay_rect(viewport, (120.0, 40.0), (0.0, 0.0), (0.0, 0.0))
    assert clamped == pytest.approx((104.0, 54.0, 224.0, 94.0))
    oversized = positioned_overlay_rect(viewport, (500.0, 300.0), (0.0, 0.0), (0.0, 0.0))
    assert normalized_overlay_position(viewport, oversized) == pytest.approx((0.5, 0.5))


def test_only_overlay_border_band_is_a_drag_target() -> None:
    rect = (10.0, 20.0, 110.0, 60.0)
    assert overlay_border_hit((12.0, 40.0), rect, 6.0)
    assert overlay_border_hit((60.0, 23.0), rect, 6.0)
    assert not overlay_border_hit((60.0, 40.0), rect, 6.0)
    assert not overlay_border_hit((0.0, 0.0), rect, 6.0)


@pytest.mark.parametrize(
    ("width", "height"),
    (
        (116.0, 44.0),
        (44.0, 162.0),
        (44.0, 44.0),
    ),
)
def test_capsule_points_keep_exact_bounds_and_circular_ends(width, height):
    points = capsule_points(7.0, 11.0, width, height)
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    assert min(xs) == pytest.approx(7.0)
    assert max(xs) == pytest.approx(7.0 + width)
    assert min(ys) == pytest.approx(11.0)
    assert max(ys) == pytest.approx(11.0 + height)

    radius = min(width, height) * 0.5
    if width >= height:
        centers = ((7.0 + radius, 11.0 + radius), (7.0 + width - radius, 11.0 + radius))
        halves = (points[len(points) // 2 :], points[: len(points) // 2])
    else:
        centers = ((7.0 + radius, 11.0 + radius), (7.0 + radius, 11.0 + height - radius))
        halves = (points[: len(points) // 2], points[len(points) // 2 :])
    for center, half in zip(centers, halves, strict=True):
        for point in half:
            assert math.dist(point, center) == pytest.approx(radius, abs=1e-9)


def test_default_overlay_scale_preserves_shared_radial_steps():
    geometry = OVERLAY_GEOMETRY
    assert (
        geometry.icon_radius,
        geometry.radial_step,
        geometry.center_step,
        geometry.tool_center_step,
        geometry.tool_group_gap,
        geometry.divider_width,
        geometry.tool_stroke,
        geometry.rotate_ring_gap,
    ) == pytest.approx((10.0, 8.0, 42.0, 42.0, 10.0, 20.0, 1.46, 0.73))
    assert geometry.rotate_ring_cap == "round"
    assert geometry.state_radius - geometry.icon_radius == pytest.approx(geometry.radial_step)
    assert geometry.shell_radius - geometry.state_radius == pytest.approx(geometry.radial_step)
    assert playback_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((222.5, 65.0))
    assert tool_column_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((65.0, 235.0))
    assert OVERLAY_GEOMETRY.tool_center_step > OVERLAY_GEOMETRY.state_radius * 2.0


def test_default_mouse_hint_geometry_matches_accepted_probe_settings():
    geometry = OVERLAY_GEOMETRY
    assert (
        geometry.hint_mouse_width,
        geometry.hint_mouse_stroke,
        geometry.hint_mouse_button_width_ratio,
        geometry.hint_mouse_button_shell_ratio,
        geometry.hint_mouse_button_height_ratio,
        geometry.hint_mouse_wheel_width_ratio,
        geometry.hint_mouse_wheel_height_ratio,
        geometry.hint_mouse_wheel_gap_ratio,
    ) == pytest.approx((14.0, 1.0, 0.40, 1.25, 0.40, 0.32, 0.40, 1.0))


def test_capsule_host_guard_band_covers_outline_and_antialiasing() -> None:
    outline_half_width = 1.4 * 0.5
    antialias_fringe = 2.0

    assert outline_half_width + antialias_fringe < OVERLAY_CLIP_PADDING


def test_tool_glyph_scale_increases_only_the_visual_paths():
    assert TOOL_GLYPH_SCALE > 1.0
    assert TOOL_GLYPH_SCALE * OVERLAY_GEOMETRY.icon_radius < OVERLAY_GEOMETRY.state_radius


def test_transient_chrome_tracks_large_global_ui_scale():
    playback = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, PLAYBACK_CHROME_SCALE)
    tools = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, TOOL_CHROME_SCALE)
    hint = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, HINT_CHROME_SCALE)

    assert playback == pytest.approx(1.4 * DEFAULT_VIEWPORT_OVERLAY_SCALE * PLAYBACK_CHROME_SCALE)
    assert hint == pytest.approx(playback)
    assert tools == playback
    assert tool_column_size(tools)[0] == pytest.approx(playback_size(playback)[1])


def test_viewport_tooltip_padding_scales_with_chrome():
    assert _viewport_tooltip_padding(1.0) == pytest.approx((7.0, 4.0))
    assert _viewport_tooltip_padding(1.5) == pytest.approx((10.5, 6.0))


class _MeasuredText:
    def text_size(self, value):
        return (len(value) * 7.0, 14.0)


class _RecordedGlyph:
    def __init__(self):
        self.paths = []
        self.lines = []
        self.fills = []
        self.circles = []
        self.filled_circles = []
        self.polylines = []
        self.rectangles = []

    def fringed_concave_fill(self, points, _color):
        self.paths.append(tuple(points))

    def concave_fill(self, points, _color):
        self.paths.append(tuple(points))

    def circle_filled(self, *args, **kwargs):
        self.filled_circles.append((args, kwargs))

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def convex_fill(self, points, _color):
        self.fills.append(tuple(points))

    def centered_label(self, *_args, **_kwargs):
        pass

    def circle(self, *args, **kwargs):
        self.circles.append((args, kwargs))

    def polyline(self, *args, **kwargs):
        self.polylines.append((args, kwargs))

    def rect_filled(self, *args, **kwargs):
        self.rectangles.append((args, kwargs))


@pytest.mark.parametrize("kind", ("play", "previous", "step"))
def test_playback_triangles_use_true_rounded_paths(kind: str):
    draw = _RecordedGlyph()

    draw_playback_glyph(draw, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), 1.0, kind)

    assert len(draw.paths) == 1
    assert len(draw.paths[0]) > 3
    assert not any(point == pytest.approx((7.0, 0.0)) for point in draw.paths[0])
    if kind in ("previous", "step"):
        assert draw.rectangles[0][1]["rounding"] > 0.0


@pytest.mark.parametrize("kind", ("previous", "step"))
def test_frame_step_glyph_is_centered_and_fills_the_icon_bound(kind: str):
    draw = _RecordedGlyph()

    draw_playback_glyph(draw, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), 1.0, kind)

    points = (*draw.paths[0], *draw.rectangles[0][0][:2])
    xs = tuple(point[0] for point in points)
    assert abs(min(xs) + max(xs)) < 0.75
    assert max(math.hypot(*point) for point in points) < OVERLAY_GEOMETRY.icon_radius

    play = _RecordedGlyph()
    draw_playback_glyph(play, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0), 1.0, "play")
    play_x = tuple(point[0] for point in play.paths[0])
    play_y = tuple(point[1] for point in play.paths[0])
    triangle_x = tuple(point[0] for point in draw.paths[0])
    triangle_y = tuple(point[1] for point in draw.paths[0])
    assert (max(triangle_x) - min(triangle_x)) / (max(play_x) - min(play_x)) == pytest.approx(0.78)
    assert (max(triangle_y) - min(triangle_y)) / (max(play_y) - min(play_y)) == pytest.approx(0.78)


def test_projection_glyph_distinguishes_converging_and_parallel_edges():
    perspective = _RecordedGlyph()
    orthographic = _RecordedGlyph()

    draw_projection_glyph(perspective, (0.0, 0.0), (1.0,) * 4, 1.0, "persp")
    draw_projection_glyph(orthographic, (0.0, 0.0), (1.0,) * 4, 1.0, "ortho")

    assert perspective.lines == orthographic.lines == []
    assert len(perspective.polylines) == len(orthographic.polylines) == 1
    for draw in (perspective, orthographic):
        args, kwargs = draw.polylines[0]
        assert len(args[0]) == 4
        assert kwargs == {"closed": True}
    perspective_path = perspective.polylines[0][0][0]
    orthographic_path = orthographic.polylines[0][0][0]
    assert perspective_path[0][1] != pytest.approx(perspective_path[1][1])
    assert orthographic_path[0][1] == pytest.approx(orthographic_path[1][1])


@pytest.mark.parametrize("scale", (0.75, 1.0, 2.0, 4.0))
@pytest.mark.parametrize(
    ("kind", "label"),
    (("persp", "persp"), ("ortho", "ortho"), ("persp", "透视"), ("ortho", "正交")),
)
def test_projection_labels_center_the_visible_pair_and_share_a_body_line(scale, kind, label):
    ink = (2.0 * scale, 2.0 * scale, (len(label) * 7.0 - 3.0) * scale, 18.0 * scale)

    class LabelDraw(_RecordedGlyph):
        def text_size(self, text):
            return len(text) * 7.0 * scale, 18.0 * scale

        def text_ink_bounds(self, text):
            if text == label:
                return ink
            assert text == ("x" if label.isascii() else "田")
            return 0.0, 6.0 * scale, 7.0 * scale, 14.0 * scale

        def text(self, pos, color, text, *, pixel_snap=True):
            assert not pixel_snap
            self.label = text, pos, color

    draw = LabelDraw()
    lo, hi = (12.25, 24.5), (172.25, 24.5 + 24.0 * scale)
    draw_projection_label(draw, lo, hi, (1.0,) * 4, scale, kind, label)

    points = draw.polylines[0][0][0]
    stroke = draw.polylines[0][0][2]
    expected_y = (lo[1] + hi[1]) * 0.5
    assert (min(p[1] for p in points) + max(p[1] for p in points)) * 0.5 == pytest.approx(
        expected_y
    )
    assert draw.label[0] == label
    assert draw.label[1][1] + 10.0 * scale == pytest.approx(expected_y)
    icon_left = min(p[0] for p in points) - stroke * 0.5
    icon_right = max(p[0] for p in points) + stroke * 0.5
    label_left = draw.label[1][0] + ink[0]
    label_right = draw.label[1][0] + ink[2]
    assert label_left - icon_right == pytest.approx(7.0 * scale)
    assert icon_left + label_right == pytest.approx(lo[0] + hi[0])


def test_move_glyph_is_one_connected_rounded_antialiased_outline():
    draw = _RecordedGlyph()
    center = (20.0, 30.0)

    draw_tool_glyph(draw, center, (1.0, 1.0, 1.0, 1.0), 1.0, "move", "world")

    assert len(draw.paths) == 1
    path = draw.paths[0]
    assert len(path) > 24
    xs = tuple(point[0] for point in path)
    ys = tuple(point[1] for point in path)
    assert center[0] - min(xs) == pytest.approx(max(xs) - center[0])
    assert center[1] - min(ys) == pytest.approx(max(ys) - center[1])
    assert max(xs) - min(xs) == pytest.approx(max(ys) - min(ys))
    sharp_tip = 9.0 * TOOL_GLYPH_SCALE
    assert all(
        not any(point == pytest.approx(expected) for point in path)
        for expected in (
            (center[0], center[1] - sharp_tip),
            (center[0] + sharp_tip, center[1]),
            (center[0], center[1] + sharp_tip),
            (center[0] - sharp_tip, center[1]),
        )
    )
    assert _polygon_area(path) > 0.0
    assert pytest.approx((_MOVE_ARROW_TIP - _MOVE_ARROW_BASE) / math.sqrt(3.0)) == (
        _MOVE_ARROW_WING
    )


def test_tool_glyphs_use_the_configured_stroke_without_hidden_scales():
    center = (20.0, 30.0)
    color = (1.0, 1.0, 1.0, 1.0)
    move = _RecordedGlyph()
    rotate = _RecordedGlyph()
    frame = _RecordedGlyph()

    draw_tool_glyph(move, center, color, 1.0, "move", "world")
    draw_tool_glyph(rotate, center, color, 1.0, "rotate", "world")
    draw_tool_glyph(frame, center, color, 1.0, "frame", "world")

    move_path = move.paths[0]
    move_shaft_half = OVERLAY_GEOMETRY.tool_stroke * 0.5
    move_shaft_corner = (
        center[0] + move_shaft_half,
        center[1] - 5.0 * TOOL_GLYPH_SCALE,
    )
    assert any(point == pytest.approx(move_shaft_corner) for point in move_path)
    move_shaft_width = move_shaft_half * 2.0
    move_top_head = [point for point in move_path if point[1] <= center[1] - 4.5 * TOOL_GLYPH_SCALE]
    move_head_width = max(point[0] for point in move_top_head) - min(
        point[0] for point in move_top_head
    )
    frame_head_width = max(point[0] for point in frame.paths[0]) - min(
        point[0] for point in frame.paths[0]
    )
    assert move_shaft_width == pytest.approx(OVERLAY_GEOMETRY.tool_stroke)
    assert len(rotate.circles) == 1
    assert rotate.circles[0][0][3] == pytest.approx(OVERLAY_GEOMETRY.tool_stroke)
    assert len(rotate.paths) == 6
    assert len(frame.lines) == 3
    assert all(
        args[3] == pytest.approx(OVERLAY_GEOMETRY.tool_stroke) for args, _kwargs in frame.lines
    )
    assert move_shaft_width < min(move_head_width, frame_head_width)
    assert move_head_width < 2.0 * _MOVE_ARROW_WING * TOOL_GLYPH_SCALE
    assert frame_head_width < 2.0 * 1.8 * TOOL_GLYPH_SCALE


def test_rotate_glyph_uses_antialiased_transparent_knockout_breaks():
    draw = _RecordedGlyph()

    draw_tool_glyph(
        draw,
        (20.0, 30.0),
        (1.0, 1.0, 1.0, 1.0),
        4.0,
        "rotate",
        "world",
    )

    # Each authored half-ring is behind at one crossing and in front at the
    # other, so all three split into two antialiased filled silhouettes.
    assert len(draw.paths) == 6
    assert not draw.polylines
    assert len(draw.circles) == 1
    assert not draw.filled_circles


def test_rotate_glyph_uses_cyclic_axis_occlusion():
    rings = _rotate_visible_ring_polygons(
        OVERLAY_GEOMETRY.tool_stroke,
        OVERLAY_GEOMETRY.rotate_ring_gap_ratio,
        "butt",
    )

    assert tuple(len(ring) for ring in rings) == (2, 2, 2)
    assert all(_polygon_area(polygon) > 0.0 for ring in rings for polygon in ring)
    assert OVERLAY_GEOMETRY.rotate_ring_gap_ratio == pytest.approx(0.5)
    assert OVERLAY_GEOMETRY.rotate_ring_gap == pytest.approx(OVERLAY_GEOMETRY.tool_stroke * 0.5)
    custom = replace(OVERLAY_GEOMETRY, tool_stroke=2.0, rotate_ring_gap_ratio=0.25)
    assert custom.rotate_ring_gap == pytest.approx(0.5)


@pytest.mark.parametrize("stroke", (1.0, 1.44, 2.2))
@pytest.mark.parametrize("gap_ratio", (0.25, 0.5, 0.9))
@pytest.mark.parametrize("cap", ("butt", "round"))
def test_rotate_shell_subtraction_supports_geometry_controls(stroke, gap_ratio, cap):
    rings = _rotate_visible_ring_polygons(stroke, gap_ratio, cap)

    assert tuple(len(ring) for ring in rings) == (2, 2, 2)
    assert all(_polygon_area(polygon) > 0.0 for ring in rings for polygon in ring)


def test_rotate_glyph_supports_butt_and_round_authored_caps():
    color = (1.0, 1.0, 1.0, 1.0)
    butt = _RecordedGlyph()
    rounded = _RecordedGlyph()

    draw_tool_glyph(
        butt,
        (20.0, 30.0),
        color,
        4.0,
        "rotate",
        "world",
        replace(OVERLAY_GEOMETRY, rotate_ring_cap="butt"),
    )
    draw_tool_glyph(
        rounded,
        (20.0, 30.0),
        color,
        4.0,
        "rotate",
        "world",
        replace(OVERLAY_GEOMETRY, rotate_ring_cap="round"),
    )

    assert len(butt.paths) == len(rounded.paths) == 6
    assert sum(map(len, rounded.paths)) > sum(map(len, butt.paths))


@pytest.mark.parametrize("scale", (0.75, 1.0, 2.0, 4.0))
def test_frame_arrows_scale_native_stroke_shafts_and_center_shell(scale: float):
    draw = _RecordedGlyph()
    center = (20.0, 30.0)

    draw_tool_glyph(
        draw,
        center,
        (1.0, 1.0, 1.0, 1.0),
        scale,
        "frame",
        "world",
    )

    assert len(draw.paths) == 3
    assert len(draw.lines) == 3
    assert not draw.fills
    heads = draw.paths
    head_areas = []
    for path in heads:
        assert len(path) > 3
        assert max(math.dist(point, center) for point in path) < 10.0 * TOOL_GLYPH_SCALE * scale
        head_areas.append(abs(_polygon_area(path)))
    assert head_areas == pytest.approx([head_areas[0]] * 3)
    clear_radius = (
        OVERLAY_GEOMETRY.frame_center_radius * TOOL_GLYPH_SCALE
        + OVERLAY_GEOMETRY.tool_stroke * OVERLAY_GEOMETRY.frame_center_gap_ratio
    )
    for (start, _end, _color, width), kwargs in draw.lines:
        assert math.dist(start, center) == pytest.approx(clear_radius * scale)
        assert width == pytest.approx(OVERLAY_GEOMETRY.tool_stroke * scale)
        assert not kwargs
    assert _FRAME_ARROW_CORNER_RADIUS_PT < 0.5
    assert len(draw.filled_circles) == 1
    center_args, center_kwargs = draw.filled_circles[0]
    assert center_args[0] == center
    assert center_args[1] == pytest.approx(
        OVERLAY_GEOMETRY.frame_center_radius * TOOL_GLYPH_SCALE * scale
    )
    assert center_args[1] * 2.0 > OVERLAY_GEOMETRY.tool_stroke * scale
    assert center_args[2] == (1.0, 1.0, 1.0, 1.0)
    assert center_kwargs["segments"] == 16


def test_status_backend_and_fps_use_independent_stable_columns():
    draw = _MeasuredText()
    two_digits = _status_performance_layout(draw, 500.0, 1.0, "OpenGL", 0.009, 99.9)
    three_digits = _status_performance_layout(draw, 500.0, 1.0, "OpenGL", 0.0167, 107.0)

    assert two_digits.backend_x == pytest.approx(three_digits.backend_x)
    assert two_digits.delta_text == "Δt 0.009 s"
    assert three_digits.delta_text == "Δt 0.0167 s"
    assert two_digits.dividers == pytest.approx(three_digits.dividers)
    assert two_digits.fps_x + draw.text_size(two_digits.fps_text)[0] == pytest.approx(500.0)
    assert three_digits.fps_x + draw.text_size(three_digits.fps_text)[0] == pytest.approx(500.0)


class _RecordedStatus(_MeasuredText):
    def __init__(self):
        self.texts = []
        self.text_positions = []
        self.text_colors = []
        self.lines = []
        self.circles = []
        self.polylines = []
        self.convex_fills = []

    def text(self, position, color, value):
        self.texts.append(value)
        self.text_positions.append((position, value))
        self.text_colors.append(color)

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def circle_filled(self, *args, **kwargs):
        self.circles.append((args, kwargs))

    def polyline(self, *args, **kwargs):
        self.polylines.append((args, kwargs))

    def convex_fill(self, *args, **kwargs):
        self.convex_fills.append((args, kwargs))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def test_status_places_simulation_state_before_selection():
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        900.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        state="paused",
        sim_time=0.2,
        step=674,
        metric_mode="steps",
        backend="OpenGL",
        dt=0.002,
        fps=60.0,
    )

    assert draw.texts[:2] == ["Paused", "01_revolute"]
    assert draw.lines[0][0][2] == THEME.border
    assert draw.lines[0][0][3] == pytest.approx(1.0)
    assert draw.lines[0][0][0][1] == pytest.approx(0.5)
    assert draw.lines[0][0][1][1] == pytest.approx(0.5)
    assert draw.circles[0][0][0][1] == pytest.approx(14.5)
    x_by_text = {text: position[0] for position, text in draw.text_positions}
    assert x_by_text["01_revolute"] < x_by_text["OpenGL"]
    assert x_by_text["OpenGL"] < x_by_text["Steps 674"] < x_by_text["Δt 0.002 s"]


def test_status_exposes_recording_controls_and_warning_divider() -> None:
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    layout = draw_status(
        draw,
        (0.0, 0.0),
        1100.0,
        24.0,
        THEME,
        1.0,
        selected="body",
        state="paused",
        sim_time=0.2,
        step=10,
        metric_mode="time",
        backend="wgpu",
        dt=0.002,
        fps=60.0,
        recording_phase="recording",
        recording_duration=65.2,
        recording_surface="viewport",
    )

    assert "VIEW 01:05" in draw.texts
    assert draw.lines[0][0][2] == THEME.warning
    assert layout.recording_pause_rect is not None
    assert layout.recording_stop_rect is not None


def test_right_aligned_telemetry_has_no_separator_against_empty_space():
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        900.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        state="paused",
        sim_time=0.2,
        step=0,
        metric_mode="steps",
        backend="wgpu",
        dt=0.002,
        fps=57.5,
    )

    backend_x = next(position[0] for position, text in draw.text_positions if text == "wgpu")
    vertical = [
        args[0][0] for args, _kwargs in draw.lines if args[0][0] == pytest.approx(args[1][0])
    ]
    assert not any(backend_x - 30.0 < x < backend_x for x in vertical)
    assert any(x > backend_x for x in vertical)


def test_status_renders_context_hints_after_core_simulation_fields():
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        1400.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        state="paused",
        sim_time=0.2,
        step=674,
        metric_mode="steps",
        backend="OpenGL",
        dt=0.002,
        fps=60.0,
        tool_hints=(ToolHint("key", "T", "Frame"),),
    )

    assert draw.texts[:4] == ["Paused", "01_revolute", "T", "Frame"]
    x_by_text = {text: position[0] for position, text in draw.text_positions}
    assert x_by_text["Frame"] < x_by_text["OpenGL"]
    assert x_by_text["OpenGL"] < x_by_text["Steps 674"] < x_by_text["Δt 0.002 s"]


def test_status_uses_muted_gray_for_chrome_and_context_hints():
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        1800.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        state="running",
        sim_time=0.2,
        step=674,
        metric_mode="steps",
        backend="OpenGL",
        dt=0.002,
        fps=60.0,
        status="Saved scene",
        status_level="success",
        tool_hints=(ToolHint("key", "T", "Frame"), ToolHint("mouse", "left", "Select")),
    )

    colors_by_text = dict(zip(draw.texts, draw.text_colors, strict=True))
    assert colors_by_text.pop("Running") == THEME.primary
    assert set(colors_by_text.values()) == {THEME.text_disabled}
    assert draw.lines[0][0][2] == THEME.primary_dim
    assert draw.lines[0][0][3] == pytest.approx(1.0)
    assert draw.circles[0][0][2] == THEME.primary
    assert draw.polylines[0][0][1] == THEME.text_disabled
    assert draw.convex_fills[0][0][1] == THEME.bg_frame_active
    status_x = next(position[0] for position, text in draw.text_positions if text == "Saved scene")
    vertical_dividers = [
        args[0][0] for args, _kwargs in draw.lines if args[0][0] == pytest.approx(args[1][0])
    ]
    assert not any(status_x - 12.0 < x < status_x for x in vertical_dividers)


@pytest.mark.parametrize("width", (48.0, 80.0, 140.0, 220.0, 300.0, 400.0, 520.0))
def test_status_progressively_collapses_without_text_overlap(width):
    from mojive.ui.theme import THEME

    draw = _RecordedStatus()
    layout = draw_status(
        draw,
        (0.0, 0.0),
        width,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        state="paused",
        sim_time=0.2,
        step=674,
        metric_mode="steps",
        backend="wgpu",
        dt=0.002,
        fps=59.8,
        status="Saved scene",
        tool_hints=(ToolHint("key", "Shift", "Snap"),),
    )

    intervals = sorted(
        (position[0], position[0] + draw.text_size(text)[0], text)
        for position, text in draw.text_positions
        if text
    )
    assert all(0.0 <= left <= right <= width for left, right, _text in intervals)
    assert all(
        left >= previous_right for (_, previous_right, _), (left, _, _) in pairwise(intervals)
    )
    if layout.metric_rect is not None:
        assert 0.0 <= layout.metric_rect[0] < layout.metric_rect[2] <= width


class _RecordedHint(_MeasuredText):
    def __init__(self):
        self.lines = []

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _RecordedMouse(_MeasuredText):
    def __init__(self):
        self.rects = []
        self.lines = []
        self.polylines = []
        self.filled_rects = []
        self.convex_fills = []
        self.events = []

    def rect(self, *args, **kwargs):
        self.rects.append((args, kwargs))
        self.events.append("shell")

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def polyline(self, *args, **kwargs):
        self.polylines.append((args, kwargs))
        self.events.append("shell")

    def rect_filled(self, *args, **kwargs):
        self.filled_rects.append((args, kwargs))
        self.events.append("fill")

    def convex_fill(self, *args, **kwargs):
        self.convex_fills.append((args, kwargs))
        self.events.append("fill")

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


@pytest.mark.parametrize("scale", (1.0, 2.0, 2.25))
@pytest.mark.parametrize("button", ("left", "right"))
def test_mouse_hint_button_replaces_its_part_of_the_blender_style_shell(
    scale: float, button: str
) -> None:
    from mojive.ui.theme import THEME

    draw = _RecordedMouse()
    width = draw_mouse_hint_glyph(draw, 10.0, 30.0, button, "", THEME, scale)

    assert width == pytest.approx(OVERLAY_GEOMETRY.hint_mouse_width * scale)
    assert draw.rects == []
    assert draw.lines == []
    assert len(draw.polylines) == 1
    assert len(draw.convex_fills) == 1
    assert draw.events == ["shell", "fill"]
    shell_args, shell_kwargs = draw.polylines[0]
    fill_args, _fill_kwargs = draw.convex_fills[0]
    shell_points, shell_color, shell_stroke = shell_args
    fill_points, fill_color = fill_args
    assert shell_kwargs == {}
    assert shell_color == THEME.text
    assert shell_stroke == pytest.approx(OVERLAY_GEOMETRY.hint_mouse_stroke * scale)
    assert fill_color == THEME.primary

    x = 10.0
    mouse_width = OVERLAY_GEOMETRY.hint_mouse_width * scale
    mouse_height = OVERLAY_GEOMETRY.hint_control_height * scale
    mouse_y = 30.0 - mouse_height * 0.5
    half_stroke = shell_stroke * 0.5
    shell_gap = shell_stroke * OVERLAY_GEOMETRY.hint_mouse_button_shell_ratio
    button_width = (mouse_width + shell_stroke) * OVERLAY_GEOMETRY.hint_mouse_button_width_ratio
    button_bottom = mouse_y + mouse_height * OVERLAY_GEOMETRY.hint_mouse_button_height_ratio
    fill_xs = [point[0] for point in fill_points]
    fill_ys = [point[1] for point in fill_points]
    assert min(fill_ys) == pytest.approx(mouse_y - half_stroke)
    assert max(fill_ys) == pytest.approx(button_bottom)
    if button == "left":
        assert min(fill_xs) == pytest.approx(x - half_stroke)
        assert max(fill_xs) - min(fill_xs) == pytest.approx(button_width)
        assert shell_points[0] == pytest.approx((max(fill_xs) + shell_gap, mouse_y))
        assert shell_points[-1] == pytest.approx((x, button_bottom + shell_gap))
    else:
        assert max(fill_xs) == pytest.approx(x + mouse_width + half_stroke)
        assert max(fill_xs) - min(fill_xs) == pytest.approx(button_width)
        assert shell_points[0] == pytest.approx((min(fill_xs) - shell_gap, mouse_y))
        assert shell_points[-1] == pytest.approx((x + mouse_width, button_bottom + shell_gap))


@pytest.mark.parametrize(
    ("scale", "pixel_size"),
    ((1.0, 1.0), (1.0, 0.5), (2.0, 1.0), (4.0, 1.0)),
)
def test_mouse_wheel_uses_configured_stroke_gap_with_a_physical_pixel_minimum(
    scale: float,
    pixel_size: float,
) -> None:
    from mojive.ui.theme import THEME

    draw = _RecordedMouse()
    draw_mouse_hint_glyph(
        draw,
        10.0,
        30.0,
        "wheel",
        "",
        THEME,
        scale,
        pixel_size=pixel_size,
    )

    shell_lo, _shell_hi = draw.rects[0][0][:2]
    wheel_args, wheel_kwargs = draw.filled_rects[0]
    wheel_lo, wheel_hi = wheel_args[:2]
    stroke = OVERLAY_GEOMETRY.hint_mouse_stroke * scale
    expected_gap = max(stroke * OVERLAY_GEOMETRY.hint_mouse_wheel_gap_ratio, pixel_size)
    assert wheel_lo[1] == pytest.approx(shell_lo[1] + stroke * 0.5 + expected_gap)
    assert wheel_hi[1] - wheel_lo[1] == pytest.approx(
        OVERLAY_GEOMETRY.hint_control_height
        * scale
        * OVERLAY_GEOMETRY.hint_mouse_wheel_height_ratio
    )
    assert wheel_kwargs["rounding"] == pytest.approx(
        OVERLAY_GEOMETRY.hint_mouse_width
        * scale
        * OVERLAY_GEOMETRY.hint_mouse_wheel_width_ratio
        * 0.42
    )


@pytest.mark.parametrize("gap_ratio", (0.5, 1.0))
def test_mouse_wheel_geometry_scales_after_the_physical_gap_floor_is_inactive(gap_ratio) -> None:
    geometry = mouse_wheel_geometry(
        10.0,
        20.0,
        14.0,
        18.0,
        outline_width=5.0,
        pixel_size=0.5,
    )
    scaled = mouse_wheel_geometry(
        30.0,
        60.0,
        42.0,
        54.0,
        outline_width=15.0,
        pixel_size=0.5,
    )

    assert geometry.gap == pytest.approx(5.0)
    assert scaled.gap == pytest.approx(15.0)
    assert scaled.lo == pytest.approx(tuple(value * 3.0 for value in geometry.lo))
    assert scaled.hi == pytest.approx(tuple(value * 3.0 for value in geometry.hi))

    custom = replace(
        OVERLAY_GEOMETRY,
        hint_mouse_wheel_width_ratio=0.36,
        hint_mouse_wheel_gap_ratio=gap_ratio,
    )
    customized = mouse_wheel_geometry(
        10.0,
        20.0,
        14.0,
        18.0,
        outline_width=5.0,
        pixel_size=0.5,
        geometry=custom,
    )
    assert customized.hi[0] - customized.lo[0] == pytest.approx(14.0 * 0.36)
    assert customized.gap == pytest.approx(5.0 * gap_ratio)


def test_toolhint_separates_each_group_with_one_subtle_short_rule():
    from mojive.ui.theme import THEME

    draw = _RecordedHint()
    draw_hint(draw, (0.0, 0.0), THEME, 1.0, "ready")
    dividers = [
        args
        for args, _kwargs in draw.lines
        if abs(args[1][1] - args[0][1]) == pytest.approx(10.0)
        and args[0][0] == pytest.approx(args[1][0])
    ]

    assert len(dividers) == len(_hint_groups("ready", DEFAULT_INPUT_BINDINGS)) - 1


def test_tool_hint_registry_can_extend_and_suppress_defaults_per_surface():
    registry = ToolHintRegistry()
    defaults = (
        ToolHint("key", "T", "Frame", hint_id="frame"),
        ToolHint("mouse", "left", "Select", hint_id="select"),
    )
    registry.add("custom", ToolHint("key", "F", "Focus"))

    assert [hint.hint_id for hint in registry.resolve(defaults)] == [
        "frame",
        "select",
        "custom",
    ]

    registry.remove("select")
    assert [hint.hint_id for hint in registry.resolve(defaults)] == ["frame", "custom"]
    assert registry.resolve(defaults, surface="scene") == defaults


def test_tool_hint_fitting_never_draws_a_partial_group():
    draw = _MeasuredText()
    hints = (
        ToolHint("key", "T", "Frame"),
        ToolHint("mouse", "left", "Select"),
    )
    first_width = tool_hints_size(draw, 1.0, hints[:1])[0]

    assert fitting_tool_hints(draw, 1.0, hints, first_width) == hints[:1]
    assert fitting_tool_hints(draw, 1.0, hints, 1.0) == ()


def test_viewport_chrome_registry_dispatches_custom_actions_and_allows_removal():
    called = []
    registry = ViewportChromeRegistry()
    control = ViewportControl("capture", icon=lambda *_args: None, tooltip="Capture")

    registry.add_playback(control, lambda: called.append("capture"))
    assert registry.playback_controls[-1] == control
    assert registry.dispatch("playback", "capture")
    assert called == ["capture"]

    registry.remove("playback", "capture")
    assert all(item.name != "capture" for item in registry.playback_controls)
    assert not registry.dispatch("playback", "capture")


def test_tool_hint_and_chrome_extension_types_are_public_ui_api():
    from mojive import ui

    assert ui.ToolHint is ToolHint
    assert ui.ToolHintRegistry is ToolHintRegistry
    assert ui.ViewportChromeRegistry is ViewportChromeRegistry
    assert ui.ViewportControl is ViewportControl


def test_single_group_toolhint_has_no_separator_rule():
    from mojive.ui.theme import THEME

    draw = _RecordedHint()
    draw_hint(draw, (0.0, 0.0), THEME, 1.0, "dragging")
    dividers = [
        args
        for args, _kwargs in draw.lines
        if abs(args[1][1] - args[0][1]) == pytest.approx(10.0)
        and args[0][0] == pytest.approx(args[1][0])
    ]

    assert not dividers


def test_ready_hint_names_the_frame_switch_action_in_both_spaces():
    draw = _MeasuredText()
    assert hint_size(draw, 1.0, "ready", space="world") == hint_size(
        draw, 1.0, "ready", space="body"
    )


def test_ready_and_dragging_hints_render_snap_like_the_other_key_hints():
    for variant in ("ready", "dragging"):
        groups = _hint_groups(variant, DEFAULT_INPUT_BINDINGS)
        assert groups[0].kind == "key"
        assert groups[0].control == "Shift"
        assert groups[0].label == "Snap"


def test_remapped_shortcut_updates_the_hint_from_the_same_binding_map():
    bindings = DEFAULT_INPUT_BINDINGS.remap(InputAction.SNAP, "x")

    groups = _hint_groups("ready", bindings)

    assert groups[0].control == "X"
    assert groups[0].label == "Snap"


def test_viewport_hint_copy_is_resolved_from_one_localized_catalog():
    labels = ViewportLabels(
        **{
            **DEFAULT_VIEWPORT_LABELS.__dict__,
            "snap": "吸附",
            "type_value": "输入数值",
        }
    )

    ready = _hint_groups("ready", DEFAULT_INPUT_BINDINGS, labels)
    minimal = _hint_groups("ready_minimal", DEFAULT_INPUT_BINDINGS, labels)

    assert ready[0].label == "吸附"
    assert ready[2].label == "输入数值"
    assert minimal[0].label == "输入数值"


def test_perturb_hint_measurement_uses_localized_labels():
    draw = _MeasuredText()
    verbose = replace(
        DEFAULT_VIEWPORT_LABELS,
        drag="localized drag",
        push="localized push",
        twist="localized twist",
    )

    assert hint_size(draw, 1.0, "perturb", labels=verbose)[0] > hint_size(draw, 1.0, "perturb")[0]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    (
        (5.64, "5.640 s"),
        (65.125, "1:05.125"),
        (3661.0, "1:01:01"),
        (90061.0, "1d 01:01:01"),
    ),
)
def test_simulation_time_compacts_long_durations(seconds, expected):
    assert format_simulation_time(seconds) == expected


@pytest.mark.parametrize(
    ("steps", "expected"),
    ((999, "999"), (1_000, "1k"), (12_340, "12.3k"), (1_250_000, "1.2M")),
)
def test_simulation_steps_use_compact_suffixes(steps, expected):
    assert format_simulation_steps(steps) == expected


def test_simulation_metric_keeps_an_exact_clipboard_value():
    shown, exact = format_simulation_metric("steps", 12.0, 12_345)
    assert shown == "Steps 12.3k"
    assert exact == "12345"

    shown, exact = format_simulation_metric("time", 65.125, 12_345)
    assert shown == "Time 1:05.125"
    assert exact == "65.125 s"


def test_mouse_button_geometry_is_mirrored_and_scales_without_pixel_corrections():
    x, y, width, height = 13.0, 17.0, 17.5, 22.5
    outline_width = 1.7
    left = mouse_button_geometry(x, y, width, height, "left", outline_width=outline_width)
    right = mouse_button_geometry(x, y, width, height, "right", outline_width=outline_width)
    assert left is not None and right is not None
    assert _polygon_area(left.fill) > 0.0
    assert _polygon_area(right.fill) == pytest.approx(_polygon_area(left.fill))
    mirror_x = x * 2.0 + width
    assert all(
        actual == pytest.approx((mirror_x - px, py))
        for actual, (px, py) in zip(right.visible_shell, left.visible_shell, strict=True)
    )
    assert all(
        actual == pytest.approx((mirror_x - px, py))
        for actual, (px, py) in zip(right.fill, reversed(left.fill), strict=True)
    )

    factor = 3.25
    scaled = mouse_button_geometry(
        x * factor,
        y * factor,
        width * factor,
        height * factor,
        "left",
        outline_width=outline_width * factor,
    )
    assert scaled is not None
    assert all(
        actual == pytest.approx((px * factor, py * factor))
        for actual, (px, py) in zip(scaled.visible_shell, left.visible_shell, strict=True)
    )
    assert all(
        actual == pytest.approx((px * factor, py * factor))
        for actual, (px, py) in zip(scaled.fill, left.fill, strict=True)
    )
