import math
from dataclasses import replace
from itertools import pairwise

import pytest

from forge_viewer.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
from forge_viewer.ui.viewport_widgets import (
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
    draw_status,
    draw_tool_glyph,
    fitting_tool_hints,
    format_simulation_metric,
    format_simulation_steps,
    format_simulation_time,
    hint_size,
    mouse_button_fill_geometry,
    playback_size,
    tool_column_size,
    tool_hints_size,
    viewport_chrome_scale,
)


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
    ) == pytest.approx((10.0, 8.0, 42.0, 42.0, 10.0, 20.0, 1.46, 1.0))
    assert geometry.rotate_ring_cap == "round"
    assert geometry.state_radius - geometry.icon_radius == pytest.approx(geometry.radial_step)
    assert geometry.shell_radius - geometry.state_radius == pytest.approx(geometry.radial_step)
    assert playback_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((170.0, 65.0))
    assert tool_column_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((65.0, 235.0))
    assert OVERLAY_GEOMETRY.tool_center_step > OVERLAY_GEOMETRY.state_radius * 2.0


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


def test_move_glyph_is_one_connected_antialiased_outline():
    draw = _RecordedGlyph()
    center = (20.0, 30.0)

    draw_tool_glyph(draw, center, (1.0, 1.0, 1.0, 1.0), 1.0, "move", (0.0,) * 4, "world")

    assert len(draw.paths) == 1
    path = draw.paths[0]
    assert len(path) == 24
    tips = (path[0], path[6], path[12], path[18])
    assert math.dist(tips[0], center) == pytest.approx(math.dist(tips[1], center))
    assert math.dist(tips[0], center) == pytest.approx(math.dist(tips[2], center))
    assert math.dist(tips[0], center) == pytest.approx(math.dist(tips[3], center))


def test_tool_glyphs_use_the_configured_stroke_without_hidden_scales():
    center = (20.0, 30.0)
    color = (1.0, 1.0, 1.0, 1.0)
    surface = (0.0,) * 4
    move = _RecordedGlyph()
    rotate = _RecordedGlyph()
    frame = _RecordedGlyph()

    draw_tool_glyph(move, center, color, 1.0, "move", surface, "world")
    draw_tool_glyph(rotate, center, color, 1.0, "rotate", surface, "world")
    draw_tool_glyph(frame, center, color, 1.0, "frame", surface, "world")

    move_path = move.paths[0]
    move_shaft_width = math.dist(move_path[2], move_path[-2])
    move_head_width = math.dist(move_path[1], move_path[-1])
    frame_head_width = math.dist(frame.fills[0][1], frame.fills[0][2])
    assert move_shaft_width == pytest.approx(OVERLAY_GEOMETRY.tool_stroke)
    assert len(rotate.circles) == 1
    assert rotate.circles[0][0][3] == pytest.approx(OVERLAY_GEOMETRY.tool_stroke)
    assert len(rotate.paths) == 6
    assert all(args[3] == pytest.approx(OVERLAY_GEOMETRY.tool_stroke) for args, _ in frame.lines)
    assert move_head_width <= frame_head_width * 1.05


def test_rotate_glyph_uses_antialiased_transparent_knockout_breaks():
    draw = _RecordedGlyph()

    draw_tool_glyph(
        draw,
        (20.0, 30.0),
        (1.0, 1.0, 1.0, 1.0),
        4.0,
        "rotate",
        (0.0,) * 4,
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
        OVERLAY_GEOMETRY.rotate_ring_gap,
        "butt",
    )

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
        (0.0,) * 4,
        "world",
        replace(OVERLAY_GEOMETRY, rotate_ring_cap="butt"),
    )
    draw_tool_glyph(
        rounded,
        (20.0, 30.0),
        color,
        4.0,
        "rotate",
        (0.0,) * 4,
        "world",
        replace(OVERLAY_GEOMETRY, rotate_ring_cap="round"),
    )

    assert len(butt.paths) == len(rounded.paths) == 6
    assert sum(map(len, rounded.paths)) > sum(map(len, butt.paths))


def test_frame_arrows_use_separate_axis_shafts_and_triangular_heads():
    draw = _RecordedGlyph()
    center = (20.0, 30.0)

    draw_tool_glyph(draw, center, (1.0, 1.0, 1.0, 1.0), 1.0, "frame", (0.0,) * 4, "world")

    assert not draw.paths
    assert len(draw.lines) == 3
    assert len(draw.fills) == 3
    head_edges = []
    areas = []
    for path in draw.fills:
        assert len(path) == 3
        head_edges.append(
            (
                math.dist(path[0], path[1]),
                math.dist(path[0], path[2]),
                math.dist(path[1], path[2]),
            )
        )
        areas.append(
            abs(
                sum(
                    a[0] * b[1] - a[1] * b[0]
                    for a, b in zip(path, (*path[1:], path[0]), strict=True)
                )
            )
            * 0.5
        )
    for edges in head_edges[1:]:
        assert edges == pytest.approx(head_edges[0])
    assert areas == pytest.approx([areas[0]] * 3)
    assert all(line[0][0] == center for line in draw.lines)
    assert draw.fills[1][0][1] == pytest.approx(draw.fills[2][0][1])


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
        self.lines = []

    def text(self, position, _color, value):
        self.texts.append(value)
        self.text_positions.append((position, value))

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def test_status_places_simulation_state_before_selection():
    from forge_viewer.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        900.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        has_selection=True,
        state="paused",
        sim_time=0.2,
        step=674,
        metric_mode="steps",
        backend="OpenGL",
        dt=0.002,
        fps=60.0,
    )

    assert draw.texts[:2] == ["Paused", "01_revolute"]
    x_by_text = {text: position[0] for position, text in draw.text_positions}
    assert x_by_text["01_revolute"] < x_by_text["OpenGL"]
    assert x_by_text["OpenGL"] < x_by_text["Steps 674"] < x_by_text["Δt 0.002 s"]


def test_right_aligned_telemetry_has_no_separator_against_empty_space():
    from forge_viewer.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        900.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        has_selection=True,
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
    from forge_viewer.ui.theme import THEME

    draw = _RecordedStatus()
    draw_status(
        draw,
        (0.0, 0.0),
        1400.0,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        has_selection=True,
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


@pytest.mark.parametrize("width", (48.0, 80.0, 140.0, 220.0, 300.0, 400.0, 520.0))
def test_status_progressively_collapses_without_text_overlap(width):
    from forge_viewer.ui.theme import THEME

    draw = _RecordedStatus()
    layout = draw_status(
        draw,
        (0.0, 0.0),
        width,
        28.0,
        THEME,
        1.0,
        selected="01_revolute",
        has_selection=True,
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
        self.filled_rects = []
        self.convex_fills = []
        self.events = []

    def rect(self, *args, **kwargs):
        self.rects.append((args, kwargs))
        self.events.append("shell")

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

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
    from forge_viewer.ui.theme import THEME

    draw = _RecordedMouse()
    width = draw_mouse_hint_glyph(draw, 10.0, 30.0, button, "", THEME, scale)

    assert width == pytest.approx(OVERLAY_GEOMETRY.hint_mouse_width * scale)
    shell_args, shell_kwargs = draw.rects[0]
    shell_width = shell_args[1][0] - shell_args[0][0]
    shell_height = shell_args[1][1] - shell_args[0][1]
    assert shell_width / shell_height == pytest.approx(14.0 / 18.0)
    assert shell_kwargs["rounding"] < shell_width * 0.25
    assert draw.lines == []
    assert len(draw.convex_fills) == 2
    assert draw.events == ["shell", "fill", "fill"]
    mask_args, _mask_kwargs = draw.convex_fills[0]
    fill_args, _fill_kwargs = draw.convex_fills[1]
    shell_lo, shell_hi = shell_args[:2]
    mask_points, mask_color = mask_args
    points, fill_color = fill_args
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mask_xs = [point[0] for point in mask_points]
    mask_ys = [point[1] for point in mask_points]
    assert mask_color == (*THEME.bg_child[:3], 1.0)
    assert fill_color == THEME.primary
    assert min(ys) - min(mask_ys) == pytest.approx(1.0)
    assert min(mask_ys) < shell_lo[1]
    seam_y = shell_lo[1] + shell_height * 0.40
    assert max(mask_ys) > seam_y > max(ys)
    if button == "left":
        assert min(xs) - min(mask_xs) == pytest.approx(1.0)
        assert min(mask_xs) < shell_lo[0]
        assert max(mask_xs) > max(xs)
        assert max(mask_xs) < (shell_lo[0] + shell_hi[0]) * 0.5
    else:
        assert max(mask_xs) - max(xs) == pytest.approx(1.0)
        assert max(mask_xs) > shell_hi[0]
        assert min(mask_xs) < min(xs)
        assert min(mask_xs) > (shell_lo[0] + shell_hi[0]) * 0.5


@pytest.mark.parametrize("scale", (1.0, 2.0, 2.25))
def test_mouse_wheel_is_shifted_down_one_fixed_pixel(scale: float) -> None:
    from forge_viewer.ui.theme import THEME

    draw = _RecordedMouse()
    draw_mouse_hint_glyph(draw, 10.0, 30.0, "wheel", "", THEME, scale)

    shell_lo, _shell_hi = draw.rects[0][0][:2]
    wheel_args, wheel_kwargs = draw.filled_rects[0]
    wheel_lo, wheel_hi = wheel_args[:2]
    assert wheel_lo[1] == pytest.approx(shell_lo[1] + 1.35 * scale + 1.0)
    assert wheel_hi[1] - wheel_lo[1] == pytest.approx(7.0 * scale)
    assert wheel_kwargs["rounding"] == pytest.approx(1.2 * scale)


def test_toolhint_separates_each_group_with_one_subtle_short_rule():
    from forge_viewer.ui.theme import THEME

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
    from forge_viewer import ui

    assert ui.ToolHint is ToolHint
    assert ui.ToolHintRegistry is ToolHintRegistry
    assert ui.ViewportChromeRegistry is ViewportChromeRegistry
    assert ui.ViewportControl is ViewportControl


def test_single_group_toolhint_has_no_separator_rule():
    from forge_viewer.ui.theme import THEME

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


@pytest.mark.parametrize("button", ("left", "right"))
def test_mouse_button_mask_covers_its_shell_edge_but_keeps_inner_gaps(button):
    x, y, width, height = 13.0, 17.0, 17.5, 22.5
    outline_width = 1.7
    safety = 0.2
    geometry = mouse_button_fill_geometry(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        safety_inset=safety,
    )
    assert geometry is not None
    mask, fill = geometry
    mask_xs = [point[0] for point in mask]
    mask_ys = [point[1] for point in mask]
    xs = [point[0] for point in fill]
    ys = [point[1] for point in fill]
    assert min(ys) - min(mask_ys) == pytest.approx(1.0)
    assert min(mask_ys) < y
    seam_y = y + height * 0.40
    assert max(mask_ys) > seam_y > max(ys)
    if button == "left":
        assert min(xs) - min(mask_xs) == pytest.approx(1.0)
        assert min(mask_xs) < x
        assert max(mask_xs) > max(xs)
        assert max(mask_xs) < x + width * 0.5
    else:
        assert max(mask_xs) - max(xs) == pytest.approx(1.0)
        assert max(mask_xs) > x + width
        assert min(mask_xs) < min(xs)
        assert min(mask_xs) > x + width * 0.5


@pytest.mark.parametrize("button", ("left", "right"))
def test_mouse_button_fill_shrinks_one_pixel_from_the_opposite_bottom_corner(button):
    x, y, width, height = 13.0, 17.0, 17.5, 22.5
    outline_width = 1.7
    safety = 0.2
    mask, fill = mouse_button_fill_geometry(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        safety_inset=safety,
    )
    mask_xs = [point[0] for point in mask]
    mask_ys = [point[1] for point in mask]
    fill_xs = [point[0] for point in fill]
    fill_ys = [point[1] for point in fill]
    join_delta = safety * 0.55 + max(0.45, safety * 0.45)

    assert max(mask_ys) - min(mask_ys) - (max(fill_ys) - min(fill_ys)) == pytest.approx(
        join_delta + 1.0
    )
    if button == "left":
        assert max(mask_xs) - max(fill_xs) == pytest.approx(join_delta)
        assert min(fill_xs) - min(mask_xs) == pytest.approx(1.0)
    else:
        assert min(fill_xs) - min(mask_xs) == pytest.approx(join_delta)
        assert max(mask_xs) - max(fill_xs) == pytest.approx(1.0)
