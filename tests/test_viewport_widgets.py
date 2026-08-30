import math
from dataclasses import replace

import pytest

from forge_viewer.ui.input_bindings import DEFAULT_INPUT_BINDINGS, InputAction
from forge_viewer.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_LABELS,
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    HINT_CHROME_SCALE,
    MAX_VIEWPORT_CHROME_STYLE_SCALE,
    OVERLAY_CLIP_PADDING,
    OVERLAY_GEOMETRY,
    PLAYBACK_CHROME_SCALE,
    TOOL_CHROME_SCALE,
    TOOL_GLYPH_SCALE,
    ViewportLabels,
    _hint_groups,
    _status_performance_layout,
    capsule_points,
    draw_hint,
    draw_tool_glyph,
    format_simulation_metric,
    format_simulation_steps,
    format_simulation_time,
    hint_size,
    mouse_button_fill_points,
    playback_size,
    tool_column_size,
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
    assert geometry.state_radius - geometry.icon_radius == pytest.approx(geometry.radial_step)
    assert geometry.shell_radius - geometry.state_radius == pytest.approx(geometry.radial_step)
    assert playback_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((145.0, 55.0))
    assert tool_column_size(DEFAULT_VIEWPORT_OVERLAY_SCALE) == pytest.approx((55.0, 217.5))
    assert OVERLAY_GEOMETRY.tool_center_step > OVERLAY_GEOMETRY.state_radius * 2.0


def test_capsule_host_guard_band_covers_outline_and_antialiasing() -> None:
    outline_half_width = 1.4 * 0.5
    antialias_fringe = 2.0

    assert outline_half_width + antialias_fringe < OVERLAY_CLIP_PADDING


def test_tool_glyph_scale_increases_only_the_visual_paths():
    assert TOOL_GLYPH_SCALE > 1.0
    assert TOOL_GLYPH_SCALE * OVERLAY_GEOMETRY.icon_radius < OVERLAY_GEOMETRY.state_radius


def test_transient_chrome_dampens_large_global_ui_scale():
    playback = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, PLAYBACK_CHROME_SCALE)
    tools = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, TOOL_CHROME_SCALE)
    hint = viewport_chrome_scale(1.4, DEFAULT_VIEWPORT_OVERLAY_SCALE, HINT_CHROME_SCALE)

    assert playback == pytest.approx(
        MAX_VIEWPORT_CHROME_STYLE_SCALE * DEFAULT_VIEWPORT_OVERLAY_SCALE * PLAYBACK_CHROME_SCALE
    )
    assert hint == pytest.approx(
        MAX_VIEWPORT_CHROME_STYLE_SCALE * DEFAULT_VIEWPORT_OVERLAY_SCALE * HINT_CHROME_SCALE
    )
    assert playback < 1.4 * DEFAULT_VIEWPORT_OVERLAY_SCALE
    assert tools == playback
    assert tool_column_size(tools)[0] == pytest.approx(playback_size(playback)[1])
    assert hint < playback


class _MeasuredText:
    def text_size(self, value):
        return (len(value) * 7.0, 14.0)


class _RecordedGlyph:
    def __init__(self):
        self.paths = []

    def fringed_concave_fill(self, points, _color):
        self.paths.append(tuple(points))

    def circle_filled(self, *_args, **_kwargs):
        pass

    def centered_label(self, *_args, **_kwargs):
        pass


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


def test_frame_arrows_share_one_antialiased_silhouette():
    draw = _RecordedGlyph()
    center = (20.0, 30.0)

    draw_tool_glyph(draw, center, (1.0, 1.0, 1.0, 1.0), 1.0, "frame", (0.0,) * 4, "world")

    assert len(draw.paths) == 3
    head_edges = []
    areas = []
    for path in draw.paths:
        assert len(path) == 7
        head_edges.append(
            (
                math.dist(path[3], path[2]),
                math.dist(path[3], path[4]),
                math.dist(path[2], path[4]),
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


def test_status_backend_and_fps_use_independent_stable_columns():
    draw = _MeasuredText()
    two_digits = _status_performance_layout(draw, 500.0, 1.0, "OpenGL", 0.009, 99.9)
    three_digits = _status_performance_layout(draw, 500.0, 1.0, "OpenGL", 0.0167, 107.0)

    assert two_digits.backend_x == pytest.approx(three_digits.backend_x)
    assert two_digits.delta_divider_x == pytest.approx(three_digits.delta_divider_x)
    assert two_digits.fps_divider_x == pytest.approx(three_digits.fps_divider_x)
    assert two_digits.fps_x + draw.text_size(two_digits.fps_text)[0] == pytest.approx(500.0)
    assert three_digits.fps_x + draw.text_size(three_digits.fps_text)[0] == pytest.approx(500.0)


class _RecordedHint(_MeasuredText):
    def __init__(self):
        self.lines = []

    def line(self, *args, **kwargs):
        self.lines.append((args, kwargs))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


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
def test_mouse_button_fill_stays_inside_shell_and_underlays_dividers(button):
    x, y, width, height = 13.0, 17.0, 17.5, 22.5
    outline_width = 1.7
    divider_width = 1.25
    safety = 0.2
    points = mouse_button_fill_points(
        x,
        y,
        width,
        height,
        button,
        outline_width=outline_width,
        divider_width=divider_width,
        safety_inset=safety,
    )
    split = y + height * 0.44
    shell_inset = outline_width * 0.5 + safety
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    assert min(xs) >= x + shell_inset - 1e-9
    assert max(xs) <= x + width - shell_inset + 1e-9
    assert min(ys) >= y + shell_inset - 1e-9
    assert max(ys) == pytest.approx(split + divider_width * 0.5)
    if button == "left":
        assert max(xs) == pytest.approx(x + width * 0.5)
    else:
        assert min(xs) == pytest.approx(x + width * 0.5)
