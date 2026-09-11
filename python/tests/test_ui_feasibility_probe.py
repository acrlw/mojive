from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from design.tools import render_ui_feasibility as probe
from design.tools.ui_redesign import _capsule_icon_color

from mojive.ui.viewport_widgets import draw_status, draw_tool_glyph

PROBE_PATH = Path(__file__).resolve().parents[2] / "design/tools/render_ui_feasibility.py"


@pytest.mark.parametrize(
    ("name", "function"),
    (("draw_tool_glyph", draw_tool_glyph), ("draw_status", draw_status)),
)
def test_shared_widget_calls_match_runtime_signatures(name, function):
    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"), filename=str(PROBE_PATH))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]

    assert calls, f"no {name} calls found in {PROBE_PATH}"
    signature = inspect.signature(function)
    for call in calls:
        assert all(keyword.arg is not None for keyword in call.keywords)
        positional = [object()] * len(call.args)
        keywords = {keyword.arg: object() for keyword in call.keywords if keyword.arg is not None}
        try:
            signature.bind(*positional, **keywords)
        except TypeError as exc:
            pytest.fail(f"{PROBE_PATH}:{call.lineno}: {name}{signature}: {exc}")


@pytest.mark.parametrize(
    ("scale", "expected"),
    (
        (0.75, (1600, 1000)),
        (1.0, (1600, 1000)),
        (2.0, (4000, 1900)),
        (2.5, (5000, 2375)),
        (4.0, (8000, 3800)),
    ),
)
def test_probe_window_preserves_canvas_at_extreme_ui_scales(scale, expected):
    assert probe._probe_window_size(1600, 1000, scale) == expected


def test_probe_window_growth_keeps_the_scaled_canvas_and_controls_visible():
    """A capture that clips the right-hand experiment controls misstates the design."""

    for scale in (1.5, 2.0, 2.5, 3.0, 4.0):
        width, height = probe._probe_window_size(1600, 1000, scale)
        assert width >= (probe.GEOMETRY_CANVAS_SIZE[0] + 400.0) * scale
        assert height >= probe.GEOMETRY_CANVAS_SIZE[1] * scale


def test_probe_tab_rows_wrap_instead_of_clipping_at_large_scales():
    """Native tab bars run past the panel edge once their labels grow."""

    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "begin_tab_bar" not in source
    assert source.count("_wrapped_tabs(") >= 4


def test_virtual_canvas_scrolls_instead_of_compressing_components():
    available = type("Available", (), {"x": 800.0, "y": 500.0})()

    assert probe._virtual_canvas_size(available, 4.0, probe.WORKSPACE_CANVAS_SIZE) == (
        6400.0,
        3840.0,
    )


def test_icon_library_canvas_reaches_the_last_row_of_the_longest_family():
    family, icons = max(probe.ICON_FAMILIES, key=lambda item: len(item[1]))
    required_height = 260.0 + (max(probe._ICON_REVIEW_SIZES) + 20.0) * len(icons)

    assert probe._icon_library_canvas_size(family)[1] >= required_height
    assert probe._icon_library_canvas_size("Overview") == probe.GEOMETRY_CANVAS_SIZE
    assert probe._icon_library_canvas_size("Capsules")[1] >= 1260.0


def test_geometry_canvas_reserves_zoomed_playback_and_tool_extents():
    state = probe.ProbeState(construction_playback_scale=4.0, construction_tool_scale=3.0)
    playback_width, playback_height = probe._geometry_canvas_size("Playback", state)
    _centers, playback_length = probe.capsule_layout(6, (3, 4), state)

    assert playback_width >= 54.0 + playback_length * 4.0 + 54.0 + 360.0 + 24.0
    assert playback_height > probe.GEOMETRY_CANVAS_SIZE[1]
    _centers, tool_length = probe.capsule_layout(5, (3,), state)
    assert probe._geometry_canvas_size("Tools", state)[1] >= max(
        probe.GEOMETRY_CANVAS_SIZE[1], 120.0 + tool_length * 3.0
    )


def test_geometry_export_names_production_overlay_fields():
    values = probe._geometry_values_text(probe.ProbeState())

    assert "icon_radius=10," in values
    assert "icon_padding_viewport_tools=0.5," in values
    assert "icon_padding_viewport_playback=0.75," in values
    assert "tool_stroke=" in values
    assert "hint_mouse_wheel_gap_ratio=" in values


def test_probe_geometry_defaults_follow_production_constants():
    state = probe.ProbeState()

    assert state.viewport_overlay_scale == probe.DEFAULT_VIEWPORT_OVERLAY_SCALE
    assert state.position_snap == probe.gizmo_ui.DEFAULT_TRANSLATION_SNAP_M
    assert state.rotation_snap == probe.gizmo_ui.DEFAULT_ROTATION_SNAP_DEG
    assert state.tick_scale == probe.gizmo_ui.DEFAULT_ROTATION_TICK_SCALE
    assert state.selection_padding == probe.DEFAULT_SELECTION_PADDING
    assert state.corner_radius == probe.OUTLINE_CORNER_RADIUS_PT
    assert not state.preview_icon_library
    assert state.icon_padding_for("Viewport tools") == 0.5
    assert state.icon_padding_for("Viewport playback") == probe.ICON_DEFAULT_PADDING
    assert state.icon_padding_for("Keyframe transport") == probe.ICON_DEFAULT_PADDING


def test_icon_layout_controls_are_stored_per_actual_component_group() -> None:
    state = probe.ProbeState()

    state.set_icon_padding_for("Viewport tools", 1.2)

    assert state.icon_padding_for("Viewport tools") == 1.2
    assert state.icon_padding_for("Viewport playback") == probe.ICON_DEFAULT_PADDING


@pytest.mark.parametrize("button", ("left", "right", "wheel"))
def test_status_mouse_adapter_preserves_the_original_control_height(button):
    height = float(probe.OVERLAY_GEOMETRY.hint_control_height)
    padding = probe.ICON_DEFAULT_PADDING
    state = probe.ProbeState()
    mouse_width = state.concept_mouse_width()
    size = probe._concept_mouse_icon_size(height, button, padding, mouse_width=mouse_width)
    metrics = probe.icon_metrics(f"status-mouse-{button}", padding=padding, mouse_width=mouse_width)
    rendered_height = (metrics.bounds[3] - metrics.bounds[1]) * size / probe.ICON_GRID

    assert rendered_height == pytest.approx(height)
    assert size > probe.OVERLAY_GEOMETRY.hint_mouse_width


@pytest.mark.parametrize("muted", (False, True))
def test_status_mouse_preview_uses_neutral_status_colors(monkeypatch, muted):
    calls = []
    state = probe.ProbeState(preview_icon_library=True)
    monkeypatch.setattr(
        probe,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    used = probe._draw_mouse_input(
        None,
        10.0,
        30.0,
        1.0,
        width=float(probe.OVERLAY_GEOMETRY.hint_mouse_width),
        height=float(probe.OVERLAY_GEOMETRY.hint_control_height),
        button="left",
        suffix="",
        state=state,
        muted=muted,
    )

    assert used == pytest.approx(probe.OVERLAY_GEOMETRY.hint_mouse_width)
    assert calls[0][0][4] == (
        probe.CONCEPT_THEME.text_disabled if muted else probe.CONCEPT_THEME.text
    )
    assert calls[0][1]["accent_color"] == probe.CONCEPT_THEME.bg_frame_active


def test_status_mouse_width_control_changes_candidate_aspect_ratio() -> None:
    state = probe.ProbeState()
    original = state.concept_mouse_width()

    state.hint_mouse_width += 4

    assert state.concept_mouse_width() > original


def test_status_mouse_family_specimen_uses_neutral_button_gray(monkeypatch) -> None:
    calls = []

    class Draw:
        def circle(self, *_args, **_kwargs):
            return None

        def rect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        probe,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe._draw_concept_icon_specimen(
        Draw(),
        (10.0, 20.0),
        112.0,
        "status-mouse-left",
        1.0,
    )

    assert calls[0][0][4] == probe.CONCEPT_THEME.text
    assert calls[0][1]["accent_color"] == probe.CONCEPT_THEME.bg_frame_active


def test_icon_specimen_square_matches_orange_circle_diameter(monkeypatch) -> None:
    rectangles = []
    circles = []

    class Draw:
        def rect(self, lo, hi, *_args, **_kwargs):
            rectangles.append((lo, hi))

        def circle(self, center, radius, *_args, **_kwargs):
            circles.append((center, radius))

    monkeypatch.setattr(probe, "draw_concept_icon", lambda *_args, **_kwargs: None)
    probe._draw_concept_icon_specimen(Draw(), (20.0, 30.0), 112.0, "playback-play", 1.0)

    (lo, hi), (center, radius) = rectangles[0], circles[0]
    assert center == (20.0, 30.0)
    assert hi[0] - lo[0] == pytest.approx(2.0 * radius)
    assert hi[1] - lo[1] == pytest.approx(2.0 * radius)


def test_capsule_record_and_stop_share_the_viewport_danger_color():
    ordinary = (0.1, 0.2, 0.3, 1.0)

    assert _capsule_icon_color("play", ordinary) == ordinary
    assert _capsule_icon_color("record", ordinary) == probe.THEME.viewport.record
    assert _capsule_icon_color("record", ordinary, 0.4) == (
        *probe.THEME.viewport.record[:3],
        probe.THEME.viewport.record[3] * 0.4,
    )


@pytest.mark.parametrize(
    ("kind", "scale", "expected"),
    (
        ("previous", 1.0, "transport-previous"),
        ("add", 1.0, "key-add"),
        ("key", 1.0, "key-snapshot"),
        ("key", 0.2, "key-keyframe"),
    ),
)
def test_keyframe_context_preview_uses_icon_library_candidates(monkeypatch, kind, scale, expected):
    calls = []
    monkeypatch.setattr(
        probe,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe._draw_icon_library_command_icon(None, (10.0, 20.0), kind, (1, 1, 1, 1), scale)

    assert calls[0][0][3] == expected
    assert calls[0][1]["padding"] == probe.ICON_DEFAULT_PADDING


def test_icon_library_reuses_production_output_severity_painter(monkeypatch):
    calls = []

    class Draw:
        def circle(self, *_args, **_kwargs):
            return None

        def rect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(probe, "severity_icon", lambda *args: calls.append(args))
    monkeypatch.setattr(
        probe,
        "draw_concept_icon",
        lambda *_args: pytest.fail("status icons must not use the concept duplicate"),
    )

    probe._draw_concept_icon_specimen(Draw(), (10.0, 20.0), 14.0, "status-warning", 1.0)

    assert len(calls) == 1
    assert calls[0][1:4] == ((10.0, 20.0), 14.0, "warning")


@pytest.mark.parametrize("name", ("status-info", "status-warning", "status-error"))
def test_production_severity_stays_centered_inside_icon_library_boundary(name):
    clearance, circle_x, circle_y, box_x, box_y = probe._icon_review_metrics(name)

    assert clearance >= 0.6
    assert circle_x == pytest.approx(0.0, abs=1e-6)
    assert circle_y == pytest.approx(0.0, abs=1e-6)
    assert box_x == pytest.approx(0.0, abs=1e-6)
    assert box_y == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ("status-info", "status-warning", "status-error"))
def test_production_severity_metrics_ignore_candidate_layout_controls(name):
    assert probe._icon_review_metrics(name, 1.0, probe.ICON_MAX_PADDING) == pytest.approx(
        probe._icon_review_metrics(name, 0.0, probe.ICON_MIN_CLEARANCE)
    )
