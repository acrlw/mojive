from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mojive.tools.ui_capsule_geometry import capsule_layout
from mojive.tools.ui_feasibility import fixtures as probe_fixtures
from mojive.tools.ui_feasibility import icon_library as probe_icon_library
from mojive.tools.ui_feasibility import layout as probe_layout
from mojive.tools.ui_feasibility import state as probe_state
from mojive.tools.ui_feasibility import tuning as probe_tuning
from mojive.tools.ui_feasibility import widgets as probe_widgets
from mojive.tools.ui_redesign import _capsule_icon_color
from mojive.ui import gizmo as gizmo_ui
from mojive.ui.icons import (
    ICON_ALIGNMENT_EDITABLE_ICONS,
    ICON_DEFAULT_PADDING,
    ICON_FAMILIES,
    ICON_GLYPH_ALIGNMENT_DEFAULTS,
    ICON_GLYPH_STROKE_DEFAULTS,
    ICON_GROUP_LAYOUT_DEFAULTS,
    ICON_GROUP_STROKE_DEFAULTS,
    ICON_MAX_PADDING,
    ICON_MIN_CLEARANCE,
    ICON_STROKE,
    draw_icon_label,
)
from mojive.ui.perturb import OUTLINE_CORNER_RADIUS_PT
from mojive.ui.theme import THEME
from mojive.ui.viewcube import DEFAULT_SELECTION_PADDING
from mojive.ui.viewport_widgets import (
    DEFAULT_VIEWPORT_OVERLAY_SCALE,
    OVERLAY_GEOMETRY,
    draw_status,
    draw_tool_glyph,
    playback_control_centers,
    playback_size,
    tool_column_size,
    tool_control_centers,
)

PROBE_PATH = Path(__file__).resolve().parents[1] / "python/tools/ui_feasibility"


@pytest.mark.parametrize(
    ("name", "function"),
    (("draw_tool_glyph", draw_tool_glyph), ("draw_status", draw_status)),
)
def test_shared_widget_calls_match_runtime_signatures(name, function):
    calls = [
        (path, node)
        for path in PROBE_PATH.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]

    assert calls, f"no {name} calls found in {PROBE_PATH}"
    signature = inspect.signature(function)
    for path, call in calls:
        assert all(keyword.arg is not None for keyword in call.keywords)
        positional = [object()] * len(call.args)
        keywords = {keyword.arg: object() for keyword in call.keywords if keyword.arg is not None}
        try:
            signature.bind(*positional, **keywords)
        except TypeError as exc:
            pytest.fail(f"{path}:{call.lineno}: {name}{signature}: {exc}")


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
    assert probe_layout._probe_window_size(1600, 1000, scale) == expected


def test_probe_window_growth_keeps_the_scaled_canvas_and_controls_visible():
    """A capture that clips the right-hand experiment controls misstates the design."""

    for scale in (1.5, 2.0, 2.5, 3.0, 4.0):
        width, height = probe_layout._probe_window_size(1600, 1000, scale)
        assert width >= (probe_fixtures.GEOMETRY_CANVAS_SIZE[0] + 400.0) * scale
        assert height >= probe_fixtures.GEOMETRY_CANVAS_SIZE[1] * scale


def test_probe_tab_rows_wrap_instead_of_clipping_at_large_scales():
    """Native tab bars run past the panel edge once their labels grow."""

    source = "\n".join(path.read_text(encoding="utf-8") for path in PROBE_PATH.glob("*.py"))
    assert "begin_tab_bar" not in source
    assert source.count("_wrapped_tabs(") >= 4


def test_virtual_canvas_scrolls_instead_of_compressing_components():
    available = type("Available", (), {"x": 800.0, "y": 500.0})()

    assert probe_layout._virtual_canvas_size(
        available, 4.0, probe_fixtures.WORKSPACE_CANVAS_SIZE
    ) == (
        6400.0,
        3840.0,
    )


def test_icon_library_canvas_reaches_the_last_row_of_the_longest_family():
    family, icons = max(ICON_FAMILIES, key=lambda item: len(item[1]))
    required_height = 260.0 + (max(probe_fixtures._ICON_REVIEW_SIZES) + 20.0) * len(icons)

    assert probe_layout._icon_library_canvas_size(family)[1] >= required_height
    assert probe_layout._icon_library_canvas_size("Overview") == probe_fixtures.GEOMETRY_CANVAS_SIZE
    assert probe_layout._icon_library_canvas_size("Capsules")[1] >= 1260.0


def test_geometry_canvas_reserves_zoomed_playback_and_tool_extents():
    state = probe_state.ProbeState(construction_playback_scale=4.0, construction_tool_scale=3.0)
    playback_width, playback_height = probe_layout._geometry_canvas_size("Playback", state)
    _centers, playback_length = capsule_layout(6, (3, 4), state)

    assert playback_width >= 54.0 + playback_length * 4.0 + 54.0 + 360.0 + 24.0
    assert playback_height > probe_fixtures.GEOMETRY_CANVAS_SIZE[1]
    _centers, tool_length = capsule_layout(5, (3,), state)
    assert probe_layout._geometry_canvas_size("Tools", state)[1] >= max(
        probe_fixtures.GEOMETRY_CANVAS_SIZE[1], 120.0 + tool_length * 3.0
    )


def test_geometry_export_names_production_overlay_fields():
    values = probe_tuning._geometry_values_text(probe_state.ProbeState())

    assert "icon_radius=8," in values
    assert f"end_padding_ratio={OVERLAY_GEOMETRY.end_padding_ratio}," in values
    assert f"icon_stroke_viewport_tools={ICON_STROKE}," in values
    assert f"icon_stroke_viewport_playback={ICON_STROKE}," in values
    assert "icon_padding_viewport_tools=0.5," in values
    assert "icon_padding_viewport_playback=2.0," in values
    assert "tool_stroke=" in values
    assert "icon_tool_move_head_scale=0.85," in values
    assert "icon_tool_scale_handle_scale=1.15," in values
    assert "icon_tool_snap_endpoint_scale=1.3," in values
    assert "icon_key_fit_arm_length=4.5," in values
    assert "icon_reset_head_scale=1.5," in values
    assert "hint_mouse_wheel_gap_ratio=" in values


@pytest.mark.parametrize("scale", (0.65, 1.0, 1.25, 1.5, 2.25))
def test_production_capsules_match_the_accepted_probe_layout(scale):
    state = probe_state.ProbeState()
    playback_centers, playback_length = capsule_layout(6, (3, 4), state)
    tool_centers, tool_length = capsule_layout(5, (3,), state)
    thickness = 2 * (state.overlay_icon_radius + 2 * state.overlay_radial_step) * scale

    assert playback_control_centers() == pytest.approx(playback_centers)
    assert tool_control_centers() == pytest.approx(tool_centers)
    assert playback_size(scale) == pytest.approx((playback_length * scale, thickness))
    assert tool_column_size(scale) == pytest.approx((thickness, tool_length * scale))


def test_icon_library_export_contains_group_glyph_and_shape_controls():
    state = probe_state.ProbeState()
    state.set_icon_padding_for_glyph("tool-move", 0.85)
    state.set_icon_stroke_for_glyph("key-fit", 2.1)
    state.move_head_scale = 1.2
    state.reset_head_scale = 1.6

    values = probe_tuning._icon_values_text(state)

    assert "icon_padding_viewport_tools=0.5," in values
    assert "icon_padding_tool_move=0.85," in values
    assert "icon_stroke_key_fit=2.1," in values
    assert "icon_tool_move_head_scale=1.2," in values
    assert "icon_reset_head_scale=1.6," in values
    assert state.icon_tuning().reset_head_scale == 1.6
    assert "icon_tool_snap_endpoint_scale=1.3," in values
    assert "icon_alignment_key_snapshot='box'," in values


@pytest.mark.parametrize("alignment", ("circle", "box"))
def test_labelled_snapshot_preview_offsets_the_selected_anchor_from_text_ink(
    monkeypatch, alignment
):
    state = probe_state.ProbeState(preview_icon_library=True)
    state.set_icon_alignment_for_glyph("key-snapshot", alignment)
    calls = []
    monkeypatch.setattr(
        probe_widgets,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe_widgets._draw_icon_library_command_icon(
        object(),
        (100.0, 50.0),
        "key",
        (1.0, 1.0, 1.0, 1.0),
        1.0,
        context_scale=1.0,
        state=state,
    )

    args, kwargs = calls[0]
    assert args[1] == pytest.approx((100.0, 50.0 - 0.84 * 16 / 24))
    assert kwargs["alignment"] == alignment


@pytest.mark.parametrize("mode", ("off", "page", "locked"))
def test_follow_label_preview_and_export_share_live_tuning(monkeypatch, mode):
    name = f"key-follow-{mode}"
    state = probe_state.ProbeState()
    state.set_icon_padding_for_glyph(name, 3.0)
    state.set_icon_stroke_for_glyph(name, 1.25)
    calls = []
    monkeypatch.setattr(probe_widgets, "draw_icon_label", lambda *a, **kw: calls.append((a, kw)))
    color = (0.4, 0.6, 0.2, 0.5)
    probe_widgets._draw_icon_library_label(
        object(),
        (10.0, 20.0),
        (90.0, 48.0),
        color,
        1.0,
        name,
        mode,
        state=state,
    )
    args, kwargs = calls[0]
    assert args[3] == color
    assert kwargs["style"].padding == 3.0
    assert kwargs["style"].stroke_width == 1.25
    exported = probe_tuning._icon_values_text(state)
    assert f"icon_padding_key_follow_{mode}=3.0," in exported
    assert f"icon_stroke_key_follow_{mode}=1.25," in exported


def test_probe_geometry_defaults_follow_production_constants():
    state = probe_state.ProbeState()

    assert state.viewport_overlay_scale == DEFAULT_VIEWPORT_OVERLAY_SCALE
    assert state.position_snap == gizmo_ui.DEFAULT_TRANSLATION_SNAP_M
    assert state.rotation_snap == gizmo_ui.DEFAULT_ROTATION_SNAP_DEG
    assert state.tick_scale == gizmo_ui.DEFAULT_ROTATION_TICK_SCALE
    assert state.selection_padding == DEFAULT_SELECTION_PADDING
    assert state.corner_radius == OUTLINE_CORNER_RADIUS_PT
    assert not state.preview_icon_library
    assert state.timeline_panel.toolbar.follow_mode_icon_drawer is draw_icon_label
    assert all(state.icon_stroke_for(group) == ICON_STROKE for group in ICON_GROUP_STROKE_DEFAULTS)
    assert state.icon_padding_for("Viewport tools") == 0.5
    assert state.icon_padding_for_glyph("tool-rotate") == 0.0
    assert state.icon_padding_for_glyph("playback-previous") == 4.0
    assert state.rotate_ring_gap_ratio == 1.0
    assert state.rotate_ring_cap == "round"
    assert state.move_head_scale == 0.85
    assert state.scale_handle_scale == 1.15
    assert state.snap_endpoint_scale == 1.3
    assert state.key_fit_arm_length == 4.5
    assert state.hint_mouse_width == 14
    assert {
        name: state.icon_stroke_for_glyph(name) for name in ICON_GLYPH_STROKE_DEFAULTS
    } == ICON_GLYPH_STROKE_DEFAULTS
    assert {
        name: state.icon_alignment_for_glyph(name) for name in ICON_ALIGNMENT_EDITABLE_ICONS
    } == ICON_GLYPH_ALIGNMENT_DEFAULTS
    assert all(
        state.icon_padding_for(group) == 2.0
        for group in ICON_GROUP_LAYOUT_DEFAULTS
        if group != "Viewport tools"
    )


def test_icon_layout_controls_are_stored_per_actual_component_group() -> None:
    state = probe_state.ProbeState()

    state.set_icon_padding_for("Viewport tools", 1.2)
    state.set_icon_stroke_for("Viewport tools", 2.0)
    state.set_icon_padding_for_glyph("tool-world", 0.8)
    state.set_icon_stroke_for_glyph("tool-world", 1.6)

    assert state.icon_padding_for("Viewport tools") == 1.2
    assert state.icon_padding_for("Viewport playback") == ICON_DEFAULT_PADDING
    assert state.icon_stroke_for("Viewport tools") == 2.0
    assert state.icon_stroke_for("Viewport playback") == ICON_STROKE
    assert state.icon_padding_for_glyph("tool-world") == 0.8
    assert state.icon_stroke_for_glyph("tool-world") == 1.6


@pytest.mark.parametrize("muted", (False, True))
def test_status_mouse_preview_keeps_the_original_production_painter(monkeypatch, muted):
    calls = []
    state = probe_state.ProbeState(preview_icon_library=True)

    def draw_mouse(*args, **kwargs):
        calls.append((args, kwargs))
        return OVERLAY_GEOMETRY.hint_mouse_width

    monkeypatch.setattr(
        probe_widgets,
        "draw_mouse_hint_glyph",
        draw_mouse,
    )

    used = probe_widgets._draw_mouse_input(
        None,
        10.0,
        30.0,
        1.0,
        width=float(OVERLAY_GEOMETRY.hint_mouse_width),
        height=float(OVERLAY_GEOMETRY.hint_control_height),
        button="left",
        suffix="",
        state=state,
        muted=muted,
    )

    assert used == pytest.approx(OVERLAY_GEOMETRY.hint_mouse_width)
    assert calls[0][0][3:7] == ("left", "", probe_fixtures.CONCEPT_THEME, 1.0)
    assert calls[0][1]["size"] == (
        OVERLAY_GEOMETRY.hint_mouse_width,
        OVERLAY_GEOMETRY.hint_control_height,
    )
    assert calls[0][1]["muted"] is muted


def test_status_mouse_width_control_changes_candidate_aspect_ratio() -> None:
    state = probe_state.ProbeState()
    original = state.concept_mouse_width()

    state.hint_mouse_width += 4

    assert state.concept_mouse_width() > original


def test_status_mouse_family_specimen_uses_original_control_color(monkeypatch) -> None:
    calls = []

    class Draw:
        def circle(self, *_args, **_kwargs):
            return None

        def rect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        probe_icon_library,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe_icon_library._draw_concept_icon_specimen(
        Draw(),
        (10.0, 20.0),
        112.0,
        "status-mouse-left",
        1.0,
    )

    assert calls[0][0][4] == probe_fixtures.CONCEPT_THEME.text
    assert calls[0][1]["accent_color"] == probe_fixtures.CONCEPT_THEME.primary
    assert calls[0][1]["stroke_width"] == ICON_STROKE


def test_icon_specimen_square_matches_orange_circle_diameter(monkeypatch) -> None:
    rectangles = []
    circles = []

    class Draw:
        def rect(self, lo, hi, *_args, **_kwargs):
            rectangles.append((lo, hi))

        def circle(self, center, radius, *_args, **_kwargs):
            circles.append((center, radius))

    monkeypatch.setattr(probe_icon_library, "draw_concept_icon", lambda *_args, **_kwargs: None)
    probe_icon_library._draw_concept_icon_specimen(
        Draw(), (20.0, 30.0), 112.0, "playback-play", 1.0
    )

    (lo, hi), (center, radius) = rectangles[0], circles[0]
    assert center == (20.0, 30.0)
    assert hi[0] - lo[0] == pytest.approx(2.0 * radius)
    assert hi[1] - lo[1] == pytest.approx(2.0 * radius)


def test_icon_specimen_draws_reference_guides_behind_the_candidate(monkeypatch) -> None:
    events = []

    class Draw:
        def rect(self, *_args, **_kwargs):
            events.append("square")

        def circle(self, *_args, **_kwargs):
            events.append("circle")

    monkeypatch.setattr(
        probe_icon_library,
        "draw_concept_icon",
        lambda *_args, **_kwargs: events.append("glyph"),
    )

    probe_icon_library._draw_concept_icon_specimen(Draw(), (20.0, 30.0), 112.0, "tool-rotate", 1.0)

    assert events == ["square", "circle", "glyph"]


def test_capsule_record_and_stop_share_the_viewport_danger_color():
    ordinary = (0.1, 0.2, 0.3, 1.0)

    assert _capsule_icon_color("play", ordinary) == ordinary
    assert _capsule_icon_color("record", ordinary) == THEME.viewport.record
    assert _capsule_icon_color("record", ordinary, 0.4) == (
        *THEME.viewport.record[:3],
        THEME.viewport.record[3] * 0.4,
    )


@pytest.mark.parametrize(
    ("kind", "scale", "expected"),
    (
        ("previous", 1.0, "transport-previous"),
        ("add", 1.0, "key-add"),
        ("key", 1.0, "key-snapshot"),
        ("key", 0.2, "key-keyframe"),
        ("key-keyframe", 0.2, "key-keyframe"),
    ),
)
def test_keyframe_context_preview_uses_icon_library_candidates(monkeypatch, kind, scale, expected):
    calls = []
    monkeypatch.setattr(
        probe_widgets,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe_widgets._draw_icon_library_command_icon(None, (10.0, 20.0), kind, (1, 1, 1, 1), scale)

    assert calls[0][0][3] == expected
    assert calls[0][1]["padding"] == ICON_DEFAULT_PADDING
    assert calls[0][1]["stroke_width"] == ICON_STROKE


def test_keyframe_context_preview_reports_unknown_production_icon_name() -> None:
    with pytest.raises(ValueError, match="unknown keyframe command icon"):
        probe_widgets._draw_icon_library_command_icon(
            None,
            (10.0, 20.0),
            "missing",
            (1, 1, 1, 1),
            1.0,
        )


def test_keyframe_context_preview_prefers_the_reviewed_glyph_stroke(monkeypatch):
    calls = []
    state = probe_state.ProbeState()
    state.set_icon_stroke_for("Keyframe transport", 2.1)
    monkeypatch.setattr(
        probe_widgets,
        "draw_concept_icon",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    probe_widgets._draw_icon_library_command_icon(
        None,
        (10.0, 20.0),
        "previous",
        (1, 1, 1, 1),
        1.0,
        state=state,
    )

    assert calls[0][1]["stroke_width"] == 2.0

    state.set_icon_stroke_for_glyph("transport-previous", 2.1)
    probe_widgets._draw_icon_library_command_icon(
        None,
        (10.0, 20.0),
        "previous",
        (1, 1, 1, 1),
        1.0,
        state=state,
    )
    assert calls[1][1]["stroke_width"] == 2.1


def test_icon_library_reuses_production_output_severity_painter(monkeypatch):
    calls = []

    class Draw:
        def circle(self, *_args, **_kwargs):
            return None

        def rect(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(probe_icon_library, "severity_icon", lambda *args: calls.append(args))
    monkeypatch.setattr(
        probe_icon_library,
        "draw_concept_icon",
        lambda *_args: pytest.fail("status icons must not use the concept duplicate"),
    )

    probe_icon_library._draw_concept_icon_specimen(
        Draw(), (10.0, 20.0), 14.0, "status-warning", 1.0
    )

    assert len(calls) == 1
    assert calls[0][1:4] == ((10.0, 20.0), 14.0, "warning")


@pytest.mark.parametrize("name", ("status-info", "status-warning", "status-error"))
def test_production_severity_stays_centered_inside_icon_library_boundary(name):
    clearance, circle_x, circle_y, box_x, box_y = probe_icon_library._icon_review_metrics(name)

    assert clearance >= 0.6
    assert circle_x == pytest.approx(0.0, abs=1e-6)
    assert circle_y == pytest.approx(0.0, abs=1e-6)
    assert box_x == pytest.approx(0.0, abs=1e-6)
    assert box_y == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ("status-info", "status-warning", "status-error"))
def test_production_severity_metrics_ignore_candidate_layout_controls(name):
    assert probe_icon_library._icon_review_metrics(name, 1.0, ICON_MAX_PADDING) == pytest.approx(
        probe_icon_library._icon_review_metrics(name, 0.0, ICON_MIN_CLEARANCE)
    )
