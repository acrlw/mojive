from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from design.tools import render_ui_feasibility as probe

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


def test_geometry_export_names_production_overlay_fields():
    values = probe._geometry_values_text(probe.ProbeState())

    assert "icon_radius=10," in values
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
    monkeypatch.setattr(probe, "draw_concept_icon", lambda *args: calls.append(args))

    probe._draw_icon_library_command_icon(None, (10.0, 20.0), kind, (1, 1, 1, 1), scale)

    assert calls[0][3] == expected


def test_icon_library_reuses_production_output_severity_painter(monkeypatch):
    calls = []

    class Draw:
        def circle(self, *_args, **_kwargs):
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
    clearance, center_x, center_y, sphere_x, sphere_y, _ink_x, _ink_y = probe._icon_review_metrics(
        name
    )

    assert clearance >= 0.6
    assert center_x == pytest.approx(0.0, abs=1e-6)
    assert center_y == pytest.approx(0.0, abs=1e-6)
    assert sphere_x == pytest.approx(0.0, abs=1e-6)
    assert sphere_y == pytest.approx(0.0, abs=1e-6)


def test_production_information_and_warning_ink_are_mirrored():
    info = probe._icon_review_metrics("status-info")
    warning = probe._icon_review_metrics("status-warning")

    assert info[5] == pytest.approx(0.0, abs=1e-6)
    assert warning[5] == pytest.approx(0.0, abs=1e-6)
    assert info[6] == pytest.approx(-warning[6], abs=1e-6)
