from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from design.tools import render_ui_feasibility as probe

from mojive.ui.viewport_widgets import draw_status, draw_tool_glyph

PROBE_PATH = Path(__file__).resolve().parents[1] / "design/tools/render_ui_feasibility.py"


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
    ((0.75, (1600, 1000)), (1.0, (1600, 1000)), (2.0, (1600, 1000)), (4.0, (3200, 2000))),
)
def test_probe_window_preserves_canvas_at_extreme_ui_scales(scale, expected):
    assert probe._probe_window_size(1600, 1000, scale) == expected


def test_virtual_canvas_scrolls_instead_of_compressing_components():
    available = type("Available", (), {"x": 800.0, "y": 500.0})()

    assert probe._virtual_canvas_size(available, 4.0, probe.WORKSPACE_CANVAS_SIZE) == (
        6400.0,
        3840.0,
    )


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
