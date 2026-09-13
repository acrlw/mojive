"""Live visibility masks preserve the retained drawing publisher's state."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mojive import RecordingConfig, ViewportLayers
from mojive.render.debugdraw import DebugDraw
from mojive.ui.app import ViewerApp
from mojive.ui.layers import visible_debug_layers


def test_visibility_groups_and_named_masks_restore_after_failed_render():
    draw = DebugDraw()
    names = ("trajectory", "contacts", "canvas2d:labels", "ui.selection", "ui.gizmo.drag")
    layers = {name: draw.layer(name) for name in names}
    layers["contacts"].visible = False
    mask = ViewportLayers(debug_2d=False, selection=False, hidden_debug_layers=("trajectory",))
    with pytest.raises(RuntimeError), visible_debug_layers(draw, mask):
        assert [name for name, layer in layers.items() if layer.visible] == ["ui.gizmo.drag"]
        raise RuntimeError("failed draw")
    assert [name for name, layer in layers.items() if layer.visible] == [
        "trajectory",
        "canvas2d:labels",
        "ui.selection",
        "ui.gizmo.drag",
    ]
    draw.layer("new trajectory")
    with visible_debug_layers(draw, replace(mask, debug_3d=False)):
        assert not draw.layer("new trajectory").visible
        assert draw.layer("ui.gizmo.drag").visible


def test_hiding_controls_cancels_drag_and_persists_without_changing_tool_policy():
    events = []
    app = ViewerApp.__new__(ViewerApp)
    app.gizmo = SimpleNamespace(cancel=lambda: events.append("cancel"))
    app._precise_gizmo_edit = object()
    app._tool_widget_rect = app._playback_widget_rect = (1, 2, 3, 4)
    app._overlay_drag_kind = "tools"
    app.localizer = SimpleNamespace(set_preferences=lambda values: events.append(values))
    app.set_viewport_layers(ViewportLayers(viewport_ui=False, gizmos=False))
    assert events[0] == "cancel"
    assert events[1]["viewport_layers"]["gizmos"] is False
    assert app._precise_gizmo_edit is None
    assert app._tool_widget_rect is app._playback_widget_rect is None
    assert app._overlay_drag_kind == ""


def test_recording_and_layer_preferences_validate_persisted_data():
    assert (
        RecordingConfig.from_mapping({"countdown": "nan", "fps": "broken", "surface": 1})
        == RecordingConfig()
    )
    assert RecordingConfig.from_mapping({"countdown": -1, "fps": 999}).countdown == 0
    assert RecordingConfig.from_mapping({"countdown": -1, "fps": 999}).fps == 240
    assert ViewportLayers.from_mapping(
        {"debug_3d": False, "gizmos": "invalid", "hidden_debug_layers": ["one", "one", None]}
    ) == ViewportLayers(debug_3d=False, hidden_debug_layers=("one",))
