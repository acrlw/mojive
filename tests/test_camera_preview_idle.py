"""Inactive preview presentation preserves clearing without resolving hidden controls."""

import pytest
from imgui_bundle import imgui

from tests.test_camera_preview_drag import _driver
from tests.test_component_appearance import appearance as appearance


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("state", ("disabled", "image_missing", "viewport_ui_hidden"))
def test_inactive_preview_clears_native_hit_and_hover_without_style_resolution(
    appearance, monkeypatch, state
):
    ui = appearance[0]
    camera, preview, paint, event, _grab = _driver(appearance)
    paint(3)
    close = ui.hits["camera-preview-close"]
    event("add_mouse_pos_event", close[0] + close[2] / 2, close[1] + close[3] / 2)
    assert preview.bounds is not None
    assert "camera_preview" in camera.workspace.rects

    if state == "disabled":
        preview.set_enabled(False)
    elif state == "image_missing":
        preview._image = None
    else:
        camera.workspace.app.viewport_layers.viewport_ui = False

    def hidden_style(*args, **kwargs):
        pytest.fail("An inactive preview resolved its hidden control styles")

    monkeypatch.setattr(ui, "component_style", hidden_style)
    # ImGui's hovered window uses the preceding frame's active window list.
    paint(2)
    assert preview.bounds is None
    assert "camera_preview" not in camera.workspace.rects
    assert not camera.workspace.input_overlays
    assert not any(key.startswith("camera-preview-") for key in ui.hits)
    hovered = imgui.get_current_context().hovered_window
    assert hovered is None or "##camera_preview" not in hovered.name
