"""Native preview pixels and keyboard navigation through actual viewport inputs."""

from dataclasses import replace

import numpy as np
import pytest
from imgui_bundle import imgui
from PIL import Image
from tests.gpu.test_ui_design_inspector import _center, _select, _visible_hit
from tests.gpu.test_ui_design_runtime import runtime as runtime

from mojive import commands as cmd
from mojive.adapters.base import NodeType

pytestmark = pytest.mark.gpu


def _click(study, point):
    study.event("add_mouse_pos_event", *point)
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_button_event", 0, False)


def _action(study, key):
    if key not in study.ui.hits:
        study.click_hit("camera-preview-more")
        study.frame(3)
    study.click_hit(key)
    study.frame(3)
    if imgui.get_current_context().open_popup_stack:
        study.event("add_key_event", imgui.Key.escape, True)
        study.event("add_key_event", imgui.Key.escape, False)
        study.frame(3)


def _scene_point(study):
    x, y, width, height = study.rects["viewport_content"]
    for ry in (0.6, 0.4, 0.7, 0.2, 0.8):
        for rx in (0.5, 0.3, 0.7, 0.15, 0.85):
            point = (x + width * rx, y + height * ry)
            if not any(
                a <= point[0] <= a + w and b <= point[1] <= b + h
                for a, b, w, h in study.rects["overlays"]
            ):
                return point
    raise AssertionError("The viewport has no free navigation point")


def test_camera_preview_toggle_presents_peer_pixels_and_shared_pin_lock(runtime, monkeypatch):
    study, app = runtime, runtime.preview
    session = app.session
    camera = next(node for node in session.nodes if node.type is NodeType.CAMERA)
    _select(study, camera)
    _visible_hit(study, "##camera_preview_enabled")
    assert not app.camera_preview.enabled
    image_rects = []
    image = imgui.image

    def observe_image(*args, **kwargs):
        image(*args, **kwargs)
        if "camera_preview" in str(imgui.internal.get_current_window().name):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            image_rects.append((lo.x, lo.y, hi.x, hi.y))

    monkeypatch.setattr(imgui, "image", observe_image)
    study.click_hit("##camera_preview_enabled")
    study.frame(5)
    assert app.camera_preview.enabled and app.camera_preview.bounds is not None
    assert image_rects
    pixels = np.asarray(study.draw(readback=True))[::-1].copy()
    peer = np.asarray(app.camera_preview._backend.target.read_color(flip=True))[..., :3]
    x, y, right, bottom = image_rects[-1]
    assert right - x > 40 * study.s and bottom - y > 20 * study.s
    px, py, end_x, end_y = app.camera_preview.bounds
    assert (x - px, end_x - right, end_y - bottom) == pytest.approx((6 * study.s,) * 3, abs=0.001)
    assert y - py == pytest.approx(30 * study.s)
    assert (right - x) / (bottom - y) == pytest.approx(peer.shape[1] / peer.shape[0])
    rendered = Image.fromarray(peer).resize(
        (round(right) - round(x), round(bottom) - round(y)), Image.Resampling.BILINEAR
    )
    actual = pixels[round(y) : round(bottom), round(x) : round(right), :3]
    expected = np.asarray(rendered)
    assert expected.std() > 3
    assert np.mean(np.abs(actual[3:-3, 3:-3].astype(float) - expected[3:-3, 3:-3])) < 20
    study.capture("camera-preview-enabled")

    _action(study, "camera-preview-pin")
    assert app.camera_preview.pinned
    assert session.submit(cmd.Select(0)).ok
    study.frame(3)
    assert app.camera_preview.bounds is not None
    _action(study, "camera-preview-lock")
    assert app.camera_preview.locked and not app.camera_preview.pinned
    camera_id = session.cameras[camera.camera_index].camera_id
    view = session.camera_view(camera_id)
    moved = replace(view, eye=np.asarray(view.eye) + np.array([0.15, 0.0, 0.0]))
    assert session.submit(cmd.SetSceneCamera(camera_id, moved)).ok
    study.frame(4)
    np.testing.assert_allclose(app.camera_preview._camera.eye, moved.eye)
    study.capture("camera-preview-locked")
    _action(study, "camera-preview-lock")
    assert session.submit(cmd.SelectNode(camera.node_id)).ok
    study.frame(3)

    start = _center(study.ui.hits["camera-preview-title"])
    before = app.camera_preview.bounds
    study.event("add_mouse_pos_event", *start)
    study.event("add_mouse_button_event", 0, True)
    study.event("add_mouse_pos_event", start[0] - 12 * study.s, start[1] - 8 * study.s)
    study.event("add_mouse_button_event", 0, False)
    study.frame(3)
    assert app.camera_preview.bounds != before
    study.capture("camera-preview-dragged")
    study.workspace_action("Review")
    study.frame(4)
    study.capture("camera-preview-short-viewport")
    assert app.camera_preview.bounds is not None, (
        app._viewport_rect,
        app._reference_camera_preview._region_key,
        app.camera_preview._image,
        app.session.selected_node,
    )
    study.click_hit("camera-preview-close")
    study.frame(3)
    assert not app.camera_preview.enabled and app.camera_preview.bounds is None
    assert "camera_preview" not in study.rects
    study.capture("camera-preview-closed")


def test_wasd_qe_fly_and_numeric_edit_ownership(runtime, monkeypatch):
    study, app = runtime, runtime.preview
    fields = {}
    drag_float = imgui.drag_float

    def observe_field(label, *args, **kwargs):
        result = drag_float(label, *args, **kwargs)
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        fields[label] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(imgui, "drag_float", observe_field)
    mode, snap = app.gizmo.mode, app._snap_latched
    for key in (imgui.Key.w, imgui.Key.a, imgui.Key.s, imgui.Key.d, imgui.Key.q, imgui.Key.e):
        before = app.camera.view().eye.copy()
        study.event("add_mouse_pos_event", *_scene_point(study))
        study.event("add_key_event", key, True)
        study.frame(5)
        study.event("add_key_event", key, False)
        assert np.linalg.norm(app.camera.view().eye - before) > 0.001
        assert app.gizmo.mode == mode and app._snap_latched == snap

    camera = next(node for node in app.session.nodes if node.type is NodeType.CAMERA)
    _select(study, camera)
    _visible_hit(study, "##inspector-camera-near")
    point = _center(fields["##inspector-camera-near-value"])
    modifier = (
        imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
    )
    study.event("add_key_event", modifier, True)
    _click(study, point)
    study.event("add_key_event", modifier, False)
    study.frame(3)
    assert imgui.get_io().want_text_input
    before = app.camera.view().eye.copy()
    for key in (imgui.Key.w, imgui.Key.s, imgui.Key.a, imgui.Key.d):
        study.event("add_key_event", key, True)
        study.frame(3)
        study.event("add_key_event", key, False)
        np.testing.assert_allclose(app.camera.view().eye, before)
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    study.frame(3)
    study.capture("camera-navigation-input-ownership")
