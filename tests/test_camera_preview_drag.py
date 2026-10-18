"""A captured camera header follows the pointer before painting each native frame."""

from math import ceil, floor

import pytest
from imgui_bundle import imgui

from tests.test_camera_preview_corners import _camera
from tests.test_component_appearance import appearance as appearance


def _driver(appearance, *, obstacles=()):
    ui, _runtime, _update, frame = appearance
    camera = _camera(ui)
    preview = camera.workspace.app.camera_preview

    def paint(count=1):
        for _ in range(count):
            ui.hits.clear()
            camera.workspace.rects["overlays"] = list(obstacles)
            camera.workspace.input_overlays.clear()
            frame(lambda: camera.draw("joint_gizmos"))

    def event(method, *args):
        getattr(imgui.get_io(), method)(*args)
        paint()

    def grab():
        paint(3)
        x, y, width, height = ui.hits["camera-preview-title"]
        point = (x + width / 2, y + height / 2)
        event("add_mouse_pos_event", *point)
        event("add_mouse_button_event", 0, True)
        return tuple(imgui.get_io().mouse_pos), preview.bounds

    return camera, preview, paint, event, grab


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("locked", (False, True))
def test_held_camera_title_follows_absolute_pointer_anchor_in_the_same_frame(appearance, locked):
    ui = appearance[0]
    camera, preview, paint, event, grab = _driver(
        appearance, obstacles=((300 * ui.s, 30 * ui.s, 180 * ui.s, 100 * ui.s),)
    )
    preview._locked = locked
    start, before = grab()
    offset = (start[0] - before[0], start[1] - before[1])
    for dx, dy in ((-20.25, -15.5), (-70.5, -50.25), (-125.75, -200.5), (-55.25, -25.75)):
        target = (start[0] + dx * ui.s, start[1] + dy * ui.s)
        event("add_mouse_pos_event", *target)
        assert preview.bounds[:2] == pytest.approx(
            (floor(target[0] - offset[0]), floor(target[1] - offset[1])), abs=1e-5
        )
        painted = preview.bounds
        paint(2)
        assert preview.bounds == painted
    event("add_mouse_button_event", 0, False)
    released = preview.bounds
    event("add_mouse_pos_event", 40, 40)
    assert preview.bounds == released
    assert preview.locked is locked
    assert camera.workspace.rects["camera_preview"][:2] == preview.bounds[:2]


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
def test_camera_drag_accumulates_subpixel_motion_and_recovers_from_viewport_clamp(appearance):
    ui = appearance[0]
    _camera_view, preview, _paint, event, grab = _driver(appearance)
    start, before = grab()
    for delta in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        event("add_mouse_pos_event", start[0] - delta, start[1] - delta)
        assert preview.bounds[:2] == pytest.approx(
            (floor(before[0] - delta), floor(before[1] - delta)), abs=1e-5
        )
    event("add_mouse_pos_event", -100, -100)
    viewport = _camera_view.workspace.app._viewport_rect
    assert preview.bounds[:2] == pytest.approx(
        (ceil(viewport[0] + 8 * ui.s), ceil(viewport[1] + 8 * ui.s))
    )
    event("add_mouse_pos_event", start[0] - 35, start[1] - 20)
    assert preview.bounds[:2] == pytest.approx((floor(before[0] - 35), floor(before[1] - 20)))
    event("add_mouse_button_event", 0, False)


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("pinned", (False, True))
def test_image_click_and_pinned_header_do_not_capture_panel_drag(appearance, pinned):
    camera, preview, paint, event, _grab = _driver(appearance)
    preview._pinned = pinned
    paint(3)
    before = preview.bounds
    if pinned:
        x, y, width, height = appearance[0].hits["camera-preview-title"]
        point = x + width / 2, y + height / 2
    else:
        point = (before[0] + before[2]) / 2, (before[1] + before[3]) / 2
    event("add_mouse_pos_event", *point)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_pos_event", point[0] - 50, point[1] - 30)
    assert preview.bounds == before
    event("add_mouse_button_event", 0, False)
    assert camera.workspace.app.camera_preview.enabled


@pytest.mark.parametrize("appearance", (1, 1.2, 1.25, 2.25), indirect=True)
def test_manual_preview_resizes_inside_all_viewport_edges_and_close_remains_clickable(appearance):
    ui = appearance[0]
    camera, preview, paint, event, grab = _driver(appearance)
    start, _before = grab()
    event("add_mouse_pos_event", start[0] - 80 * ui.s, start[1] - 60 * ui.s)
    event("add_mouse_button_event", 0, False)
    camera.workspace.app._viewport_rect = (20 * ui.s, 30 * ui.s, 210 * ui.s, 180 * ui.s)
    paint(4)
    x, y, width, height = camera.workspace.app._viewport_rect
    a, b, right, bottom = preview.bounds
    assert a >= ceil(x + 8 * ui.s) and b >= ceil(y + 8 * ui.s)
    assert right <= floor(x + width - 8 * ui.s)
    assert bottom <= floor(y + height - 8 * ui.s)
    close = ui.hits["camera-preview-close"]
    point = close[0] + close[2] / 2, close[1] + close[3] / 2
    event("add_mouse_pos_event", *point)
    event("add_mouse_button_event", 0, True)
    event("add_mouse_button_event", 0, False)
    paint(2)
    assert not preview.enabled and preview.bounds is None


@pytest.mark.parametrize("manual", (False, True))
def test_capture_collision_gate_distinguishes_user_preview_placement(manual):
    from types import SimpleNamespace

    from examples.ui_design.native.docking_study import frames

    preview, toolbar = (20, 20, 50, 40), (40, 30, 40, 20)
    snapshot = {"viewport_content": (0, 0, 100, 100), "overlays": (toolbar, preview)}
    study = SimpleNamespace(
        draw=lambda **kwargs: None,
        snapshot=lambda: snapshot,
        rects={"camera_preview": preview},
        preview=SimpleNamespace(
            camera_preview=SimpleNamespace(_position=(20, 20) if manual else None)
        ),
    )
    if manual:
        assert frames(None, study, count=1) == snapshot
        snapshot["overlays"] = (toolbar, (41, 31, 20, 10), preview)
    with pytest.raises(AssertionError, match="Overlapping viewport controls"):
        frames(None, study, count=1)
    snapshot["overlays"] = ((95, 95, 20, 20),)
    with pytest.raises(AssertionError, match="Overlay outside viewport"):
        frames(None, study, count=1)
