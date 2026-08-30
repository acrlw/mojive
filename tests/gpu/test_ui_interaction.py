from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.physics]

pytest.importorskip("glfw")
pytest.importorskip("mujoco")

from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.composition import build, build_workspace  # noqa: E402

W, H = 1280, 800


@pytest.fixture(autouse=True, scope="module")
def _pin_ui_scale(tmp_path_factory):
    # These tests assume scale-1 layout geometry (CI displays); pin the UI
    # scale so HiDPI machines produce the same coordinates.
    old = os.environ.get("FORGE_VIEWER_UI_SCALE")
    old_settings = os.environ.get("FORGE_VIEWER_SETTINGS")
    os.environ["FORGE_VIEWER_UI_SCALE"] = "1"
    os.environ["FORGE_VIEWER_SETTINGS"] = str(
        tmp_path_factory.mktemp("ui-interaction-settings") / "settings.json"
    )
    try:
        yield
    finally:
        if old is None:
            del os.environ["FORGE_VIEWER_UI_SCALE"]
        else:
            os.environ["FORGE_VIEWER_UI_SCALE"] = old
        if old_settings is None:
            del os.environ["FORGE_VIEWER_SETTINGS"]
        else:
            os.environ["FORGE_VIEWER_SETTINGS"] = old_settings


@pytest.fixture(scope="module")
def viewer():

    v = build(resolve("pick_scene"), "mujoco", paused=True, vsync=False, width=W, height=H)
    try:
        for _ in range(14):
            v.sync()
        yield v
    finally:
        v.release()


def snap(v) -> np.ndarray:

    px = v.window.read_frame()
    assert px is not None
    return np.asarray(px)[::-1][..., :3].copy()


def viewport_snap(v) -> np.ndarray:

    image = snap(v)
    x, y, w, h = v.window.points_to_pixels(v.app._viewport_rect)
    x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
    return image[y0:y1, x0:x1]


def center(v) -> tuple[float, float]:
    x, y, w, h = v.app._viewport_rect
    return x + w * 0.5, y + h * 0.5


def drag(v, io, x0, y0, dx, dy, steps=12, button=0):
    io.add_mouse_pos_event(x0, y0)
    io.add_mouse_button_event(button, True)
    v.sync()
    for i in range(1, steps + 1):
        io.add_mouse_pos_event(x0 + dx * i / steps, y0 + dy * i / steps)
        v.sync()
    io.add_mouse_button_event(button, False)
    v.sync()


def click(v, io, point, button=0):
    io.add_mouse_pos_event(*point)
    v.sync()
    io.add_mouse_button_event(button, True)
    v.sync()
    io.add_mouse_button_event(button, False)
    v.sync()


def activate_panel(v, name):

    from imgui_bundle import imgui

    window = imgui.internal.find_window_by_name(name)
    assert window is not None
    node = window.dock_node
    if node is None or node.selected_tab_id == window.tab_id:
        return
    tab = next(t for t in node.tab_bar.tabs if t.id_ == window.tab_id)
    bar = node.tab_bar.bar_rect
    point = (bar.min.x + tab.offset + tab.width * 0.5, (bar.min.y + bar.max.y) * 0.5)
    click(v, imgui.get_io(), point)


def _scroll_panel(v, name, wheel_y):
    from imgui_bundle import imgui

    window = imgui.internal.find_window_by_name(name)
    assert window is not None
    io = imgui.get_io()
    io.add_mouse_pos_event(window.pos.x + window.size.x * 0.5, window.pos.y + window.size.y * 0.7)
    io.add_mouse_wheel_event(0.0, wheel_y)
    for _ in range(3):
        v.sync()


def item_rect(v, function_name, label):

    from imgui_bundle import imgui

    original = getattr(imgui, function_name)
    found = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append(((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5))
        return result

    setattr(imgui, function_name, spy)
    try:
        v.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def item_bounds(v, function_name, label):

    from imgui_bundle import imgui

    original = getattr(imgui, function_name)
    found = []

    def spy(item_label, *args, **kwargs):
        result = original(item_label, *args, **kwargs)
        if item_label == label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            found.append((lo.x, lo.y, hi.x, hi.y))
        return result

    setattr(imgui, function_name, spy)
    try:
        v.sync()
    finally:
        setattr(imgui, function_name, original)
    assert found
    return found[-1]


def test_viewport_gets_real_estate(viewer):

    _x, _y, w, h = viewer.app._viewport_rect
    pw, ph = viewer.window.size_points
    assert w > 200 and h > 200
    assert (w * h) / (pw * ph) > 0.20


def test_interactive_entry_uses_adapter_camera_hint(viewer):
    hint = viewer.session.camera_hint()
    assert hint is not None
    view = viewer.app.camera.view()
    np.testing.assert_allclose(view.eye, hint.eye, atol=1e-6)
    np.testing.assert_allclose(view.target, hint.target, atol=1e-6)
    np.testing.assert_allclose(view.forward(), hint.forward(), atol=1e-6)
    assert view.fov_y == pytest.approx(hint.fov_y)
    assert view.near == pytest.approx(hint.near)
    assert view.far == pytest.approx(hint.far)
    assert view.orthographic is hint.orthographic


def test_all_panels_docked_not_stacked(viewer):

    from forge_viewer.ui.window import Window

    laid_out = set(
        Window._LAYOUT_LEFT
        + Window._LAYOUT_RIGHT_TOP
        + Window._LAYOUT_RIGHT_BOTTOM
        + Window._LAYOUT_BOTTOM
    )
    declared = {p.name for p in viewer.app.panels.panels if not p.standalone and not p.modal}
    assert declared <= laid_out


def test_output_panel_filters_visible_messages(viewer):
    panel = viewer.app.panels.get("Output")
    output = viewer.app.output
    output.clear()
    output.write("[forge/ui] FILTER_KEEP", level="warning", timestamp="10:00:00")
    output.write("[forge/window] FILTER_HIDE", level="info", timestamp="10:00:01")
    panel._filter_text = "forge/ui"
    panel._level_filter = 0
    panel._filter_cache_key = None
    activate_panel(viewer, "Output")

    try:
        viewer.sync()
        item_rect(viewer, "input_text_with_hint", "##output-filter")
        item_rect(viewer, "combo", "##output-level")
    finally:
        panel._filter_text = ""
        panel._level_filter = 0
        panel._filter_cache_key = None
        output.clear()

    assert [entry.text for entry in panel._filtered_entries] == ["[forge/ui] FILTER_KEEP"]


def test_hierarchy_search_clear_button_resets_filter(viewer):
    from imgui_bundle import imgui

    panel = viewer.app.panels.get("Hierarchy")
    panel._filter = "revolute"
    activate_panel(viewer, "Hierarchy")

    try:
        point = item_rect(viewer, "invisible_button", "##clear_filter")
        click(viewer, imgui.get_io(), point)
        assert panel._filter == ""
    finally:
        panel._filter = ""


def test_hierarchy_visibility_toggle_does_not_select_the_row(viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    activate_panel(viewer, "Hierarchy")
    target = next(n for n in viewer.session.nodes if n.object_id and n.visible)
    other = next(n for n in viewer.session.nodes if n.object_id and n is not target)
    selected_before = viewer.session.selected
    viewer.session.submit(cmd.Select(other.object_id))
    point = item_rect(viewer, "invisible_button", f"##vis{target.node_id}")

    click(viewer, imgui.get_io(), point)
    assert not target.visible
    assert viewer.session.selected == other.object_id

    click(viewer, imgui.get_io(), point)
    assert target.visible
    viewer.session.submit(cmd.Select(selected_before))


def test_environment_inspector_controls_render_flags(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeType
    from forge_viewer.render.backend import RenderFlag

    selected_before = viewer.session.selected
    environment = next(n for n in viewer.session.nodes if n.type is NodeType.ENVIRONMENT)
    viewer.session.submit(cmd.Select(environment.object_id))
    activate_panel(viewer, "Inspector")
    item_rect(viewer, "combo", "texture##skybox")
    item_rect(viewer, "combo", "mode##haze")
    _scroll_panel(viewer, "Inspector", -7.0)
    point = item_rect(viewer, "checkbox", "enabled##fog")
    before = viewer.backend.get_flag(RenderFlag.FOG)

    click(viewer, imgui.get_io(), point)
    assert viewer.backend.get_flag(RenderFlag.FOG) is not before

    click(viewer, imgui.get_io(), point)
    viewer.session.submit(cmd.Select(selected_before))


def test_material_inspector_exposes_instance_and_shared_controls(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeType

    selected_before = viewer.session.selected
    target = next(
        node
        for node in viewer.session.nodes
        if node.type is NodeType.LINK and node.name == "ball_00"
    )
    viewer.session.submit(cmd.Select(target.object_id))
    activate_panel(viewer, "Inspector")
    _scroll_panel(viewer, "Inspector", 100.0)
    header = item_rect(viewer, "collapsing_header", "material")
    click(viewer, imgui.get_io(), header)

    item_rect(viewer, "input_text", "##entity_name")
    item_rect(viewer, "color_edit4", "instance color")
    _scroll_panel(viewer, "Inspector", -6.0)
    contact = item_rect(viewer, "collapsing_header", "contact properties")
    click(viewer, imgui.get_io(), contact)
    _scroll_panel(viewer, "Inspector", -5.0)
    item_rect(viewer, "drag_float3", "friction (slide spin roll)")
    item_rect(viewer, "combo", "contact dimension")
    item_rect(viewer, "input_int", "collision type mask")
    item_rect(viewer, "begin_combo", "assigned material")
    item_rect(viewer, "small_button", "New material")
    item_rect(viewer, "small_button", "Duplicate material")
    item_rect(viewer, "small_button", "Import texture")
    item_rect(viewer, "color_edit4", "base color")
    item_rect(viewer, "begin_combo", "preset")
    item_rect(viewer, "drag_float", "specular")
    viewer.session.submit(cmd.Select(selected_before))


def test_orbit_moves_camera_and_picture(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    cx, cy = center(viewer)
    before_yaw = viewer.app.camera.yaw
    a = snap(viewer)

    drag(viewer, io, cx, cy, 144, -36)

    assert abs(viewer.app.camera.yaw - before_yaw) > 5.0
    diff = np.abs(snap(viewer).astype(np.int16) - a.astype(np.int16)).mean()
    assert diff > 1.0


def test_localized_viewport_accepts_camera_input(viewer):
    from imgui_bundle import imgui

    language = viewer.app.localizer.language
    try:
        viewer.app.set_language("zh_CN")
        viewer.sync()
        viewer.sync()
        cx, cy = center(viewer)
        before_yaw = viewer.app.camera.yaw

        drag(viewer, imgui.get_io(), cx, cy, 96.0, 0.0)

        assert abs(viewer.app.camera.yaw - before_yaw) > 5.0
    finally:
        viewer.app.set_language(language)
        viewer.sync()


def test_wheel_dollies(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    cx, cy = center(viewer)
    io.add_mouse_pos_event(cx, cy)
    viewer.sync()
    before = viewer.app.camera.distance
    a = snap(viewer)

    for _ in range(6):
        io.add_mouse_wheel_event(0.0, -1.0)
        viewer.sync()

    assert viewer.app.camera.distance > before * 1.05
    diff = np.abs(snap(viewer).astype(np.int16) - a.astype(np.int16)).mean()
    assert diff > 0.5


def test_floating_panel_over_the_viewport_blocks_camera_input(viewer):

    from imgui_bundle import imgui

    real_draw = viewer.app.panels.draw
    x, y, w, h = viewer.app._viewport_rect
    panel_pos = (x + w * 0.35, y + h * 0.35)

    def draw_with_overlay(ctx):
        real_draw(ctx)
        imgui.set_next_window_pos(imgui.ImVec2(*panel_pos), imgui.Cond_.always)
        imgui.set_next_window_size(imgui.ImVec2(240.0, 150.0), imgui.Cond_.always)
        _expanded, _open = imgui.begin("Floating Inspector test", True)
        imgui.text("dragging here must not orbit")
        imgui.end()

    viewer.app.panels.draw = draw_with_overlay
    io = imgui.get_io()
    try:
        point = (panel_pos[0] + 90.0, panel_pos[1] + 70.0)
        io.add_mouse_pos_event(*point)
        viewer.sync()
        before = (viewer.app.camera.yaw, viewer.app.camera.pitch)
        drag(viewer, io, *point, 50.0, -25.0)
        after = (viewer.app.camera.yaw, viewer.app.camera.pitch)
        assert after == pytest.approx(before, abs=1e-9)
    finally:
        viewer.app.panels.draw = real_draw
        io.add_mouse_button_event(0, False)
        viewer.sync()


def test_floating_viewport_separates_window_and_scene_gestures(viewer):
    from imgui_bundle import imgui

    v = build(resolve("pick_scene"), "mujoco", paused=True, vsync=False, width=W, height=H)
    try:
        for _ in range(8):
            v.sync()
        imgui.set_current_context(v.window._imgui_context)
        io = imgui.get_io()
        assert io.config_windows_move_from_title_bar_only
        viewport = imgui.internal.find_window_by_name("Viewport")
        imgui.internal.dock_context_process_undock_window(
            imgui.get_current_context(), viewport, True
        )
        v.sync()
        viewport = imgui.internal.find_window_by_name("Viewport")
        assert viewport.dock_node is None

        window_before = (viewport.pos.x, viewport.pos.y)
        yaw_before = v.app.camera.yaw
        x, y, width, height = v.app._viewport_rect
        drag(v, io, x + width * 0.5, y + height * 0.5, 80.0, 0.0)
        viewport = imgui.internal.find_window_by_name("Viewport")
        assert (viewport.pos.x, viewport.pos.y) == pytest.approx(window_before)
        assert abs(v.app.camera.yaw - yaw_before) > 5.0

        window_before = (viewport.pos.x, viewport.pos.y)
        yaw_before = v.app.camera.yaw
        title = viewport.title_bar_rect()
        title_x = (title.min.x + title.max.x) * 0.5
        title_y = (title.min.y + title.max.y) * 0.5
        drag(v, io, title_x, title_y, 70.0, 35.0)
        viewport = imgui.internal.find_window_by_name("Viewport")
        assert viewport.pos.x > window_before[0] + 50.0
        assert viewport.pos.y > window_before[1] + 20.0
        assert v.app.camera.yaw == pytest.approx(yaw_before)

        size_before = (viewport.size.x, viewport.size.y)
        yaw_before = v.app.camera.yaw
        edge = (viewport.pos.x + viewport.size.x - 1.0, viewport.pos.y + viewport.size.y * 0.5)
        drag(v, io, *edge, 50.0, 0.0)
        viewport = imgui.internal.find_window_by_name("Viewport")
        assert viewport.size.x > size_before[0] + 30.0
        assert viewport.size.y == pytest.approx(size_before[1])
        assert v.app.camera.yaw == pytest.approx(yaw_before)
    finally:
        v.release()
        imgui.set_current_context(viewer.window._imgui_context)


def _project(cam, world, rect):

    clip = cam.proj_matrix() @ (cam.view_matrix() @ np.array([*world, 1.0], np.float64))
    if clip[3] <= 0.0:
        return None
    ndc = clip[:3] / clip[3]
    if not (-1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0):
        return None
    x, y, w, h = rect

    return x + w * (ndc[0] * 0.5 + 0.5), y + h * (0.5 - ndc[1] * 0.5)


def test_click_picks_the_object_actually_under_the_cursor(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    viewer.app._frame_scene(animate=False)
    viewer.sync()

    for _ in range(3):
        viewer.app.camera.dolly(-1.0)
    viewer.app.camera.pan(0.0, 90.0, viewer.app._viewport_rect[3])
    for _ in range(3):
        viewer.sync()

    cam = viewer.app.camera.view()
    frame = viewer.session.frame
    assert frame.body_xpos is not None
    rect = viewer.app._viewport_rect
    _x, y, _w, h = rect
    mid_y = y + h * 0.5

    projected = []
    for node in viewer.session.nodes:
        if node.object_id == 0 or node.body_index < 0:
            continue
        world = np.asarray(frame.body_xpos[node.body_index], np.float64)
        pt = _project(cam, world, rect)
        if pt is not None:
            projected.append((float(np.linalg.norm(world - np.asarray(cam.eye))), node, pt))
    assert projected

    CLEAR_PT = 30.0
    candidates = []
    for dist, node, (px, py) in projected:
        if abs(py - mid_y) < CLEAR_PT * 2:
            continue
        mx, my = px, 2 * mid_y - py
        if any(
            np.hypot(qx - mx, qy - my) < CLEAR_PT
            for _d, other, (qx, qy) in projected
            if other is not node
        ):
            continue
        candidates.append((dist, node, (px, py)))
    assert candidates
    _dist, target, (px, py) = min(candidates, key=lambda t: t[0])

    io.add_mouse_pos_event(px, py)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    viewer.sync()

    got = viewer.session.selected_node
    assert got is not None
    assert got.object_id == target.object_id

    img = viewer.app._viewport_image
    assert img is not None
    hit = img.pixel_from_viewport_point((px, py), rect)
    assert hit is not None
    assert int(viewer.backend.pick(*hit)) == target.object_id


def test_selection_reaches_the_outline_in_the_window(viewer):

    from forge_viewer import commands as cmd

    target = next(n for n in viewer.session.nodes if n.name == "cluster")
    viewer.session.submit(cmd.Select(0))
    viewer.app._frame_scene(animate=False)
    viewer.sync()
    viewer.sync()
    before = viewport_snap(viewer)

    viewer.session.submit(cmd.Select(target.object_id))
    viewer.sync()
    viewer.sync()
    after = viewport_snap(viewer)

    color = np.array([255, 161, 51], np.int16)
    before_count = np.all(np.abs(before.astype(np.int16) - color) <= 3, axis=-1).sum()
    after_count = np.all(np.abs(after.astype(np.int16) - color) <= 3, axis=-1).sum()
    assert after_count > before_count + 20


def test_view_gizmo_fits_the_corner(viewer):

    balls = viewer.app.view_cube.balls
    assert balls
    xs = [b.screen[0] for b in balls]
    ys = [b.screen[1] for b in balls]
    r = max(b.radius for b in balls)
    left, right = min(xs) - r, max(xs) + r
    top, bottom = min(ys) - r, max(ys) + r

    x, y, w, h = viewer.app._viewport_rect
    assert x <= left and right <= x + w
    assert y <= top and bottom <= y + h
    share = ((right - left) * (bottom - top)) / (w * h)

    assert share < 0.05


@pytest.mark.parametrize(
    ("axis", "sign", "yaw", "pitch"),
    [(0, 1.0, 0.0, 0.0), (0, -1.0, 180.0, 0.0), (1, 1.0, 90.0, 0.0), (1, -1.0, 270.0, 0.0)],
)
def test_view_gizmo_click_snaps_to_that_axis(viewer, axis, sign, yaw, pitch):

    from imgui_bundle import imgui

    io = imgui.get_io()
    viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
    viewer.sync()

    ball = next(b for b in viewer.app.view_cube.balls if b.axis == axis and b.sign == sign)
    io.add_mouse_pos_event(*ball.screen)
    viewer.sync()

    hit = viewer.app.view_cube.hovered
    assert hit is not None and (hit.axis, hit.sign) == (axis, sign)
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)

    for _ in range(240):
        viewer.sync()

    got_yaw = viewer.app.camera.yaw % 360.0
    assert abs((got_yaw - yaw + 180.0) % 360.0 - 180.0) < 1.0
    assert abs(viewer.app.camera.pitch - pitch) < 1.0


def test_view_gizmo_click_frames_the_selected_object(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeType

    node = next(
        node
        for node in viewer.session.nodes
        if node.object_id > 0
        and node.body_index > 0
        and node.type in (NodeType.ROBOT, NodeType.LINK)
    )
    viewer.session.submit(cmd.Select(node.object_id))
    viewer.app.view_cube.selection_padding = 1.6
    viewer.app.camera.pivot = np.array((20.0, -15.0, 9.0))
    viewer.app.camera.distance = 30.0
    viewer.sync()
    focus = viewer.app._selected_view_focus()
    assert focus is not None
    center, radius = focus

    ball = next(b for b in viewer.app.view_cube.balls if b.axis == 0 and b.sign == 1.0)
    click(viewer, imgui.get_io(), ball.screen)
    viewer.app.camera.advance(1.0, viewer.app.camera_out)
    viewer.sync()

    expected_distance, _height = viewer.app.camera._framing_distance(radius, 1.6)
    assert viewer.app.camera.pivot == pytest.approx(center, abs=1e-5)
    assert viewer.app.camera.distance == pytest.approx(expected_distance)

    viewer.session.submit(cmd.Select(0))
    viewer.app.view_cube.selection_padding = 1.2


def test_view_gizmo_axis_points_at_you_when_you_look_down_it(viewer):

    viewer.app.camera.look_from(0.0, 0.0, viewer.app.camera_out, animate=False)
    viewer.sync()

    from forge_viewer.ui import viewcube as vc

    s = viewer.window.style_scale
    cx, cy = vc.widget_center(viewer.app._viewport_rect, s)
    reach = vc.RADIUS_PT * s

    for b in viewer.app.view_cube.balls:
        d = float(np.hypot(b.screen[0] - cx, b.screen[1] - cy))
        if b.axis == 0:
            assert d < reach * 0.1
        else:
            assert d > reach * 0.9


def _ball_and_ink(frame, ball, scale):

    cx, cy = ball.screen[0] * scale, ball.screen[1] * scale
    r = ball.radius * scale
    x0, y0 = int(cx - r - 3), int(cy - r - 3)
    sub = frame[y0 : int(cy + r + 4), x0 : int(cx + r + 4)].astype(np.int16)
    r_, g_, b_ = sub[..., 0], sub[..., 1], sub[..., 2]
    dominance = (r_ - np.maximum(g_, b_), g_ - np.maximum(r_, b_), b_ - np.maximum(r_, g_))[
        ball.axis
    ]
    disc = dominance > 25
    ink = (sub.min(axis=2) > 165) & ((sub.max(axis=2) - sub.min(axis=2)) < 40)
    if not disc.any() or not ink.any():
        return None

    def box(mask):
        ys, xs = np.nonzero(mask)
        return (
            (xs.min() + xs.max()) / 2 + x0,
            (ys.min() + ys.max()) / 2 + y0,
            xs.max() - xs.min() + 1,
        )

    bx, by, bw = box(disc)
    ix, iy, _ = box(ink)
    return (bx, by), (ix, iy), bw


def test_gizmo_label_sits_in_the_middle_of_its_ball(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    dxs, dys = [], []
    for yaw in (-135.0, -100.0, -62.0, -20.0, 25.0, 70.0):
        viewer.app.camera.look_from(yaw, 25.0, viewer.app.camera_out, animate=False)
        viewer.sync()
        for axis in range(3):
            target = next(
                b for b in viewer.app.view_cube.balls if b.axis == axis and not b.positive
            )
            io.add_mouse_pos_event(*target.screen)
            for _ in range(2):
                viewer.sync()
            hit = viewer.app.view_cube.hovered
            if hit is None or (hit.axis, hit.sign) != (axis, target.sign):
                continue
            b = next(x for x in viewer.app.view_cube.balls if (x.axis, x.sign) == (axis, -1.0))
            got = _ball_and_ink(snap(viewer), b, viewer.window.pixel_scale)
            if got is None:
                continue
            (bx, by), (ix, iy), _bw = got
            dxs.append(ix - bx)
            dys.append(iy - by)

    assert len(dxs) >= 9
    mx, my = float(np.mean(dxs)), float(np.mean(dys))
    assert abs(mx) < 1.5
    assert abs(my) < 1.5


class _RecordingDrawList:
    def __init__(self, inner, balls):
        self._inner = inner
        self._balls = balls
        self.calls: list[tuple[str, int]] = []

    @property
    def flags(self):
        return self._inner.flags

    @flags.setter
    def flags(self, value):
        self._inner.flags = value

    def _nearest(self, pos):
        best, bd = -1, 1e18
        for i, b in enumerate(self._balls):
            d = (b.screen[0] - pos.x) ** 2 + (b.screen[1] - pos.y) ** 2
            if d < bd:
                best, bd = i, d
        return best if bd <= 50.0**2 else -1

    def _record(self, event_type, pos):
        index = self._nearest(pos)
        if index >= 0:
            self.calls.append((event_type, index))

    def add_line(self, a, b, *rest):
        self._record("line", b)
        return self._inner.add_line(a, b, *rest)

    def add_concave_poly_filled(self, points, *rest):
        self._record("lollipop", points[len(points) // 2])
        return self._inner.add_concave_poly_filled(points, *rest)

    def add_circle_filled(self, pos, r, *rest):
        if r < 30.0:
            self._record("disc", pos)
        return self._inner.add_circle_filled(pos, r, *rest)

    def add_circle(self, pos, r, *rest):
        self._record("ring", pos)
        return self._inner.add_circle(pos, r, *rest)

    def add_image(self, tex, p_min, p_max, *rest):
        self._record("label", p_min)
        return self._inner.add_image(tex, p_min, p_max, *rest)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_gizmo_draws_each_axis_as_one_unit(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    viewer.app.camera.look_from(-118.0, 22.0, viewer.app.camera_out, animate=False)
    viewer.sync()
    balls = list(viewer.app.view_cube.balls)
    io.add_mouse_pos_event(*next(b for b in balls if not b.positive).screen)
    viewer.sync()

    real = imgui.get_window_draw_list
    rec: list[_RecordingDrawList] = []

    def spy():
        current = imgui.get_current_context().current_window
        if current is None or str(current.name) != "Viewport":
            return real()
        dl = _RecordingDrawList(real(), balls)
        rec.append(dl)
        return dl

    imgui.get_window_draw_list = spy
    try:
        viewer.sync()
    finally:
        imgui.get_window_draw_list = real

    calls = [c for dl in rec for c in dl.calls]
    assert calls

    span: dict[int, tuple[int, int]] = {}
    for pos, (_kind, idx) in enumerate(calls):
        lo, hi = span.get(idx, (pos, pos))
        span[idx] = (min(lo, pos), max(hi, pos))

    for i in sorted(span):
        for j in sorted(span):
            if i < j:
                assert span[i][1] < span[j][0]

    for idx, (lo, hi) in span.items():
        event_types = [event_type for event_type, i in calls[lo : hi + 1] if i == idx]
        if balls[idx].positive:
            assert event_types.count("lollipop") == 1
        if "label" in event_types:
            body = "lollipop" if balls[idx].positive else "disc"
            assert event_types.index(body) < event_types.index("label")


def test_negative_balls_are_dark_and_opaque(viewer):

    from imgui_bundle import imgui

    imgui.get_io().add_mouse_pos_event(0.0, 0.0)

    def sample(yaw):
        viewer.app.camera.look_from(yaw, 25.0, viewer.app.camera_out, animate=False)
        for _ in range(3):
            viewer.sync()
        frame = snap(viewer)
        s = viewer.window.pixel_scale
        out = {}
        for b in viewer.app.view_cube.balls:
            px = int(b.screen[0] * s)
            py = int(b.screen[1] * s - b.radius * s * 0.5)
            bg_y = int((b.screen[1] + b.radius * 2.2) * s)
            out[(b.axis, b.positive)] = (
                frame[py, px].astype(float),
                frame[min(bg_y, frame.shape[0] - 1), px].astype(float),
            )
        return out

    a = sample(-135.0)
    b = sample(-52.0)

    for axis in range(3):
        pos = a[(axis, True)][0]
        neg = a[(axis, False)][0]

        assert neg[axis] < pos[axis] * 0.6

    moved_bg = 0
    for key in a:
        if key[1]:
            continue
        fill_a, bg_a = a[key]
        fill_b, bg_b = b[key]
        if float(np.abs(bg_a - bg_b).max()) < 8.0:
            continue
        moved_bg += 1
        assert float(np.abs(fill_a - fill_b).max()) < 6.0
    assert moved_bg >= 1


def test_hover_does_not_resize_the_ball(viewer):

    from imgui_bundle import imgui

    io = imgui.get_io()
    viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
    io.add_mouse_pos_event(0.0, 0.0)
    for _ in range(3):
        viewer.sync()
    target = next(ball for ball in viewer.app.view_cube.balls if ball.positive)
    plain_radius = target.radius

    io.add_mouse_pos_event(*target.screen)
    for _ in range(3):
        viewer.sync()
    assert viewer.app.view_cube.hovered is not None
    hovered = next(
        ball
        for ball in viewer.app.view_cube.balls
        if (ball.axis, ball.sign) == (target.axis, target.sign)
    )
    assert hovered.radius == plain_radius


def test_top_view_is_canonical_x_right_y_up(viewer):

    from forge_viewer.ui import viewcube as vc
    from forge_viewer.ui.camera import camera_basis

    yaw, pitch = vc.yaw_pitch_for(2, 1.0, 37.0)
    viewer.app.camera.look_from(yaw, pitch, viewer.app.camera_out, animate=False)
    viewer.sync()
    right, up, _fwd = camera_basis(viewer.app.camera.view())
    assert right[0] > 0.99
    assert up[1] > 0.99


def test_clicking_during_a_transition_does_not_strand_the_camera(viewer):

    from imgui_bundle import imgui

    from forge_viewer.ui import viewcube as vc

    io = imgui.get_io()
    viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
    viewer.sync()

    ball = next(b for b in viewer.app.view_cube.balls if b.axis == 2 and b.positive)
    io.add_mouse_pos_event(*ball.screen)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()
    assert viewer.app.camera.animating

    cx, cy = center(viewer)
    io.add_mouse_pos_event(cx, cy)
    viewer.sync()
    io.add_mouse_button_event(0, True)
    viewer.sync()
    io.add_mouse_pos_event(cx + 1.0, cy + 1.0)
    viewer.sync()
    io.add_mouse_button_event(0, False)
    viewer.sync()

    for _ in range(400):
        viewer.sync()

    assert not viewer.app.camera.animating
    assert abs(viewer.app.camera.pitch - vc.PITCH_LIMIT) < 0.5
    assert abs((viewer.app.camera.yaw - vc.TOP_YAW + 180.0) % 360.0 - 180.0) < 0.5


def test_font_is_monospace(viewer):

    from imgui_bundle import imgui

    widths = {c: imgui.calc_text_size(c).x for c in "iWml.0X"}
    assert len(set(widths.values())) == 1


def test_cjk_glyphs_present(viewer):

    from imgui_bundle import imgui

    if not viewer.window.font_report.cjk:
        pytest.skip(f"CJK font unavailable: {viewer.window.font_report.notes}")
    font = imgui.get_io().fonts.fonts[0]
    sample = "\u6ca1\u6709\u753b\u9762\u540e\u7aef\u8fd9\u4e00\u5e27\u753b\u4e0d\u51fa\u6765"
    missing = [c for c in sample if not font.is_glyph_in_font(ord(c))]
    assert not missing

    assert not font.is_glyph_in_font(0x10FFFD)


def test_font_size_is_in_layout_space(viewer):

    from imgui_bundle import imgui

    want = viewer.window.config.font_size_pt * viewer.window.style_scale
    got = imgui.get_font_size()
    assert abs(got - want) < 0.5

    style = imgui.get_style()
    button_h = imgui.get_frame_height()
    assert imgui.get_text_line_height() + 2 * style.frame_padding.y <= button_h + 0.01


@pytest.fixture(scope="module")
def free_body_viewer():

    v = build(resolve("perturb_ghost"), "mujoco", paused=True, vsync=False, width=W, height=H)
    try:
        for _ in range(14):
            v.sync()
        yield v
    finally:
        v.release()


def test_inspector_transform_resets_and_copies_without_gesture_conflicts(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    for _ in range(3):
        v.sync()
    old_pos = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64).copy()
    old_mat = (
        np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3).copy()
    )
    wanted = np.array((1.25, -0.85, 0.06), np.float32)
    v.session.submit(cmd.SetPose(node.node_id, wanted, old_mat))
    for _ in range(3):
        v.sync()

    activate_panel(v, "Inspector")
    position_label = item_bounds(v, "text", "position")
    x_axis = item_bounds(v, "button", f"X##position_0_{node.node_id}")
    x_value = item_bounds(v, "drag_float", f"##position_0_{node.node_id}")
    assert position_label[2] < x_axis[0]
    assert (position_label[1] + position_label[3]) * 0.5 == pytest.approx(
        (x_axis[1] + x_axis[3]) * 0.5,
        abs=0.1,
    )
    assert 2.0 <= x_value[0] - x_axis[2] <= 6.0

    from forge_viewer.ui.panels.inspector import _mix_color
    from forge_viewer.ui.theme import THEME

    readonly_x = item_bounds(v, "button", f"X##linear velocity_0_{node.node_id}")
    image = snap(v)
    expected_axis = np.rint(
        np.asarray(_mix_color(THEME.bg_frame, THEME.axis_color(0), 0.56)[:3]) * 255.0
    ).astype(np.int16)
    center_y = (readonly_x[1] + readonly_x[3]) * 0.5
    for x0, x1 in (
        (readonly_x[0] + 3.0, readonly_x[0] + 7.0),
        (readonly_x[2] - 2.5, readonly_x[2] - 0.5),
    ):
        px0, py0 = v.window.points_to_pixels((x0, center_y - 3.0))
        px1, py1 = v.window.points_to_pixels((x1, center_y + 3.0))
        sample = image[round(py0) : round(py1), round(px0) : round(px1)].astype(np.int16)
        median = np.median(sample.reshape(-1, 3), axis=0)
        assert np.max(np.abs(median - expected_axis)) <= 2

    x_reset = item_rect(v, "button", f"X##position_0_{node.node_id}")
    click(v, io, x_reset)
    v.sync()
    got = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    assert got == pytest.approx((0.0, -0.85, 0.06), abs=1e-5)

    def copied_from(point):
        copied = []
        original = imgui.set_clipboard_text
        imgui.set_clipboard_text = copied.append
        try:
            click(v, io, point, button=1)
        finally:
            imgui.set_clipboard_text = original
        assert copied
        return copied[-1]

    y_value = item_rect(v, "drag_float", f"##position_1_{node.node_id}")
    assert copied_from(y_value) == "-0.850"
    group = item_rect(v, "text", "position")
    assert copied_from(group) == "0, -0.85, 0.06"

    v.session.submit(cmd.SetPose(node.node_id, old_pos, old_mat))
    for _ in range(2):
        v.sync()


def test_inspector_drag_stays_ui_owned_after_crossing_into_viewport(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    activate_panel(v, "Inspector")
    for _ in range(3):
        v.sync()

    old_pos = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64).copy()
    old_mat = (
        np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3).copy()
    )
    start = item_rect(v, "drag_float", f"##position_0_{node.node_id}")
    vx, _vy, vw, _vh = v.app._viewport_rect
    destination = (vx + vw * 0.45, start[1])
    camera_before = (v.app.camera.yaw, v.app.camera.pitch)

    try:
        io.add_mouse_pos_event(*start)
        v.sync()
        io.add_mouse_button_event(0, True)
        v.sync()
        for alpha in np.linspace(0.1, 1.0, 10):
            io.add_mouse_pos_event(
                start[0] + (destination[0] - start[0]) * float(alpha), destination[1]
            )
            v.sync()
        io.add_mouse_button_event(0, False)
        v.sync()

        moved = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
        assert abs(moved[0] - old_pos[0]) > 0.1
        assert (v.app.camera.yaw, v.app.camera.pitch) == pytest.approx(camera_before, abs=1e-9)
    finally:
        io.add_mouse_button_event(0, False)
        v.session.submit(cmd.SetPose(node.node_id, old_pos, old_mat))
        for _ in range(2):
            v.sync()


def test_inspector_rotation_y_drags_continuously_past_gimbal_lock(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer import math3d

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    old_pos = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64).copy()
    old_mat = (
        np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3).copy()
    )
    v.session.submit(cmd.Select(node.object_id))
    v.session.submit(cmd.SetPose(node.node_id, old_pos, np.eye(3)))
    activate_panel(v, "Inspector")
    for _ in range(3):
        v.sync()

    start = item_rect(v, "drag_float", f"##rotation_1_{node.node_id}")
    panel = v.app.panels.get("Inspector")
    samples = []
    try:
        io.add_mouse_pos_event(*start)
        v.sync()
        io.add_mouse_button_event(0, True)
        v.sync()
        for dx in np.linspace(-20.0, -300.0, 15):
            io.add_mouse_pos_event(start[0] + float(dx), start[1])
            v.sync()
            samples.append(float(panel._rotation_euler[1]))
        io.add_mouse_button_event(0, False)
        v.sync()

        assert min(samples) < -100.0
        assert np.max(np.diff(samples)) < 1.0
        displayed = np.asarray(panel._rotation_euler, np.float64)
        actual = np.asarray(v.session.frame.body_xmat[node.body_index]).reshape(3, 3)
        assert actual == pytest.approx(math3d.euler_xyz_to_mat3(np.radians(displayed)), abs=2e-5)
    finally:
        io.add_mouse_button_event(0, False)
        v.session.submit(cmd.SetPose(node.node_id, old_pos, old_mat))
        for _ in range(2):
            v.sync()


def test_gizmo_is_live_for_a_free_body(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    for _ in range(4):
        v.sync()
    assert v.app.gizmo.last_verdict.ok

    cam = v.app.camera.view()
    x, y, w, h = v.app._viewport_rect
    world = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    clip = cam.proj_matrix() @ (cam.view_matrix() @ np.array([*world, 1.0]))
    ndc = clip[:3] / clip[3]
    px = x + w * (ndc[0] * 0.5 + 0.5)
    py = y + h * (0.5 - ndc[1] * 0.5)

    io.add_mouse_pos_event(px, py)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered

    io.add_mouse_pos_event(x + 12.0, y + h - 12.0)
    for _ in range(2):
        v.sync()
    assert not v.app.gizmo.hovered


def test_tool_column_hides_without_actions_and_centers_when_available(free_body_viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    assert v.session.submit(cmd.Select(0))
    for _ in range(3):
        v.sync()
    tools = imgui.internal.find_window_by_name("Tools###viewport_tools")
    assert tools is None or not tools.active

    node = next(item for item in v.session.nodes if item.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    for _ in range(3):
        v.sync()
    tools = imgui.internal.find_window_by_name("Tools###viewport_tools")
    assert tools is not None and tools.active
    _x, viewport_y, _width, viewport_height = v.app._viewport_rect
    assert tools.pos.y + tools.size.y * 0.5 == pytest.approx(
        viewport_y + viewport_height * 0.5,
        abs=2.0,
    )


@pytest.mark.parametrize("workspace", (False, True), ids=("viewer", "editor"))
def test_joint_gizmo_is_live_in_the_real_viewer_pipeline(workspace):
    """Viewer and editor must retain the diagnostics requested by a joint gizmo."""

    import forge_viewer.commands as cmd

    factory = build_workspace if workspace else build
    v = factory(
        resolve("joint_types"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    try:
        for _ in range(10):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == "hinge_body")
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(3):
            v.sync()

        assert v.session.frame.diagnostics is not None
        assert v.app.frame_needs().diagnostics
        assert v.app.gizmo.last_verdict.ok
        assert v.app.gizmo.visible
    finally:
        v.release()


def test_joint_limit_label_click_sets_the_endpoint_in_the_real_viewer() -> None:
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = build(
        resolve("joint_types"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    try:
        for _ in range(10):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == "hinge_body")
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(4):
            v.sync()

        target, reason = v.app.gizmo._joint_target(v.session, node)
        assert target is not None, reason
        lower_hit, upper_hit = v.app.gizmo.joint_limit_hits
        assert lower_hit.value == pytest.approx(target.joint.range[0])
        assert upper_hit.value == pytest.approx(target.joint.range[1])
        x0, y0, x1, y1 = lower_hit.rect
        click(v, imgui.get_io(), ((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        for _ in range(2):
            v.sync()

        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(target.joint.range[0])
    finally:
        v.release()


def test_limited_hinge_drag_keeps_feedback_and_claim_until_mouse_release() -> None:
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, project, world_scale

    v = build(
        resolve("joint_types"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        for _ in range(8):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == "hinge_body")
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(4):
            v.sync()

        target, reason = v.app.gizmo._joint_target(v.session, node)
        assert target is not None, reason
        pose = v.app.gizmo._target_pose(v.session, node, target)
        assert pose is not None
        position, basis = pose
        cam = v.app._camera_view()
        rect = v.app._viewport_rect
        scale = world_scale(cam, position, rect[3], SIZE_PT)

        def cursor(angle: float) -> np.ndarray:
            world = (
                position
                + (basis[:, 0] * np.cos(angle) + basis[:, 1] * np.sin(angle)) * scale * RING_RADIUS
            )
            return project(cam, (world,), rect)[0, :2]

        start_angle = next(
            angle
            for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            if v.app.gizmo.update_hover(
                v.session,
                cam,
                rect,
                tuple(cursor(angle)),
            )
            is GizmoHandle.ROTATE_Z
        )
        start = cursor(start_angle)
        io.add_mouse_pos_event(*start)
        v.sync()
        io.add_mouse_button_event(0, True)
        v.sync()
        assert v.app.gizmo.using

        qpos = v.session.frame.qpos
        assert qpos is not None
        origin = float(qpos[target.joint.qpos_adr])
        lower = float(target.joint.range[0])
        overtravel = lower - origin - 0.35
        for delta in np.linspace(0.0, overtravel, 40)[1:]:
            io.add_mouse_pos_event(*cursor(start_angle + delta))
            v.sync()
        v.sync()

        assert v.app.gizmo.using
        assert v.app.gizmo.active_handle is GizmoHandle.ROTATE_Z
        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(lower)

        class Recorder:
            def __init__(self, inner):
                self.inner = inner
                self.fans = 0

            def triangle_fan_fill(self, points, color):
                self.fans += 1
                return self.inner.triangle_fan_fill(points, color)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        original_draw = v.app.gizmo.draw_overlay
        recorders = []

        def spy(cam, rect, overlay, *, style_scale=1.0):
            recorder = Recorder(overlay)
            recorders.append(recorder)
            return original_draw(cam, rect, recorder, style_scale=style_scale)

        v.app.gizmo.draw_overlay = spy
        try:
            v.sync()
        finally:
            v.app.gizmo.draw_overlay = original_draw
        assert recorders and recorders[-1].fans == 1
        assert v.app.gizmo.using

        io.add_mouse_pos_event(*cursor(start_angle + overtravel + 0.05))
        v.sync()
        v.sync()
        assert v.session.frame.qpos is not None
        returned = float(v.session.frame.qpos[target.joint.qpos_adr])
        assert lower + 0.03 < returned < lower + 0.07
        assert v.app.gizmo.using

        io.add_mouse_button_event(0, False)
        v.sync()
        assert not v.app.gizmo.using
    finally:
        if imgui.is_mouse_down(0):
            io.add_mouse_button_event(0, False)
            v.sync()
        v.release()


def test_limited_slide_drag_keeps_feedback_and_claim_until_mouse_release() -> None:
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import GizmoHandle

    v = build(
        resolve("joint_types"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        for _ in range(8):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == "slide_body")
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(4):
            v.sync()

        target, reason = v.app.gizmo._joint_target(v.session, node)
        assert target is not None, reason
        pose = v.app.gizmo._target_pose(v.session, node, target)
        assert pose is not None
        position, basis = pose
        cam = v.app._camera_view()
        rect = v.app._viewport_rect
        state = v.app.gizmo._joint_range_state(v.session, target)
        assert state is not None
        slide = v.app.gizmo._slide_range_projection(
            cam,
            rect,
            v.window.style_scale,
            state,
            position,
            basis,
        )
        assert slide is not None
        arrow = v.app.gizmo._slide_arrow_polygon(slide, v.window.style_scale)
        start = np.mean(arrow, axis=0)
        io.add_mouse_pos_event(*start)
        v.sync()
        assert v.app.gizmo.hovered_handle is GizmoHandle.Z
        io.add_mouse_button_event(0, True)
        v.sync()
        assert v.app.gizmo.using
        drag_origin = v.app.gizmo._drag_origin_pos.copy()

        qpos = v.session.frame.qpos
        assert qpos is not None
        origin = float(qpos[target.joint.qpos_adr])
        lower = float(target.joint.range[0])
        overtravel = lower - origin - 0.12
        end = start + v.app.gizmo._axis_screen * (overtravel / v.app.gizmo._world_per_pt)
        for cursor in np.linspace(start, end, 32)[1:]:
            io.add_mouse_pos_event(*cursor)
            v.sync()
        v.sync()

        assert v.app.gizmo.using
        assert v.app.gizmo.active_handle is GizmoHandle.Z
        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(lower)
        assert v.app.gizmo._drag_origin_pos == pytest.approx(drag_origin)
        assert np.linalg.norm(v.app.gizmo._start_pos - drag_origin) > 0.1

        farther = end - v.app.gizmo._axis_screen * (0.08 / v.app.gizmo._world_per_pt)
        io.add_mouse_pos_event(*farther)
        v.sync()
        assert v.app.gizmo.using
        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(lower)

        inward = farther + v.app.gizmo._axis_screen * (0.02 / v.app.gizmo._world_per_pt)
        io.add_mouse_pos_event(*inward)
        v.sync()
        v.sync()
        assert v.session.frame.qpos is not None
        returned = float(v.session.frame.qpos[target.joint.qpos_adr])
        assert lower + 0.01 < returned < lower + 0.025
        assert v.app.gizmo.using

        io.add_mouse_button_event(0, False)
        v.sync()
        assert not v.app.gizmo.using
    finally:
        if imgui.is_mouse_down(0):
            io.add_mouse_button_event(0, False)
            v.sync()
        v.release()


def test_multi_joint_viewport_picker_selects_the_gizmo_target():
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    try:
        for _ in range(10):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == "05_multi_joint")
        vx, vy, vw, vh = v.app._viewport_rect
        imgui.get_io().add_mouse_pos_event(vx + vw * 0.55, vy + vh * 0.45)
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(3):
            v.sync()

        choices = v.app.gizmo.joint_choices(v.session)
        assert [joint.name for joint in choices] == [
            "05_multi_slide_x",
            "05_multi_slide_y",
            "05_multi_revolute_z",
        ]
        assert not v.app.gizmo.last_verdict.ok
        picker = imgui.internal.find_window_by_name("Joint gizmo###viewport_joint_gizmo")
        assert picker is not None and picker.active
        assert not picker.flags & imgui.WindowFlags_.no_move.value
        assert abs(picker.pos.x - (vx + vw * 0.55)) < 320.0
        assert abs(picker.pos.y - (vy + vh * 0.45)) < 240.0
        label = f"05_multi_revolute_z  (hinge)##viewport-joint-{choices[2].joint_id}"
        click(v, imgui.get_io(), item_rect(v, "selectable", label))
        for _ in range(3):
            v.sync()

        assert v.app.gizmo.selected_joint_id(node.body_index) == choices[2].joint_id
        assert v.app.gizmo.last_verdict.ok
        assert v.app.gizmo.visible

        labels = []
        original_selectable = imgui.selectable

        def record_selectable(item_label, *args, **kwargs):
            labels.append(item_label)
            return original_selectable(item_label, *args, **kwargs)

        assert v.session.submit(cmd.Play())
        imgui.selectable = record_selectable
        try:
            v.sync()
        finally:
            imgui.selectable = original_selectable
        assert not any("##viewport-joint-" in label for label in labels)
    finally:
        if not v.session.paused:
            v.session.submit(cmd.Pause())
        v.release()


def test_viewport_playback_widget_controls_simulation(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = viewer
    v.session.submit(cmd.Pause())
    v.session.submit(cmd.Reset())
    v.sync()

    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-toggle"))
    assert not v.session.paused

    for _ in range(3):
        v.sync()
    assert v.session.frame.step > 0
    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-stop"))
    assert v.session.paused
    v.sync()
    assert v.session.frame.step == 0

    before = v.session.frame.step
    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-step"))
    assert v.session.paused
    v.sync()
    assert v.session.frame.step == before + 1


def test_status_simulation_metric_switches_and_copies_exact_value(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = viewer
    v.session.submit(cmd.Pause())
    v.app._status_metric_mode = "time"
    v.sync()

    metric = item_rect(v, "invisible_button", "##status_simulation_metric")
    click(v, imgui.get_io(), metric)
    assert v.app._status_metric_mode == "steps"

    copied = []
    original = imgui.set_clipboard_text
    imgui.set_clipboard_text = copied.append
    try:
        metric = item_rect(v, "invisible_button", "##status_simulation_metric")
        click(v, imgui.get_io(), metric, button=1)
    finally:
        imgui.set_clipboard_text = original
    assert copied == [str(v.session.frame.step)]


def test_gizmo_disappears_without_an_editable_body(free_body_viewer):

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeType

    v = free_body_viewer
    node = next(n for n in v.session.nodes if n.type is NodeType.WORLD)
    v.session.submit(cmd.SelectNode(node.node_id))
    for _ in range(4):
        v.sync()

    assert not v.app.gizmo.last_verdict.ok
    assert v.app.gizmo.last_verdict.reason
    assert not v.app.gizmo.hovered


def test_dragging_the_gizmo_moves_the_object_not_the_camera(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import SIZE_PT, project, world_scale

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("translate")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(4):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    world = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    scale = world_scale(cam, world, rect[3], SIZE_PT * v.window.style_scale)
    axis = np.array([0.0, 0.0, 1.0])
    cursor = project(cam, (world + axis * scale * 0.55,), rect)[0, :2]

    io.add_mouse_pos_event(*cursor)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered

    before_z = float(v.session.frame.body_xpos[node.body_index][2])
    before_yaw = v.app.camera.yaw
    io.add_mouse_button_event(0, True)
    v.sync()
    for i in range(1, 12):
        io.add_mouse_pos_event(*(cursor + np.array([0.0, -i * 4.0])))
        v.sync()
    io.add_mouse_button_event(0, False)
    for _ in range(2):
        v.sync()

    after_z = float(v.session.frame.body_xpos[node.body_index][2])
    assert after_z - before_z > 0.1
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6


def test_double_clicking_a_scalar_gizmo_opens_and_applies_precise_input(
    free_body_viewer, monkeypatch
):
    from dataclasses import replace

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import SIZE_PT, GizmoHandle, project, world_scale

    v = free_body_viewer
    v.sync()
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("translate")
    v.app.gizmo.set_space("body")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(4):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    before = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64).copy()
    rotation = np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3)
    scale = world_scale(cam, before, rect[3], SIZE_PT * v.window.style_scale)
    cursor = project(cam, (before + rotation[:, 2] * scale * 0.55,), rect)[0, :2]
    before_yaw = v.app.camera.yaw

    io.add_mouse_pos_event(*cursor)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered_handle is GizmoHandle.Z
    for _ in range(2):
        io.add_mouse_button_event(0, True)
        v.sync()
        io.add_mouse_button_event(0, False)
        v.sync()

    edit = v.app._precise_gizmo_edit
    assert edit is not None
    assert (edit.action, edit.label, edit.unit) == ("Move", "Z", "m")
    assert v.session.frame.body_xpos[node.body_index] == pytest.approx(before, abs=1e-7)
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6

    input_bounds = []
    input_flags = []
    popup_bounds = []
    title_bounds = []
    original_input_double = imgui.input_double
    original_text = imgui.text

    def record_input(*args, **kwargs):
        result = original_input_double(*args, **kwargs)
        input_flags.append(args[5] if len(args) > 5 else kwargs.get("flags", 0))
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        input_bounds.append((lo.x, hi.x))
        window_pos = imgui.get_window_pos()
        window_size = imgui.get_window_size()
        popup_bounds.append((window_pos.x, window_size.x))
        return result

    def record_text(value, *args, **kwargs):
        result = original_text(value, *args, **kwargs)
        if str(value).startswith("Move "):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            title_bounds.append((lo.x, hi.x))
        return result

    monkeypatch.setattr(imgui, "input_double", record_input)
    monkeypatch.setattr(imgui, "text", record_text)
    v.app._precise_gizmo_edit = replace(
        edit,
        label="01_revolute_y_with_a_long_joint_name",
        joint_id=123,
    )
    v.sync()
    popup_left, popup_width = popup_bounds[-1]
    content_right = popup_left + popup_width - imgui.get_style().window_padding.x
    assert popup_width == pytest.approx(204.0 * v.window.style_scale, abs=2.0)
    assert input_bounds[-1][1] <= content_right + 1.0
    assert input_flags[-1] & imgui.InputTextFlags_.auto_select_all.value
    assert title_bounds[-1][0] >= popup_left + imgui.get_style().window_padding.x - 1.0
    assert title_bounds[-1][1] <= content_right + 1.0
    v.app._precise_gizmo_edit = edit
    monkeypatch.setattr(imgui, "input_double", original_input_double)
    monkeypatch.setattr(imgui, "text", original_text)
    for _ in range(2):
        v.sync()

    input_center = item_rect(v, "input_double", "##precise_gizmo_value")
    click(v, io, input_center)
    assert v.app._precise_gizmo_edit is not None
    active = []
    original_input_double = imgui.input_double

    def record_active(*args, **kwargs):
        result = original_input_double(*args, **kwargs)
        if args[0] == "##precise_gizmo_value":
            active.append(imgui.is_item_active())
        return result

    monkeypatch.setattr(imgui, "input_double", record_active)
    v.sync()
    monkeypatch.setattr(imgui, "input_double", original_input_double)
    assert active[-1]

    # Clicking elsewhere dismisses Type Value, but that physical click is
    # consumed by the popup.  It must not fall through to scene picking and
    # clear the joint/body selection that the user intends to keep editing.
    selected_before_dismiss = v.session.selected
    pick_calls = []
    original_pick = v.app._pick_at

    def record_pick(point):
        pick_calls.append(point)
        return original_pick(point)

    monkeypatch.setattr(v.app, "_pick_at", record_pick)
    x, y, width, height = v.app._viewport_rect
    click(v, io, (x + width - 24.0, y + height - 48.0))
    monkeypatch.setattr(v.app, "_pick_at", original_pick)
    assert v.app._precise_gizmo_edit is None
    assert v.session.selected == selected_before_dismiss
    assert not pick_calls

    # Escape remains a distinct cancellation route and likewise preserves the
    # stable target.  Reopen directly so both dismissal paths are covered.
    v.app._begin_precise_gizmo_input(edit)
    v.sync()
    io.add_key_event(imgui.Key.escape, True)
    v.sync()
    io.add_key_event(imgui.Key.escape, False)
    v.sync()
    assert v.app._precise_gizmo_edit is None
    assert v.session.selected == selected_before_dismiss
    io.add_mouse_pos_event(*cursor)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered_handle is GizmoHandle.Z

    # Reopening focuses and selects the value. Model one Enter-returning edit
    # after separately verifying that clicking the real input keeps it active.
    v.app._begin_precise_gizmo_input(edit)
    original_input_double = imgui.input_double

    def submit_value(*args, **kwargs):
        original_input_double(*args, **kwargs)
        return True, 0.125

    monkeypatch.setattr(imgui, "input_double", submit_value)
    v.sync()
    monkeypatch.setattr(imgui, "input_double", original_input_double)
    for _ in range(2):
        v.sync()

    expected = before + rotation[:, 2] * 0.125
    assert v.app._precise_gizmo_edit is None
    assert v.session.frame.body_xpos[node.body_index] == pytest.approx(expected, abs=1e-5)
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6

    assert v.session.submit(cmd.SetPose(node.node_id, before, rotation))
    for _ in range(2):
        v.sync()


def test_precise_input_error_has_copy_button(free_body_viewer, monkeypatch):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import GizmoHandle

    v = free_body_viewer
    node = next(n for n in v.session.nodes if n.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("translate")
    v.app.gizmo._hovered = GizmoHandle.Z
    edit = v.app.gizmo.precise_input(v.session)
    assert edit is not None
    v.app._precise_gizmo_edit = edit
    v.app._precise_gizmo_error = "Enter a finite numeric value"
    v.app._open_precise_gizmo_popup = True

    copied = []
    original_small_button = imgui.small_button

    def press_copy_error(label, *args, **kwargs):
        shown = original_small_button(label, *args, **kwargs)
        return True if label == "Copy error##precise-gizmo" else shown

    monkeypatch.setattr(imgui, "set_clipboard_text", copied.append)
    monkeypatch.setattr(imgui, "small_button", press_copy_error)
    v.sync()

    assert copied == ["Enter a finite numeric value"]
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    v.sync()
    io.add_key_event(imgui.Key.escape, False)
    v.sync()
    assert v.app._precise_gizmo_edit is None


def test_precise_rotation_input_switches_to_radians_with_u(free_body_viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer import math3d
    from forge_viewer.gizmo import GizmoHandle

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("rotate")
    v.app.gizmo.set_space("body")
    for _ in range(2):
        v.sync()

    before_pos = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64).copy()
    before_mat = (
        np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3).copy()
    )
    v.app.gizmo._hovered = GizmoHandle.ROTATE_Z
    edit = v.app.gizmo.precise_input(v.session)
    assert edit is not None and edit.unit == "°"
    v.app._precise_gizmo_edit = edit
    v.app._precise_gizmo_value = 180.0
    v.app._precise_gizmo_angle_unit = "degrees"
    v.app._open_precise_gizmo_popup = True
    v.sync()

    io.add_key_event(imgui.Key.u, True)
    v.sync()
    io.add_key_event(imgui.Key.u, False)
    v.sync()
    assert v.app._precise_gizmo_angle_unit == "radians"
    assert v.app._precise_gizmo_value == pytest.approx(np.pi)

    io.add_key_event(imgui.Key.enter, True)
    v.sync()
    io.add_key_event(imgui.Key.enter, False)
    for _ in range(2):
        v.sync()

    axis = before_mat[:, 2]
    expected = math3d.rotvec_to_mat3(axis * np.pi) @ before_mat
    assert v.app._precise_gizmo_edit is None
    assert v.session.frame.body_xpos[node.body_index] == pytest.approx(before_pos, abs=1e-5)
    assert v.session.frame.body_xmat[node.body_index] == pytest.approx(expected, abs=1e-5)

    assert v.session.submit(cmd.SetPose(node.node_id, before_pos, before_mat))
    for _ in range(2):
        v.sync()


def test_precise_input_choice_memory_can_be_disabled(free_body_viewer):
    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import GizmoHandle

    v = free_body_viewer
    node = next(n for n in v.session.nodes if n.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("rotate")
    v.app.gizmo.set_space("world")
    v.app.gizmo._hovered = GizmoHandle.ROTATE_Z
    edit = v.app.gizmo.precise_input(v.session)
    assert edit is not None and edit.absolute_value is not None

    v.app.gizmo.remember_precise_input_choices = True
    v.app._precise_gizmo_preferred_absolute = True
    v.app._precise_gizmo_angle_unit = "radians"
    v.app._begin_precise_gizmo_input(edit)
    assert v.app._precise_gizmo_absolute
    assert v.app._precise_gizmo_angle_unit == "radians"
    assert v.app._precise_gizmo_value == pytest.approx(np.radians(edit.absolute_value))

    v.app._precise_gizmo_value += 0.25
    intended_absolute = v.app._precise_gizmo_value
    v.app._set_precise_gizmo_absolute(edit, False)
    assert v.app._precise_gizmo_value == pytest.approx(0.25)
    v.app._set_precise_gizmo_absolute(edit, True)
    assert v.app._precise_gizmo_value == pytest.approx(intended_absolute)

    v.app.gizmo.set_space("body")
    body_edit = v.app.gizmo.precise_input(v.session)
    assert body_edit is not None and body_edit.absolute_value is None
    v.app._begin_precise_gizmo_input(body_edit)
    assert not v.app._precise_gizmo_absolute
    assert v.app._precise_gizmo_preferred_absolute
    v.app.gizmo.set_space("world")
    v.app._begin_precise_gizmo_input(edit)
    assert v.app._precise_gizmo_absolute

    v.app.gizmo.remember_precise_input_choices = False
    v.app._begin_precise_gizmo_input(edit)
    assert not v.app._precise_gizmo_absolute
    assert v.app._precise_gizmo_angle_unit == "degrees"
    assert v.app._precise_gizmo_value == 0.0

    v.app.gizmo.remember_precise_input_choices = True
    v.app._precise_gizmo_edit = None
    v.app._precise_gizmo_absolute = False
    v.app._precise_gizmo_preferred_absolute = False
    v.app._open_precise_gizmo_popup = False


@pytest.mark.parametrize(("style", "arrow_count"), (("2d", 1), ("3d", 0)))
def test_gizmo_drag_feedback_matches_in_2d_and_3d(free_body_viewer, style, arrow_count):
    """2D/3D share one compound GPU drag link and the same value label."""
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import SIZE_PT, project, world_scale
    from forge_viewer.render.debugdraw import PrimitiveType

    class Recorder:
        """Counts the gizmo's Draw2D calls while forwarding to the real overlay."""

        def __init__(self, inner):
            self.inner = inner
            self.arrows = 0
            self.lines = 0
            self.circles = 0
            self.texts = []

        def concave_fill(self, points, color):
            self.arrows += 1
            return self.inner.concave_fill(points, color)

        def line(self, *args, **kwargs):
            self.lines += 1
            return self.inner.line(*args, **kwargs)

        def circle(self, *args, **kwargs):
            self.circles += 1
            return self.inner.circle(*args, **kwargs)

        def text(self, pos, color, text):
            self.texts.append(text)
            return self.inner.text(pos, color, text)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("translate")
    v.app.gizmo.set_style(style)
    v.app.gizmo.set_space("body")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(3):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    origin = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    rotation = np.asarray(v.session.frame.body_xmat[node.body_index]).reshape(3, 3)
    scale = world_scale(cam, origin, rect[3], SIZE_PT * v.window.style_scale)
    cursor = project(cam, (origin + rotation[:, 2] * scale * 0.55,), rect)[0, :2]
    io.add_mouse_pos_event(*cursor)
    v.sync()
    io.add_mouse_button_event(0, True)
    v.sync()
    axis_screen = project(cam, (origin, origin + rotation[:, 2] * scale), rect)[:, :2]
    direction = axis_screen[1] - axis_screen[0]
    direction /= np.linalg.norm(direction)
    io.add_mouse_pos_event(*(cursor + direction * 36.0))
    v.sync()

    orig_draw = v.app.gizmo.draw_overlay
    recorders = []

    def spy(cam, rect, overlay, *, style_scale=1.0):
        recorder = Recorder(overlay)
        recorders.append(recorder)
        return orig_draw(cam, rect, recorder, style_scale=style_scale)

    v.app.gizmo.draw_overlay = spy
    try:
        v.sync()
        drag_count = v.backend.debug.layer("ui.gizmo.drag").count_of(PrimitiveType.DRAG_LINK)
    finally:
        v.app.gizmo.draw_overlay = orig_draw
        io.add_mouse_button_event(0, False)
        v.sync()
        v.session.submit(cmd.Reset())
        v.sync()

    gizmo_draw = recorders[0]
    assert gizmo_draw.arrows == arrow_count
    assert drag_count == 1, f"{style} gizmo drag feedback is not one compound GPU shape"
    assert gizmo_draw.lines == gizmo_draw.circles == 0
    labels = gizmo_draw.texts
    assert any(text.startswith("Z ") and text.endswith(" m") for text in labels), labels


@pytest.mark.parametrize("style", ("2d", "3d"))
def test_rotation_feedback_matches_in_2d_and_3d(free_body_viewer, style, monkeypatch):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import (
        RING_RADIUS,
        SIZE_PT,
        GizmoHandle,
        GizmoMode,
        hit_test,
        project,
        world_scale,
    )

    v = free_body_viewer
    imgui.set_current_context(v.window._imgui_context)
    io = imgui.get_io()
    # Both parameter cases reuse one Viewer and run faster than a human double
    # click. Disable double-click recognition so the second synthetic drag does
    # not open precise input and leave a modal active for later interaction tests.
    monkeypatch.setattr(io, "mouse_double_click_time", 0.0)
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("rotate")
    v.app.gizmo.set_space("body")
    v.app.gizmo.set_style(style)
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(4):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    origin = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    rotation = np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3)
    style_scale = v.window.style_scale
    scale = world_scale(cam, origin, rect[3], SIZE_PT * style_scale)

    def ring_point(angle):
        return origin + scale * RING_RADIUS * (
            np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1]
        )

    candidates = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    start_angle = next(
        angle
        for angle in candidates
        if hit_test(
            cam,
            origin,
            rotation,
            rect,
            tuple(np.floor(project(cam, (ring_point(angle),), rect)[0, :2])),
            GizmoMode.ROTATE,
            style_scale,
        )[0]
        is GizmoHandle.ROTATE_Z
    )
    start = np.floor(project(cam, (ring_point(start_angle),), rect)[0, :2])
    io.add_mouse_pos_event(*start)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered_handle is GizmoHandle.ROTATE_Z

    before_pos = origin.copy()
    before_mat = rotation.copy()
    before_yaw = v.app.camera.yaw
    io.add_mouse_button_event(0, True)
    v.sync()
    angle = np.radians(5.0)
    for i in range(1, 13):
        cursor = project(cam, (ring_point(start_angle + angle * i / 12),), rect)[0, :2]
        io.add_mouse_pos_event(*cursor)
        v.sync()

    class Recorder:
        """Counts the gizmo's Draw2D calls while forwarding to the real overlay."""

        def __init__(self, inner):
            self.inner = inner
            self.sectors = 0
            self.arc_strokes = 0
            self.open_arcs = 0
            self.closed_arcs = 0
            self.radials = 0
            self.center_dots = 0
            self.texts = []

        def convex_fill(self, points, color):
            self.sectors += 1
            return self.inner.convex_fill(points, color)

        def triangle_fan_fill(self, points, color):
            self.sectors += 1
            return self.inner.triangle_fan_fill(points, color)

        def fringed_concave_fill(self, points, color):
            self.arc_strokes += 1
            return self.inner.fringed_concave_fill(points, color)

        def polyline(self, points, color, width, *, closed=False):
            if closed:
                self.closed_arcs += 1
            else:
                self.open_arcs += 1
            return self.inner.polyline(points, color, width, closed=closed)

        def line(self, *args, **kwargs):
            self.radials += 1
            return self.inner.line(*args, **kwargs)

        def circle_filled(self, *args, **kwargs):
            self.center_dots += 1
            return self.inner.circle_filled(*args, **kwargs)

        def text(self, pos, color, text):
            self.texts.append(text)
            return self.inner.text(pos, color, text)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    orig_draw = v.app.gizmo.draw_overlay
    recorders = []

    def spy(cam, rect, overlay, *, style_scale=1.0):
        recorder = Recorder(overlay)
        recorders.append(recorder)
        return orig_draw(cam, rect, recorder, style_scale=style_scale)

    v.app.gizmo.draw_overlay = spy
    try:
        v.sync()
        v.app.gizmo._rotation_raw_angle += 2.0 * np.pi
        v.sync()
        after_pos = np.asarray(v.session.frame.body_xpos[node.body_index]).copy()
        after_mat = np.asarray(v.session.frame.body_xmat[node.body_index]).reshape(3, 3).copy()
    finally:
        v.app.gizmo.draw_overlay = orig_draw
        io.add_mouse_button_event(0, False)
        v.sync()
        v.session.submit(cmd.Reset())
        v.sync()

    gizmo_draw = recorders[0]
    assert gizmo_draw.sectors == 1
    assert gizmo_draw.arc_strokes == 1
    assert gizmo_draw.open_arcs == 0
    assert all(draw.closed_arcs == gizmo_draw.closed_arcs for draw in recorders[1:])
    assert gizmo_draw.radials == 0
    assert gizmo_draw.center_dots == 0
    assert any(text.startswith("Z ") and text.endswith("°") for text in gizmo_draw.texts)
    assert after_pos == pytest.approx(before_pos, abs=1e-5)
    assert np.linalg.norm(after_mat - before_mat) > 0.05
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6


@pytest.mark.parametrize("orthographic", (False, True), ids=("perspective", "orthographic"))
def test_pressed_screen_rotation_ring_keeps_its_idle_pixel_geometry(
    free_body_viewer,
    orthographic,
):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import SCREEN_RING_RADIUS, SIZE_PT, GizmoHandle, project

    class Recorder:
        def __init__(self, inner):
            self.inner = inner
            self.closed = []

        def polyline(self, points, color, width, *, closed=False):
            if closed:
                self.closed.append(np.asarray(points).copy())
            return self.inner.polyline(points, color, width, closed=closed)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("rotate")
    v.app.gizmo.set_style("3d")
    v.app.gizmo.set_space("body")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    v.app.camera.orthographic = orthographic
    for _ in range(4):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    origin = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    center = project(cam, (origin,), rect)[0, :2]
    radius = SCREEN_RING_RADIUS * SIZE_PT * v.window.style_scale
    cursor = center + np.array((radius, 0.0))
    io.add_mouse_pos_event(*cursor)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered_handle is GizmoHandle.ROTATE_SCREEN

    recorders = []
    original = v.app.gizmo.draw_overlay

    def spy(cam, rect, overlay, *, style_scale=1.0):
        recorder = Recorder(overlay)
        recorders.append(recorder)
        return original(cam, rect, recorder, style_scale=style_scale)

    try:
        io.add_mouse_button_event(0, True)
        v.sync()
        assert v.app.gizmo.active_handle is GizmoHandle.ROTATE_SCREEN
        v.app.gizmo.draw_overlay = spy
        v.sync()
    finally:
        v.app.gizmo.draw_overlay = original
        io.add_mouse_button_event(0, False)
        v.sync()
        v.app.camera.orthographic = False
        v.sync()

    reference = recorders[0].closed[0]
    radii = np.linalg.norm(reference[:, :2] - center, axis=1)
    assert radii == pytest.approx(np.full(len(radii), radius), abs=1e-4)


def test_gizmo_stays_drawn_while_the_camera_is_being_dragged(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    for _ in range(3):
        v.sync()

    x, y, w, h = v.app._viewport_rect

    sx, sy = x + w * 0.15, y + h * 0.85
    io.add_mouse_pos_event(sx, sy)
    v.sync()
    io.add_mouse_button_event(0, True)
    v.sync()
    seen_drawn_while_dragging = []
    for i in range(1, 10):
        io.add_mouse_pos_event(sx + i * 10.0, sy)
        v.sync()
        seen_drawn_while_dragging.append((v.app.gizmo.last_drawn, v.app.gizmo.interactive))
    io.add_mouse_button_event(0, False)
    v.sync()

    assert all(drawn for drawn, _ in seen_drawn_while_dragging)
    assert not any(inter for _, inter in seen_drawn_while_dragging)


def test_holding_axis_key_uses_the_exact_gizmo_axis_without_a_mouse_click(free_body_viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import GizmoHandle, project

    class Recorder:
        def __init__(self, inner):
            self.inner = inner
            self.lines = []

        def add_line(self, *args):
            self.lines.append(args)
            return self.inner.add_line(*args)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    turn_z_90 = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    v.session.submit(cmd.Select(node.object_id))
    v.session.submit(
        cmd.SetPose(
            node_id=node.node_id,
            position=np.zeros(3, np.float32),
            rotation=turn_z_90.astype(np.float32),
        )
    )
    v.app.gizmo.set_mode("translate")
    v.app.gizmo.set_space("body")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(3):
        v.sync()

    rect = v.app._viewport_rect
    cursor = np.array((rect[0] + rect[2] * 0.18, rect[1] + rect[3] * 0.78))
    io.add_mouse_pos_event(*cursor)
    io.add_key_event(imgui.Key.x, True)
    v.sync()
    assert v.app.gizmo.keyboard_using
    assert v.app.gizmo.active_handle is GizmoHandle.X

    real = imgui.get_window_draw_list
    recorders = []

    def spy():
        recorder = Recorder(real())
        recorders.append(recorder)
        return recorder

    imgui.get_window_draw_list = spy
    try:
        v.sync()
    finally:
        imgui.get_window_draw_list = real

    candidates = []
    for recorder in recorders:
        for candidate in recorder.lines:
            segment = np.array(((candidate[0].x, candidate[0].y), (candidate[1].x, candidate[1].y)))
            inside = (
                np.all(segment[:, 0] >= rect[0] - 1.0)
                and np.all(segment[:, 0] <= rect[0] + rect[2] + 1.0)
                and np.all(segment[:, 1] >= rect[1] - 1.0)
                and np.all(segment[:, 1] <= rect[1] + rect[3] + 1.0)
            )
            if inside and np.linalg.norm(segment[1] - segment[0]) > min(rect[2], rect[3]):
                candidates.append(segment)
    assert candidates
    line = max(candidates, key=lambda segment: np.linalg.norm(segment[1] - segment[0]))
    cam = v.app.camera.view()
    axis = project(cam, (np.zeros(3), turn_z_90[:, 0]), rect)[:, :2]
    line_dir = line[1] - line[0]
    axis_dir = axis[1] - axis[0]
    line_dir /= np.linalg.norm(line_dir)
    axis_dir /= np.linalg.norm(axis_dir)
    assert abs(float(np.dot(line_dir, axis_dir))) > 0.9999
    assert np.linalg.norm(line[1] - line[0]) > min(rect[2], rect[3])

    before = np.asarray(v.session.frame.body_xpos[node.body_index]).copy()
    io.add_mouse_pos_event(*(cursor + axis_dir * 36.0))
    v.sync()
    after = np.asarray(v.session.frame.body_xpos[node.body_index]).copy()
    assert after[1] - before[1] > 0.1
    assert abs(after[0] - before[0]) < 1e-6

    io.add_key_event(imgui.Key.x, False)
    v.sync()
    assert not v.app.gizmo.keyboard_using
    v.session.submit(cmd.Reset())
    v.sync()


def test_the_keyboard_shortcuts_are_not_swallowed(free_body_viewer):

    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(n for n in v.session.nodes if n.posable)
    v.session.submit(cmd.Select(node.object_id))
    for _ in range(3):
        v.sync()
    assert not io.want_text_input

    def press(key):
        io.add_key_event(key, True)
        v.sync()
        io.add_key_event(key, False)
        v.sync()

    v.app.gizmo.set_mode("translate")
    press(imgui.Key.r)
    assert v.app.gizmo.mode == "rotate"
    press(imgui.Key.g)
    assert v.app.gizmo.mode == "translate"
    v.app.gizmo.set_space("body")
    press(imgui.Key.t)
    assert v.app.gizmo.space == "world"
    press(imgui.Key.t)
    assert v.app.gizmo.space == "body"

    paused = v.session.paused
    press(imgui.Key.space)
    assert v.session.paused is not paused
    press(imgui.Key.space)
    assert v.session.paused is paused

    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    v.app.camera.distance = 99.0
    v.sync()
    press(imgui.Key.f)
    for _ in range(120):
        v.sync()
    assert v.app.camera.distance < 50.0


def test_blocking_modal_owns_all_viewport_input_and_hides_context_hint(
    free_body_viewer,
) -> None:
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd

    v = free_body_viewer
    io = imgui.get_io()
    node = next(item for item in v.session.nodes if item.posable)
    assert v.session.submit(cmd.Pause())
    assert v.session.submit(cmd.Select(node.object_id))
    for _ in range(3):
        v.sync()
    hint = imgui.internal.find_window_by_name("Hints###viewport_hints")
    assert hint is not None and hint.active
    camera_eye = np.asarray(v.app._camera_view().eye, np.float64).copy()

    try:
        v.app._pending_document_action = ("new_scene", None)
        for _ in range(2):
            v.sync()
        popup = imgui.internal.find_window_by_name("Unsaved changes")
        assert popup is not None and popup.active
        hint = imgui.internal.find_window_by_name("Hints###viewport_hints")
        assert hint is None or not hint.active

        x, y, width, height = v.app._viewport_rect
        io.add_mouse_pos_event(x + width * 0.5, y + height * 0.5)
        io.add_key_event(imgui.Key.left_ctrl, True)
        io.add_key_event(imgui.Key.space, True)
        io.add_mouse_button_event(0, True)
        io.add_mouse_pos_event(x + width * 0.6, y + height * 0.6)
        v.sync()

        assert v.session.paused
        assert not v.app.gizmo.using
        assert not v.session.perturb.active
        assert v.app.view_cube.hovered is None
        assert np.asarray(v.app._camera_view().eye, np.float64) == pytest.approx(camera_eye)
        hint = imgui.internal.find_window_by_name("Hints###viewport_hints")
        assert hint is None or not hint.active
    finally:
        io.add_mouse_button_event(0, False)
        io.add_key_event(imgui.Key.space, False)
        io.add_key_event(imgui.Key.left_ctrl, False)
        io.add_key_event(imgui.Key.escape, True)
        v.sync()
        io.add_key_event(imgui.Key.escape, False)
        v.sync()


def test_g_and_r_switch_modes_and_there_is_no_scale(free_body_viewer):

    v = free_body_viewer
    g = v.app.gizmo
    g.set_mode("rotate")
    assert g.mode == "rotate"
    g.set_mode("translate")
    assert g.mode == "translate"
    g.set_mode("scale")
    assert g.mode == "translate"


def test_closing_one_window_leaves_glfw_alive_for_the_other():

    import glfw as _glfw

    from forge_viewer.ui.window import Window, WindowConfig

    a = Window(WindowConfig(title="a", width=320, height=240, vsync=False, ini_path=""))
    b = Window(WindowConfig(title="b", width=320, height=240, vsync=False, ini_path=""))
    try:
        a.close()

        assert _glfw.get_window_size(b._window) == (320, 240)
        _glfw.poll_events()
    finally:
        b.close()

    c = Window(WindowConfig(title="c", width=320, height=240, vsync=False, ini_path=""))
    c.close()
