from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.physics]

pytest.importorskip("glfw")
pytest.importorskip("mujoco")

from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.composition import build  # noqa: E402

W, H = 1280, 800


@pytest.fixture(autouse=True, scope="module")
def _pin_ui_scale():
    # These tests assume scale-1 layout geometry (CI displays); pin the UI
    # scale so HiDPI machines produce the same coordinates.
    old = os.environ.get("FORGE_VIEWER_UI_SCALE")
    os.environ["FORGE_VIEWER_UI_SCALE"] = "1"
    try:
        yield
    finally:
        if old is None:
            del os.environ["FORGE_VIEWER_UI_SCALE"]
        else:
            os.environ["FORGE_VIEWER_UI_SCALE"] = old


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


def test_viewport_gets_real_estate(viewer):

    _x, _y, w, h = viewer.app._viewport_rect
    pw, ph = viewer.window.size_points
    assert w > 200 and h > 200
    assert (w * h) / (pw * ph) > 0.20


def test_all_panels_docked_not_stacked(viewer):

    from forge_viewer.ui.window import Window

    laid_out = set(Window._LAYOUT_LEFT + Window._LAYOUT_RIGHT + Window._LAYOUT_BOTTOM)
    declared = {p.name for p in viewer.app.panels.panels}
    assert declared <= laid_out


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
    from forge_viewer.adapters.base import NodeKind
    from forge_viewer.render.backend import RenderFlag

    selected_before = viewer.session.selected
    environment = next(n for n in viewer.session.nodes if n.kind is NodeKind.ENVIRONMENT)
    viewer.session.submit(cmd.Select(environment.object_id))
    activate_panel(viewer, "Inspector")
    point = item_rect(viewer, "checkbox", "enabled##fog")
    before = viewer.backend.get_flag(RenderFlag.FOG)

    click(viewer, imgui.get_io(), point)
    assert viewer.backend.get_flag(RenderFlag.FOG) is not before

    click(viewer, imgui.get_io(), point)
    viewer.session.submit(cmd.Select(selected_before))


def test_material_inspector_exposes_instance_and_shared_controls(viewer):
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeKind

    selected_before = viewer.session.selected
    target = next(
        node
        for node in viewer.session.nodes
        if node.kind is NodeKind.LINK and node.name == "ball_00"
    )
    viewer.session.submit(cmd.Select(target.object_id))
    activate_panel(viewer, "Inspector")
    header = item_rect(viewer, "collapsing_header", "material")
    click(viewer, imgui.get_io(), header)

    item_rect(viewer, "color_edit4", "instance color")
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

    def _record(self, kind, pos):
        index = self._nearest(pos)
        if index >= 0:
            self.calls.append((kind, index))

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
        kinds = [k for k, i in calls[lo : hi + 1] if i == idx]
        if balls[idx].positive:
            assert kinds.count("lollipop") == 1
        if "label" in kinds:
            body = "lollipop" if balls[idx].positive else "disc"
            assert kinds.index(body) < kinds.index("label")


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

    yaw, pitch = vc.yaw_pitch_for(2, 1.0, viewer.app.camera.yaw)
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


def test_gizmo_disappears_without_a_free_body(free_body_viewer):

    import forge_viewer.commands as cmd
    from forge_viewer.adapters.base import NodeKind

    v = free_body_viewer

    fr = v.session.frame
    node = max(
        (
            n
            for n in v.session.nodes
            if not n.posable
            and n.object_id
            and n.body_index >= 0
            and n.kind in (NodeKind.ROBOT, NodeKind.LINK)
        ),
        key=lambda n: float(np.linalg.norm(fr.body_xpos[n.body_index])),
    )
    v.session.submit(cmd.Select(node.object_id))
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


@pytest.mark.parametrize(("style", "arrow_count"), (("2d", 1), ("3d", 0)))
def test_gizmo_drag_feedback_matches_in_2d_and_3d(free_body_viewer, style, arrow_count):
    """2D/3D share one compound GPU drag link and the same value label."""
    from imgui_bundle import imgui

    import forge_viewer.commands as cmd
    from forge_viewer.gizmo import SIZE_PT, project, world_scale
    from forge_viewer.render.debugdraw import Prim

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
        drag_count = v.backend.debug.layer("ui.gizmo.drag").count_of(Prim.DRAG_LINK)
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
def test_rotation_feedback_matches_in_2d_and_3d(free_body_viewer, style):

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
            self.open_arcs = 0
            self.closed_arcs = 0
            self.radials = 0
            self.center_dots = 0
            self.texts = []

        def convex_fill(self, points, color):
            self.sectors += 1
            return self.inner.convex_fill(points, color)

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
    assert gizmo_draw.open_arcs == 0
    assert all(draw.closed_arcs == gizmo_draw.closed_arcs for draw in recorders[1:])
    assert gizmo_draw.radials == 0
    assert gizmo_draw.center_dots == 3
    assert any(text.startswith("Z ") and text.endswith("°") for text in gizmo_draw.texts)
    assert after_pos == pytest.approx(before_pos, abs=1e-5)
    assert np.linalg.norm(after_mat - before_mat) > 0.05
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6


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

    constraint = recorders[0].lines[0]
    line = np.array(((constraint[0].x, constraint[0].y), (constraint[1].x, constraint[1].y)))
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
