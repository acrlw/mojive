from __future__ import annotations

import ast
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.physics]

pytest.importorskip("glfw")
pytest.importorskip("mujoco")

from mojive.assets import resolve  # noqa: E402
from mojive.composition import build, build_workspace  # noqa: E402

W, H = 1280, 800


@pytest.fixture(autouse=True, scope="module")
def _pin_ui_scale(tmp_path_factory):
    # These tests assume scale-1 layout geometry (CI displays); pin the UI
    # scale so HiDPI machines produce the same coordinates.
    old = os.environ.get("MOJIVE_UI_SCALE")
    old_settings = os.environ.get("MOJIVE_SETTINGS")
    os.environ["MOJIVE_UI_SCALE"] = "1"
    os.environ["MOJIVE_SETTINGS"] = str(
        tmp_path_factory.mktemp("ui-interaction-settings") / "settings.json"
    )
    try:
        yield
    finally:
        if old is None:
            del os.environ["MOJIVE_UI_SCALE"]
        else:
            os.environ["MOJIVE_UI_SCALE"] = old
        if old_settings is None:
            del os.environ["MOJIVE_SETTINGS"]
        else:
            os.environ["MOJIVE_SETTINGS"] = old_settings


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


def test_escape_clears_paused_selection_and_is_advertised_in_status(viewer):
    from imgui_bundle import imgui

    from mojive import commands as cmd

    target = next(node for node in viewer.session.nodes if node.object_id)
    viewer.session.submit(cmd.Select(target.object_id))
    x, y, width, height = viewer.app._viewport_rect
    imgui.get_io().add_mouse_pos_event(x + width * 0.5, y + height * 0.5)
    viewer.sync()

    hints = viewer.app._status_tool_hints(loading=False)
    assert any(hint.hint_id == "selection.clear" for hint in hints)

    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()

    assert viewer.session.selected == 0
    assert viewer.session.selected_node is None


@pytest.mark.parametrize("node_name", ("01_revolute_y", "02_prismatic"))
@pytest.mark.parametrize(
    ("button", "shift", "view_cube"),
    (
        (0, False, False),
        (1, False, False),
        (2, False, False),
        (0, True, False),
        (0, False, True),
        (None, False, False),
    ),
    ids=("orbit", "pan-right", "pan-middle", "pan-shift", "view-cube", "zoom"),
)
def test_selection_status_survives_navigation_and_escape_consumes_release(
    viewer, monkeypatch, node_name, button, shift, view_cube
):
    from imgui_bundle import imgui

    from mojive import commands as cmd
    from mojive.ui import app as app_module
    from mojive.ui.gestures import Claim

    v = build(resolve("joint_gizmo"), "mujoco", paused=True, vsync=False, width=W, height=H)
    io = imgui.get_io()
    rendered_hints = []
    draw_status = app_module.draw_status

    def record_status(*args, **kwargs):
        rendered_hints.append(kwargs["tool_hints"])
        return draw_status(*args, **kwargs)

    monkeypatch.setattr(app_module, "draw_status", record_status)
    try:
        for _ in range(8):
            v.sync()
        node = next(item for item in v.session.nodes if item.name == node_name)
        v.session.submit(cmd.SelectNode(node.node_id))
        v.sync()
        x, y, width, height = v.app._viewport_rect
        empty = (x + width * 0.6, y + height * 0.75)
        start = next(b.screen for b in v.app.view_cube.balls if b.positive) if view_cube else empty
        io.add_mouse_pos_event(*start)
        v.sync()
        hints = v.app._status_tool_hints(loading=False)
        assert hints[0].hint_id == "selection.clear"
        qpos = v.session.frame.qpos.copy()
        camera_before = v.app.camera.view().view_matrix().copy()
        distance_before = v.app.camera.distance
        io.add_key_event(imgui.Key.mod_shift, shift)
        if button is not None:
            io.add_mouse_button_event(button, True)
        for step in range(6):
            if button is None:
                io.add_mouse_wheel_event(0.0, 1.0)
            else:
                io.add_mouse_pos_event(start[0] - step * 5.0, start[1] + step * 3.0)
            v.sync()
            assert v.app.router.claim is (Claim.VIEW_CUBE if view_cube else Claim.CAMERA)
            assert v.session.selected_node is node
            assert v.app._status_tool_hints(loading=False) == hints
            assert rendered_hints[-1] == hints
        if button is not None:
            io.add_mouse_button_event(button, False)
        io.add_key_event(imgui.Key.mod_shift, False)
        v.sync()
        assert v.app._status_tool_hints(loading=False) == hints
        np.testing.assert_array_equal(v.session.frame.qpos, qpos)
        assert (
            not np.allclose(v.app.camera.view().view_matrix(), camera_before)
            or v.app.camera.distance != distance_before
        )

        # A second short press is still inside click slop. Esc must not let
        # its release re-pick an object, even with a navigation modifier held.
        pick_calls = []
        monkeypatch.setattr(
            v.app, "_pick_at", lambda point: pick_calls.append(point) or node.object_id
        )
        io.add_mouse_pos_event(*empty)
        v.sync()
        io.add_mouse_button_event(0, True)
        v.sync()
        assert v.app.router.claim is Claim.CAMERA
        io.add_key_event(imgui.Key.mod_ctrl, True)
        io.add_key_event(imgui.Key.escape, True)
        v.sync()
        assert v.session.selected_node is None
        assert v.app._consume_scene_pointer_until_release
        io.add_key_event(imgui.Key.escape, False)
        io.add_key_event(imgui.Key.mod_ctrl, False)
        v.sync()
        io.add_mouse_button_event(0, False)
        v.sync()
        v.sync()
        assert not pick_calls
        assert not v.app._consume_scene_pointer_until_release
        assert v.session.selected_node is None
        assert all(
            hint.hint_id != "selection.clear" for hint in v.app._status_tool_hints(loading=False)
        )
    finally:
        for mouse_button in range(3):
            io.add_mouse_button_event(mouse_button, False)
        for key in (imgui.Key.escape, imgui.Key.mod_ctrl, imgui.Key.mod_shift):
            io.add_key_event(key, False)
        v.release()
        viewer.sync()


def test_all_panels_docked_not_stacked(viewer):
    from mojive.ui.window import Window

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
    output.write("[mojive/ui] FILTER_KEEP", level="warning", timestamp="10:00:00")
    output.write("[mojive/window] FILTER_HIDE", level="info", timestamp="10:00:01")
    panel._filter_text = "mojive/ui"
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

    assert [entry.text for entry in panel._filtered_entries] == ["[mojive/ui] FILTER_KEEP"]


def test_output_row_hover_does_not_repeat_the_message_as_a_tooltip(viewer, monkeypatch):
    from imgui_bundle import imgui

    output = viewer.app.output
    output.clear()
    output.write("NO_DUPLICATE_TOOLTIP", level="info", timestamp="10:00:02")
    entry = output.entries()[0]
    activate_panel(viewer, "Output")

    try:
        point = item_rect(viewer, "invisible_button", f"##output-row-{entry.sequence}")
        imgui.get_io().add_mouse_pos_event(*point)
        tooltips = []
        monkeypatch.setattr(imgui, "set_tooltip", tooltips.append)
        viewer.sync()
        assert entry.text not in tooltips
    finally:
        output.clear()


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

    import mojive.commands as cmd

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


def test_hierarchy_filter_strip_accepts_the_ordinary_mouse_wheel(viewer):
    from imgui_bundle import imgui

    activate_panel(viewer, "Hierarchy")
    point = item_rect(viewer, "button", "all##hierarchy-type-all")
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()

    targets = []
    original = imgui.set_scroll_x

    def record(value):
        targets.append(float(value))
        original(value)

    imgui.set_scroll_x = record
    try:
        io.add_mouse_wheel_event(0.0, -1.0)
        viewer.sync()
    finally:
        imgui.set_scroll_x = original

    assert targets and targets[-1] > 0.0


def test_keyframe_timeline_owns_the_wheel_while_zooming(viewer):
    from imgui_bundle import imgui

    panel = viewer.app.panels.get("Keyframes")
    assert panel is not None
    panel.open = True
    viewer.sync()
    viewer.sync()
    activate_panel(viewer, "Keyframes")
    viewer.sync()
    viewer.sync()
    point = item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
    window = imgui.internal.find_window_by_name("Keyframes")
    assert window is not None
    io = imgui.get_io()
    io.add_mouse_pos_event(*point)
    viewer.sync()
    click(viewer, io, point)
    assert [hint.hint_id for hint in viewer.app._panel_status_hints] == [
        "keyframes.playhead",
        "keyframes.zoom",
        "keyframes.pan",
    ]
    before_span = panel._view_end - panel._view_start
    before_scroll = float(window.scroll.y)

    io.add_mouse_wheel_event(0.0, -1.0)
    viewer.sync()

    assert panel._view_end - panel._view_start > before_span
    assert float(window.scroll.y) == pytest.approx(before_scroll)


def test_control_click_owns_status_and_right_click_copies(viewer) -> None:
    from imgui_bundle import imgui

    v = build(
        resolve("actuator_visuals"),
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
        activate_panel(v, "Control")
        for _ in range(2):
            v.sync()
        name = "hinge_drive"
        point = item_rect(v, "text_disabled", name)
        io.add_mouse_pos_event(*point)
        v.sync()
        click(v, io, point)

        assert [hint.hint_id for hint in v.app._panel_status_hints] == ["panel.copy-name"]
        click(v, io, point, button=1)
        assert imgui.get_clipboard_text() == name

        first = item_bounds(v, "text_disabled", "hinge_drive")
        second = item_bounds(v, "text_disabled", "slide_drive")
        gap_point = (first[0] + 2.0, (first[3] + second[1]) * 0.5)
        assert first[3] <= gap_point[1] <= second[1]
        io.add_mouse_pos_event(*gap_point)
        v.sync()
        assert [hint.hint_id for hint in v.app._panel_status_hints] == ["panel.copy-name"]

        io.add_mouse_pos_event(*center(v))
        v.sync()
        assert v.app._status_panel == "Control"
        assert [hint.hint_id for hint in v.app._status_tool_hints(loading=False)] == [
            "panel.copy-name"
        ]
        click(v, io, center(v))
        assert v.app._status_panel == "Viewport"
        viewport_hints = v.app._status_tool_hints(loading=False)
        assert viewport_hints
        io.add_mouse_pos_event(*point)
        v.sync()
        io.add_mouse_wheel_event(0.0, -1.0)
        v.sync()
        assert v.app._status_panel == "Viewport"
        assert v.app._status_tool_hints(loading=False) == viewport_hints

        window = imgui.internal.find_window_by_name("Control")
        assert window is not None and window.dock_node is not None
        node = window.dock_node
        tab = next(item for item in node.tab_bar.tabs if item.id_ == window.tab_id)
        bar = node.tab_bar.bar_rect
        tab_point = (bar.min.x + tab.offset + tab.width * 0.5, (bar.min.y + bar.max.y) * 0.5)
        io.add_mouse_pos_event(*tab_point)
        io.add_mouse_button_event(0, True)
        v.sync()
        v.sync()
        assert [hint.hint_id for hint in v.app._status_tool_hints(loading=False)] == [
            "panel.copy-name"
        ]
        io.add_mouse_button_event(0, False)
        v.sync()
        activate_panel(v, "Camera")
        assert v.app._status_panel == "Camera"
        assert v.app._panel_status_hints == ()
        activate_panel(v, "Control")
        assert v.app._status_panel == "Control"
    finally:
        io.add_mouse_button_event(0, False)
        io.add_mouse_button_event(1, False)
        v.release()
        viewer.sync()


def test_truncated_joint_name_has_full_tooltip_and_copy_action(viewer) -> None:
    from imgui_bundle import imgui

    v = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    tooltips: list[str] = []
    original_tooltip = imgui.set_tooltip
    try:
        v.app.panels.open_panel("Joints")
        for _ in range(8):
            v.sync()
        activate_panel(v, "Joints")
        for _ in range(2):
            v.sync()
        joint = next(item for item in v.session.joints if item.name == "05_multi_revolute_z")
        item_label = f"{joint.name}##joint-select-{joint.joint_id}"
        point = item_rect(v, "selectable", item_label)
        io.add_mouse_pos_event(*point)

        def record_tooltip(text, *args, **kwargs):
            tooltips.append(str(text))
            return original_tooltip(text, *args, **kwargs)

        imgui.set_tooltip = record_tooltip
        v.sync()
        assert joint.name in tooltips
        click(v, io, point)
        assert [hint.hint_id for hint in v.app._panel_status_hints] == [
            "panel.focus-item",
            "panel.copy-name",
        ]
        assert [hint.hint_id for hint in v.app._status_tool_hints(loading=False)] == [
            "selection.clear",
            "panel.focus-item",
            "panel.copy-name",
        ]
        click(v, io, point, button=1)
        assert imgui.get_clipboard_text() == joint.name
    finally:
        imgui.set_tooltip = original_tooltip
        io.add_mouse_button_event(1, False)
        v.release()
        viewer.sync()


def test_free_joint_row_keeps_joint_selection_and_exposes_the_transform_gizmo(viewer) -> None:
    from imgui_bundle import imgui

    v = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        v.app.panels.open_panel("Joints")
        for _ in range(8):
            v.sync()
        activate_panel(v, "Joints")
        joint = next(item for item in v.session.joints if item.name == "04_free_6dof")
        point = item_rect(v, "selectable", f"{joint.name}##joint-select-{joint.joint_id}")

        click(v, io, point)
        for _ in range(2):
            v.sync()
        selected = v.session.selected_node
        assert selected is not None
        assert selected.name == joint.name
        assert not selected.posable
        assert selected.body_index == joint.body
        assert v.app.gizmo.last_verdict.ok
        assert v.app.gizmo.visible

        click(v, io, point)
        assert v.session.selected_node is selected
        assert v.app.camera.animating
        v.app.camera.advance(1.0, v.app.camera_out)

        hierarchy = v.app.panels.get("Hierarchy")
        assert hierarchy is not None
        hierarchy._filter = joint.name
        activate_panel(v, "Hierarchy")
        for _ in range(2):
            v.sync()
        point = item_rect(v, "invisible_button", f"##hierarchy-node-{selected.node_id}")

        click(v, io, point)
        assert v.session.selected_node is selected
        assert v.app.gizmo.last_verdict.ok
        assert v.app.gizmo.visible
        click(v, io, point)
        assert v.session.selected_node is selected
        assert v.app.camera.animating
    finally:
        io.add_mouse_button_event(0, False)
        v.release()
        viewer.sync()


def test_joint_and_control_copy_buttons_export_complete_state_vectors(viewer) -> None:
    from imgui_bundle import imgui

    v = build(
        resolve("actuator_visuals"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        v.app.panels.open_panel("Joints")
        v.app.panels.open_panel("Control")
        for _ in range(8):
            v.sync()
        state = v.session.adapter.capture_state()
        assert state is not None and len(state.act) > 0

        activate_panel(v, "Joints")
        qpos_bounds = item_bounds(v, "button", "Copy qpos")
        qvel_bounds = item_bounds(v, "button", "Copy qvel")
        assert qpos_bounds[1] == pytest.approx(qvel_bounds[1], abs=1.0)
        assert qpos_bounds[2] <= qvel_bounds[0]
        for label, expected in (("Copy qpos", state.qpos), ("Copy qvel", state.qvel)):
            click(v, io, item_rect(v, "button", label))
            copied = np.asarray(ast.literal_eval(imgui.get_clipboard_text()), np.float64)
            assert copied == pytest.approx(expected)

        activate_panel(v, "Control")
        ctrl_bounds = item_bounds(v, "button", "Copy ctrl")
        act_bounds = item_bounds(v, "button", "Copy act")
        assert ctrl_bounds[1] == pytest.approx(act_bounds[1], abs=1.0)
        assert ctrl_bounds[2] <= act_bounds[0]
        for label, expected in (("Copy ctrl", state.ctrl), ("Copy act", state.act)):
            click(v, io, item_rect(v, "button", label))
            copied = np.asarray(ast.literal_eval(imgui.get_clipboard_text()), np.float64)
            assert copied == pytest.approx(expected)
    finally:
        io.add_mouse_button_event(0, False)
        v.release()
        viewer.sync()


def test_double_clicking_joint_and_hierarchy_rows_focuses_the_camera(viewer) -> None:
    from imgui_bundle import imgui

    from mojive.adapters.base import NodeType
    from mojive.gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, project, world_scale

    v = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        v.app.panels.open_panel("Joints")
        for _ in range(8):
            v.sync()
        activate_panel(v, "Joints")
        joint = next(item for item in v.session.joints if item.name == "01_revolute_y")
        label = f"{joint.name}##joint-select-{joint.joint_id}"
        point = item_rect(v, "selectable", label)

        click(v, io, point)
        assert v.app._status_panel == "Joints"
        assert [hint.hint_id for hint in v.app._panel_status_hints] == [
            "panel.focus-item",
            "panel.copy-name",
        ]
        click(v, io, point)
        assert v.session.selected_node is not None
        assert v.session.selected_node.joint_index == joint.joint_id
        assert v.session.selection_highlight_object_id > 0
        assert v.backend._selected == v.session.selection_highlight_object_id
        outline = (
            v.backend._outline if v.backend.caps.name == "wgpu" else v.backend._passes["outline"]
        )
        assert outline.xray
        assert v.app.camera.animating
        v.app.camera.advance(1.0, v.app.camera_out)

        diagnostics = v.session.frame.diagnostics
        assert diagnostics is not None
        axis = diagnostics.joint_xaxis[joint.joint_id]
        center = diagnostics.joint_xpos[joint.joint_id]
        axis_angle = np.degrees(
            np.arccos(np.clip(abs(float(np.dot(v.app.camera.direction(), axis))), -1.0, 1.0))
        )
        assert 35.0 - 1e-5 <= axis_angle <= 55.0 + 1e-5
        assert v.app.camera.pitch >= 35.0 - 1e-5
        assert v.app.camera.direction()[2] > 0.0
        assert v.app.camera.pivot == pytest.approx(center, abs=1e-5)

        selected = v.session.selected_node
        target, reason = v.app.gizmo._joint_target(v.session, selected)
        assert target is not None, reason
        pose = v.app.gizmo._target_pose(v.session, selected, target)
        assert pose is not None
        position, rotation = pose
        cam = v.app._camera_view()
        rect = v.app._viewport_rect
        scale = world_scale(cam, position, rect[3], SIZE_PT * v.window.style_scale)

        def ring_point(angle):
            world = position + scale * RING_RADIUS * (
                np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1]
            )
            return project(cam, (world,), rect)[0, :2]

        ring_cursor = next(
            ring_point(angle)
            for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
            if v.app.gizmo.update_hover(
                v.session,
                cam,
                rect,
                tuple(ring_point(angle)),
                style_scale=v.window.style_scale,
            )
            is GizmoHandle.ROTATE_Z
        )
        io.add_mouse_pos_event(*ring_cursor)
        v.sync()
        assert all(
            hint.hint_id != "gizmo.type_value" for hint in v.app._status_tool_hints(loading=False)
        )
        assert v.app._gizmo_hint_hover.entered_at is not None
        v.app._gizmo_hint_hover.entered_at -= 1.0
        v.sync()
        assert [hint.hint_id for hint in v.app._status_tool_hints(loading=False)] == [
            "selection.clear",
            "gizmo.type_value",
        ]

        target = next(
            item
            for item in v.session.nodes
            if item.type is NodeType.LINK and item.name == "03_ball"
        )
        bounds = v.session.node_world_bounds(target.node_id)
        assert bounds is not None
        hierarchy = v.app.panels.get("Hierarchy")
        assert hierarchy is not None
        hierarchy._filter = target.name
        activate_panel(v, "Hierarchy")
        for _ in range(2):
            v.sync()
        point = item_rect(v, "invisible_button", f"##hierarchy-node-{target.node_id}")

        click(v, io, point)
        assert v.app._status_panel == "Hierarchy"
        assert [hint.hint_id for hint in v.app._panel_status_hints] == ["panel.focus-item"]
        click(v, io, point)
        assert v.session.selected_node is target
        assert not outline.xray
        assert v.app.camera.animating
        v.app.camera.advance(1.0, v.app.camera_out)
        assert v.app.camera.pivot == pytest.approx(bounds[0], abs=1e-5)

        camera_node = next(item for item in v.session.nodes if item.type is NodeType.CAMERA)
        camera_info = v.session.cameras[camera_node.camera_index]
        camera_view = v.session.camera_view(camera_info.camera_id)
        assert camera_view is not None
        hierarchy._filter = camera_node.name
        for _ in range(2):
            v.sync()
        point = item_rect(v, "invisible_button", f"##hierarchy-node-{camera_node.node_id}")

        click(v, io, point)
        click(v, io, point)
        assert v.session.selected_node is camera_node
        assert v.app.camera.animating
        v.app.camera.advance(1.0, v.app.camera_out)
        assert v.app.camera.pivot == pytest.approx(camera_view.eye, abs=1e-5)
    finally:
        io.add_mouse_button_event(0, False)
        v.release()
        viewer.sync()


def test_environment_inspector_controls_render_flags(viewer):
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.adapters.base import NodeType
    from mojive.render.backend import RenderFlag

    selected_before = viewer.session.selected
    environment = next(n for n in viewer.session.nodes if n.type is NodeType.ENVIRONMENT)
    viewer.session.submit(cmd.Select(environment.object_id))
    activate_panel(viewer, "Inspector")
    item_rect(viewer, "combo", "##environment-skybox-texture")
    item_rect(viewer, "combo", "##environment-haze-mode")
    _scroll_panel(viewer, "Inspector", -7.0)
    point = item_rect(viewer, "checkbox", "##environment-fog-enabled")
    before = viewer.backend.get_flag(RenderFlag.FOG)

    click(viewer, imgui.get_io(), point)
    assert viewer.backend.get_flag(RenderFlag.FOG) is not before

    click(viewer, imgui.get_io(), point)
    viewer.session.submit(cmd.Select(selected_before))


def test_material_inspector_exposes_instance_and_shared_controls(viewer):
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.adapters.base import NodeType

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
    item_rect(viewer, "color_edit4", "##geometry_instance_color")
    _scroll_panel(viewer, "Inspector", -6.0)
    contact = item_rect(viewer, "collapsing_header", "contact properties")
    click(viewer, imgui.get_io(), contact)
    _scroll_panel(viewer, "Inspector", -5.0)
    item_rect(viewer, "drag_float3", "##contact_friction")
    item_rect(viewer, "combo", "##contact_dimension")
    item_rect(viewer, "input_int", "##collision_type_mask")
    item_rect(viewer, "begin_combo", "##assigned_material")
    item_rect(viewer, "small_button", "New material##create material-0")
    item_rect(viewer, "small_button", "Duplicate material##material actions-0")
    item_rect(viewer, "small_button", "Import texture##texture import-0")
    item_rect(viewer, "color_edit4", "##material_base_color")
    item_rect(viewer, "begin_combo", "##material_preset")
    item_rect(viewer, "drag_float", "##material_specular")
    viewer.session.submit(cmd.Select(selected_before))


def test_light_inspector_groups_property_tables_and_joined_vectors(viewer):
    import mojive.commands as cmd
    from mojive.adapters.base import NodeType

    selected_before = viewer.session.selected
    target = next(node for node in viewer.session.nodes if node.type is NodeType.LIGHT)
    viewer.session.submit(cmd.Select(target.object_id))
    activate_panel(viewer, "Inspector")

    item_rect(viewer, "collapsing_header", "light properties")
    item_rect(viewer, "checkbox", "##light_enabled")
    item_rect(viewer, "combo", "##light_type")
    item_rect(viewer, "collapsing_header", "light transform")
    item_rect(viewer, "button", f"X##light_position_0_{target.node_id}")
    item_rect(viewer, "button", f"X##light_direction_0_{target.node_id}")
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
    from mojive import commands as cmd

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

    import mojive.commands as cmd
    from mojive.adapters.base import NodeType

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

    from mojive.ui import viewcube as vc

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

    def add_polyline(self, points, *rest):
        if len(points) == 2:
            self._record("line", points[-1])
        return self._inner.add_polyline(points, *rest)

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
    from mojive.ui import viewcube as vc
    from mojive.ui.camera import camera_basis

    yaw, pitch = vc.yaw_pitch_for(2, 1.0)
    viewer.app.camera.look_from(yaw, pitch, viewer.app.camera_out, animate=False)
    viewer.sync()
    right, up, _fwd = camera_basis(viewer.app.camera.view())
    assert right[0] > 0.99
    assert up[1] > 0.99


def test_clicking_during_a_transition_does_not_strand_the_camera(viewer):
    from imgui_bundle import imgui

    from mojive.ui import viewcube as vc

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

    import mojive.commands as cmd

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
    # The axis badge and value are one compound field: their square inner
    # corners meet exactly, while only the outer corners stay rounded.
    assert x_value[0] - x_axis[2] == pytest.approx(0.0, abs=0.1)

    from mojive.ui.panels.inspector import _mix_color
    from mojive.ui.theme import THEME

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

    import mojive.commands as cmd

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

    import mojive.commands as cmd
    from mojive import math3d

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

    import mojive.commands as cmd

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

    import mojive.commands as cmd

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


@pytest.mark.parametrize("kind", ("playback", "tools"))
def test_viewport_capsules_drag_from_their_border(free_body_viewer, kind):
    from dataclasses import replace

    from imgui_bundle import imgui

    import mojive.commands as cmd

    v = free_body_viewer
    node = next(item for item in v.session.nodes if item.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.set_viewport_overlays(
        replace(
            v.app.viewport_overlays,
            playback_position=None,
            tool_position=None,
            movable=True,
        ),
        persist=False,
    )
    for _ in range(3):
        v.sync()
    rect = v.app._playback_widget_rect if kind == "playback" else v.app._tool_widget_rect
    assert rect is not None
    before = ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5)

    drag(
        v,
        imgui.get_io(),
        rect[0] + 2.0,
        (rect[1] + rect[3]) * 0.5,
        90.0,
        45.0,
        steps=4,
    )
    moved = v.app._playback_widget_rect if kind == "playback" else v.app._tool_widget_rect
    assert moved is not None
    after = ((moved[0] + moved[2]) * 0.5, (moved[1] + moved[3]) * 0.5)
    assert after[0] - before[0] == pytest.approx(90.0, abs=4.0)
    assert after[1] - before[1] == pytest.approx(45.0, abs=4.0)
    v.app.set_viewport_overlays(
        replace(
            v.app.viewport_overlays,
            playback_position=None,
            tool_position=None,
        ),
        persist=True,
    )
    x, y, width, height = v.app._viewport_rect
    imgui.get_io().add_mouse_pos_event(x + width - 8.0, y + height - 8.0)
    v.sync()


@pytest.mark.parametrize("workspace", (False, True), ids=("viewer", "editor"))
def test_joint_gizmo_is_live_in_the_real_viewer_pipeline(workspace):
    """Viewer and editor must retain the joint frames requested by a joint gizmo."""

    import mojive.commands as cmd

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
        assert v.app.frame_needs().joint_frames
        assert not v.app.frame_needs().diagnostics
        assert v.app.gizmo.last_verdict.ok
        assert v.app.gizmo.visible

        target, reason = v.app.gizmo._joint_target(v.session, node)
        assert target is not None, reason
        assert v.app.request_joint_focus(target.joint.joint_id)
        assert v.app.frame_needs().joint_frames
        v.sync()
        assert v.app.camera.animating
        v.app.camera.advance(1.0, v.app.camera_out)
        diagnostics = v.session.frame.diagnostics
        assert diagnostics is not None
        axis = diagnostics.joint_xaxis[target.joint.joint_id]
        center = diagnostics.joint_xpos[target.joint.joint_id]
        axis_angle = np.degrees(
            np.arccos(np.clip(abs(float(np.dot(v.app.camera.direction(), axis))), -1.0, 1.0))
        )
        assert 35.0 - 1e-5 <= axis_angle <= 55.0 + 1e-5
        assert v.app.camera.pitch >= 35.0 - 1e-5
        assert v.app.camera.direction()[2] > 0.0
        assert v.app.camera.pivot == pytest.approx(center, abs=1e-5)
    finally:
        v.release()


def test_joint_limit_tick_click_sets_the_endpoint_in_the_real_viewer() -> None:
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import GizmoHandle

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
        hinge = v.app.gizmo._hinge_range_projection(
            v.app._camera_view(),
            v.app._viewport_rect,
            v.window.style_scale,
            v.app.gizmo._joint_range,
        )
        assert hinge is not None and hinge.upper_tick is not None
        assert (
            v.app.gizmo.update_hover(
                v.session,
                v.app._camera_view(),
                v.app._viewport_rect,
                tuple(hinge.upper_tick[0]),
                style_scale=v.window.style_scale,
            )
            is GizmoHandle.ROTATE_Z
        )
        assert v.app.gizmo.hovered_joint_limit is None
        assert (
            v.app.gizmo.update_hover(
                v.session,
                v.app._camera_view(),
                v.app._viewport_rect,
                tuple(hinge.upper_tick[1]),
                style_scale=v.window.style_scale,
            )
            is GizmoHandle.ROTATE_Z
        )
        assert v.app.gizmo.hovered_joint_limit is not None
        assert v.app.gizmo.hovered_joint_limit.label.startswith("MAX")
        x0, y0, x1, y1 = lower_hit.rect
        click(v, imgui.get_io(), ((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        for _ in range(2):
            v.sync()

        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(target.joint.range[0])
    finally:
        v.release()


@pytest.mark.parametrize(
    "node_name",
    ("06_precision_hinge", "07_precision_slide"),
)
def test_compact_joint_range_expands_to_a_drag_track_in_the_real_viewer(
    node_name: str,
) -> None:
    from imgui_bundle import imgui

    import mojive.commands as cmd

    v = build(
        resolve("joint_gizmo"),
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
        node = next(item for item in v.session.nodes if item.name == node_name)
        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(4):
            v.sync()

        target, reason = v.app.gizmo._joint_target(v.session, node)
        assert target is not None, reason
        range_state = v.app.gizmo._joint_range
        assert range_state is not None
        limit_hits = v.app.gizmo.joint_limit_hits
        if target.joint.type == "hinge":
            projection = v.app.gizmo._hinge_range_projection(
                v.app._camera_view(),
                v.app._viewport_rect,
                v.window.style_scale,
                range_state,
            )
            assert projection is not None
            assert projection.allowed is not None
            assert projection.current_tick is not None
            hover_points = (
                tuple((projection.allowed[0] + projection.allowed[-1]) * 0.5),
                tuple(projection.current_tick[1]),
                *(hit.tick_end for hit in limit_hits),
            )
        else:
            pose = v.app.gizmo._target_pose(v.session, node, target)
            assert pose is not None
            projection = v.app.gizmo._slide_range_projection(
                v.app._camera_view(),
                v.app._viewport_rect,
                v.window.style_scale,
                range_state,
                pose[0],
                v.app.gizmo._target_basis(pose[1], target),
            )
            assert projection is not None
            hover_points = (
                tuple((projection.lower + projection.upper) * 0.5),
                tuple(projection.current + projection.normal * 10.0 * v.window.style_scale),
                *(hit.tick_end for hit in limit_hits),
                *(
                    tuple(point)
                    for point in v.app.gizmo._slide_arrow_targets(
                        projection,
                        v.window.style_scale,
                    )
                ),
            )

        dwell_started = None
        for point in hover_points:
            io.add_mouse_pos_event(*point)
            v.sync()
            assert v.app.gizmo._joint_precision_dwell_key == (
                target.joint.joint_id,
                target.joint.qpos_adr,
            )
            if dwell_started is None:
                dwell_started = v.app.gizmo._joint_precision_dwell_started
            else:
                assert v.app.gizmo._joint_precision_dwell_started == dwell_started
        v.app.gizmo._joint_precision_dwell_started -= 1.0
        v.sync()

        rail = v.app.gizmo._joint_precision
        assert rail is not None
        assert v.app.gizmo.joint_precision_visible
        assert v.app.gizmo.joint_precision_hit_rect == rail.panel_rect
        cursor = rail.start + 0.75 * (rail.end - rail.start)
        click(v, io, tuple(cursor))
        for _ in range(2):
            v.sync()

        expected = target.joint.range[0] + 0.75 * (target.joint.range[1] - target.joint.range[0])
        assert v.session.frame.qpos is not None
        assert v.session.frame.qpos[target.joint.qpos_adr] == pytest.approx(expected, abs=5e-4)
    finally:
        io.add_mouse_button_event(0, False)
        v.release()


def test_limited_hinge_drag_keeps_feedback_and_claim_until_mouse_release() -> None:
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, project, world_scale

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

    import mojive.commands as cmd
    from mojive.gizmo import GizmoHandle

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
        start = v.app.gizmo._slide_arrow_targets(slide, v.window.style_scale)[0]
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
        # ImGui delivers integer framebuffer cursor coordinates on this path.
        # Allow one pixel of projection quantization around the requested
        # 0.02 m return instead of coupling this invariant to dock-bar height.
        pixel_error = v.app.gizmo._world_per_pt
        assert lower + 0.02 - pixel_error <= returned <= lower + 0.02 + pixel_error
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

    import mojive.commands as cmd

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

        activate_panel(v, "Joints")
        for _ in range(2):
            v.sync()
        row_labels = [f"{joint.name}##joint-select-{joint.joint_id}" for joint in choices]
        before_y = [item_rect(v, "selectable", label)[1] for label in row_labels]
        click(v, imgui.get_io(), item_rect(v, "selectable", row_labels[1]))
        after_y = [item_rect(v, "selectable", label)[1] for label in row_labels]
        assert after_y == pytest.approx(before_y)
        assert v.session.selected_node is not None
        assert v.session.selected_node.joint_index == choices[1].joint_id
        assert v.app.gizmo.visible
        assert v.app.gizmo.selected_joint_id(node.body_index) == choices[1].joint_id

        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(2):
            v.sync()
        slider_joint = choices[0]
        click(
            v,
            imgui.get_io(),
            item_rect(v, "slider_float", f"##joint-qpos-{slider_joint.qpos_adr}"),
        )
        assert v.session.selected_node is not None
        assert v.session.selected_node.joint_index == slider_joint.joint_id
        assert v.app.gizmo.visible
        assert v.app.gizmo.selected_joint_id(node.body_index) == slider_joint.joint_id
        selected_joint_node = v.session.selected_node

        activate_panel(v, "Inspector")
        item_rect(
            v,
            "button",
            f"X##joint_axis_0_{selected_joint_node.node_id}",
        )
        item_rect(v, "checkbox", "##joint_limited")
        item_rect(v, "drag_float2", "##joint_range")
        item_rect(v, "drag_float", "##joint_damping")
        item_rect(v, "combo", "##joint_advanced_group")
        item_rect(v, "drag_float2", "##joint_limit_solver_reference")

        assert v.session.submit(cmd.Select(node.object_id))
        for _ in range(2):
            v.sync()

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

    import mojive.commands as cmd

    v = viewer
    v.session.submit(cmd.Pause())
    v.session.submit(cmd.Reset())
    v.sync()

    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-toggle"))
    assert not v.session.paused

    for _ in range(3):
        v.sync()
    assert v.session.frame.step > 0
    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-reset"))
    assert v.session.paused
    v.sync()
    assert v.session.frame.step == 0

    before = v.session.frame.step
    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-step"))
    assert v.session.paused
    v.sync()
    assert v.session.frame.step == before + 1

    click(v, imgui.get_io(), item_rect(v, "invisible_button", "##viewport-playback-previous"))
    v.sync()
    assert v.session.frame.step == before


def test_status_simulation_metric_switches_and_copies_exact_value(viewer):
    from imgui_bundle import imgui

    import mojive.commands as cmd

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
    import mojive.commands as cmd
    from mojive.adapters.base import NodeType

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

    import mojive.commands as cmd
    from mojive.gizmo import SIZE_PT, project, world_scale

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


def test_dimension_gizmo_resizes_authored_geometry_in_the_real_viewer(
    free_body_viewer, monkeypatch
):
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.adapters.base import NodeType
    from mojive.gizmo import AXIS_END, SIZE_PT, GizmoHandle, project, world_scale

    v = free_body_viewer
    io = imgui.get_io()
    monkeypatch.setattr(io, "mouse_double_click_time", 0.0)
    node = next(
        item for item in v.session.nodes if item.type is NodeType.GEOM and item.name == "wall_left"
    )
    assert v.session.submit(cmd.SelectNode(node.node_id))
    v.set_gizmo_mode("dimensions")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(4):
        v.sync()

    target, reason = v.app.gizmo._dimension_target(v.session, v.session.selected_node)
    assert target is not None, reason
    original = target.size.copy()
    pose = v.app.gizmo._dimension_pose(v.session, v.session.selected_node, target)
    assert pose is not None
    origin, rotation = pose
    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    scale = world_scale(cam, origin, rect[3], SIZE_PT * v.window.style_scale)

    handle = GizmoHandle.NONE
    cursor = np.zeros(2)
    for mapping in target.dimensions.handles:
        if mapping.axis is None:
            continue
        candidate = project(
            cam,
            (origin + rotation[:, mapping.axis] * scale * AXIS_END,),
            rect,
        )[0, :2]
        io.add_mouse_pos_event(*candidate)
        v.sync()
        expected = (GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z)[mapping.axis]
        if v.app.gizmo.hovered_handle is expected:
            handle, cursor = expected, candidate
            break
    assert handle is not GizmoHandle.NONE

    io.add_mouse_button_event(0, True)
    v.sync()
    assert v.app.gizmo.active_handle is handle
    io.add_mouse_pos_event(*(cursor + v.app.gizmo._axis_screen * 32.0))
    for _ in range(3):
        v.sync()
    io.add_mouse_button_event(0, False)
    v.sync()
    assert not v.app.gizmo.using

    changed, reason = v.app.gizmo._dimension_target(v.session, v.session.selected_node)
    assert changed is not None, reason
    assert not np.allclose(changed.size, original)
    assert v.session.submit(cmd.SetGeometrySize(node.node_id, original))
    restored, reason = v.app.gizmo._dimension_target(v.session, v.session.selected_node)
    assert restored is not None, reason
    assert restored.size == pytest.approx(original)

    # This fixture is shared by later parametrized interaction tests. ImGui
    # retains the timestamp of a completed click even when double-click timing
    # is temporarily disabled, so clear that history before the next test.
    io.mouse_clicked_time[0] = -float("inf")
    io.mouse_clicked_count[0] = 0
    io.mouse_clicked_last_count[0] = 0


def test_double_clicking_a_scalar_gizmo_opens_and_applies_precise_input(
    free_body_viewer, monkeypatch
):
    from dataclasses import replace

    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import SIZE_PT, GizmoHandle, project, world_scale

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
    assert input_flags[-1] & imgui.InputTextFlags_.chars_scientific.value
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

    import mojive.commands as cmd
    from mojive.gizmo import GizmoHandle

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

    import mojive.commands as cmd
    from mojive import math3d
    from mojive.gizmo import GizmoHandle

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
    # Native keyboard input also arrives through the character queue. The
    # unit shortcut must not replace the selected numeric buffer with "u".
    io.add_input_character(ord("u"))
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
    import mojive.commands as cmd
    from mojive.gizmo import GizmoHandle

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
def test_gizmo_drag_feedback_matches_in_2d_and_3d(
    free_body_viewer, style, arrow_count, monkeypatch
):
    """2D/3D share one compound GPU drag link and the same value label."""
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import SIZE_PT, project, world_scale
    from mojive.render.debugdraw import PrimitiveType

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
    # Parameter cases share one Viewer and execute too quickly to represent two
    # intentional user clicks. Keep the second drag from becoming a double-click.
    monkeypatch.setattr(io, "mouse_double_click_time", 0.0)
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

    import mojive.commands as cmd
    from mojive.gizmo import (
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
    # Free-body axis rotations keep one finite viewport-clipped guide; the
    # sector itself still has no center-to-arc radial strokes.
    assert gizmo_draw.radials == 1
    assert all(draw.radials == 1 for draw in recorders[1:])
    assert gizmo_draw.center_dots == 0
    assert any(text.startswith("Z ") and text.endswith("°") for text in gizmo_draw.texts)
    assert after_pos == pytest.approx(before_pos, abs=1e-5)
    assert np.linalg.norm(after_mat - before_mat) > 0.05
    assert abs(v.app.camera.yaw - before_yaw) < 1e-6


def test_rotation_gizmo_accepts_a_straight_drag_without_orbiting(free_body_viewer):
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import RING_RADIUS, SIZE_PT, GizmoHandle, project, world_scale

    v = free_body_viewer
    io = imgui.get_io()
    node = next(item for item in v.session.nodes if item.posable)
    assert v.session.submit(cmd.Select(node.object_id))
    v.app.gizmo.set_mode("rotate")
    v.app.gizmo.set_space("body")
    v.app.gizmo.set_style("2d")
    v.app.camera.look_from(-135.0, 25.0, v.app.camera_out, animate=False)
    for _ in range(4):
        v.sync()

    cam = v.app.camera.view()
    rect = v.app._viewport_rect
    position = np.asarray(v.session.frame.body_xpos[node.body_index], np.float64)
    rotation = np.asarray(v.session.frame.body_xmat[node.body_index], np.float64).reshape(3, 3)
    scale = world_scale(cam, position, rect[3], SIZE_PT * v.window.style_scale)
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    ring = np.array(
        [
            position
            + scale
            * RING_RADIUS
            * (np.cos(angle) * rotation[:, 0] + np.sin(angle) * rotation[:, 1])
            for angle in angles
        ]
    )
    projected = project(cam, ring, rect)[:, :2]
    start = projected[int(np.argmax(projected[:, 0]))]
    io.add_mouse_pos_event(*start)
    for _ in range(2):
        v.sync()
    assert v.app.gizmo.hovered_handle is GizmoHandle.ROTATE_Z

    before_mat = rotation.copy()
    before_yaw = v.app.camera.yaw
    try:
        io.add_mouse_button_event(0, True)
        v.sync()
        io.add_mouse_pos_event(*(start + np.array((20.0, 0.0))))
        v.sync()
        assert v.app.gizmo._rotation_linear
        io.add_mouse_pos_event(*(start + np.array((60.0, 0.0))))
        v.sync()
        after_mat = np.asarray(v.session.frame.body_xmat[node.body_index]).reshape(3, 3)
        assert np.linalg.norm(after_mat - before_mat) > 0.05
        assert abs(v.app.camera.yaw - before_yaw) < 1e-6
    finally:
        io.add_mouse_button_event(0, False)
        v.sync()
        v.session.submit(cmd.Reset())
        v.sync()


@pytest.mark.parametrize("orthographic", (False, True), ids=("perspective", "orthographic"))
def test_pressed_screen_rotation_ring_keeps_its_idle_pixel_geometry(
    free_body_viewer,
    orthographic,
):
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.gizmo import SCREEN_RING_RADIUS, SIZE_PT, GizmoHandle, project

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

    import mojive.commands as cmd

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

    import mojive.commands as cmd
    from mojive.gizmo import GizmoHandle, project

    class Recorder:
        def __init__(self, inner):
            self.inner = inner
            self.lines = []

        def add_line(self, *args):
            self.lines.append(args)
            return self.inner.add_line(*args)

        def add_polyline(self, points, color, thickness, flags):
            if len(points) == 2 and not (flags & imgui.ImDrawFlags_.closed.value):
                self.lines.append((*points, color, thickness))
            return self.inner.add_polyline(points, color, thickness, flags)

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
        inner = real()
        window = imgui.internal.get_current_window_read()
        if window is None or not window.name.startswith("Viewport"):
            return inner
        recorder = Recorder(inner)
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

    import mojive.commands as cmd

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


def test_policy_input_recovers_after_hierarchy_search_focus(viewer):
    from imgui_bundle import imgui

    from mojive import InputClaim

    v = viewer
    # Fixtures own separate ImGui contexts; make this viewer current before
    # retrieving its IO object instead of depending on prior test order.
    v.sync()
    io = imgui.get_io()
    activate_panel(v, "Hierarchy")
    search = item_rect(v, "input_text_with_hint", "##filter")
    click(v, io, search)
    v.sync()
    assert io.want_text_input

    observations = []

    def policy_input(context):
        observations.append((context.viewport_focused, context.blocked, context.key_down("w")))
        return InputClaim(keys=frozenset({"w", "a", "s", "d"}))

    v.set_input_handler(policy_input)
    try:
        click(v, io, center(v))
        v.sync()
        assert not io.want_text_input

        io.add_key_event(imgui.Key.w, True)
        v.sync()
        assert observations[-1] == (True, False, True)
        io.add_key_event(imgui.Key.w, False)
        v.sync()
    finally:
        io.add_key_event(imgui.Key.w, False)
        v.set_input_handler(None)


def test_backspace_steps_history_backward_and_repeats_while_held(viewer) -> None:
    import time

    from imgui_bundle import imgui

    from mojive import commands as cmd
    from mojive.ui.app import STEP_BACK_REPEAT_DELAY_SECONDS, STEP_BACK_REPEAT_RATE_SECONDS

    v = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=W,
        height=H,
    )
    io = imgui.get_io()
    try:
        for _ in range(5):
            assert v.session.submit(cmd.Step(1))
            v.sync()
        assert v.session.frame.step == 5
        assert v.session.can_step_back
        hint = next(
            hint
            for hint in v.app._status_tool_hints(loading=False)
            if hint.hint_id == "playback.previous"
        )
        assert (hint.control, hint.label) == ("Backspace", "Rewind")

        io.add_key_event(imgui.Key.backspace, True)
        v.sync()
        assert v.session.frame.step == 4

        deadline = (
            time.monotonic() + STEP_BACK_REPEAT_DELAY_SECONDS + 2.0 * STEP_BACK_REPEAT_RATE_SECONDS
        )
        while time.monotonic() < deadline and v.session.frame.step == 4:
            time.sleep(0.02)
            v.sync()
        assert v.session.frame.step < 4
    finally:
        io.add_key_event(imgui.Key.backspace, False)
        v.sync()
        v.release()
        viewer.sync()


def test_blocking_modal_owns_all_viewport_input_and_hides_context_hint(
    free_body_viewer,
) -> None:
    from imgui_bundle import imgui

    import mojive.commands as cmd
    from mojive.ui.viewport_widgets import ToolHint

    v = free_body_viewer
    io = imgui.get_io()
    v.app.tool_hints.add(
        "modal.scene",
        ToolHint("mouse", "left", "Scene hint"),
        surface="scene",
    )
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
        v.app.tool_hints.remove("modal.scene", surface="scene")
        v.app.tool_hints.restore("modal.scene", surface="scene")
        io.add_mouse_button_event(0, False)
        io.add_key_event(imgui.Key.space, False)
        io.add_key_event(imgui.Key.left_ctrl, False)
        io.add_key_event(imgui.Key.escape, True)
        v.sync()
        io.add_key_event(imgui.Key.escape, False)
        v.sync()


def test_g_and_r_switch_transform_modes_and_dimensions_is_not_scale(free_body_viewer):
    v = free_body_viewer
    g = v.app.gizmo
    g.set_mode("rotate")
    assert g.mode == "rotate"
    g.set_mode("translate")
    assert g.mode == "translate"
    g.set_mode("scale")
    assert g.mode == "translate"
    g.set_mode("dimensions")
    assert g.mode == "dimensions"


def test_closing_one_window_leaves_glfw_alive_for_the_other():
    import glfw as _glfw

    from mojive.ui.window import Window, WindowConfig

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
