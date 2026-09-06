"""Native layout boundaries must own input before scene gesture routing."""

from __future__ import annotations

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import build
from mojive import commands as cmd
from mojive.assets import resolve
from mojive.ui.panels.keyframes import timeline_status_hints

pytestmark = pytest.mark.gpu


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    with build(
        resolve("joint_gizmo"), paused=True, vsync=False, show_window=False, width=1280, height=900
    ) as viewer:
        viewer.panels.open_panel("Keyframes")
        for _ in range(10):
            viewer.sync()
        yield viewer


def _camera_state(viewer):
    view = viewer.app._camera_view()
    return np.concatenate((view.eye, view.target, [view.fov_y]))


def _drag(viewer, start, delta, button=0):
    io = imgui.get_io()
    io.add_mouse_pos_event(*start)
    viewer.sync()
    io.add_mouse_button_event(button, True)
    viewer.sync()
    for fraction in np.linspace(0.1, 1.0, 10):
        io.add_mouse_pos_event(start[0] + delta[0] * fraction, start[1] + delta[1] * fraction)
        viewer.sync()
    io.add_mouse_button_event(button, False)
    viewer.sync()
    viewer.sync()


@pytest.mark.parametrize("inset", [1.0, 3.0, 4.0])
def test_dock_splitter_drag_does_not_reach_the_scene(viewer, inset):
    window = imgui.internal.find_window_by_name("Keyframes")
    start = (window.pos.x + window.size.x * 0.5, window.pos.y - inset)
    size_before = float(window.size.y)
    camera_before = _camera_state(viewer)
    selection_before = viewer.session.selected_node
    _drag(viewer, start, (25.0, -70.0))
    assert window.size.y > size_before + 40.0, "The drag must actually resize the dock split"
    np.testing.assert_allclose(_camera_state(viewer), camera_before, atol=1e-7)
    assert viewer.session.selected_node is selection_before
    assert not viewer.app.gizmo.using


def test_dock_splitter_is_blocked_for_embedding_input_handlers(viewer):
    contexts = []
    viewer.set_input_handler(lambda context: contexts.append(context))
    window = imgui.internal.find_window_by_name("Keyframes")
    start = (window.pos.x + window.size.x * 0.5, window.pos.y - 2.0)
    _drag(viewer, start, (0.0, -50.0))
    assert all(context.blocked and not context.viewport_hovered for context in contexts[2:-2])


def test_escape_dismisses_menu_without_clearing_selection(viewer):
    from mojive.tools.ui_runtime import _dismiss_popup, _open_main_menu

    node = next(node for node in viewer.session.nodes if node.name == "02_prismatic")
    viewer.session.submit(cmd.SelectNode(node.node_id))
    _open_main_menu(viewer, "Window")
    assert imgui.get_current_context().open_popup_stack
    assert viewer.session.selected_node is node
    _dismiss_popup(viewer)
    assert not imgui.get_current_context().open_popup_stack
    assert viewer.session.selected_node is node
    # A fresh Escape still performs the advertised scene action.
    _dismiss_popup(viewer)
    assert viewer.session.selected_node is None


def test_wrapped_checkbox_label_and_keyboard_activate_the_same_control(viewer, monkeypatch):
    from mojive.tools.ui_runtime import _activate_panel, _click
    from mojive.ui.panels import themed_checkbox

    state = {"value": False, "focus": False, "disabled": False}
    bounds = []

    def draw(_ctx):
        imgui.begin_child("checkbox-test", (150, 120))
        if state.pop("focus", False):
            imgui.set_keyboard_focus_here()
        imgui.begin_disabled(state["disabled"])
        _, state["value"] = themed_checkbox("Clear selection on empty click", state["value"])
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        state["id"] = imgui.get_item_id()
        bounds[:] = [lo.x, lo.y, hi.x, hi.y]
        imgui.end_disabled()
        imgui.end_child()

    monkeypatch.setattr(viewer.panels.get("Keyframes"), "draw", draw)
    _activate_panel(viewer, "Keyframes")
    viewer.sync()
    assert bounds[3] - bounds[1] > imgui.get_frame_height()
    point = (bounds[0] + imgui.get_frame_height() + 16, (bounds[1] + bounds[3]) * 0.5)
    _click(viewer, point)
    assert state["value"]
    state["focus"] = True
    viewer.sync()
    viewer.sync()
    assert imgui.get_current_context().nav_id == state["id"]
    imgui.get_io().add_key_event(imgui.Key.tab, True)
    viewer.sync()
    imgui.get_io().add_key_event(imgui.Key.tab, False)
    viewer.sync()
    paused = viewer.session.paused
    imgui.get_io().add_key_event(imgui.Key.space, True)
    viewer.sync()
    imgui.get_io().add_key_event(imgui.Key.space, False)
    viewer.sync()
    assert not state["value"]
    assert viewer.session.paused == paused
    state["disabled"] = True
    _click(viewer, point)
    assert not state["value"]


def test_timeline_right_drag_and_wheel_have_distinct_effects(viewer):
    from mojive.tools.ui_runtime import _activate_panel, _item_rect, _settle

    _activate_panel(viewer, "Keyframes")
    _settle(viewer, 3)
    panel = viewer.panels.get("Keyframes")
    lo, hi = _item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
    window = imgui.internal.find_window_by_name("Keyframes")
    point = (lo[0] + (hi[0] - lo[0]) * 0.65, min(lo[1] + 35, window.inner_clip_rect.max.y - 4))
    camera_before = _camera_state(viewer)
    initial = (panel._view_start, panel._view_end)
    _drag(viewer, point, (60.0, 0.0), button=1)
    assert panel._view_start < initial[0]
    assert panel._view_end - panel._view_start == pytest.approx(initial[1] - initial[0])
    np.testing.assert_allclose(_camera_state(viewer), camera_before, atol=1e-7)
    span = panel._view_end - panel._view_start
    imgui.get_io().add_mouse_pos_event(*point)
    viewer.sync()
    imgui.get_io().add_mouse_wheel_event(0.0, 1.0)
    viewer.sync()
    assert panel._view_end - panel._view_start < span
    hints = {hint.hint_id: hint for hint in timeline_status_hints(str)}
    assert hints["keyframes.pan"].control == "right"
    assert hints["keyframes.zoom"].control == "wheel"
    np.testing.assert_allclose(_camera_state(viewer), camera_before, atol=1e-7)


def test_hierarchy_skips_offscreen_drawing_and_preserves_scrolled_selection(viewer, monkeypatch):
    from mojive.tools.ui_runtime import _click, _item_rect

    panel = viewer.panels.get("Hierarchy")
    panel._open_state.update({node.node_id: True for node in viewer.session.nodes if node.children})
    original = panel._visibility_toggle
    rows = []

    def record(ctx, node, row_y, row_height):
        window = imgui.get_current_context().current_window
        rows.append((node, row_y, row_height, window))
        original(ctx, node, row_y, row_height)

    monkeypatch.setattr(panel, "_visibility_toggle", record)
    viewer.sync()
    rows.clear()
    viewer.sync()
    first_ids = {row[0].node_id for row in rows}
    assert 0 < len(rows) < panel._rows_drawn
    window = rows[0][3]
    scroll_height = float(window.scroll_max.y)
    assert scroll_height > 100
    open_state = dict(panel._open_state)
    io = imgui.get_io()
    io.add_mouse_pos_event(window.pos.x + window.size.x * 0.5, window.pos.y + window.size.y * 0.6)
    io.add_mouse_wheel_event(0, -8)
    for _ in range(4):
        rows.clear()
        viewer.sync()
    assert float(window.scroll.y) > 100
    assert {row[0].node_id for row in rows} != first_ids
    assert window.scroll_max.y == pytest.approx(scroll_height)
    assert panel._open_state == open_state
    node, _y, _height, _ = next(
        row
        for row in rows
        if not row[0].children
        and row[1] > window.inner_clip_rect.min.y
        and row[1] + row[2] < window.inner_clip_rect.max.y
    )
    lo, hi = _item_rect(viewer, "invisible_button", f"##hierarchy-node-{node.node_id}")
    _click(viewer, ((lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5))
    assert viewer.session.selected_node is node
    assert panel._batch_selected == {node.node_id}
