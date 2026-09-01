"""HiDPI window and overlay scaling regressions."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")

from mojive import commands as cmd  # noqa: E402
from mojive.assets import resolve  # noqa: E402
from mojive.composition import build, build_scene  # noqa: E402
from mojive.scene import Scene  # noqa: E402
from mojive.ui import viewcube  # noqa: E402
from mojive.ui.viewport_widgets import ToolHint  # noqa: E402


def test_view_gizmo_and_font_share_the_explicit_ui_scale(monkeypatch):
    from imgui_bundle import imgui

    monkeypatch.setenv("MOJIVE_UI_SCALE", "2")
    viewer = build_scene(Scene(), vsync=False, width=960, height=640)
    try:
        viewer.sync()
        scale = viewer.window.style_scale

        assert scale == pytest.approx(2.0)
        assert viewer.window.font_report.size_pt == pytest.approx(
            viewer.window.config.font_size_pt * scale
        )
        assert imgui.get_style().font_scale_dpi == pytest.approx(1.0)
        radii = {ball.radius for ball in viewer.app.view_cube.balls}
        assert len(radii) == 1
        assert next(iter(radii)) == pytest.approx(viewcube.BALL_PT * scale)
    finally:
        viewer.release()


def test_hidpi_capsule_hosts_are_clipped_and_modal_width_tracks_layout_scale(monkeypatch):
    from imgui_bundle import imgui

    monkeypatch.setenv("MOJIVE_UI_SCALE", "2.25")
    viewer = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=1180,
        height=1400,
        show_window=False,
    )
    io = imgui.get_io()
    try:
        viewer.app.tool_hints.add(
            "hidpi.scene",
            ToolHint("mouse", "left", "Scene hint"),
            surface="scene",
        )
        node = next(item for item in viewer.session.nodes if item.name == "02_prismatic")
        assert viewer.session.submit(cmd.Select(node.object_id))
        for _ in range(10):
            viewer.sync()

        viewport_x, viewport_y, viewport_width, viewport_height = viewer.app._viewport_rect
        viewport_right = viewport_x + viewport_width
        viewport_bottom = viewport_y + viewport_height
        for name in (
            "Playback###viewport_playback",
            "Tools###viewport_tools",
            "Hints###viewport_hints",
        ):
            window = imgui.internal.find_window_by_name(name)
            assert window is not None and window.active, name
            assert window.pos.x >= viewport_x - 1.0
            assert window.pos.y >= viewport_y - 1.0
            assert window.pos.x + window.size.x <= viewport_right + 1.0
            assert window.pos.y + window.size.y <= viewport_bottom + 1.0
        assert viewer.app.gizmo.joint_limit_hits
        for hit in viewer.app.gizmo.joint_limit_hits:
            name = f"Joint {hit.label}###joint_limit_{hit.joint_id}_{hit.label[:3]}"
            window = imgui.internal.find_window_by_name(name)
            assert window is not None and window.active, name
            assert window.pos.x >= viewport_x - 1.0
            assert window.pos.y >= viewport_y - 1.0
            assert window.pos.x + window.size.x <= viewport_right + 1.0
            assert window.pos.y + window.size.y <= viewport_bottom + 1.0

        viewer.app._pending_document_action = ("new_scene", None)
        for _ in range(3):
            viewer.sync()
        popup = imgui.internal.find_window_by_name("Unsaved changes")
        assert popup is not None and popup.active
        viewport = imgui.get_main_viewport()
        expected_width = min(
            360.0 * viewer.window.style_scale,
            viewport.work_size.x - 32.0 * viewer.window.style_scale,
        )
        assert popup.size.x == pytest.approx(expected_width, abs=1.0)
        assert popup.size.y <= viewport.work_size.y - 32.0 * viewer.window.style_scale + 1.0
    finally:
        io.add_key_event(imgui.Key.escape, True)
        viewer.sync()
        io.add_key_event(imgui.Key.escape, False)
        viewer.release()


def test_hidpi_viewport_overlays_keep_a_hard_clip_after_splitter_collapse(monkeypatch):
    from imgui_bundle import imgui

    monkeypatch.setenv("MOJIVE_UI_SCALE", "2.25")
    viewer = build(
        resolve("joint_gizmo"),
        "mujoco",
        paused=True,
        vsync=False,
        width=1180,
        height=1000,
        show_window=False,
    )
    io = imgui.get_io()
    try:
        node = next(item for item in viewer.session.nodes if item.name == "02_prismatic")
        assert viewer.session.submit(cmd.Select(node.object_id))
        for _ in range(8):
            viewer.sync()

        viewport_window = imgui.internal.find_window_by_name("Viewport")
        assert viewport_window is not None and viewport_window.dock_node is not None
        dock = viewport_window.dock_node
        drag_y = dock.pos.y + dock.size.y * 0.5
        io.add_mouse_pos_event(dock.pos.x + dock.size.x, drag_y)
        io.add_mouse_button_event(0, True)
        viewer.sync()
        io.add_mouse_pos_event(dock.pos.x + 40.0, drag_y)
        viewer.sync()
        io.add_mouse_button_event(0, False)
        for _ in range(4):
            viewer.sync()

        viewport_x, viewport_y, viewport_width, viewport_height = viewer.app._viewport_rect
        viewport_right = viewport_x + viewport_width
        viewport_bottom = viewport_y + viewport_height
        assert viewport_width < 80.0
        clip_left, clip_top = viewer.window.points_to_pixels((viewport_x, viewport_y))
        clip_right, clip_bottom = viewer.window.points_to_pixels((viewport_right, viewport_bottom))

        clipped_windows = [imgui.internal.find_window_by_name("Playback###viewport_playback")]
        for hit in viewer.app.gizmo.joint_limit_hits:
            name = f"Joint {hit.label}###joint_limit_{hit.joint_id}_{hit.label[:3]}"
            window = imgui.internal.find_window_by_name(name)
            if window is not None and window.active:
                clipped_windows.append(window)
        assert len(clipped_windows) >= 2
        for window in clipped_windows:
            assert window is not None and window.active
            commands = [item for item in window.draw_list.cmd_buffer if item.elem_count]
            assert commands
            for item in commands:
                assert item.clip_rect.x >= clip_left - 1.0
                assert item.clip_rect.y >= clip_top - 1.0
                assert item.clip_rect.z <= clip_right + 1.0
                assert item.clip_rect.w <= clip_bottom + 1.0
    finally:
        io.add_mouse_button_event(0, False)
        viewer.release()
