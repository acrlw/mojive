"""HiDPI window and overlay scaling regressions."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("glfw")

from mojive import commands as cmd  # noqa: E402
from mojive.app.composition import build, build_scene  # noqa: E402
from mojive.scene import Scene  # noqa: E402
from mojive.scene.assets import resolve  # noqa: E402
from mojive.ui import viewcube  # noqa: E402
from mojive.ui.viewport_widgets import ToolHint  # noqa: E402


@pytest.mark.parametrize("initial_scale", [1.0, 2.0])
def test_relative_fonts_follow_live_scale_changes_once(initial_scale, tmp_path, monkeypatch):
    from imgui_bundle import imgui

    from mojive.config import LayoutConfig, ViewerConfig
    from mojive.tools.ui_runtime import _activate_panel, _settle
    from mojive.ui.panels.inspector.model import _Model

    monkeypatch.setenv("MOJIVE_UI_SCALE", str(initial_scale))
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_LANGUAGE", "en")
    scene = Scene()
    box = scene.box(name="Workpiece")
    measured = {}
    identity, wrapped, text = _Model._identity, imgui.text_wrapped, imgui.text

    def observe_identity(panel, ctx, node):
        measured["name"] = imgui.get_font_size()
        return identity(panel, ctx, node)

    def observe_wrapped(value):
        if value.startswith("node id "):
            measured["identity"] = imgui.get_font_size()
        return wrapped(value)

    def observe_text(value):
        if value == "Recording starts in":
            measured["countdown_label"] = imgui.get_font_size()
        elif value.endswith(" s") and value[:-2].isdigit():
            measured["countdown"] = imgui.get_font_size()
        return text(value)

    monkeypatch.setattr(_Model, "_identity", observe_identity)
    monkeypatch.setattr(imgui, "text_wrapped", observe_wrapped)
    monkeypatch.setattr(imgui, "text", observe_text)
    with build_scene(
        scene,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
        vsync=False,
        show_window=False,
        width=1600,
        height=1000,
    ) as viewer:
        assert viewer.session.submit(cmd.Select(box.object_id))
        _activate_panel(viewer, "Inspector")
        viewer.start_recording(tmp_path / "cancelled.mp4", countdown=60)
        try:
            for scale, main in ((1, 1), (1.6, 1), (0.75, 1), (2.25, 1), (1.5, 1.2), (1, 1)):
                measured.clear()
                viewer.window._scale_override = scale
                imgui.get_style().font_scale_main = main
                _settle(viewer, 3)
                assert measured["identity"] == pytest.approx(measured["name"] * 0.85, abs=1)
                # Both atlas sizes are rounded: 0.5 * 1.6 + 0.5 pixels of error.
                assert measured["countdown"] == pytest.approx(
                    measured["countdown_label"] * 1.6, abs=1.3
                )
        finally:
            viewer.app.stop_recording(report=False)


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
        viewer.set_gizmo_mode("translate")
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
            assert window is None or not window.active, name
            assert hit.rect[0] >= viewport_x - 1.0
            assert hit.rect[1] >= viewport_y - 1.0
            assert hit.rect[2] <= viewport_right + 1.0
            assert hit.rect[3] <= viewport_bottom + 1.0

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
    clip_rects = []
    render = imgui.render

    def capture_logical_clips():
        render()
        # OpenGL's ImGui adapter scales clip rectangles in place on submission;
        # the native backend leaves them in points. Inspect the shared pre-submission domain.
        clip_rects.clear()
        window = imgui.internal.find_window_by_name("Playback###viewport_playback")
        if window is not None and window.active:
            clip_rects.extend(
                (item.clip_rect.x, item.clip_rect.y, item.clip_rect.z, item.clip_rect.w)
                for item in window.draw_list.cmd_buffer
                if item.elem_count
            )

    monkeypatch.setattr(imgui, "render", capture_logical_clips)
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
        for hit in viewer.app.gizmo.joint_limit_hits:
            name = f"Joint {hit.label}###joint_limit_{hit.joint_id}_{hit.label[:3]}"
            window = imgui.internal.find_window_by_name(name)
            assert window is None or not window.active, name
        assert clip_rects
        for left, top, right, bottom in clip_rects:
            assert left >= viewport_x - 1.0
            assert top >= viewport_y - 1.0
            assert right <= viewport_right + 1.0
            assert bottom <= viewport_bottom + 1.0
    finally:
        io.add_mouse_button_event(0, False)
        viewer.release()
