"""Focused scene browsing and contextual controls through real native input."""

from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.capture import CaptureSurface
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.themes import select_theme
from mojive.tools.native_ui_review import _NativeInput
from mojive.tools.native_workspace import capture
from mojive.ui.workspace.inspector_components import inspector_tabs

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


def test_timeline_keyframe_menu_and_keyboard(backend_name, monkeypatch, tmp_path):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    output = Path("output/focused-completion") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1000,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False),
                debug_server=False,
                threaded_physics=False,
                live_model_updates=True,
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        app.set_language("zh_CN")
        assert session.submit(cmd.Pause()).ok
        model_id = session.scene_models[0].model_id
        assert session.submit(cmd.AddModelKeyframe(model_id, "pose")).ok
        inputs.sync(10)
        workspace.activate("Timeline")
        inputs.sync()
        editor = workspace.timeline_panel.editor
        track = workspace.timeline_panel.last_layout
        inputs.click_at((track.track_left + 80, track.lane_y(0)))
        modifier = (
            imgui.Key.mod_super if imgui.get_io().config_mac_osx_behaviors else imgui.Key.mod_ctrl
        )

        def shortcut(key):
            inputs.event("add_key_event", modifier, True)
            inputs.event("add_key_event", key, True)
            inputs.event("add_key_event", key, False)
            inputs.event("add_key_event", modifier, False)
            for _ in range(120):
                inputs.sync(1)
                if not app._model_load_queue and app._model_load_future is None:
                    break
            inputs.sync()

        shortcut(imgui.Key.a)
        assert len(editor.selected_keyframes) == 1
        inputs.event("add_mouse_button_event", 1, True)
        inputs.event("add_mouse_button_event", 1, False)
        inputs.sync()
        assert "timeline-Copy keyframes" in workspace.ui.hits
        capture(viewer, output / "keyframe-context.png")
        inputs.click("timeline-Copy keyframes")
        assert len(editor._clipboard) == 1
        shortcut(imgui.Key.v)
        assert len(session.keyframes) == 2
        shortcut(imgui.Key.d)
        assert len(session.keyframes) == 3, (
            editor.selected_keyframes,
            editor.error,
            editor.edit_lane,
            imgui.get_current_context().nav_window.name,
            len(editor._clipboard),
            session.document_id,
        )
        assert session.submit(cmd.Undo()).ok
        assert len(session.keyframes) == 2
        assert session.submit(cmd.Undo()).ok
        assert len(session.keyframes) == 1


def test_camera_optics_and_output_controls(backend_name, monkeypatch, tmp_path):
    """Native numeric entry edits the camera and controls actual capture dimensions."""
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    output = Path("output/focused-completion") / backend_name
    output.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1600,
            height=1100,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        ui = workspace.ui
        app.set_language("zh_CN")
        inputs.sync(10)
        workspace.activate("Cameras")
        inputs.sync()
        camera_id = session.cameras[0].camera_id
        inputs.click(f"camera-select-{camera_id}")

        def enter(key, text):
            modifier = (
                imgui.Key.mod_super
                if imgui.get_io().config_mac_osx_behaviors
                else imgui.Key.mod_ctrl
            )
            inputs.sync(30)
            inputs.click(key)
            inputs.click(key)
            assert imgui.get_io().want_text_input
            inputs.event("add_key_event", modifier, True)
            inputs.event("add_key_event", imgui.Key.a, True)
            inputs.event("add_key_event", imgui.Key.a, False)
            inputs.event("add_key_event", modifier, False)
            inputs.event("add_input_characters_utf8", text)
            inputs.event("add_key_event", imgui.Key.enter, True)
            inputs.event("add_key_event", imgui.Key.enter, False)
            inputs.sync()

        enter("##camera-Focal length", "50")
        np.testing.assert_allclose(session.camera_view(camera_id).focal_length, (0.05, 0.05))
        enter("##camera-Sensor width", "32")
        assert session.camera_view(camera_id).sensor_size[0] == pytest.approx(0.032)
        inputs.click("##camera-output-preset")
        inputs.click("##camera-output-preset/choice/1")
        assert session.camera_view(camera_id).output_size == (1280, 720)
        enter("##camera-output-Width", "640")
        enter("##camera-output-Height", "480")
        assert session.camera_view(camera_id).output_size == (640, 480)
        inputs.click("inspector-camera-view")
        for _ in range(30):
            app._dt = 1 / 60
            app._sync_model_camera()
            inputs.sync(1)
        assert app._surface_image(CaptureSurface.SCENE, None).shape[:2] == (480, 640)
        inputs.move((800, 400))
        capture(viewer, output / "camera-optics-output.png")
        projection = f"camera-inspector-projection-{session.selected_node.node_id}"
        inputs.click(projection + "-1")
        assert session.camera_view(camera_id).orthographic
        assert "##inspector-camera-ortho height" in ui.hits
        assert "##inspector-camera-vertical fov" not in ui.hits
        assert "##camera-Focal length" not in ui.hits
        capture(viewer, output / "camera-orthographic.png")


@pytest.mark.parametrize("scale", (1, 1.25))
def test_focused_workspace(backend_name, monkeypatch, tmp_path, scale):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        show_window=False,
        width=1700,
        height=1100,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    output = Path("output/focused-workspace") / f"{backend_name}-{scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        ui = workspace.ui
        app.set_language("zh_CN" if scale == 1 else "en")
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            for theme in ("light", "dark"):
                select_theme(app.design, theme, persist=False)
                workspace.activate("Assets")
                inputs.sync()
                assert "material-Mist blue ceramic" not in ui.hits
                assert "##asset-filter" in ui.hits
                capture(viewer, output / f"{theme}-resources.png")

                workspace.activate("Cameras")
                inputs.sync()
                inputs.click("camera-scene")
                assert not any(key.startswith("tab-") for key in ui.hits)
                assert "##camera-distance" in ui.hits
                capture(viewer, output / f"{theme}-cameras.png")
                node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
                camera = session.cameras[node.camera_index]
                inputs.click(f"camera-select-{camera.camera_id}")
                assert session.selected_node.node_id == node.node_id
                assert inspector_tabs(session, app.entity) == ("Properties",)
                assert not any(key.startswith("tab-") for key in ui.hits)

                geom = next(
                    node
                    for node in session.nodes
                    if node.type is NodeType.GEOM and node.body_index > 0
                )
                assert session.submit(cmd.SelectNode(geom.node_id)).ok
                inputs.sync()
                inputs.click("tab-Physics")
                assert any("contact_friction" in key for key in ui.hits), sorted(ui.hits)
                assert "##contact_solver_reference" not in ui.hits
                capture(viewer, output / f"{theme}-physics.png")
                inputs.click("tab-Material")
                inputs.click("material-color")
                capture(viewer, output / f"{theme}-color.png")
                inputs.escape()

                workspace.activate("Pose")
                inputs.sync()
                joint = next(joint for joint in session.joints if joint.type == "hinge")
                key = f"##pose-joint-{joint.joint_id}"
                assert ui.hits[key + "-value"][1] < ui.hits[key + "-track"][1]
                initial = session.frame.qpos[joint.qpos_adr]
                x, y, width, height = ui.hits[key + "-track"]
                inputs.move((x + width * 0.5, y + height / 2))
                inputs.event("add_mouse_button_event", 0, True)
                inputs.move((x + width * 0.8, y + height / 2))
                assert session.editing
                assert session.frame.qpos[joint.qpos_adr] != pytest.approx(initial)
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync()
                assert not session.editing
                assert session.submit(cmd.Undo()).ok
                inputs.sync()
                assert session.frame.qpos[joint.qpos_adr] == pytest.approx(initial)
                inputs.move((x + width * 0.3, y + height / 2))
                inputs.event("add_mouse_button_event", 0, True)
                inputs.move((x + width * 0.7, y + height / 2))
                inputs.escape()
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync()
                assert not session.editing
                assert session.frame.qpos[joint.qpos_adr] == pytest.approx(initial)
                inputs.click_at((x + width * 0.8, y + height / 2))
                vx, vy, vw, vh = ui.hits[key + "-value"]
                inputs.move((vx + vw / 2, vy + vh / 2))
                inputs.event("add_mouse_button_event", 1, True)
                inputs.event("add_mouse_button_event", 1, False)
                inputs.sync()
                assert session.frame.qpos[joint.qpos_adr] == pytest.approx(initial)
                capture(viewer, output / f"{theme}-joints.png")
                workspace.activate("Control")
                inputs.sync()
                if session.actuators:
                    assert ui.hits["##ctrl-0-value"][1] < ui.hits["##ctrl-0-track"][1]
                    initial_ctrl = session.frame.ctrl.copy()
                    x, y, width, height = ui.hits["##ctrl-0-track"]
                    inputs.click_at((x + width * 0.8, y + height / 2))
                    assert session.frame.ctrl[0] != pytest.approx(initial_ctrl[0])
                    inputs.click("##ctrl-0-reset")
                    np.testing.assert_allclose(session.frame.ctrl, initial_ctrl)
                capture(viewer, output / f"{theme}-actuators.png")
                assert session.submit(cmd.StartStateTakeRecording()).ok
                inputs.sync(12)
                assert session.submit(cmd.StopStateTakeRecording()).ok
                workspace.activate("Timeline")
                inputs.sync()
                assert "playback-options" not in ui.hits
                if "##timeline-speed" not in ui.hits:
                    inputs.click("timeline-overflow")
                    assert "##timeline-speed" in ui.hits
                    inputs.escape()
                inputs.move((800, 500))
                capture(viewer, output / f"{theme}-timeline.png")
    finally:
        viewer.release()


@pytest.mark.parametrize("scale", (1, 1.25))
def test_electron_layout_and_camera_operations(backend_name, monkeypatch, tmp_path, scale):
    """Exercise browser actions, popovers and navigation through production input."""
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_types"),
        renderer=backend_name,
        show_window=False,
        width=2100 if scale == 1 else 1700,
        height=1100,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    output = Path("output/electron-alignment") / f"{backend_name}-{scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        ui = workspace.ui
        app.set_language("zh_CN")
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            workspace.activate("Cameras")
            inputs.sync()
            inputs.click("header-Cameras-add")
            node = session.selected_node
            assert node is not None and node.type is NodeType.CAMERA
            assert node.model_id < 0
            camera_id = session.cameras[node.camera_index].camera_id
            editor_view = app.camera.view()
            inputs.click(f"camera-pilot-{camera_id}")
            assert app._camera_pilot.camera_id == camera_id
            original = session.camera_view(camera_id)
            x, y, width, height = app._viewport_rect
            point = x + width * 0.7, y + height * 0.65
            inputs.move(point)
            inputs.event("add_mouse_button_event", 1, True)
            inputs.move((point[0] + 40, point[1] + 25))
            assert session.editing
            assert not np.allclose(session.camera_view(camera_id).eye, original.eye)
            inputs.event("add_mouse_button_event", 1, False)
            inputs.sync()
            assert not session.editing
            assert session.submit(cmd.Undo()).ok
            inputs.sync()
            np.testing.assert_allclose(session.camera_view(camera_id).eye, original.eye, atol=1e-5)
            np.testing.assert_allclose(app.camera.view().eye, editor_view.eye, atol=1e-5)
            inputs.move(point)
            inputs.event("add_mouse_button_event", 1, True)
            inputs.move((point[0] - 35, point[1] + 20))
            inputs.escape()
            inputs.event("add_mouse_button_event", 1, False)
            inputs.sync()
            assert not session.editing
            np.testing.assert_allclose(session.camera_view(camera_id).eye, original.eye, atol=1e-5)
            inputs.click("camera-scene")
            assert app._camera_pilot is None and app._model_camera_id < 0
            inputs.click(f"camera-select-{camera_id}")
            inputs.click("inspector-camera-view")
            assert app._model_camera_id == camera_id and app._camera_pilot is None
            assert app._camera_transition is not None
            for _ in range(24):
                if app._model_camera_id >= 0:
                    app._dt = 1 / 60
                    app._sync_model_camera()
                else:
                    app._advance_camera(1 / 60)
                inputs.sync(1)
            assert app._camera_transition is None
            inputs.click("inspector-camera-view")
            assert app._model_camera_id < 0
            assert app._camera_transition is not None
            for _ in range(24):
                if app._model_camera_id >= 0:
                    app._dt = 1 / 60
                    app._sync_model_camera()
                else:
                    app._advance_camera(1 / 60)
                inputs.sync(1)
            assert app._camera_transition is None
            app.camera.pan(40, 10, app._viewport_rect[3])
            aligned = app.camera.view()
            inputs.click("inspector-camera-align")
            np.testing.assert_allclose(session.camera_view(camera_id).eye, aligned.eye, atol=1e-5)
            assert session.submit(cmd.Undo()).ok
            inputs.sync()
            np.testing.assert_allclose(session.camera_view(camera_id).eye, original.eye, atol=1e-5)
            requested = []
            monkeypatch.setattr(
                app, "request_capture", lambda: requested.append(app._model_camera_id)
            )
            inputs.click("inspector-camera-capture")
            assert requested == [camera_id]
            inputs.click("camera-scene")
            for theme in ("dark", "light"):
                select_theme(app.design, theme, persist=False)
                workspace.activate("Cameras")
                inputs.sync()
                inputs.click(f"camera-select-{camera_id}")
                inputs.move((0, 0))
                capture(viewer, output / f"{theme}-cameras.png")
                inputs.click("viewport-layers")
                assert "viewport-layers-popup" in ui.hits
                assert workspace.left_active == "Cameras"
                assert "rail-Layers" not in ui.hits
                before = app.viewport_layers.helpers
                inputs.click("layer-helpers")
                assert app.viewport_layers.helpers != before
                inputs.move((0, 0))
                capture(viewer, output / f"{theme}-layers.png")
                inputs.click("layers-reset")
                assert app.viewport_layers.helpers
                inputs.escape()
                assert "viewport-layers-popup" not in ui.hits
                workspace.activate("Output")
                inputs.sync()
                panel = app.panels.get("output")
                if not panel.show_search:
                    inputs.click("output-search")
                query = ui.hits["##output-filter"]
                assert query[1] <= ui.hits["output-copy"][1] + 1
                inputs.event("add_input_characters_utf8", "camera")
                assert panel._filter_text == "camera"
                inputs.move((0, 0))
                capture(viewer, output / f"{theme}-output.png")
                inputs.click("output-search")
                for panel_name, file_name in (("Pose", "joints"), ("Control", "actuators")):
                    workspace.activate(panel_name)
                    inputs.sync()
                    inputs.move((0, 0))
                    capture(viewer, output / f"{theme}-{file_name}.png")
                workspace.activate("Timeline")
                inputs.sync()
                inputs.move((0, 0))
                capture(viewer, output / f"{theme}-timeline.png")
            # View edits remain local even with a recorded playback interval.
            assert session.submit(cmd.StartStateTakeRecording()).ok
            inputs.sync(6)
            assert session.submit(cmd.StopStateTakeRecording()).ok
            editor = workspace.timeline_panel.editor
            editor.set_view_range(0, 0.5)
            inputs.sync()
            limits = session.state_take_range
            for key in ("##timeline-range-end", "##timeline-zoom"):
                if key not in ui.hits:
                    inputs.click("timeline-overflow")
                before = (editor.view_start, editor.view_end, editor.manual_view_range)
                x, y, width, height = ui.hits[key]
                point = x + width / 2, y + height / 2
                inputs.move(point)
                inputs.event("add_mouse_button_event", 0, True)
                inputs.move((point[0] + 35, point[1]))
                assert editor.view_end != pytest.approx(before[1])
                assert session.state_take_range == limits
                inputs.escape()
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync()
                assert (editor.view_start, editor.view_end, editor.manual_view_range) == before
            inputs.escape()
            workspace.activate("Cameras")
            inputs.sync()
            inputs.click(f"camera-more-{camera_id}")
            inputs.click(f"camera-delete-{camera_id}")
            assert session.camera_view(camera_id) is None
            assert session.submit(cmd.Undo()).ok
            inputs.sync()
            assert session.camera_view(camera_id) is not None
            workspace.activate("Output")
            inputs.sync()
            window = imgui.internal.find_window_by_name("Output")
            imgui.internal.set_window_dock(window, 0, imgui.Cond_.always)
            imgui.set_window_size("Output", (330 * scale, 460 * scale), imgui.Cond_.always)
            imgui.set_window_pos("Output", (400, 250), imgui.Cond_.always)
            inputs.sync()
            inputs.click("output-search")
            assert ui.hits["##output-filter"][1] < ui.hits["output-copy"][1]
            capture(viewer, output / "light-output-narrow.png")
    finally:
        viewer.release()


def test_model_camera_pilot_uses_existing_writeback_and_undo(backend_name, monkeypatch, tmp_path):
    """Model cameras use the same edit transaction as cameras created in the editor."""
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    output = Path("output/electron-alignment") / f"model-camera-{backend_name}"
    output.mkdir(parents=True, exist_ok=True)
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=backend_name,
            show_window=False,
            width=1440,
            height=1000,
            vsync=False,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
            ),
        ) as viewer,
        _NativeInput(viewer) as inputs,
    ):
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        app.set_language("zh_CN")
        inputs.sync(10)
        workspace.activate("Cameras")
        inputs.sync()
        node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
        assert node.model_id >= 0
        camera_id = session.cameras[node.camera_index].camera_id
        original = session.camera_view(camera_id)
        editor = app.camera.view()
        inputs.click(f"camera-select-{camera_id}")
        inputs.click(f"camera-pilot-{camera_id}")
        assert app._camera_pilot.camera_id == camera_id
        x, y, width, height = app._viewport_rect
        point = (x + width * 0.7, y + height * 0.65)
        inputs.move(point)
        inputs.event("add_mouse_button_event", 1, True)
        for dx, dy in ((20, 10), (40, 25)):
            inputs.move((point[0] + dx, point[1] + dy))
        assert session.editing
        changed = session.camera_view(camera_id)
        assert not np.allclose(changed.eye, original.eye)
        np.testing.assert_allclose(
            session.adapter.camera_view(camera_id).eye, changed.eye, atol=1e-5
        )
        inputs.event("add_mouse_button_event", 1, False)
        inputs.sync()
        assert not session.editing
        capture(viewer, output / "pilot.png")
        assert session.submit(cmd.Undo()).ok
        inputs.sync()
        np.testing.assert_allclose(session.camera_view(camera_id).eye, original.eye, atol=1e-5)
        np.testing.assert_allclose(
            session.adapter.camera_view(camera_id).eye, original.eye, atol=1e-5
        )
        assert session.submit(cmd.Redo()).ok
        inputs.sync()
        np.testing.assert_allclose(session.camera_view(camera_id).eye, changed.eye, atol=1e-5)
        assert session.submit(cmd.Undo()).ok
        inputs.sync()
        # The Inspector entry point and cancellation retain the same model ownership.
        inputs.click("camera-scene")
        inputs.click(f"camera-select-{camera_id}")
        inputs.click("inspector-camera-pilot")
        assert app._camera_pilot.camera_id == camera_id
        inputs.move(point)
        inputs.event("add_mouse_button_event", 1, True)
        inputs.move((point[0] - 35, point[1] + 20))
        inputs.escape()
        inputs.event("add_mouse_button_event", 1, False)
        inputs.sync()
        assert not session.editing
        np.testing.assert_allclose(
            session.adapter.camera_view(camera_id).eye, original.eye, atol=1e-5
        )
        inputs.click("camera-scene")
        np.testing.assert_allclose(app.camera.view().eye, editor.eye, atol=1e-5)
        for _ in range(24):
            app._advance_camera(1 / 60)
            inputs.sync(1)
        assert app._camera_transition is None
        inputs.click(f"camera-select-{camera_id}")
        app.camera.pan(40, 10, app._viewport_rect[3])
        aligned = app.camera.view()
        inputs.click("inspector-camera-align")
        np.testing.assert_allclose(
            session.adapter.camera_view(camera_id).eye, aligned.eye, atol=1e-5
        )
        assert session.submit(cmd.Undo()).ok
        inputs.sync()
        np.testing.assert_allclose(
            session.adapter.camera_view(camera_id).eye, original.eye, atol=1e-5
        )
        inputs.move((0, 0))
        capture(viewer, output / "actions.png")


@pytest.mark.parametrize("scale", (1, 1.25))
def test_reference_scene_density_and_layer_operations(backend_name, monkeypatch, tmp_path, scale):
    """Review the reference scene at desktop width, including retained design profiles."""
    import os

    from mojive.design.store import atomic_write, read_spec
    from mojive.render.backend import RenderFlag

    requested = os.environ.get("MOJIVE_REVIEW_DESIGN")
    if requested:
        path = tmp_path / "review-design.json"
        atomic_write(path, read_spec(requested))
        monkeypatch.setenv("MOJIVE_DESIGN_SPEC", str(path))
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    viewer = build_workspace(
        resolve("joint_gizmo"),
        renderer=backend_name,
        show_window=False,
        width=1440,
        height=1000,
        vsync=False,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), debug_server=False, threaded_physics=False
        ),
    )
    profile = "desktop-profile" if requested else "reference"
    output = Path("output/electron-alignment") / f"{profile}-{backend_name}-{scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        app, session, workspace = viewer.app, viewer.session, viewer.app.workspace
        ui = workspace.ui
        app.set_language("zh_CN")
        with _NativeInput(viewer) as inputs:
            inputs.sync(10)
            for theme in ("light", "dark"):
                select_theme(app.design, theme, persist=False)
                workspace.activate("Pose")
                inputs.sync(4)
                joint = next(j for j in session.joints if j.type == "hinge")
                heading = ui.hits[f"pose-joint-{joint.joint_id}"]
                value = ui.hits[f"##pose-joint-{joint.joint_id}-value"]
                assert heading[0] + heading[2] <= value[0] + 1
                assert heading[2] >= 90 * scale
                card = ui.hits[f"pose-fields-{joint.joint_id}"]
                assert value[1] - card[1] == pytest.approx(
                    card[0] + card[2] - value[0] - value[2], abs=1
                )
                inputs.move((value[0] + value[2] / 2, value[1] + value[3] / 2))
                capture(viewer, output / f"{theme}-scalar-card-hover.png")
                inputs.click(f"pose-joint-{joint.joint_id}")
                inputs.move((900, 500))
                capture(viewer, output / f"{theme}-joints-status.png")
                # Caption and whitespace belong to the same native switch item.
                sx, sy, _sw, sh = ui.hits["##joint_limited"]
                inputs.move((sx + 12 * scale, sy + sh / 2))
                capture(viewer, output / f"{theme}-switch-label-hover.png")
                inputs.event("add_mouse_button_event", 0, True)
                capture(viewer, output / f"{theme}-switch-label-held.png")
                inputs.move((800, 450))
                inputs.event("add_mouse_button_event", 0, False)
                assert next(j for j in session.joints if j.joint_id == joint.joint_id).limited
                inputs.click_at((sx + 12 * scale, sy + sh / 2))
                assert not next(j for j in session.joints if j.joint_id == joint.joint_id).limited
                assert session.submit(cmd.Undo()).ok
                inputs.sync(4)
                scalar_keys = ("##joint_damping", "##joint_stiffness")
                fields = [ui.hits[key] for key in scalar_keys]
                assert all(rect[0] == pytest.approx(fields[0][0], abs=1) for rect in fields)
                assert all(rect[2] == pytest.approx(fields[0][2], abs=1) for rect in fields)
                lower, upper = (ui.hits[f"##joint_range-{i}"] for i in range(2))
                assert lower[0] == pytest.approx(fields[0][0], abs=1)
                assert upper[0] + upper[2] == pytest.approx(fields[0][0] + fields[0][2], abs=1)
                free = next(j for j in session.joints if j.type == "free")
                assert f"pose-joint-{free.joint_id}-expand" not in ui.hits
                axis = f"##pose-joint-{free.joint_id}-positionX"
                badge = ui.hits[axis + "-reset"]
                numeric = ui.hits[axis + "-numeric"]
                for component in ("Y", "Z"):
                    sibling = ui.hits[axis[:-1] + component + "-numeric"]
                    assert sibling[1] == pytest.approx(numeric[1])
                assert badge[1] == pytest.approx(numeric[1])
                assert badge[3] == pytest.approx(numeric[3])
                assert numeric[0] - badge[0] - badge[2] == pytest.approx(4 * scale)
                inputs.move((numeric[0] + numeric[2] / 2, numeric[1] + numeric[3] / 2))
                capture(viewer, output / f"{theme}-joint-value-hover.png")
                original = session.frame.qpos.copy()
                ax, ay, aw, ah = ui.hits[axis]
                inputs.move((ax + aw * 0.7, ay + ah / 2))
                inputs.event("add_mouse_button_event", 0, True)
                inputs.move((ax + aw * 0.7 + 20 * scale, ay + ah / 2))
                inputs.event("add_mouse_button_event", 0, False)
                inputs.sync()
                assert session.frame.qpos[free.qpos_adr] != pytest.approx(original[free.qpos_adr])
                inputs.click(axis + "-reset")
                np.testing.assert_allclose(session.frame.qpos, original)
                inputs.sync(30)
                inputs.click(axis + "-numeric")
                inputs.click(axis + "-numeric")
                capture(viewer, output / f"{theme}-joint-value-entry.png")
                modifier = (
                    imgui.Key.mod_super
                    if imgui.get_io().config_mac_osx_behaviors
                    else imgui.Key.mod_ctrl
                )
                inputs.event("add_key_event", modifier, True)
                inputs.event("add_key_event", imgui.Key.a, True)
                inputs.event("add_key_event", imgui.Key.a, False)
                inputs.event("add_key_event", modifier, False)
                inputs.event("add_input_characters_utf8", "1.625")
                inputs.event("add_key_event", imgui.Key.enter, True)
                inputs.event("add_key_event", imgui.Key.enter, False)
                assert session.frame.qpos[free.qpos_adr] == pytest.approx(1.625)
                inputs.click(axis + "-reset")
                np.testing.assert_allclose(session.frame.qpos, original)

                inputs.click(f"pose-joint-{free.joint_id}")
                workspace.inspector_tab = "Joint"
                inputs.sync(4)
                inputs.move((800, 450))
                capture(viewer, output / f"{theme}-free-inspector.png")

                inputs.click("shading")
                popup = workspace.menus.observed["Shading"]
                assert [row["icon"] for row in popup["rows"]] == ["shading", "sphere"]
                capture(viewer, output / f"{theme}-shading-menu.png")
                row = popup["rows"][1]["rect"]
                inputs.click_at((row[0] + row[2] / 2, row[1] + row[3] / 2))
                assert workspace.menus.shading == "Wireframe"
                capture(viewer, output / f"{theme}-wireframe.png")
                inputs.click("shading")
                row = workspace.menus.observed["Shading"]["rows"][0]["rect"]
                inputs.click_at((row[0] + row[2] / 2, row[1] + row[3] / 2))
                assert workspace.menus.shading == "Solid"

                workspace.activate("Control")
                inputs.sync(4)
                capture(viewer, output / f"{theme}-actuators.png")
                workspace.activate("Cameras")
                inputs.sync(4)
                camera_node = next(node for node in session.nodes if node.type is NodeType.CAMERA)
                camera_id = session.cameras[camera_node.camera_index].camera_id
                inputs.click(f"camera-select-{camera_id}")
                projection = ui.hits[f"camera-inspector-projection-{camera_node.node_id}"]
                for name in ("vertical fov", "near", "far"):
                    field = ui.hits[f"##inspector-camera-{name}-value"]
                    assert field[0] == pytest.approx(projection[0], abs=1)
                    assert field[0] + field[2] == pytest.approx(
                        projection[0] + projection[2], abs=1
                    )
                capture(viewer, output / f"{theme}-scene-camera.png")
                inputs.click("camera-scene")
                capture(viewer, output / f"{theme}-editor-camera.png")
                # Compact editor properties share rows with their labels.
                assert "##camera-distance-value" in ui.hits
                assert "##camera-distance-track" not in ui.hits
                label = ui.hits["camera-label-" + ui.translate("distance")]
                value = ui.hits["##camera-distance-value"]
                assert abs(label[1] - value[1]) <= 2 * scale
                assert label[0] + label[2] <= value[0]
                panel_before = workspace.left_active
                inputs.click("viewport-layers")
                popup = ui.hits["viewport-layers-popup"]
                assert popup[2] <= 240 * scale
                assert popup[3] <= 440 * scale
                rows = [
                    ui.hits["layer-" + key]
                    for key in (
                        "geometry",
                        "viewport_ui",
                        "gizmos",
                        "helpers",
                        "influence",
                        "view_cube",
                        "perturbation",
                        "selection",
                    )
                ]
                assert all(row[0] == pytest.approx(rows[0][0]) for row in rows)
                capture(viewer, output / f"{theme}-layers.png")
                gutter = rows[0][0] - popup[0]
                assert gutter == pytest.approx(4 * scale, abs=1)
                last = rows[-1]
                assert popup[0] + popup[2] - last[0] - last[2] == pytest.approx(gutter, abs=1)
                assert popup[1] + popup[3] - last[1] - last[3] == pytest.approx(gutter, abs=1)
                # The last hover surface challenges both adjacent shell corners.
                painted = []
                captions = []
                original_background = ui.design.primitives.field_background
                original_text = ui.design.primitives.aligned_text

                def observe_text(
                    style, content, text, *, records=captions, paint=original_text, **options
                ):
                    if "switch" in style.part_ids:
                        records.append(
                            (
                                text,
                                ui.measure(
                                    text,
                                    style.get("fontSize", 14),
                                    style=style,
                                    translate=False,
                                    font_resolved=True,
                                ),
                                content.width,
                            )
                        )
                    return paint(style, content, text, **options)

                def observe_background(
                    style, box, color=None, *, records=painted, paint=original_background, **options
                ):
                    if "switch" in style.part_ids:
                        records.append((box, style.get("radius", 0) * scale))
                    return paint(style, box, color, **options)

                with monkeypatch.context() as patch:
                    patch.setattr(ui.design.primitives, "field_background", observe_background)
                    patch.setattr(ui.design.primitives, "aligned_text", observe_text)
                    inputs.move((last[0] + 2 * scale, last[1] + last[3] / 2))
                    capture(viewer, output / f"{theme}-layers-last-hover.png")
                assert captions and all(width <= available for _, width, available in captions), (
                    captions
                )
                box, radius = painted[-1]
                assert box.frame.tuple() == pytest.approx(last)
                assert box.padding[0] == box.padding[2] == pytest.approx(8 * scale)
                assert radius + gutter == pytest.approx(8 * scale, abs=1)
                helper = ui.hits["layer-helpers"]
                padding_point = (helper[0] + 2 * scale, helper[1] + helper[3] / 2)
                inputs.move(padding_point)
                capture(viewer, output / f"{theme}-layers-switch-hover.png")
                before_helpers = app.viewport_layers.helpers
                inputs.click_at(padding_point)
                assert app.viewport_layers.helpers != before_helpers
                inputs.click_at(padding_point)
                assert app.viewport_layers.helpers == before_helpers
                selected = session.selected_node
                inputs.click("layer-geometry")
                assert not app.backend.get_flag(RenderFlag.SCENE_GEOMETRY)
                assert app.backend._builder.scene.count == 0
                assert session.selected_node == selected
                capture(viewer, output / f"{theme}-geometry-hidden.png")
                inputs.click("layer-view_cube")
                assert not app.viewport_layers.view_cube
                inputs.click("layers-reset")
                assert app.backend.get_flag(RenderFlag.SCENE_GEOMETRY)
                assert app.backend._builder.scene.count > 0
                inputs.escape()
                assert workspace.left_active == panel_before

                workspace.activate("Timeline")
                inputs.sync(4)
                assert "timeline-take" not in ui.hits if not session.state_takes else True
                assert "timeline-new-take" in ui.hits
                toolbar = ui.hits["timeline-tools"]
                for key in (
                    "timeline-first",
                    "timeline-record",
                    "timeline-fit",
                    "timeline-follow",
                    "##timeline-range-start",
                    "##timeline-range-end",
                ):
                    if key not in ui.hits:
                        inputs.click("timeline-overflow")
                    rect = ui.hits[key]
                    containers = (toolbar, ui.hits.get("timeline-overflow-content", toolbar))
                    assert any(
                        container[0] - 1 <= rect[0]
                        and rect[0] + rect[2] <= container[0] + container[2] + 1
                        and container[1] - 1 <= rect[1]
                        and rect[1] + rect[3] <= container[1] + container[3] + 1
                        for container in containers
                    )
                capture(viewer, output / f"{theme}-timeline.png")
                if "timeline-overflow-content" in ui.hits:
                    capture(viewer, output / f"{theme}-timeline-overflow.png")
                    inputs.escape()
                inputs.click("timeline-settings")
                row = ui.hits["timeline-pause-at-end"]
                label = ui.hits["##timeline-pause-label"]
                assert label[0] - row[0] >= 3 * scale
                padding_point = (row[0] + 2 * scale, row[1] + row[3] / 2)
                inputs.move(padding_point)
                capture(viewer, output / f"{theme}-timeline-switch-hover.png")
                before_pause = session.state_take_pause_at_end
                inputs.click_at(padding_point)
                assert session.state_take_pause_at_end != before_pause
                inputs.click_at(padding_point)
                assert session.state_take_pause_at_end == before_pause
                inputs.escape()
                inputs.click("timeline-new-take")
                assert session.state_takes
                first_id = session.active_state_take_id
                inputs.click("timeline-new-take")
                second_id = session.active_state_take_id
                inputs.click("timeline-new-take")
                third_id = session.active_state_take_id
                inputs.click("timeline-take")
                inputs.click(f"timeline-take-choice-{first_id}-remove")
                assert session.active_state_take_id == third_id
                assert len(session.state_takes) == 2
                inputs.escape()
                assert second_id != first_id
                assert "timeline-take" in ui.hits
                inputs.click("timeline-take")
                rx, ry, rw, rh = ui.hits[f"timeline-take-choice-{third_id}-remove"]
                inputs.move((rx + rw / 2, ry + rh / 2))
                capture(viewer, output / f"{theme}-take-selector.png")
                popup = workspace.menus.observed["timeline-take-choices"]
                assert popup["rect"][2] >= 70 * scale
                inputs.click(f"timeline-take-choice-{third_id}-remove")
                inputs.escape()
                assert session.active_state_take_id == second_id
                assert len(session.state_takes) == 1
                track = workspace.timeline_panel.last_layout
                inputs.click_at((track.track_left + 40 * scale, track.lane_y(1)))
                inputs.event("add_mouse_button_event", 1, True)
                inputs.event("add_mouse_button_event", 1, False)
                inputs.sync(3)
                assert "timeline-delete-take" in ui.hits
                capture(viewer, output / f"{theme}-timeline-context.png")
                inputs.click("timeline-delete-take")
                assert not session.state_takes
                for take in tuple(session.state_takes):
                    assert session.submit(cmd.RemoveStateTake(take.take_id)).ok
                inputs.sync(3)
            camera = app.camera
            distance, yaw = camera.distance, camera.yaw
            camera.distance = distance * 0.3
            inputs.sync(4)
            capture(viewer, output / "floor-close.png")
            camera.yaw += 8
            inputs.sync(4)
            capture(viewer, output / "floor-orbit.png")
            camera.distance, camera.yaw = distance, yaw
            assert session.submit(cmd.NewScene()).ok
            for theme in ("light", "dark"):
                select_theme(app.design, theme, persist=False)
                empty_origins = []
                for name, key in (
                    ("Pose", "pose-empty"),
                    ("Control", "control-empty"),
                    ("Sensors", "sensor-empty"),
                    ("Cameras", "camera-empty"),
                    ("Assets", "resources-empty"),
                ):
                    workspace.activate(name)
                    inputs.sync(3)
                    assert key in ui.hits
                    if name in ("Pose", "Control", "Sensors"):
                        empty_origins.append(ui.hits[key][0])
                    capture(viewer, output / f"{theme}-empty-{name.lower()}.png")
                assert empty_origins == pytest.approx([empty_origins[0]] * 3)
    finally:
        viewer.release()
