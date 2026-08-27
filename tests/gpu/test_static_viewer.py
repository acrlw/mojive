from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")

from forge_viewer import commands as cmd  # noqa: E402
from forge_viewer.bridge import DebugClient  # noqa: E402
from forge_viewer.composition import build_scene  # noqa: E402
from forge_viewer.demos import canvas_scene  # noqa: E402
from forge_viewer.gizmo import SIZE_PT, project, world_scale  # noqa: E402
from forge_viewer.render.debugdraw import PrimitiveType  # noqa: E402
from forge_viewer.scene import Scene  # noqa: E402
from forge_viewer.types import DEFAULT_MATERIAL, CameraView, Material, MeshShape  # noqa: E402
from forge_viewer.ui.scene_entities import HELPER_LAYER  # noqa: E402


@pytest.fixture(scope="module")
def canvas():
    scene = canvas_scene()
    viewer = build_scene(scene, vsync=False, width=1100, height=720)
    try:
        viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
        for _ in range(8):
            viewer.sync()
        yield viewer, scene
    finally:
        viewer.release()


def snap(viewer) -> np.ndarray:
    px = viewer.window.read_frame()
    assert px is not None
    return np.asarray(px)[::-1][..., :3].copy()


def viewport_snap(viewer) -> np.ndarray:
    """Crop the window snapshot to the viewport panel.

    The docked stats panel redraws its frame-time plot every sync, so a
    whole-window diff measures UI churn instead of the scene edit — and the
    churn differs between the forge and wgpu frame loops.
    """
    image = snap(viewer)
    x, y, w, h = viewer.window.points_to_pixels(viewer.app._viewport_rect)
    x0, y0, x1, y1 = round(x), round(y), round(x + w), round(y + h)
    return image[y0:y1, x0:x1]


def test_canvas_opens_without_importing_mujoco(canvas):
    viewer, _scene = canvas
    image = snap(viewer)

    assert "mujoco" not in sys.modules
    assert viewer.session.adapter.caps.name == "static"
    assert viewer.session.paused
    assert viewer.backend.stats.instances == 4
    assert float(image.std()) > 10.0


def test_canvas_selection_reaches_antialiased_outline(canvas):
    viewer, _scene = canvas
    target = next(n for n in viewer.session.nodes if n.name == "crate")
    viewer.session.submit(cmd.Select(target.object_id))
    viewer.sync()
    viewer.sync()
    image = snap(viewer).astype(np.int16)
    outline = np.array([255, 161, 51], np.int16)

    assert np.all(np.abs(image - outline) <= 3, axis=-1).sum() > 100


def test_canvas_pose_update_changes_the_window(canvas):
    viewer, scene = canvas
    before = snap(viewer)
    scene.object("ball").set_pose((1.4, -0.3, 0.42))
    viewer.sync()
    viewer.sync()
    after = snap(viewer)

    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 500


def test_canvas_material_edits_change_the_window(canvas):
    viewer, scene = canvas
    before = viewport_snap(viewer)
    crate = scene.object("crate")
    crate.set_color((0.1, 0.8, 0.9, 1.0))
    crate.set_material(Material(name="emissive", emission=0.65, specular=0.1))
    viewer.sync()
    viewer.sync()
    after = viewport_snap(viewer)

    diff = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=-1)
    assert np.count_nonzero(diff > 10) > 150

    crate.set_color((0.92, 0.42, 0.18, 1.0))
    crate.set_material(DEFAULT_MATERIAL)
    viewer.sync()


def test_canvas_detects_dynamic_structure_changes(canvas):
    viewer, scene = canvas
    generation = viewer.session.structure_generation
    obj = scene.sphere(name="spawned", position=(0.0, 1.2, 0.25), size=(0.25, 0.25, 0.25))
    viewer.sync()

    assert viewer.session.structure_generation == generation + 1
    assert viewer.session.node_by_object_id(obj.object_id).name == "spawned"
    assert viewer.backend.stats.instances == 5


def test_canvas_receives_external_debug_draw(canvas):
    viewer, _scene = canvas
    before = viewer.backend.debug.stats().primitives
    with DebugClient(pid=os.getpid()) as client:
        client.send(
            op="line",
            layer="external-test",
            id="socket-line",
            a=[-1.0, 0.0, 1.0],
            b=[1.0, 0.0, 1.0],
            color=[1.0, 0.0, 1.0, 1.0],
            width_px=4.0,
        )
        deadline = time.monotonic() + 1.0
        while viewer.backend.debug.stats().primitives == before and time.monotonic() < deadline:
            viewer.sync()

    assert viewer.bridge.stats.applied >= 1
    assert viewer.backend.debug.stats().primitives == before + 1


def test_canvas_records_streaming_video(canvas, tmp_path):
    from imageio_ffmpeg import read_frames

    viewer, scene = canvas
    output = tmp_path / "canvas.mp4"

    def move(index, _viewer):
        scene.object("ball").set_pose((0.3 + 0.15 * index, -0.3, 0.42))

    viewer.record(output, frames=4, fps=24, before_frame=move)
    reader = read_frames(str(output))
    metadata = next(reader)
    count = sum(1 for _ in reader)
    reader.close()

    assert output.stat().st_size > 1000
    assert count == 4
    assert metadata["fps"] == pytest.approx(24.0)


def test_editor_actions_save_and_restore_an_authored_scene(tmp_path, monkeypatch):
    viewer = build_scene(Scene(), vsync=False, width=960, height=640)
    document = tmp_path / "editor.forge.json"
    try:
        viewer.sync()
        viewer.app._add_scene_object(MeshShape.BOX, "box")
        assert viewer.session.selected
        viewer.app._duplicate_selected()
        assert viewer.session.source.instance_count == 2
        assert viewer.session.dirty

        assert viewer.app.save_scene(document)
        assert not viewer.session.dirty
        viewer.app._execute_document_action("new_scene")
        assert viewer.session.source.instance_count == 0
        assert viewer.app.open_scene(document)
        viewer.sync()
        assert viewer.backend.stats.instances == 2
        assert [node.name for node in viewer.session.nodes if node.object_id][:2] == [
            "box",
            "box Copy",
        ]

        from imgui_bundle import imgui

        box = next(node for node in viewer.session.nodes if node.name == "box")
        viewer.session.submit(cmd.Select(box.object_id))
        viewer.app.request_rename(box.object_id)
        viewer.sync()
        imgui.get_io().add_key_event(imgui.Key.escape, True)
        viewer.sync()
        imgui.get_io().add_key_event(imgui.Key.escape, False)
        viewer.sync()

        viewer.app._add_scene_object(MeshShape.SPHERE, "sphere")
        viewer.app._request_document_action("new_scene")
        original_button = imgui.button

        def discard_changes(label, *args, **kwargs):
            clicked = original_button(label, *args, **kwargs)
            return True if label == "Discard" else clicked

        monkeypatch.setattr(imgui, "button", discard_changes)
        viewer.sync()
        assert viewer.session.source.instance_count == 0
    finally:
        viewer.release()


def test_finite_authored_plane_is_pickable_in_the_viewport():
    scene = Scene()
    floor = scene.plane(name="floor", size=(2.0, 2.0, 0.02))
    viewer = build_scene(scene, vsync=False, width=960, height=640)
    try:
        viewer.app.camera.look_from(-135.0, 25.0, viewer.app.camera_out, animate=False)
        viewer.session.submit(cmd.Select(floor.object_id))
        for _ in range(3):
            viewer.sync()
        viewer.session.submit(cmd.Select(0))
        viewer.sync()
        cursor = project(
            viewer.app._camera_view(),
            [np.array((0.8, 0.0, 0.0), np.float32)],
            viewer.app._viewport_rect,
        )[0, :2]

        assert viewer.app._pick_at((float(cursor[0]), float(cursor[1]))) == floor.object_id
    finally:
        viewer.release()


def test_editor_modals_are_readable_and_follow_the_resized_window_center(monkeypatch):
    from imgui_bundle import imgui

    viewer = build_scene(Scene(), vsync=False, width=900, height=600)

    def geometry(name):
        popup = imgui.internal.find_window_by_name(name)
        viewport = imgui.get_main_viewport()
        center = (popup.pos.x + popup.size.x * 0.5, popup.pos.y + popup.size.y * 0.5)
        expected = (viewport.get_center().x, viewport.get_center().y)
        return popup, center, expected

    try:
        viewer.sync()
        viewer.app._report_model_error(
            "Unable to open the selected workspace because referenced resources are missing."
        )
        for _ in range(3):
            viewer.sync()
        error, center, expected = geometry("File operation failed")
        assert error.size.x == pytest.approx(440.0)
        assert error.size.x > error.size.y
        assert center == pytest.approx(expected, abs=1.0)

        original_button = imgui.button

        def accept_error(label, *args, **kwargs):
            return True if label == "OK" else original_button(label, *args, **kwargs)

        monkeypatch.setattr(imgui, "button", accept_error)
        viewer.sync()
        monkeypatch.setattr(imgui, "button", original_button)

        viewer.app._add_scene_object(MeshShape.BOX, "box")
        viewer.app._request_document_action("quit")
        for _ in range(3):
            viewer.sync()
        width, height = viewer.window.size_points
        glfw.set_window_size(viewer.window._window, width + 400, height + 300)
        for _ in range(3):
            viewer.sync()
        _prompt, center, expected = geometry("Unsaved changes")
        assert center == pytest.approx(expected, abs=1.0)
    finally:
        viewer.release()


def test_editor_undo_redo_updates_the_render_backend():
    viewer = build_scene(Scene(), vsync=False, width=960, height=640)
    try:
        viewer.sync()
        viewer.app._add_scene_object(MeshShape.BOX, "box")
        viewer.sync()
        assert viewer.backend.stats.instances == 1

        assert viewer.session.submit(cmd.Undo())
        viewer.sync()
        assert viewer.backend.stats.instances == 0

        assert viewer.session.submit(cmd.Redo())
        viewer.sync()
        assert viewer.backend.stats.instances == 1
    finally:
        viewer.release()


def test_settings_window_is_modal_and_centered(canvas):
    from imgui_bundle import imgui

    viewer, _scene = canvas
    settings = viewer.app.panels.get("Settings")
    assert settings is not None
    try:
        settings.open = True
        viewer.sync()
        viewer.sync()
        translated = viewer.app.localizer.text(settings.name)
        title = settings.name if translated == settings.name else f"{translated}###{settings.name}"
        window = imgui.internal.find_window_by_name(title)
        assert window is not None
        assert window.dock_node is None
        center = window.pos + window.size * 0.5
        expected = imgui.get_main_viewport().get_center()
        assert center.x == pytest.approx(expected.x, abs=1.0)
        assert center.y == pytest.approx(expected.y, abs=1.0)
    finally:
        settings.open = False


def test_scene_camera_helper_is_pickable_and_transformable():
    from imgui_bundle import imgui

    scene = Scene()
    camera_id = scene.add_camera(
        "shot",
        CameraView(
            eye=np.array((0.0, 0.0, 1.0), np.float32),
            target=np.array((0.0, 1.0, 1.0), np.float32),
            near=0.1,
            far=2.0,
        ),
    )
    viewer = build_scene(scene, vsync=False, width=960, height=640)
    try:
        viewer.sync()
        editor_camera = CameraView(
            eye=np.array((0.0, -5.0, 2.0), np.float32),
            target=np.array((0.0, 0.0, 1.0), np.float32),
            aspect=1.5,
        )
        viewer.app.camera.adopt(editor_camera)
        viewer.app.camera.publish(viewer.app.camera_out)
        viewer.sync()

        saved_editor_view = viewer.app.camera.view()
        viewer.app.select_model_camera(camera_id)
        viewer.sync()
        assert viewer.app._model_camera_view is not None
        viewer.app.select_model_camera(-1)
        viewer.sync()
        restored_editor_view = viewer.app.camera.view()
        assert restored_editor_view.eye == pytest.approx(saved_editor_view.eye)
        assert restored_editor_view.target == pytest.approx(saved_editor_view.target)
        assert restored_editor_view.fov_y == pytest.approx(saved_editor_view.fov_y)

        node = next(node for node in viewer.session.nodes if node.name == "shot")
        screen = project(
            viewer.app._camera_view(), [np.array((0.0, 0.0, 1.0))], viewer.app._viewport_rect
        )[0]
        assert viewer.app._pick_at((float(screen[0]), float(screen[1]))) == node.object_id

        viewer.session.submit(cmd.Select(node.object_id))
        viewer.sync()
        assert viewer.app.camera_preview._image is not None
        assert viewer.app.camera_preview._image.aspect == pytest.approx(16.0 / 9.0, rel=0.01)
        layer = viewer.backend.debug.layer(HELPER_LAYER)
        assert layer.count_of(PrimitiveType.POINT) == 1
        assert layer.count_of(PrimitiveType.LINE) == 20

        viewer.app.gizmo.set_mode("translate")
        viewer.app.gizmo.set_space("world")
        view = viewer.session.camera_view(camera_id)
        before = np.asarray(view.eye, np.float64).copy()
        scale = world_scale(
            viewer.app._camera_view(),
            before,
            viewer.app._viewport_rect[3],
            SIZE_PT * viewer.window.style_scale,
        )
        axis = np.array((0.0, 0.0, 1.0))
        cursor = project(
            viewer.app._camera_view(), (before + axis * scale * 0.55,), viewer.app._viewport_rect
        )[0, :2]
        io = imgui.get_io()
        io.add_mouse_pos_event(*cursor)
        viewer.sync()
        io.add_mouse_button_event(0, True)
        viewer.sync()
        axis_screen = project(
            viewer.app._camera_view(), (before, before + axis * scale), viewer.app._viewport_rect
        )[:, :2]
        direction = axis_screen[1] - axis_screen[0]
        direction /= np.linalg.norm(direction)
        io.add_mouse_pos_event(*(cursor + direction * 40.0))
        viewer.sync()
        io.add_mouse_button_event(0, False)
        viewer.sync()

        after = np.asarray(viewer.session.camera_view(camera_id).eye)
        assert after[2] - before[2] > 0.1
    finally:
        imgui.get_io().add_mouse_button_event(0, False)
        viewer.release()
