"""Native controls and displayed-pose tracking in both rendering backends."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mojive import CameraTrackingConfig, ViewerConfig, build_scene
from mojive import commands as cmd
from mojive.scene import Scene
from mojive.tools.camera_tracking import exercise_controls
from mojive.types import CameraView

pytestmark = pytest.mark.gpu


@pytest.mark.physics
@pytest.mark.parametrize("name", ("world_camera", "body_camera", "world_light", "body_light"))
@pytest.mark.parametrize("entrypoint", ("hierarchy", "viewport"))
def test_double_click_focuses_camera_and_light_positions(
    tmp_path, monkeypatch, backend_name, name, entrypoint
):
    mujoco = pytest.importorskip("mujoco")
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.app.composition import build_from_adapter
    from mojive.interaction.gizmo import project
    from mojive.scene.queries import node_world_pose
    from mojive.tools.camera_tracking import click
    from mojive.tools.ui_runtime import _item_center
    from mojive.ui.camera import FOCUS_DURATION

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(
        mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom type="box" size="6 4 .1" pos="0 0 -.1"/>
          <camera name="world_camera" pos="-2 -1 2"/>
          <light name="world_light" pos="-2 1 3"/>
          <body pos="0 0 1"><freejoint/>
            <geom type="box" size=".3 .2 .4" pos="1 0 0"/>
            <camera name="body_camera" pos="-.4 -.5 .8"/>
            <light name="body_light" pos=".4 .5 1.8"/>
          </body>
        </worldbody></mujoco>
    """)
    )
    with build_from_adapter(
        adapter, vsync=False, width=1440, height=1000, show_window=False
    ) as viewer:
        for _ in range(4):
            viewer.sync()
        adapter.data.qpos[0] += 2.0
        mujoco.mj_forward(adapter.model, adapter.data)
        viewer.set_camera(CameraView(eye=(6, -9, 6), target=(0, 0, 1)))
        viewer.sync()
        node = next(node for node in viewer.session.nodes if node.name == name)
        expected, _ = node_world_pose(viewer.session, node)
        if entrypoint == "hierarchy":
            viewer.panels.get("Hierarchy")._filter = name
            for _ in range(3):
                viewer.sync()
            cursor = _item_center(viewer, "invisible_button", f"##hierarchy-node-{node.node_id}")
        else:
            cursor = project(viewer.session.camera, [expected], viewer.app._viewport_rect)[0, :2]
            assert viewer.app._pick_at(tuple(cursor)) == node.object_id
        click(viewer, cursor)
        click(viewer, cursor)
        assert viewer.session.selected == node.object_id
        assert viewer.app.camera.animating
        viewer.app.camera.advance(FOCUS_DURATION, viewer.app.camera_out)
        assert viewer.app.camera.pivot == pytest.approx(expected, abs=1e-5)
        viewer.sync()
        assert viewer.app._model_camera_id == -1
        output = Path("output/camera-helpers")
        output.mkdir(parents=True, exist_ok=True)
        viewer.capture(output / f"focus-{backend_name}-{name}-{entrypoint}.png", surface="viewport")


@pytest.mark.parametrize("scale,language", [(1, "en"), (1.5, "zh_CN")])
def test_tracking_controls_and_navigation_follow_current_scene_pose(
    tmp_path, monkeypatch, scale, language
):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    monkeypatch.setenv("MOJIVE_LANGUAGE", language)
    scene = Scene()
    root = scene.box(name="root", position=(0, 0, 1))
    other = scene.box(name="other", position=(2, 0, 1))
    with build_scene(scene, config=ViewerConfig(), vsync=False, width=1600, height=1100) as viewer:
        viewer.sync()
        node = viewer.session.node_by_object_id(root.object_id)
        exercise_controls(viewer, node.node_id)
        viewer.set_camera(CameraView(eye=np.array((5, -7, 4)), target=np.array((0, 0, 1))))
        assert viewer.tracking_node_id is None
        viewer.configure_tracking(CameraTrackingConfig(smoothing=0))
        viewer.track_node(node.node_id)
        for position in ((1, 2, 8), (-1, 1, 3)):
            root.set_pose(position)
            viewer.sync()
            assert viewer.session.camera.target == pytest.approx((*position[:2], 1))
        viewer.session.submit(cmd.Select(other.object_id))
        viewer.sync()
        assert viewer.tracking_node_id == node.node_id
        viewer.configure_tracking(CameraTrackingConfig(axes="xyz", smoothing=0))
        root.set_pose((2, 3, 4))
        viewer.sync()
        assert viewer.session.camera.target == pytest.approx((2, 3, 4))
        viewer.configure_tracking(replace(viewer.app.camera_tracker.config, smoothing=0.25))
        root.set_pose((20, 30, 40))
        viewer.sync()
        assert 2 < viewer.session.camera.target[0] < 20
        root.remove()
        viewer.sync()
        assert viewer.tracking_node_id is None


@pytest.mark.parametrize(
    "entrypoint,pose",
    [
        ("hierarchy", "offset"),
        ("viewport", "offset"),
        ("hierarchy", "away"),
        ("hierarchy", "below"),
    ],
)
def test_double_click_focuses_from_the_current_side_and_above(
    tmp_path, monkeypatch, backend_name, entrypoint, pose
):
    from itertools import product

    from PIL import Image

    from mojive.interaction.gizmo import project
    from mojive.tools.camera_tracking import click
    from mojive.tools.ui_runtime import _item_center
    from mojive.ui.camera import FOCUS_DURATION

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    center = np.array((2.0, -1.5, 1.0))
    half = np.full(3, 0.3 / np.sqrt(3))
    scene = Scene()
    entity = scene.box(name="offset_target", position=center, size=half, color=(0.2, 0.7, 0.9, 1))
    scene.box(
        name="context", position=(-1, 1, -0.5), size=(0.3, 0.3, 0.3), color=(0.7, 0.4, 0.2, 1)
    )
    corners = center + np.array(list(product((-1, 1), repeat=3))) * half
    with build_scene(scene, vsync=False, width=1440, height=1000, show_window=False) as viewer:
        for _ in range(5):
            viewer.sync()
        initial_eye = center + np.array((0, 0, -4)) if pose == "below" else np.array((6, 0, 0))
        initial_target = (
            initial_eye + np.array((2, 0, 3)) if pose in ("away", "below") else np.zeros(3)
        )
        viewer.set_camera(CameraView(eye=initial_eye, target=initial_target))
        viewer.sync()
        # Drive only the animation clock deterministically; picking, layout and
        # presentation still run through the production native input/frame path.
        monkeypatch.setattr(viewer.app, "_advance_camera", lambda _dt: None)
        node = viewer.session.node_by_object_id(entity.object_id)
        if entrypoint == "hierarchy":
            viewer.panels.get("Hierarchy")._filter = node.name
            for _ in range(3):
                viewer.sync()
            cursor = _item_center(viewer, "invisible_button", f"##hierarchy-node-{node.node_id}")
        else:
            cursor = project(viewer.session.camera, [center], viewer.app._viewport_rect)[0, :2]
            assert viewer.app._pick_at(tuple(cursor)) == node.object_id
        click(viewer, cursor)
        click(viewer, cursor)
        assert viewer.session.selected == entity.object_id
        assert viewer.app.camera.animating
        output = Path("output/camera-focus")
        output.mkdir(parents=True, exist_ok=True)
        captures = []
        animation = []
        eyes = []
        pitches = []
        previous_rotation = viewer.session.camera.view_matrix()[:3, :3]
        for frame in range(25):
            view = viewer.session.camera
            rect = viewer.app._viewport_rect
            pixels = project(view, corners, rect)[:, :2]
            if pose == "offset" or frame == 24:
                assert np.all(pixels >= np.array(rect[:2]) - 1e-3)
                assert np.all(pixels <= np.array(rect[:2]) + np.array(rect[2:]) + 1e-3)
            rotation = view.view_matrix()[:3, :3]
            assert np.isfinite(rotation).all()
            assert rotation[0, 2] == pytest.approx(0, abs=1e-6)
            eyes.append(view.eye.copy())
            pitches.append(np.arcsin(np.clip(-view.forward()[2], -1, 1)))
            turn = np.arccos(np.clip((np.trace(previous_rotation.T @ rotation) - 1) * 0.5, -1, 1))
            assert np.degrees(turn) < 30
            previous_rotation = rotation
            if pose != "below":
                assert np.dot((view.eye - center)[:2], (initial_eye - center)[:2]) > 0
            for _ in range(3):
                viewer.sync()
            path = output / f"{backend_name}-{entrypoint}-{pose}-{frame:02d}.png"
            viewer.capture(path, surface="viewport")
            with Image.open(path) as image:
                image.thumbnail((720, 540))
                animation.append(image.copy())
                if frame in (0, 6, 12, 18, 24):
                    image.thumbnail((480, 360))
                    captures.append(image.copy())
            viewer.app.camera.advance(FOCUS_DURATION / 24, viewer.app.camera_out)
            viewer.sync()
        assert viewer.app.camera.pivot == pytest.approx(center, abs=1e-5)
        assert viewer.app.camera.pitch == pytest.approx(30, abs=1e-5)
        eyes = np.asarray(eyes)
        travel = eyes[-1] - eyes[0]
        assert np.cross(eyes - eyes[0], travel) == pytest.approx(0, abs=1e-5)
        assert np.all(np.diff(eyes, axis=0) * travel >= -1e-7)
        assert np.all(np.diff(pitches) * (pitches[-1] - pitches[0]) >= -1e-7)
        offset = viewer.session.camera.eye - center
        if pose != "below":
            assert offset[:2] / np.linalg.norm(offset[:2]) == pytest.approx(
                (initial_eye - center)[:2] / np.linalg.norm((initial_eye - center)[:2]), abs=1e-5
            )
        width, height = captures[0].size
        strip = Image.new("RGB", (width * len(captures), height))
        for index, image in enumerate(captures):
            strip.paste(image, (width * index, 0))
        strip.save(output / f"{backend_name}-{entrypoint}-{pose}-sequence.png")
        animation[0].save(
            output / f"{backend_name}-{entrypoint}-{pose}-slow.gif",
            save_all=True,
            append_images=animation[1:],
            duration=[600] + [40] * 23 + [1000],
            loop=0,
        )
        # GIF timestamps use centiseconds; preserve the configured transition
        # duration so review does not rely only on slow-motion captures.
        durations = np.diff(np.rint(np.linspace(0, FOCUS_DURATION * 100, len(animation)))) * 10
        durations[0] += 600
        animation[0].save(
            output / f"{backend_name}-{entrypoint}-{pose}-realtime.gif",
            save_all=True,
            append_images=animation[1:],
            duration=[*durations.astype(int), 1000],
            loop=0,
        )
