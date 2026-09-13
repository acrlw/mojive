"""Native controls and displayed-pose tracking in both rendering backends."""

from dataclasses import replace

import numpy as np
import pytest

from mojive import CameraTrackingConfig, ViewerConfig, build_scene
from mojive import commands as cmd
from mojive.scene import Scene
from mojive.tools.camera_tracking import exercise_controls
from mojive.types import CameraView

pytestmark = pytest.mark.gpu


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
