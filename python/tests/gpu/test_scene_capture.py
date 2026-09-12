"""Scene exports must not inherit any displayed editor annotations."""

import time

import numpy as np
import pytest
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.application.composition import build_scene
from mojive.tools.scene_entities import acceptance_scene
from mojive.types import CameraView

pytestmark = pytest.mark.gpu


def test_scene_capture_and_video_exclude_helpers_and_selection(tmp_path, monkeypatch):
    from mojive.capture.recording import VideoRecorder

    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    recorded = []
    append = VideoRecorder.append

    def save_frame(recorder, image):
        recorded.append(image.copy())
        append(recorder, image)

    monkeypatch.setattr(VideoRecorder, "append", save_frame)
    with build_scene(
        acceptance_scene(), vsync=False, show_window=False, width=1100, height=800
    ) as viewer:
        for _ in range(3):
            viewer.sync()
        viewer.set_camera(
            CameraView(
                eye=np.array((5.6, -7.2, 4.8), np.float32),
                target=np.array((0.0, 0.1, 0.9), np.float32),
                near=0.02,
                far=30,
            )
        )
        for _ in range(5):
            viewer.sync()
        clean = viewer.capture_array(surface="scene")
        peer = viewer.app._scene_capture._backend
        for kind in (NodeType.CAMERA, NodeType.LIGHT, NodeType.GEOM):
            node = next(
                n
                for n in viewer.session.nodes
                if n.type is kind and (kind is not NodeType.GEOM or n.name == "subject.geom")
            )
            assert viewer.session.submit(cmd.SelectNode(node.node_id))
            assert viewer.session.selected_node is not None
            viewer.app.gizmo.set_mode("translate")
            for _ in range(3):
                viewer.sync()
            image = viewer.capture_array(surface="scene")
            np.testing.assert_array_equal(image, clean)
            displayed = viewer.backend.target.read_rgb(flip=True)
            assert np.count_nonzero(displayed != clean) > 100
        assert viewer.app._scene_capture._backend is peer
        path = tmp_path / "scene-only.mp4"
        viewer.start_recording(path, surface="scene", fps=30, countdown=0)
        for _ in range(4):
            viewer.app._last_time = time.perf_counter() - 1 / 30
            viewer.sync()
        viewer.stop_recording()
        assert path.is_file() and recorded
        for frame in recorded:
            np.testing.assert_array_equal(frame, clean)
        recorded.clear()
        viewer.record(tmp_path / "scene-batch.mp4", frames=3)
        assert len(recorded) == 3
        for frame in recorded:
            np.testing.assert_array_equal(frame, clean)
        Image.fromarray(clean).save(tmp_path / "scene.png")
        viewer.capture(tmp_path / "window.png", surface="window")
