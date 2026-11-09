"""Camera output metadata survives authoring and feeds the scene capture owner."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mojive.capture import CaptureSurface
from mojive.scene.io import _camera_document, _camera_from_document
from mojive.scene.state import apply_camera_bookmark, camera_bookmark
from mojive.types import CameraView
from mojive.ui.app.capture import _Capture


def test_camera_output_roundtrips_and_old_documents_keep_viewport_size():
    view = CameraView(output_size=(1920, 1080))
    assert _camera_from_document(_camera_document(view)).output_size == (1920, 1080)
    assert apply_camera_bookmark(camera_bookmark(None, view), None).output_size == (1920, 1080)
    document = _camera_document(view)
    del document["output_size"]
    assert _camera_from_document(document).output_size is None


@pytest.mark.parametrize("size", ((0, 100), (100,), (12.5, 10), (True, 10)))
def test_output_size_rejects_invalid_dimensions(size):
    with pytest.raises(ValueError, match="positive integers"):
        CameraView(output_size=size)


def test_scene_capture_uses_camera_size_and_explicit_capture_size_takes_priority():
    requests = []
    view = CameraView(output_size=(1280, 720))
    app = SimpleNamespace(
        window=SimpleNamespace(points_to_pixels=lambda rect: rect),
        _viewport_rect=(0, 0, 600, 400),
        _camera_view=lambda: view,
        backend=None,
        session=None,
        _scene_capture=SimpleNamespace(read=lambda *args, **kwargs: requests.append(kwargs)),
    )
    _Capture._surface_image(app, CaptureSurface.SCENE, None)
    assert requests[-1]["size"] == (1280, 720)
    app._fixed_render_size = (320, 240)
    _Capture._surface_image(app, CaptureSurface.SCENE, None)
    assert requests[-1]["size"] == (320, 240)
    app._fixed_render_size = None
    view = replace(view, output_size=None)
    _Capture._surface_image(app, CaptureSurface.SCENE, None)
    assert requests[-1]["size"] == (600, 400)


@pytest.mark.physics
def test_model_camera_optics_output_undo_and_mjcf_roundtrip(tmp_path):
    from mojive import commands as cmd
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.workspace import WorkspaceAdapter
    from mojive.assets import resolve
    from mojive.session import Session

    session = Session(WorkspaceAdapter(MuJoCoAdapter(resolve("joint_gizmo"))))
    try:
        initial = session.camera_view(0)
        edited = replace(
            initial,
            focal_length=np.array((0.05, 0.05), np.float32),
            sensor_size=np.array((0.036, 0.024), np.float32),
            output_size=(640, 480),
        )
        assert session.submit(cmd.SetSceneCamera(0, edited)).ok
        assert session.camera_view(0).output_size == (640, 480)
        assert session.submit(cmd.Undo()).ok
        assert session.camera_view(0).output_size == initial.output_size
        assert session.submit(cmd.Redo()).ok
        path = tmp_path / "camera.xml"
        assert session.submit(cmd.SaveScene(path)).ok
        restored = MuJoCoAdapter(path)
        try:
            actual = restored.camera_view(0)
            assert actual.output_size == (640, 480)
            np.testing.assert_allclose(actual.focal_length, edited.focal_length)
            np.testing.assert_allclose(actual.sensor_size, edited.sensor_size)
        finally:
            restored.release()
    finally:
        session.release()
