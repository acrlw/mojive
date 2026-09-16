"""Scene capture inherits rendering choices independently of editor UI and target size."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from mojive.capture import CaptureSurface
from mojive.types import CameraView
from mojive.ui.app.capture import _Capture
from mojive.ui.scene_capture import SceneCapture


def test_capture_copies_visual_state_and_reuses_peer():
    main = Mock()
    main.render_options.return_value = ("inertia", "scaled_inertia", "contactforce")
    main.get_flag.return_value = True
    main.target.width, main.target.height = 1920, 1080
    peer = main.create_peer.return_value
    session = SimpleNamespace(source=object(), frame=object(), structure_generation=1)
    capture = SceneCapture()
    camera = CameraView()
    try:
        for _ in range(2):
            capture.render(main, session, camera, size=(640, 360))
        main.create_peer.assert_called_once_with(640, 360)
        peer.set_scene.assert_called_once_with(session.source)
        for flag in main.render_options():
            peer.set_flag.assert_any_call(flag, True)
        for name in (
            "debug_view",
            "label_mode",
            "frame_mode",
            "bvh_depth",
            "geometry_style",
            "background",
        ):
            getattr(peer, f"set_{name}").assert_called_with(getattr(main, f"get_{name}")())
        main.resize.assert_not_called()
    finally:
        capture.release()


def test_scene_default_size_matches_viewport_crop_at_fractional_dpi():
    capture = Mock()
    app = SimpleNamespace(
        window=SimpleNamespace(points_to_pixels=lambda rect: tuple(v * 1.5 for v in rect)),
        _viewport_rect=(3.25, 5.25, 201.5, 121.5),
        _scene_capture=capture,
        backend=object(),
        session=object(),
        _camera_view=lambda: CameraView(),
    )
    presented = np.zeros((300, 500, 4), np.uint8)
    viewport = _Capture._surface_image(app, CaptureSurface.VIEWPORT, presented)
    _Capture._surface_image(app, CaptureSurface.SCENE, None)
    assert capture.read.call_args.kwargs["size"] == (viewport.shape[1], viewport.shape[0])
    app._fixed_render_size = (640, 480)
    _Capture._surface_image(app, CaptureSurface.SCENE, None)
    assert capture.read.call_args.kwargs["size"] == (640, 480)
