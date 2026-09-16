"""Camera preview resources can be rebuilt after release."""

from unittest.mock import Mock

from mojive.adapters.base import SceneFrame, SceneSource
from mojive.types import CameraView
from mojive.ui.camera_preview import CameraPreview


def test_preview_reloads_scene_into_recreated_peer():
    main = Mock()
    main.render_options.return_value = ()
    peers = [Mock(), Mock()]
    main.create_peer.side_effect = peers
    source, frame, camera = SceneSource(), SceneFrame(), CameraView()
    preview = CameraPreview()
    preview.set_enabled(True)
    for peer in peers:
        preview.update(main, source, 7, frame, camera, (320, 180))
        peer.set_scene.assert_called_once_with(source)
        preview.release()
        preview.release()
        peer.release.assert_called_once()
