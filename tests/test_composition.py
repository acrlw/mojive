"""Ownership tests for the high-level viewer composition object."""

from __future__ import annotations

import pytest

from forge_viewer.composition import Viewer


class Resource:
    def __init__(self) -> None:
        self.releases = 0
        self.closes = 0

    def release(self) -> None:
        self.releases += 1

    def close(self) -> None:
        self.closes += 1


class App:
    def __init__(self, backend, session, bridge) -> None:
        self.backend = backend
        self.session = session
        self.bridge = bridge
        self.runs = 0
        self.releases = 0

    def run(self, max_frames=None) -> None:
        del max_frames
        self.runs += 1

    def release(self) -> None:
        self.releases += 1
        self.bridge.close()
        self.backend.release()
        self.session.release()


def _viewer():
    backend = Resource()
    session = Resource()
    bridge = Resource()
    window = Resource()
    app = App(backend, session, bridge)
    return Viewer(app, session, backend, window, bridge), app, backend, session, bridge, window


def test_viewer_run_does_not_destroy_resources_before_the_caller_is_done():
    viewer, app, backend, session, bridge, window = _viewer()

    viewer.run(max_frames=1)

    assert app.runs == 1
    assert app.releases == 0
    assert backend.releases == session.releases == bridge.closes == window.closes == 0


def test_viewer_context_manager_releases_every_owner_once():
    viewer, app, backend, session, bridge, window = _viewer()

    with viewer as entered:
        assert entered is viewer
    viewer.release()

    assert app.releases == 1
    assert backend.releases == 1
    assert session.releases == 1
    assert bridge.closes == 1
    assert window.closes == 1


@pytest.mark.parametrize("frames", (0.5, "3"))
def test_viewer_record_rejects_non_integer_frame_counts(tmp_path, frames):
    viewer, *_ = _viewer()

    with pytest.raises(TypeError, match="frame count must be an integer"):
        viewer.record(tmp_path / "capture.mp4", frames)


@pytest.mark.parametrize("fps", (0.0, -1.0, float("nan"), float("inf")))
def test_viewer_record_rejects_invalid_frame_rates(tmp_path, fps):
    viewer, *_ = _viewer()

    with pytest.raises(ValueError, match="frame rate must be finite and positive"):
        viewer.record(tmp_path / "capture.mp4", 1, fps=fps)
