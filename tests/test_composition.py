"""Ownership tests for the high-level viewer composition object."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest

from mojive.composition import Viewer


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
        self.syncs = 0
        self.releases = 0
        self.fixed_render_size = None
        self.fixed_render_size_changes = []

    def run(self, max_frames=None) -> None:
        del max_frames
        self.runs += 1

    def sync(self) -> None:
        self.syncs += 1

    def release(self) -> None:
        self.releases += 1
        self.bridge.close()
        self.backend.release()
        self.session.release()

    def set_fixed_render_size(self, width, height) -> None:
        self.fixed_render_size = (width, height)
        self.fixed_render_size_changes.append(self.fixed_render_size)

    def clear_fixed_render_size(self) -> None:
        self.fixed_render_size = None
        self.fixed_render_size_changes.append(None)


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


@pytest.mark.parametrize("previous_size", (None, (800, 600)))
def test_viewer_record_restores_the_previous_render_size_on_failure(tmp_path, previous_size):
    viewer, app, *_ = _viewer()
    app.fixed_render_size = previous_size

    def fail_before_frame(_index, _viewer):
        raise RuntimeError("stop recording")

    with pytest.raises(RuntimeError, match="stop recording"):
        viewer.record(
            tmp_path / "capture.mp4",
            1,
            size=(320, 240),
            before_frame=fail_before_frame,
        )

    assert app.fixed_render_size == previous_size
    assert app.fixed_render_size_changes == [(320, 240), previous_size]


def test_viewer_record_restores_render_size_after_success(tmp_path, monkeypatch):
    from mojive import recording

    viewer, app, backend, *_ = _viewer()
    backend.target = type(
        "Target",
        (),
        {"read_color": lambda self, flip=True: np.zeros((24, 32, 4), np.uint8)},
    )()
    recorders = []

    class Recorder:
        def __init__(self, path, size, fps):
            self.path, self.size, self.fps = path, size, fps
            self.frames = []
            self.closed = 0
            recorders.append(self)

        def append(self, frame):
            self.frames.append(frame)

        def close(self):
            self.closed += 1

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)

    output = viewer.record(tmp_path / "capture.mp4", 2, size=(320, 240))

    assert output == tmp_path / "capture.mp4"
    assert app.syncs == 2
    assert app.fixed_render_size is None
    assert app.fixed_render_size_changes == [(320, 240), None]
    assert len(recorders[0].frames) == 2
    assert recorders[0].closed == 1


def test_viewer_record_pipelines_async_readback_in_frame_order(tmp_path, monkeypatch):
    from mojive import recording

    viewer, app, backend, *_ = _viewer()

    class Target:
        def __init__(self):
            self.index = 0

        def read_rgb_async(self, flip=True):
            assert flip
            future = Future()
            future.set_result(np.full((4, 6, 3), self.index, np.uint8))
            self.index += 1
            return future

    backend.target = Target()
    recorders = []

    class Recorder:
        def __init__(self, path, size, fps):
            self.path, self.size, self.fps = path, size, fps
            self.values = []
            self.closed = 0
            recorders.append(self)

        def append(self, frame):
            self.values.append(int(frame[0, 0, 0]))

        def close(self):
            self.closed += 1

    monkeypatch.setattr(recording, "VideoRecorder", Recorder)

    viewer.record(tmp_path / "capture.mp4", 5)

    assert app.syncs == 5
    assert recorders[0].values == [0, 1, 2, 3, 4]
    assert recorders[0].closed == 1
