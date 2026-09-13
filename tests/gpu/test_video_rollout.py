"""Run the public rollout example through real rendering and video decoding."""

from __future__ import annotations

import contextlib
import runpy
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from mojive import Renderer  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("fps", (8.0, 200.0))
def test_rollout_example_streams_annotated_rgb_at_video_not_physics_rate(
    tmp_path, monkeypatch, fps
):
    path = tmp_path / "falling.xml"
    path.write_text(
        """<mujoco>
          <option timestep="0.01"/>
          <worldbody>
            <light pos="0 -2 4"/>
            <geom type="plane" size="3 3 .1"/>
            <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
          </worldbody>
        </mujoco>""",
        encoding="utf-8",
    )
    output = tmp_path / "rollout.mp4"
    program = runpy.run_path(str(Path(__file__).parents[2] / "examples/mujoco_video.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mujoco_video.py",
            str(path),
            "--output",
            str(output),
            "--width",
            "159",
            "--height",
            "119",
            "--frames",
            "5",
            "--fps",
            str(fps),
            "--label",
            "Policy A",
        ],
    )
    times = []
    original = Renderer.update_scene

    def update(renderer, data, *args, **kwargs):
        times.append(float(data.time))
        assert renderer.model.opt.timestep == 0.01
        return original(renderer, data, *args, **kwargs)

    monkeypatch.setattr(Renderer, "update_scene", update)
    with pytest.warns(RuntimeWarning, match="edge-padded"):
        program["main"]()
    assert len(times) == 5
    for index, time in enumerate(times):
        assert time >= index / fps - 1e-10
        assert time < index / fps + 0.01 + 1e-10
    with contextlib.closing(imageio_ffmpeg.read_frames(str(output))) as reader:
        metadata = next(reader)
        frames = [np.frombuffer(frame, np.uint8).reshape(120, 160, 3) for frame in reader]
    assert metadata["fps"] == pytest.approx(fps)
    assert len(frames) == 5
    assert metadata["size"] == (160, 120)
    assert np.ptp(frames[0]) > 100
    assert np.any(np.all(frames[0][5:30, 5:150] > 200, axis=2))  # subtitle pixels
    if fps == 8.0:
        assert not np.array_equal(frames[0][35:], frames[-1][35:])
