"""GPU coverage for MuJoCo horizon haze semantics."""

from __future__ import annotations

import numpy as np
import pytest

from forge_viewer.render.backend import RenderFlag
from forge_viewer.tools._harness import OffscreenHarness
from forge_viewer.types import CameraView

pytestmark = pytest.mark.gpu


def test_mujoco_haze_changes_sky_below_horizon_without_fogging_objects(tmp_path):
    scene = tmp_path / "horizon-haze.xml"
    scene.write_text(
        """
        <mujoco>
          <visual>
            <rgba haze="0.9 0.5 0.1 1"/>
            <map haze="0.3"/>
          </visual>
          <asset>
            <texture type="skybox" builtin="gradient" width="64" height="384"
                     rgb1="0.05 0.15 0.35" rgb2="0.3 0.55 0.9"/>
          </asset>
          <worldbody>
            <geom type="plane" size="0 0 .05" rgba=".2 .25 .3 1"/>
            <geom pos="0 0 .5" type="sphere" size=".5" rgba=".2 .8 .4 1"/>
            <camera pos="0 -4 4" xyaxes="1 0 0 0 .707 .707"/>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    with OffscreenHarness(scene, 320, 240) as harness:
        camera = CameraView(
            eye=np.array([0.0, -5.0, 1.0], np.float32),
            target=np.array([0.0, 0.0, 0.7], np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            near=0.01,
            far=50.0,
        )
        harness.camera = camera
        harness.backend.set_camera(camera)
        harness.backend.set_flag(RenderFlag.SKYBOX, True)
        harness.backend.set_flag(RenderFlag.HAZE, False)
        harness.warmup(2)
        clear = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)

        harness.backend.set_flag(RenderFlag.HAZE, True)
        harness.step_and_render(0)
        haze = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)

    difference = np.max(np.abs(haze - clear), axis=2)
    sphere = (clear[..., 1] > clear[..., 0] + 20) & (clear[..., 1] > clear[..., 2] + 20)
    assert np.count_nonzero(sphere) > 100
    assert np.count_nonzero(difference > 5) > 100
    assert np.count_nonzero(difference[sphere]) == 0
