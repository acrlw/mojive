"""Backend-neutral spatial-tendon rendering tests.

Runs under both backends via the ``backend_name`` fixture (conftest.py);
scene setup mirrors test_horizon_haze.py: a handwritten MJCF rendered through
OffscreenHarness.  Two sphere bodies are joined by a spatial tendon that
bends over a fixed world site, so the capsule chain has two visible segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge_viewer.adapters.base import FrameNeeds
from forge_viewer.render.backend import RenderFlag
from forge_viewer.tools._harness import OffscreenHarness
from forge_viewer.types import CameraView

pytestmark = pytest.mark.gpu

W, H = 320, 240

MJCF = """
<mujoco>
  <worldbody>
    <geom type="plane" size="4 4 0.1" rgba="0.25 0.27 0.30 1"/>
    <site name="pulley" pos="0 0 1.15"/>
    <body pos="-0.85 0 0.35">
      <geom type="sphere" size="0.18" rgba="0.75 0.25 0.20 1"/>
      <site name="left" pos="0 0 0.10"/>
    </body>
    <body pos="0.85 0 0.35">
      <geom type="sphere" size="0.18" rgba="0.20 0.30 0.75 1"/>
      <site name="right" pos="0 0 0.10"/>
    </body>
  </worldbody>
  <tendon>
    <spatial width="0.035" rgba="0.95 0.85 0.15 1">
      <site site="left"/>
      <site site="pulley"/>
      <site site="right"/>
    </spatial>
  </tendon>
</mujoco>
"""


def _camera() -> CameraView:
    return CameraView(
        eye=np.array([0.0, -3.4, 1.3], np.float32),
        target=np.array([0.0, 0.0, 0.55], np.float32),
        up=np.array([0.0, 0.0, 1.0], np.float32),
        near=0.01,
        far=50.0,
    )


@pytest.fixture
def harness(tmp_path, backend_name):
    scene = tmp_path / "tendon.xml"
    scene.write_text(MJCF, encoding="utf-8")
    with OffscreenHarness(scene, W, H) as h:
        assert h.backend.caps.name == ("wgpu" if backend_name == "wgpu" else "forge")
        h.needs = FrameNeeds(poses=True, tendons=True)
        camera = _camera()
        h.camera = camera
        h.backend.set_camera(camera)
        h.warmup(2)
        yield h


def _capsule_count(backend) -> int:
    """Segment count lives on the tendon pass in both backends (no public API)."""
    if backend.caps.name == "wgpu":
        return backend._tendons.capsule_count
    return backend._passes["tendon"].capsule_count


def _render(harness) -> np.ndarray:
    harness.step_and_render(0)
    return harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def _tendon_mask(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    return np.max(np.abs(on - off), axis=2) > 5


def test_tendon_chain_is_visible(harness):
    backend = harness.backend
    backend.set_flag(RenderFlag.TENDON, False)
    off = _render(harness)
    backend.set_flag(RenderFlag.TENDON, True)
    on = _render(harness)

    mask = _tendon_mask(on, off)
    assert int(np.count_nonzero(mask)) > 200
    rows, cols = np.nonzero(mask)
    # The chain bends over the pulley site: two arms span the screen center
    # and the apex sits above the horizon line.
    assert cols.min() < W // 2 < cols.max()
    assert rows.min() < H // 2
    # The lit tendon keeps its yellow hue (r > b) on both backends.
    pixels = on[mask]
    assert float(pixels[:, 0].mean()) > float(pixels[:, 2].mean()) + 10.0


def test_tendon_flag_off_restores_the_baseline(harness):
    backend = harness.backend
    backend.set_flag(RenderFlag.TENDON, False)
    off = _render(harness)
    backend.set_flag(RenderFlag.TENDON, True)
    on = _render(harness)
    assert int(np.count_nonzero(_tendon_mask(on, off))) > 200

    backend.set_flag(RenderFlag.TENDON, False)
    off_again = _render(harness)
    assert np.array_equal(off, off_again)


def test_tendon_segments_pack_three_capsules_each(harness):
    harness.step_and_render(0)
    # Two wrap segments (left->pulley, pulley->right); the pass stores the
    # segment count, three capsule instances each.
    assert _capsule_count(harness.backend) == 2
