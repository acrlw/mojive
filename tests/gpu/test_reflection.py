from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.physics]

pytest.importorskip("glfw")
pytest.importorskip("moderngl")
pytest.importorskip("mujoco")

from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.render.backend import RenderFlag  # noqa: E402
from forge_viewer.tools._harness import OffscreenHarness  # noqa: E402
from forge_viewer.types import CameraView  # noqa: E402

W, H = 640, 480

CAMERA = CameraView(
    eye=np.array([3.2, -3.2, 1.4], np.float32),
    target=np.array([0.0, 0.0, 0.3], np.float32),
    up=np.array([0.0, 0.0, 1.0], np.float32),
    fov_y=float(np.radians(45.0)),
    near=0.05,
    far=60.0,
    aspect=W / H,
)


@pytest.fixture(scope="module")
def rig():
    with OffscreenHarness(resolve("reflection"), W, H) as h:
        h.camera = CAMERA
        h.backend.set_camera(h.camera)
        h.warmup(4)
        yield h


def _linear(img: np.ndarray) -> np.ndarray:

    c = np.asarray(img, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def shot(rig, *, reflection: bool) -> np.ndarray:
    rig.backend.set_flag(RenderFlag.REFLECTION, reflection)
    rig.step_and_render(0)
    return rig.backend.target.read_color(flip=True)[..., :3].astype(np.int32)


@pytest.fixture(scope="module")
def pair(rig):

    return shot(rig, reflection=True), shot(rig, reflection=False)


def test_the_flag_actually_turns_reflection_on_and_off(rig, pair):

    on, off = pair
    changed = int((np.abs(on - off).sum(axis=2) > 4).sum())
    assert changed > 5000

    again = shot(rig, reflection=False)
    assert np.array_equal(again, off)


def test_reflection_only_touches_the_reflective_plane(pair):

    on, off = pair
    sky = np.abs(on[:120] - off[:120]).max()
    assert sky == 0


def test_reflection_is_added_not_mixed(pair):

    on, off = pair
    far = (slice(H - 60, H - 20), slice(W - 80, W - 20))
    diff = np.abs(on[far] - off[far]).max()
    assert diff == 0


def test_reflection_scales_with_reflectance(tmp_path):

    text = resolve("reflection").read_text(encoding="utf-8")
    assert 'reflectance="0.35"' in text

    def increment(value: str) -> float:
        path = tmp_path / f"refl_{value}.xml"
        path.write_text(text.replace('reflectance="0.35"', f'reflectance="{value}"'), "utf-8")
        with OffscreenHarness(path, W, H) as h:
            h.camera = CAMERA
            h.backend.set_camera(h.camera)
            h.backend.set_flag(RenderFlag.TONEMAP, False)
            h.warmup(4)
            lit = _linear(shot(h, reflection=True))
            dark = _linear(shot(h, reflection=False))
        band = (slice(H - 140, H - 10), slice(20, W - 20))
        return float((lit[band] - dark[band]).mean())

    low = increment("0.35")
    high = increment("0.70")

    assert low > 0.004
    ratio = high / low
    assert 1.7 < ratio < 2.3


def test_geometry_below_the_plane_stays_out_of_the_reflection(pair):

    on, off = pair
    delta = on - off
    green_excess = delta[..., 1] - np.maximum(delta[..., 0], delta[..., 2])
    leaked = int((green_excess > 3).sum())
    assert leaked == 0


def test_reflection_shows_outer_faces_not_the_inside(pair):

    on, off = pair
    mask = np.abs(on - off).sum(axis=2) > 4
    assert mask.sum() > 5000
    mean = float(on[mask].mean())
    assert mean > 65.0


def test_transparent_geometry_appears_in_reflections(tmp_path):
    scene = tmp_path / "transparent_reflection.xml"
    scene.write_text(
        """
<mujoco>
  <visual>
    <headlight ambient="0.3 0.3 0.3" diffuse="0.7 0.7 0.7"/>
  </visual>
  <asset>
    <material name="mirror" rgba="0.15 0.17 0.20 1" reflectance="0.7"/>
    <material name="glass" rgba="0.95 0.15 0.10 0.45"/>
  </asset>
  <worldbody>
    <geom type="plane" size="0 0 0.05" material="mirror"/>
    <geom type="sphere" pos="0 0 0.8" size="0.35" material="glass"/>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    with OffscreenHarness(scene, W, H) as harness:
        harness.camera = CAMERA
        harness.backend.set_camera(CAMERA)
        harness.backend.set_flag(RenderFlag.REFLECTION, True)
        harness.step_and_render(0)
        reflected = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)
        harness.backend.set_flag(RenderFlag.REFLECTION, False)
        harness.step_and_render(0)
        direct = harness.backend.target.read_color(flip=True)[..., :3].astype(np.int16)

    changed = np.abs(reflected - direct).sum(axis=2) > 6
    assert int(changed.sum()) > 500
