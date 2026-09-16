"""Real rendering of MuJoCo viewer user_scn geometry through both scene backends."""

import numpy as np
import pytest

from mojive import SceneRenderer
from mojive.app.mujoco_viewer.visuals import draw_user_geometries, snapshot_geometries

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("backend", ["opengl", "wgpu"])
def test_mujoco_user_scene_spheres_render_and_clear(backend):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    scene = mujoco.MjvScene(model, maxgeom=2)
    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [0.3, 0, 0],
        [0, 0, 0],
        np.eye(3).reshape(-1),
        [1, 0, 0, 1],
    )
    scene.ngeom = 1
    with SceneRenderer(width=96, height=72, samples=0, renderer=backend) as renderer:
        empty = renderer.render().copy()
        draw_user_geometries(renderer.debug, snapshot_geometries(scene))
        red = renderer.render().copy()
        assert np.any(red != empty)
        assert red[36, 48, 0] > red[36, 48, 2] + 30
        scene.geoms[0].rgba[:] = (0, 0, 1, 1)
        draw_user_geometries(renderer.debug, snapshot_geometries(scene))
        blue = renderer.render().copy()
        assert blue[36, 48, 2] > blue[36, 48, 0] + 30
        scene.ngeom = 0
        draw_user_geometries(renderer.debug, snapshot_geometries(scene))
        np.testing.assert_array_equal(renderer.render(), empty)
