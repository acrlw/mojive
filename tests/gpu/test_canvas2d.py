"""Canvas fills use the shared debug GPU stream on every production backend."""

import numpy as np
import pytest

from mojive import Scene, SceneRenderer
from mojive.canvas2d import PathBuilder2D

pytestmark = pytest.mark.gpu


def test_canvas_hole_and_alpha_have_no_internal_triangle_seams(backend_name):
    scene = Scene()
    with SceneRenderer(
        scene.source, width=200, height=200, samples=4, renderer=backend_name
    ) as renderer:
        canvas = renderer.canvas2d
        layer = canvas.layer("fill")
        path = PathBuilder2D(fill_rule="evenodd")
        for lo, hi in ((-0.9, 0.9), (-0.3, 0.3)):
            path.move_to(lo, lo).line_to(hi, lo).line_to(hi, hi).line_to(lo, hi).close()
        layer.fill_path("hole", path.finish(), (1, 0, 0, 0.5))
        renderer.update(scene.frame, camera=canvas.camera((-1, -1, 1, 1), aspect=1, padding=0))
        image = renderer.render()
        assert np.max(image[80:120, 80:120, :3]) < 3
        # Interior crosses tessellation edges: one alpha application everywhere.
        strip = image[20:50, 20:180, 0]
        assert np.ptp(strip.astype(int)) <= 2
        assert 120 <= np.median(strip) <= 135
        layer.visible = False
        assert np.max(renderer.render()[:, :, :3]) < 3
        layer.visible = True
        layer.fill_path("hole", PathBuilder2D().finish(), (1, 0, 0, 0.5))
        assert np.max(renderer.render()[:, :, :3]) < 3
