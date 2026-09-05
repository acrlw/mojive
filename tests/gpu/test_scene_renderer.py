"""Backend-neutral images, provider updates, and offscreen context ownership."""

from __future__ import annotations

import numpy as np
import pytest

from mojive import RenderProduct, Scene, SceneRenderer
from mojive.adapters.static import StaticSceneAdapter

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("backend", ["opengl", "wgpu"])
def test_authored_scene_outputs_follow_pose_and_structure(backend):
    scene = Scene()
    box = scene.box(color=(1, 0, 0, 1))
    adapter = StaticSceneAdapter(scene)
    with SceneRenderer(width=96, height=72, samples=0, renderer=backend) as renderer:
        renderer.update_from(adapter)
        rgb = renderer.render()
        assert rgb.shape == (72, 96, 3) and rgb.dtype == np.uint8
        assert rgb[36, 48, 0] > rgb[36, 48, 1] + 30
        ids = renderer.render(product=RenderProduct.OBJECT_ID)
        assert ids.dtype == np.uint32 and ids[36, 48] == box.object_id
        depth = renderer.render(product=RenderProduct.METRIC_DEPTH)
        assert depth.dtype == np.float32 and depth[36, 48] == pytest.approx(3.5, abs=0.01)
        out = np.empty((72, 96, 3), np.uint8)
        assert renderer.render(out=out) is out
        np.testing.assert_array_equal(out, rgb)
        strided = np.empty((72, 192, 3), np.uint8)[:, ::2]
        assert renderer.render(out=strided) is strided
        np.testing.assert_array_equal(strided, rgb)

        # All products use top-left orientation: raising the box moves it upward.
        box.set_pose((0, 0, 0.7))
        renderer.update_from(adapter)
        raised = renderer.render(product=RenderProduct.OBJECT_ID)
        assert np.nonzero(raised == box.object_id)[0].mean() < 30
        box.remove()
        renderer.update_from(adapter)
        assert not renderer.render(product=RenderProduct.OBJECT_ID).any()
        renderer.resize(64, 48)
        assert renderer.render().shape == (48, 64, 3)
    assert adapter.scene is scene
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render()
    renderer.close()


@pytest.mark.parametrize("backend", ["opengl", "wgpu"])
def test_independent_renderers_can_alternate_without_losing_context(backend):
    red, green = Scene(), Scene()
    red.box(color=(1, 0, 0, 1))
    green.box(color=(0, 1, 0, 1))
    with SceneRenderer(red.source, width=64, height=48, samples=0, renderer=backend) as a:
        a.update(red.frame)
        expected = a.render()
        with SceneRenderer(green.source, width=64, height=48, samples=0, renderer=backend) as b:
            b.update(green.frame)
            other = b.render()
            assert other[24, 32, 1] > other[24, 32, 0]
            np.testing.assert_array_equal(a.render(), expected)
            np.testing.assert_array_equal(b.render(), other)
        np.testing.assert_array_equal(a.render(), expected)
