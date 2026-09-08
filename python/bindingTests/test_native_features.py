"""Native GPU cache invalidation and camera-dependent public behavior."""

from dataclasses import replace

import numpy as np
from PIL import Image

from mojive import (
    CameraView,
    Light,
    LightSet,
    Material,
    RenderFlag,
    RenderProduct,
    Scene,
    SceneRenderer,
)


def test_static_shadow_and_reflection_cache_invalidates_on_pose_and_light_changes(tmp_path):
    light = Light(direction=np.array([0.3, 0.5, -1]), diffuse=np.array([0.8, 0.8, 0.8]))
    scene = Scene(lights=LightSet(lights=(light,)))
    scene.plane(size=(3, 3, 0.01), material=Material(reflectance=0.4))
    box = scene.box(position=(0, 0, 0.6), color=(0.8, 0.3, 0.2, 1))
    camera = CameraView(eye=np.array([3, -4, 3]), target=np.array([0, 0, 0.3]))
    with SceneRenderer(
        scene.source, width=200, height=160, renderer="bgfx", samples=0, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        runtime = renderer._backend.runtime
        runtime.advance()
        first = renderer.render()
        initial = renderer._backend.target.frame.statistics
        second = renderer.render()
        cached = renderer._backend.target.frame.statistics
        np.testing.assert_array_equal(first, second)
        assert cached.draw_calls < initial.draw_calls
        box.set_pose((0.8, 0, 0.6))
        renderer.update(scene.frame)
        moved = renderer.render()
        changed = renderer._backend.target.frame.statistics
        assert changed.draw_calls > cached.draw_calls
        assert np.count_nonzero(np.max(np.abs(first.astype(int) - moved), axis=2) > 5) > 300
        frame = scene.frame
        frame.lights = LightSet(lights=(replace(light, direction=np.array([-0.8, 0.3, -1])),))
        renderer.update(frame)
        relit = renderer.render()
        assert renderer._backend.target.frame.statistics.draw_calls > cached.draw_calls
        assert np.count_nonzero(np.max(np.abs(relit.astype(int) - moved), axis=2) > 5) > 100
        renderer.set_flag(RenderFlag.SHADOW, False)
        without = renderer.render()
        assert not np.array_equal(without, relit)
        renderer.set_flag(RenderFlag.SHADOW, True)
        np.testing.assert_array_equal(renderer.render(), relit)


def test_capture_rebuilds_camera_dependent_overlay_and_preserves_target(tmp_path):
    scene = Scene()
    scene.box()
    with SceneRenderer(scene.source, width=160, height=120, renderer="bgfx", samples=0) as renderer:
        renderer.update(scene.frame)
        backend = renderer._backend
        backend.debug.layer("test").text("label", (0, 0, 0.5), "Native label", (0.9, 0.2, 0.1, 1))
        expected = renderer.render()
        ids = renderer.render(product=RenderProduct.OBJECT_ID)
        camera = replace(backend._camera, eye=np.array([4, -3, 2]))
        path = tmp_path / "capture.png"
        assert backend.capture(path, camera=camera, size=(220, 180))
        capture = np.asarray(Image.open(path))
        assert capture.shape == (180, 220, 3) and capture.std() > 10
        np.testing.assert_array_equal(renderer.render(), expected)
        np.testing.assert_array_equal(renderer.render(product=RenderProduct.OBJECT_ID), ids)


def test_pose_and_camera_updates_keep_resources_and_pending_readbacks():
    scene = Scene()
    box = scene.box(color=(0.8, 0.2, 0.1, 1))
    with SceneRenderer(scene.source, width=160, height=120, renderer="bgfx", samples=0) as renderer:
        renderer.update(scene.frame)
        first = renderer.render()
        backend = renderer._backend
        runtime = backend.runtime
        ticket = runtime.readback(backend.target.frame, backend.api.Product.COLOR)
        revision = backend._revision
        box.set_pose((0.8, 0, 0))
        renderer.update(scene.frame, camera=replace(backend._camera, eye=np.array([4, -3, 2])))
        moved = renderer.render()
        assert backend._revision == revision
        assert not np.array_equal(first, moved)
        captured = runtime.wait(ticket)
        assert captured.state == backend.api.ReadbackState.READY
        np.testing.assert_array_equal(captured.image, first)


def test_msaa_switch_matches_single_sample_and_restores_coverage():
    scene = Scene()
    scene.box(color=(0.8, 0.3, 0.1, 1))
    camera = CameraView(eye=np.array([3.2, -4, 2.7]), target=np.zeros(3))
    with SceneRenderer(
        scene.source, width=160, height=120, renderer="bgfx", samples=4, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        smooth = renderer.render()
        backend = renderer._backend
        ticket = backend.runtime.readback(backend.target.frame, backend.api.Product.COLOR)
        renderer.set_flag(RenderFlag.MSAA, False)
        sharp = renderer.render()
        prior = backend.runtime.wait(ticket)
        assert prior.state == backend.api.ReadbackState.READY
        np.testing.assert_array_equal(prior.image, smooth)
        with SceneRenderer(
            scene.source, width=160, height=120, renderer="bgfx", samples=0, camera=camera
        ) as single:
            single.update(scene.frame)
            np.testing.assert_array_equal(sharp, single.render())
        assert np.count_nonzero(np.max(np.abs(sharp.astype(float) - smooth), axis=2) > 5) > 50
        renderer.set_flag(RenderFlag.MSAA, True)
        np.testing.assert_array_equal(renderer.render(), smooth)
