"""Native GPU cache invalidation and camera-dependent public behavior."""

from dataclasses import replace

import numpy as np
import pytest
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


@pytest.mark.parametrize("direction", [(0, 0, 0), (0, 0, -1e-12), (0, 0, -7)])
def test_degenerate_light_direction_matches_the_default(direction):
    light = Light(
        direction=np.array(direction, np.float32), cast_shadow=False, diffuse=np.ones(3, np.float32)
    )
    scene = Scene(lights=LightSet(lights=(light,), ambient=np.zeros(3, np.float32)))
    scene.box(size=(0.5, 0.5, 0.5), color=(1, 1, 1, 1))
    camera = CameraView(eye=np.array([0, 0, 4]), up=np.array([0, 1, 0]))
    with SceneRenderer(
        scene.source, renderer="bgfx", width=80, height=60, samples=0, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        first = renderer.render()
        scene.set_light_at(0, replace(light, direction=np.array([0, 0, -1], np.float32)))
        renderer.update(scene.frame)
        reference = renderer.render()
        assert reference[30, 40, 0] > 200
        np.testing.assert_array_equal(first, reference)


@pytest.mark.parametrize("orthographic", [False, True])
def test_normalized_depth_keeps_the_rendered_camera(orthographic):
    scene = Scene()
    scene.box(size=(0.5, 0.5, 0.5))
    camera = CameraView(
        eye=np.array([0, 0, 4]), up=np.array([0, 1, 0]), near=0.1, far=20, orthographic=orthographic
    )
    with SceneRenderer(
        scene.source, renderer="bgfx", width=80, height=60, samples=0, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        renderer.render(product=RenderProduct.METRIC_DEPTH)
        backend = renderer._backend
        before = backend.target.read_depth()
        backend.set_camera(replace(camera, near=0.5, far=40, orthographic=not orthographic))
        np.testing.assert_array_equal(before, backend.target.read_depth())
        renderer.update(scene.frame, camera=replace(camera, near=0.5, far=40))
        renderer.render(product=RenderProduct.METRIC_DEPTH)
        assert abs(float(before[30, 40]) - float(backend.target.read_depth()[30, 40])) > 0.05


def test_public_eight_sample_target_survives_toggle_resize_and_peer():
    scene = Scene()
    scene.box()
    with SceneRenderer(scene.source, renderer="bgfx", width=80, height=60, samples=8) as renderer:
        renderer.update(scene.frame)
        backend = renderer._backend
        assert backend.caps.msaa_samples == backend.target.samples == 8
        first = renderer.render()
        renderer.set_flag(RenderFlag.MSAA, False)
        assert not np.array_equal(first, renderer.render())
        renderer.set_flag(RenderFlag.MSAA, True)
        np.testing.assert_array_equal(first, renderer.render())
        renderer.resize(100, 75)
        assert renderer.render().shape == (75, 100, 3)
        peer = backend.create_peer(80, 60)
        try:
            assert peer.target.samples == 8
            peer.set_scene(scene.source)
            peer.update(scene.frame)
            peer.render()
            assert peer.target.read_rgb().shape == (60, 80, 3)
        finally:
            peer.release()


def test_pass_timings_follow_independent_targets_and_delayed_frames():
    scene = Scene()
    box = scene.box()
    with (
        SceneRenderer(scene.source, renderer="bgfx", width=100, height=80, samples=0) as color,
        SceneRenderer(scene.source, renderer="bgfx", width=100, height=80, samples=0) as depth,
    ):
        color_submissions, depth_submissions = set(), set()
        seen_gpu = set()
        for frame in range(40):
            box.set_pose((frame * 0.001, 0, 0))
            color.update(scene.frame)
            depth.update(scene.frame)
            color.render()
            color_submissions.add(color._backend.target.frame.submission)
            depth.render(product=RenderProduct.METRIC_DEPTH)
            depth_submissions.add(depth._backend.target.frame.submission)
            for renderer, expected, owned in (
                (color, "color", color_submissions),
                (depth, "scene data", depth_submissions),
            ):
                stats = renderer._backend.stats
                if stats.cpu_ms:
                    assert set(stats.cpu_ms) == {expected}
                    assert stats.notes["cpu submission"] in owned
                    assert all(np.isfinite(value) and value >= 0 for value in stats.cpu_ms.values())
                if stats.gpu_ms:
                    assert set(stats.gpu_ms) == {expected}
                    assert stats.notes["gpu submission"] in owned
                    assert all(np.isfinite(value) and value > 0 for value in stats.gpu_ms.values())
                    seen_gpu.add(expected)
        assert seen_gpu == {"color", "scene data"}
        assert color._backend.caps.pass_timing and color._backend.caps.gpu_timing


def test_ui_target_exposes_delayed_pass_timings():
    with SceneRenderer(renderer="bgfx", width=80, height=60) as renderer:
        runtime = renderer._backend.runtime
        target = runtime.create_target(80, 60)
        seen_cpu = seen_gpu = False
        try:
            for _ in range(24):
                frame = runtime.render_ui(
                    80, 60, np.empty(0, np.uint8), np.empty(0, np.uint32), [], target
                )
                runtime.read(frame, renderer._backend.api.Product.COLOR)
                if frame.statistics.cpu_ms:
                    assert set(frame.statistics.cpu_ms) == {"ui"}
                    seen_cpu = True
                if frame.statistics.gpu_pass_ms:
                    assert set(frame.statistics.gpu_pass_ms) == {"ui"}
                    assert frame.statistics.gpu_pass_ms["ui"] > 0
                    seen_gpu = True
            assert seen_cpu and seen_gpu
        finally:
            runtime.destroy(target)


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


@pytest.mark.parametrize(
    "product",
    [
        RenderProduct.COLOR,
        RenderProduct.OBJECT_ID,
        RenderProduct.SEGMENTATION,
        RenderProduct.METRIC_DEPTH,
    ],
)
def test_output_cache_retains_results_and_invalidates_on_pose_camera_and_resize(product):
    scene = Scene()
    box = scene.box(position=(0, 0, 0.5))
    camera = CameraView(eye=np.array([3.0, -4.0, 3.0]), target=np.array([0.0, 0.0, 0.5]))
    source = replace(scene.source, geom_segmentation=np.array([[7, 3]], np.int32))
    with SceneRenderer(
        source, width=160, height=120, renderer="bgfx", samples=0, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        first = renderer.render(product=product)
        again = renderer.render(product=product)
        np.testing.assert_array_equal(first, again)
        assert renderer._backend.target.frame.statistics.draw_calls == 0
        box.set_pose((0.7, 0, 0.5))
        renderer.update(scene.frame)
        moved = renderer.render(product=product)
        assert renderer._backend.target.frame.statistics.draw_calls > 0
        assert np.count_nonzero(first != moved) > 100
        renderer.update(scene.frame, camera=replace(camera, eye=np.array([-3.0, -4.0, 2.0])))
        turned = renderer.render(product=product)
        assert renderer._backend.target.frame.statistics.draw_calls > 0
        assert np.count_nonzero(moved != turned) > 100
        renderer.resize(180, 140)
        resized = renderer.render(product=product)
        assert resized.shape[:2] == (140, 180)
        assert renderer._backend.target.frame.statistics.draw_calls > 0
        renderer.render(product=product)
        assert renderer._backend.target.frame.statistics.draw_calls == 0


def test_pose_only_updates_retain_the_latest_material_and_color():
    scene = Scene()
    box = scene.box(color=(0.8, 0.2, 0.1, 1))
    camera = CameraView(eye=np.array([3, -4, 3]), target=np.zeros(3))
    with SceneRenderer(
        scene.source, width=160, height=120, renderer="bgfx", camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        original = renderer.render()
        box.set_color((0.1, 0.3, 0.9, 1))
        renderer.set_scene(scene.source)
        renderer.update(scene.frame)
        recolored = renderer.render()
        assert not np.array_equal(original, recolored)
        box.set_pose((0.6, 0, 0))
        renderer.update(scene.frame)
        moved = renderer.render()
        assert not np.array_equal(recolored, moved)
        box.set_pose((0, 0, 0))
        renderer.update(scene.frame)
        np.testing.assert_array_equal(renderer.render(), recolored)


def test_shadow_culling_keeps_moving_casters_and_matches_opengl():
    light = Light(direction=np.array([0.3, 0.5, -1]), diffuse=np.array([0.8, 0.8, 0.8]))
    camera = CameraView(eye=np.array([3, -4, 3]), target=np.array([0, 0, 0.3]))
    images = {}
    for backend in ("opengl", "bgfx"):
        scene = Scene(lights=LightSet(lights=(light,)))
        scene.plane(size=(3, 3, 0.01))
        scene.box(position=(0, 0, 0.6), color=(0.8, 0.3, 0.2, 1))
        caster = scene.box(position=(1000, 0, 0.6), color=(0.2, 0.7, 0.2, 1))
        with SceneRenderer(
            scene.source, width=240, height=180, renderer=backend, samples=0, camera=camera
        ) as renderer:
            rows = []
            for position in ((1000, 0, 0.6), (0.8, 0, 0.6), (1000, 0, 0.6)):
                caster.set_pose(position)
                renderer.update(scene.frame)
                rows.append(renderer.render())
            np.testing.assert_array_equal(rows[0], rows[2])
            assert not np.array_equal(rows[0], rows[1])
            if backend == "bgfx":
                assert renderer._backend.stats.notes["culled shadow instances"] > 0
            images[backend] = rows
    for expected, actual in zip(images["opengl"], images["bgfx"], strict=True):
        assert np.abs(actual.astype(int) - expected).mean() < 0.5


def test_many_targets_reuse_frame_views_without_cross_target_state():
    from contextlib import ExitStack

    from mojive.render.backend import RenderRequest

    request = RenderRequest(RenderProduct.COLOR | RenderProduct.METRIC_DEPTH)
    with ExitStack() as stack:
        targets = []
        for index in range(40):
            scene = Scene()
            scene.box(color=((index + 1) / 50, 0.3, 0.8, 1))
            camera = CameraView(far=20 + index)
            renderer = stack.enter_context(
                SceneRenderer(
                    scene.source, renderer="bgfx", width=40, height=30, samples=1, camera=camera
                )
            )
            renderer.update(scene.frame)
            renderer.set_debug_view("albedo")
            renderer._backend.render(request=request)
            targets.append(renderer)
        # Read in reverse order after other scenes have reused the same view IDs.
        expected = []
        for index in reversed(range(len(targets))):
            backend = targets[index]._backend
            rgb = backend.target.read_rgb()
            depth = backend.target.read_metric_depth()
            assert depth[0, 0] == pytest.approx(20 + index)
            assert depth[15, 20] < 10
            expected.append(rgb)
        assert len({image[15, 20, 0] for image in expected}) == len(targets)
        targets[20].close()
        targets[0].resize(53, 37)
        for index in (1, 19, 21, 39):
            backend = targets[index]._backend
            backend.render(request=request)
            np.testing.assert_array_equal(backend.target.read_rgb(), expected[39 - index])


def test_canceled_queued_readbacks_release_capacity_and_leave_out_untouched():
    import threading

    import mujoco

    from mojive import Renderer

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    with Renderer(model, width=40, height=30, renderer="bgfx") as renderer:
        renderer.update_scene(data)
        reference = renderer.render()

        def occupy(index):
            started[index].set()
            release.wait(10)

        workers = [renderer._backend.device.readbacks.submit(occupy, i) for i in range(2)]
        outputs = [np.full((30, 40, 3), 17, np.uint8) for _ in range(8)]
        try:
            assert all(event.wait(2) for event in started)
            futures = [renderer.render_async(out=out) for out in outputs]
            with pytest.raises(RuntimeError, match="Readback queue is full"):
                renderer.render_async()
            assert all(future.cancel() for future in futures)
        finally:
            release.set()
            for worker in workers:
                worker.result(timeout=2)
        # No private poll or cleanup call is required after cancellation.
        np.testing.assert_array_equal(renderer.render(), reference)
        np.testing.assert_array_equal(renderer.render_async().result(timeout=10), reference)
        assert all(future.cancelled() for future in futures)
        for out in outputs:
            np.testing.assert_array_equal(out, 17)


def test_source_recompile_reuses_assets_and_uploads_only_replacements():
    from mojive.render.mesh import builtin_mesh
    from mojive.types import MeshKey, MeshShape, TextureData, TextureType

    scene = Scene()
    mesh = builtin_mesh(MeshKey(MeshShape.BOX))
    item = scene.mesh(mesh, size=(0.5, 0.5, 0.5), material=Material(texture="albedo"))
    texture = TextureData("albedo", TextureType.TWO_D, np.full((16, 16, 3), 220, np.uint8))
    scene.add_texture(texture)
    with SceneRenderer(scene.source, renderer="bgfx", width=100, height=80, samples=0) as first:
        first.update(scene.frame)
        baseline = first.render()
        runtime = first._backend.runtime
        initial = runtime.resource_stats()
        # Scene replaces arrays with fresh immutable objects, as recompilation does.
        scene.replace_mesh(item.mesh_key, mesh)
        scene.add_texture(texture)
        first.set_scene(scene.source)
        first.update(scene.frame)
        np.testing.assert_array_equal(first.render(), baseline)
        same = runtime.resource_stats()
        assert (same.mesh_uploads, same.texture_uploads, same.upload_bytes) == (
            initial.mesh_uploads,
            initial.texture_uploads,
            initial.upload_bytes,
        )
        with SceneRenderer(scene.source, renderer="bgfx", width=100, height=80, samples=0) as peer:
            peer.update(scene.frame)
            np.testing.assert_array_equal(peer.render(), baseline)
            assert runtime.resource_stats().upload_bytes == initial.upload_bytes
            scene.add_texture(
                replace(texture, pixels=np.full((16, 16, 3), [30, 80, 240], np.uint8))
            )
            first.set_scene(scene.source)
            first.update(scene.frame)
            assert not np.array_equal(first.render(), baseline)
            after = runtime.resource_stats()
            assert after.mesh_uploads == initial.mesh_uploads
            assert after.texture_uploads == initial.texture_uploads + 1
            # Replacing one scene's material must not change its peer's pixels.
            np.testing.assert_array_equal(peer.render(), baseline)
        scene.replace_mesh(item.mesh_key, replace(mesh, positions=mesh.positions * 0.6))
        first.set_scene(scene.source)
        first.update(scene.frame)
        first.render()
        assert runtime.resource_stats().mesh_uploads == initial.mesh_uploads + 1


def test_shared_geometry_deformation_is_local_and_source_restore_is_immutable():
    from mojive.render.mesh import builtin_mesh
    from mojive.types import MeshKey, MeshShape, MeshUpdate

    scene = Scene()
    item = scene.mesh(builtin_mesh(MeshKey(MeshShape.BOX)), size=(0.5, 0.5, 0.5))
    source = scene.source
    mesh = source.meshes[item.mesh_key]
    with (
        SceneRenderer(source, renderer="bgfx", width=100, height=80, samples=0) as first,
        SceneRenderer(source, renderer="bgfx", width=100, height=80, samples=0) as peer,
    ):
        first.update(scene.frame)
        peer.update(scene.frame)
        baseline = first.render()
        np.testing.assert_array_equal(peer.render(), baseline)
        first.set_flag(RenderFlag.WIREFRAME, True)
        peer.set_flag(RenderFlag.WIREFRAME, True)
        wire = peer.render()
        for scale in (0.7, 0.4):
            first.update(
                replace(
                    scene.frame,
                    mesh_updates={item.mesh_key: MeshUpdate(mesh.positions * scale, mesh.normals)},
                )
            )
            assert not np.array_equal(first.render(), wire)
            np.testing.assert_array_equal(peer.render(), wire)
        first.set_scene(source)
        first.update(scene.frame)
        np.testing.assert_array_equal(first.render(), wire)
