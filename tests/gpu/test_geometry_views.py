"""Real images, depth, identity and resource retention across geometry views."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mojive import GeometryView, Renderer, RenderProduct, Scene, SceneRenderer
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.render.backend import RenderFlag
from mojive.types import CameraView
from mojive.ui.scene_capture import SceneCapture

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.gpu
ASSET = Path(__file__).parents[2] / "assets/geometry_views.xml"
BACKENDS = ["opengl", "wgpu"] + (["bgfx"] if os.environ.get("MOJIVE_RENDERER") == "bgfx" else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_switch_geometry_views_reuses_resources_and_restores_images(backend, monkeypatch):
    a = MuJoCoAdapter(ASSET)
    source = a.scene_source()
    cam = CameraView(
        eye=np.array((0, -5, 3), np.float32),
        target=np.array((0, 0, 0.5), np.float32),
        up=np.array((0, 0, 1), np.float32),
    )
    try:
        with SceneRenderer(source, width=240, height=160, renderer=backend, camera=cam) as view:
            view.update(a.frame(FrameNeeds()))
            if backend == "bgfx":
                # Native uploads only visible meshes. Warm the collision meshes
                # once before checking that repeated switches retain GPU assets.
                view.set_geometry_view(GeometryView.BOTH)
                view.render()
                view.set_geometry_view(GeometryView.DEFAULT)
            before = view.render().copy()

            def forbidden(*args, **kwargs):
                pytest.fail("View switch reuploaded mesh/texture resources")

            if backend == "bgfx":
                uploads = view._backend.runtime.resource_stats()
                monkeypatch.setattr(view._backend.api, "prepare_mesh", forbidden)
                monkeypatch.setattr(view._backend.api, "prepare_texture", forbidden)
            else:
                monkeypatch.setattr(view._backend.meshes, "sync", forbidden)
                monkeypatch.setattr(view._backend.textures, "sync", forbidden)
            builds = []
            build = view._backend._builder._build

            def record_build(*args, **kwargs):
                builds.append(1)
                return build(*args, **kwargs)

            monkeypatch.setattr(view._backend._builder, "_build", record_build)
            images, ids, depth = {}, {}, {}
            for mode in (GeometryView.VISUAL, GeometryView.COLLISION, GeometryView.BOTH):
                builds.clear()
                assert view.set_geometry_view(mode)
                assert len(builds) == 1
                view.set_geometry_view(mode)
                assert len(builds) == 1
                # No update() between switches: transforms must remain current.
                images[mode] = view.render().copy()
                ids[mode] = view.render(product=RenderProduct.OBJECT_ID).copy()
                depth[mode] = view.render(product=RenderProduct.METRIC_DEPTH).copy()
            assert np.count_nonzero(np.any(images["visual"] != images["collision"], axis=2)) > 200
            assert np.any(depth["visual"] != depth["collision"])
            # Both body IDs survive; a transparent proxy becomes opaque in collision mode.
            for mode in (GeometryView.VISUAL, GeometryView.COLLISION):
                assert {1, 2} <= set(np.unique(ids[mode]))
            view.set_geometry_view(GeometryView.DEFAULT)
            np.testing.assert_array_equal(view.render(), before)
            assert a.scene_source() is source and a.data.time == 0
            view.set_flag(RenderFlag.ISLAND, True)
            view.update(a.frame(FrameNeeds(islands=True)))
            view.set_geometry_view(GeometryView.VISUAL)
            np.testing.assert_array_equal(view.render(), images["visual"])
            if backend == "bgfx":
                current = view._backend.runtime.resource_stats()
                assert current.mesh_uploads == uploads.mesh_uploads
                assert current.texture_uploads == uploads.texture_uploads
                assert current.upload_bytes == uploads.upload_bytes
                view._backend.set_render_scene(view._backend._builder.scene)
                np.testing.assert_array_equal(view.render(), images["visual"])
                view.update(a.frame(FrameNeeds()))
                np.testing.assert_array_equal(view.render(), images["visual"])
                view.set_scene(Scene().source)
                assert not view._backend._mesh_indices, (
                    "A new source must release retained mesh slots"
                )
    finally:
        a.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_compatible_renderer_includes_hidden_proxy_and_correct_segmentation(backend):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with Renderer(model, width=240, height=160, renderer=backend) as renderer:
        for mode, included, excluded in (
            ("visual", "visual_shell", "collision_proxy"),
            ("collision", "collision_proxy", "visual_shell"),
            ("both", "collision_proxy", None),
            ("default", "visual_shell", "collision_proxy"),
        ):
            renderer.set_geometry_view(mode)
            renderer.update_scene(data, camera="demo")
            renderer.enable_segmentation_rendering()
            pixels = renderer.render()
            geometries = set(pixels[..., 0][pixels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)])
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, included) in geometries
            if excluded:
                assert (
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, excluded) not in geometries
                )
            renderer.disable_segmentation_rendering()
            source = renderer._source
            renderer.set_geometry_view(mode)
            renderer.update_scene(data, camera="demo")
            assert renderer._source is source


@pytest.mark.parametrize("backend", BACKENDS)
def test_scene_screenshots_and_video_frames_follow_geometry_mode(backend):
    adapter = MuJoCoAdapter(ASSET)
    capture = SceneCapture()
    source = adapter.scene_source()
    frame = adapter.frame(FrameNeeds())
    session = SimpleNamespace(source=source, frame=frame, structure_generation=1)
    camera = CameraView(
        eye=np.array((0, -5, 3), np.float32),
        target=np.array((0, 0, 0.5), np.float32),
        up=np.array((0, 0, 1), np.float32),
    ).with_aspect(240 / 160)
    try:
        with SceneRenderer(source, width=240, height=160, renderer=backend, camera=camera) as view:
            view.update(frame)
            # Match the desktop's background; SceneRenderer defaults to black.
            with view._current():
                view._backend.set_background((0.13, 0.14, 0.16, 1.0))
            try:
                for mode in GeometryView:
                    view.set_geometry_view(mode)
                    expected = view.render().copy()
                    with view._current():
                        pixels = capture.read(view._backend, session, camera)
                    np.testing.assert_array_equal(pixels, expected)
            finally:
                with view._current():
                    capture.release()
    finally:
        adapter.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_coverage_matches_color_depth_and_picking(backend):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <body pos="0 -.5 0"><geom type="box" size="1 .05 1"
          rgba="1 0 0 1" contype="0" conaffinity="0"/></body>
        <body pos="0 .5 0"><geom type="box" size="1 .05 1" rgba="0 1 0 1"/></body>
      </worldbody></mujoco>
    """)
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    camera = CameraView(
        eye=np.array((0, -5, 0), np.float32),
        target=np.zeros(3, np.float32),
        up=np.array((0, 0, 1), np.float32),
    )
    try:
        with SceneRenderer(
            adapter.scene_source(), width=160, height=160, renderer=backend, camera=camera
        ) as view:
            view.update(adapter.frame(FrameNeeds()))
            view.set_geometry_view("both")
            rgb = view.render()[60:100, 60:100]
            ids = view.render(product=RenderProduct.OBJECT_ID)[60:100, 60:100]
            depth = view.render(product=RenderProduct.METRIC_DEPTH)[60:100, 60:100]
            red = rgb[..., 0] > rgb[..., 1]
            assert 0.2 < red.mean() < 0.4
            assert set(np.unique(ids)) == {1, 2}
            np.testing.assert_array_equal(red, ids == 1)
            np.testing.assert_array_equal(red, depth < 5)
    finally:
        adapter.release()
