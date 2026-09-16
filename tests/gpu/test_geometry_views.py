"""Real images, depth, identity and resource retention across geometry views."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mojive import GeometryStyle, GeometryView, Renderer, RenderProduct, Scene, SceneRenderer
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
                default = expected.copy()
                view.set_geometry_style(GeometryStyle((0.25, 0.5, 0.7), 0.9, 0.2))
                expected = view.render().copy()
                assert np.count_nonzero(expected != default) > 100
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
            assert 0.55 < red.mean() < 0.75
            assert set(np.unique(ids)) == {1, 2}
            np.testing.assert_array_equal(red, ids == 1)
            np.testing.assert_array_equal(red, depth < 5)
    finally:
        adapter.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_capture_retains_mujoco_visuals_and_custom_comparison_style(backend):
    from mojive.render.backend import DebugView, FrameMode, LabelMode

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="plane" size="2 2 .1" rgba=".1 .1 .1 1"/>
      <body name="subject" pos="0 0 .2"><freejoint/>
        <inertial pos="0 0 .4" mass="1" diaginertia=".2 .3 .4"/>
        <geom type="box" size=".3 .2 .25" rgba=".2 .5 .8 1"/>
      </body>
      <geom name="grouped" type="sphere" pos=".7 0 .3" size=".2" group="3" rgba=".8 .2 .2 1"/>
    </worldbody></mujoco>""")
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    adapter.prepare_frame(FrameNeeds(bvh=True))
    camera = CameraView(
        eye=np.array((2, -3, 2), np.float32), target=np.array((0, 0, 0.2), np.float32), aspect=1.5
    )
    capture = SceneCapture()
    try:
        source = adapter.scene_source()
        frame = adapter.frame(FrameNeeds(diagnostics=True, contacts=True, bvh=True))
        assert len(frame.contacts)
        session = SimpleNamespace(source=source, frame=frame, structure_generation=0)
        with SceneRenderer(source, width=300, height=200, renderer=backend, camera=camera) as view:

            def capture_image():
                with view._current():
                    return capture.read(view._backend, session, camera)

            try:
                view.update(frame)
                baseline = view.render().copy()
                cases = [
                    (RenderFlag.INERTIA, RenderFlag.SCLINERTIA),
                    (RenderFlag.CONTACTPOINT, RenderFlag.CONTACTFORCE),
                    (RenderFlag.BODYBVH,),
                ]
                for flags in cases:
                    for flag in flags:
                        view.set_flag(flag, True)
                    view.update(frame)
                    displayed = view.render().copy()
                    assert np.count_nonzero(displayed != baseline) > 20, flags
                    np.testing.assert_array_equal(capture_image(), displayed)
                    for flag in flags:
                        view.set_flag(flag, False)
                view._backend.set_label_mode(LabelMode.BODY)
                view._backend.set_frame_mode(FrameMode.BODY)
                view.update(frame)
                displayed = view.render().copy()
                assert np.count_nonzero(displayed != baseline) > 20
                np.testing.assert_array_equal(capture_image(), displayed)
                view._backend.set_label_mode(LabelMode.NONE)
                view._backend.set_frame_mode(FrameMode.NONE)
                for visible in (True, False):
                    assert adapter.set_visual_group("geom", 3, visible)
                    session.source = adapter.scene_source()
                    session.frame = adapter.frame(
                        FrameNeeds(diagnostics=True, contacts=True, bvh=True)
                    )
                    session.structure_generation += 1
                    view.set_scene(session.source)
                    view.update(session.frame)
                    displayed = view.render().copy()
                    np.testing.assert_array_equal(capture_image(), displayed)
                    assert (np.count_nonzero(displayed != baseline) > 20) == visible
                view.set_geometry_view(GeometryView.BOTH)
                style = GeometryStyle((0.2, 0.6, 0.8), 0.9, 0.15)
                view.set_geometry_style(style)
                view.update(frame)
                np.testing.assert_array_equal(capture_image(), view.render())
                assert capture._backend.get_geometry_style() == style
                view.set_debug_view(DebugView.NORMAL)
                np.testing.assert_array_equal(capture_image(), view.render())
            finally:
                with view._current():
                    capture.release()
    finally:
        adapter.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_independent_opacity_keeps_inner_collision_visible_and_pickable(backend):
    from mojive import GeometryRole

    scene = Scene()
    scene.box(name="shell", position=(0, -0.5, 0), size=(2, 0.1, 2), color=(1, 0, 0, 1))
    scene.box(name="collision", position=(0, 0.5, 0), size=(2, 0.1, 2))
    source = scene.source
    source.geom_role = np.array((GeometryRole.VISUAL, GeometryRole.COLLISION), np.uint8)
    camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.update(scene.frame)
        view.set_geometry_view(GeometryView.BOTH)
        for visual, collision in ((0.75, 0.25), (1, 0), (0, 0.25), (0, 0)):
            view.set_geometry_style(GeometryStyle((0, 1, 1), visual, collision))
            rgb = view.render()[60:100, 60:100]
            ids = view.render(product=RenderProduct.OBJECT_ID)[60:100, 60:100]
            depth = view.render(product=RenderProduct.METRIC_DEPTH)[60:100, 60:100]
            red = rgb[..., 0] > rgb[..., 1]
            cyan = rgb[..., 1] > rgb[..., 0]
            assert red.mean() == pytest.approx(visual, abs=0.01)
            assert cyan.mean() == pytest.approx(collision, abs=0.01)
            np.testing.assert_array_equal(ids == source.geom_object_id[0], red)
            np.testing.assert_array_equal(ids == source.geom_object_id[1], cyan)
            if red.any() and cyan.any():
                assert depth[red].max() < depth[cyan].min()
