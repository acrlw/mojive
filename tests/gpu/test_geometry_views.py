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
BACKENDS = ["opengl"] + (["bgfx"] if os.environ.get("MOJIVE_RENDERER") == "bgfx" else [])


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
def test_comparison_blends_continuously_and_picks_the_nearest_surface(backend):
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
            assert np.all(rgb[..., 0] > rgb[..., 1])
            # Neighboring pixels belong to one surface, never complementary holes.
            assert np.abs(np.diff(rgb.astype(float), axis=1)).max() <= 2
            assert np.all(ids == 1)
            assert np.all(depth < 5)
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
def test_independent_opacity_keeps_inner_collision_visible_with_stable_picking(backend):
    from mojive import GeometryRole
    from mojive.render.backend import DebugView

    scene = Scene()
    scene.box(name="shell", position=(0, -0.5, 0), size=(2, 0.1, 2), color=(1, 0, 0, 1))
    scene.box(name="collision", position=(0, 0.5, 0), size=(2, 0.1, 2))
    source = scene.source
    source.geom_role = np.array((GeometryRole.VISUAL, GeometryRole.COLLISION), np.uint8)
    camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.update(scene.frame)
        view.set_geometry_view(GeometryView.BOTH)
        view.set_debug_view(DebugView.ALBEDO)
        for visual, collision in ((0.75, 0.25), (1, 0), (0, 0.25), (0, 0)):
            view.set_geometry_style(GeometryStyle((0, 1, 1), visual, collision))
            rgb = view.render()[60:100, 60:100]
            ids = view.render(product=RenderProduct.OBJECT_ID)[60:100, 60:100]
            depth = view.render(product=RenderProduct.METRIC_DEPTH)[60:100, 60:100]
            if visual == 1:
                expected = (255, 0, 0)
            elif visual + collision:
                weights = np.array((visual, collision))
                opacity = min(visual + collision, 1)
                expected = (
                    255 * opacity * np.array((weights[0], weights[1], weights[1])) / weights.sum()
                )
            else:
                expected = (0, 0, 0)
            np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=1.5)
            nearest = source.geom_object_id[0 if visual else 1] if visual or collision else 0
            assert np.all(ids == nearest)
            if visual:
                assert np.all(depth < 5)
            elif collision:
                assert np.all(depth > 5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("msaa", [False, True])
def test_comparison_has_no_screen_pattern_or_sorting_jump_during_pan(backend, msaa):
    from mojive import GeometryRole
    from mojive.render.backend import DebugView

    frames = []
    for reverse in (False, True):
        scene = Scene()
        roles = (GeometryRole.VISUAL, GeometryRole.COLLISION)
        for role in reversed(roles) if reverse else roles:
            scene.box(
                position=(0, -0.5 if role == GeometryRole.VISUAL else 0.5, 0),
                size=(2, 0.1, 2),
                color=(1, 0, 0, 1),
            )
        source = scene.source
        source.geom_role = np.array(tuple(reversed(roles)) if reverse else roles, np.uint8)
        camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
        with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
            view.update(scene.frame)
            view.set_geometry_view("both")
            view.set_geometry_style(GeometryStyle((0, 1, 1), 0.65, 0.35))
            view.set_debug_view(DebugView.ALBEDO)
            view.set_flag(RenderFlag.MSAA, msaa)
            # Diagnostic layers remain visible when ordinary authored alpha is hidden.
            view.set_flag(RenderFlag.TRANSPARENT, False)
            for offset in np.linspace(-0.03, 0.03, 13):
                camera.eye[0] = camera.target[0] = offset
                view.update(scene.frame, camera=camera)
                pixels = view.render()[60:100, 60:100].copy()
                assert pixels[..., 0].min() > 100
                assert pixels[..., 1].min() > 30
                # Flat overlapping surfaces must not alternate at one-pixel frequency.
                assert np.ptp(pixels.astype(float), axis=(0, 1)).max() <= 1
                frames.append(pixels)
    assert np.ptp(np.array(frames).astype(float), axis=0).max() <= 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_respects_opaque_occlusion_and_combined_viewport_identity(backend):
    from mojive import GeometryRole
    from mojive.render.backend import DebugView

    scene = Scene()
    scene.box(position=(0, 0, 0), size=(2, 0.1, 2), color=(1, 0, 0, 1))
    scene.box(position=(0, -0.5, 0), size=(0.3, 0.1, 2), color=(0, 0, 1, 1))
    source = scene.source
    source.geom_role = np.array((GeometryRole.VISUAL, GeometryRole.BOTH), np.uint8)
    camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.update(scene.frame)
        view.set_geometry_view("both")
        view.set_debug_view(DebugView.ALBEDO)
        expected = view.render().copy()
        with view._current():
            view._backend.render()
            combined = view._backend.target.read_rgb(flip=True)
            ids = view._backend.target.read_ids(flip=True)
        np.testing.assert_array_equal(combined, expected)
        assert np.all(combined[60:100, 76:84] == (0, 0, 255))
        assert np.all(ids[60:100, 76:84] == source.geom_object_id[1])
        assert np.all(ids[60:100, 45:55] == source.geom_object_id[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_remains_visible_in_debug_and_wire_views(backend):
    from mojive import GeometryRole
    from mojive.render.backend import DebugView

    scene = Scene()
    scene.box(color=(1, 0.2, 0.1, 1))
    source = scene.source
    source.geom_role = np.array((GeometryRole.VISUAL,), np.uint8)
    camera = CameraView(eye=np.array((2, -3, 2), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.update(scene.frame)
        view.set_geometry_view("both")
        for debug in (
            DebugView.ALBEDO,
            DebugView.NORMAL,
            DebugView.DEPTH,
            DebugView.OVERDRAW,
            DebugView.WIREFRAME,
            DebugView.SHADED,
        ):
            view.set_debug_view(debug)
            assert np.count_nonzero(np.any(view.render() > 5, axis=2)) > 80, debug


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_opacity_comparison_does_not_leave_a_selection_outline(backend):
    from mojive import GeometryRole

    scene = Scene()
    scene.box(color=(1, 0, 0, 1))
    source = scene.source
    source.geom_role = np.array((GeometryRole.VISUAL,), np.uint8)
    camera = CameraView(eye=np.array((2, -3, 2), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.update(scene.frame)
        view.set_geometry_view("both")
        view.set_geometry_style(GeometryStyle(visual_opacity=0, collision_opacity=0))
        hidden = view.render().copy()
        view._backend.highlight(int(source.geom_object_id[0]), xray=True)
        np.testing.assert_array_equal(view.render(), hidden)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("reverse", [False, True])
def test_comparison_uses_only_the_outer_visual_and_nearest_collision_surface(backend, reverse):
    from mojive import GeometryRole
    from mojive.render.backend import DebugView

    # Nested visual parts and overlapping proxies must not add extra opacity or color.
    parts = [
        ((0, 0, 0), (2, 2, 2), (1, 0, 0, 1), GeometryRole.VISUAL),
        ((0, -0.2, 0), (1.5, 0.2, 1.5), (0, 1, 0, 1), GeometryRole.VISUAL),
        ((0, 0.2, 0), (1.5, 0.2, 1.5), (0, 1, 0, 1), GeometryRole.VISUAL),
        ((0, -0.4, 0), (1.5, 0.2, 1.5), (1, 1, 1, 1), GeometryRole.COLLISION),
        ((0, 0.4, 0), (1.5, 0.2, 1.5), (1, 1, 1, 1), GeometryRole.COLLISION),
    ]
    if reverse:
        parts.reverse()
    scene = Scene()
    for position, size, color, _ in parts:
        scene.box(position=position, size=size, color=color)
    source = scene.source
    source.geom_role = np.array([part[3] for part in parts], np.uint8)
    camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
    with SceneRenderer(source, width=160, height=160, renderer=backend, camera=camera) as view:
        view.set_geometry_view("both")
        view.set_geometry_style(GeometryStyle((0, 0, 1), 0.65, 0.35))
        view.set_debug_view(DebugView.ALBEDO)
        for offset in (-0.02, 0, 0.02):
            camera.eye[0] = camera.target[0] = offset
            view.update(scene.frame, camera=camera)
            pixels = view.render()[60:100, 60:100]
            expected = np.broadcast_to((0.65 * 255, 0, 0.35 * 255), pixels.shape)
            np.testing.assert_allclose(pixels, expected, atol=1.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moving_internal_visual_parts_do_not_pollute_the_outer_surface(backend):
    from mojive.render.backend import DebugView

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom type="box" size="1 1 1" rgba="1 0 0 1" contype="0" conaffinity="0"/>
      <body><freejoint/>
        <geom type="box" size=".4 .4 .4" rgba="0 1 0 1" contype="0" conaffinity="0"/>
      </body>
    </worldbody></mujoco>""")
    adapter = MuJoCoAdapter()
    adapter.load_model(model)
    camera = CameraView(eye=np.array((0, -5, 0), np.float32), target=np.zeros(3, np.float32))
    try:
        with SceneRenderer(
            adapter.scene_source(), width=160, height=160, renderer=backend, camera=camera
        ) as view:
            view.set_geometry_view("both")
            view.set_debug_view(DebugView.ALBEDO)
            for y, channel in ((0, 0), (-0.2, 0), (-1.4, 1), (0, 0)):
                adapter.data.qpos[1] = y
                mujoco.mj_forward(model, adapter.data)
                view.update(adapter.frame(FrameNeeds()))
                pixels = view.render()[70:90, 70:90]
                assert np.all(pixels[..., channel] > 150)
                assert np.all(pixels[..., 1 - channel] == 0)
    finally:
        adapter.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_comparison_surfaces_recover_after_resize_and_msaa_changes(backend):
    adapter = MuJoCoAdapter(ASSET)
    camera = CameraView(
        eye=np.array((0, -5, 3), np.float32), target=np.array((0, 0, 0.5), np.float32)
    )
    try:
        with SceneRenderer(
            adapter.scene_source(), width=240, height=160, renderer=backend, camera=camera
        ) as view:
            view.update(adapter.frame(FrameNeeds()))
            view.set_geometry_view("both")
            before = view.render().copy()
            view.resize(300, 200)
            assert view.render().shape == (200, 300, 3)
            view.set_flag(RenderFlag.MSAA, False)
            assert view.render().any()
            view.set_flag(RenderFlag.MSAA, True)
            view.resize(240, 160)
            np.testing.assert_array_equal(view.render(), before)
    finally:
        adapter.release()


@pytest.mark.parametrize("backend", BACKENDS)
def test_hidden_comparison_proxies_do_not_cast_shadows_on_visual_surfaces(backend):
    adapter = MuJoCoAdapter(ASSET)
    camera = CameraView(
        eye=np.array((2.8, -4.8, 3.5), np.float32),
        target=np.array((0, 0, 0.4), np.float32),
        near=0.02,
        far=40,
    )
    try:
        with SceneRenderer(
            adapter.scene_source(), width=480, height=320, renderer=backend, camera=camera
        ) as view:
            view.update(adapter.frame(FrameNeeds()))
            view.set_geometry_view("visual")
            visual = view.render().astype(float)
            view.set_geometry_style(GeometryStyle(visual_opacity=1, collision_opacity=0))
            view.set_geometry_view("both")
            both = view.render().astype(float)
            # Removing diagnostic opacity must recover the original appearance,
            # including the lit face beside a slightly expanded convex proxy.
            assert np.abs(visual - both).mean() < 0.1
    finally:
        adapter.release()
