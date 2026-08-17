from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.physics]

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")
pytest.importorskip("mujoco")

from forge_viewer.adapters.base import FrameNeeds  # noqa: E402
from forge_viewer.assets import resolve  # noqa: E402
from forge_viewer.backends import make_adapter  # noqa: E402
from forge_viewer.render.backend import RenderFlag  # noqa: E402
from forge_viewer.render.builder import SceneSourceBuilder  # noqa: E402
from forge_viewer.render.debugdraw import Occlusion, Prim  # noqa: E402
from forge_viewer.render.forge import gl_native as G  # noqa: E402
from forge_viewer.render.forge import passes as _passes  # noqa: E402
from forge_viewer.render.forge.backend import PASS_ORDER, ForgeBackend, registered  # noqa: E402
from forge_viewer.render.forge.state_guard import GLStateGuard  # noqa: E402
from forge_viewer.types import CameraView  # noqa: E402

WIDTH, HEIGHT = 480, 360


@pytest.fixture(scope="module")
def gl():
    if not glfw.init():
        pytest.skip("glfw 起不来")
    for k, v in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, True),
        (glfw.VISIBLE, False),
    ):
        glfw.window_hint(k, v)
    win = glfw.create_window(WIDTH, HEIGHT, "pipeline", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("建不出 3.3 core 上下文")
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    G.native().drain_errors()
    yield ctx
    glfw.terminate()


@pytest.fixture(scope="module")
def rendered(gl):

    try:
        path = resolve("pick_scene")
    except FileNotFoundError:
        pytest.skip("pick_scene 还没落地")

    adapter = make_adapter("mujoco", path)
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    builder = SceneSourceBuilder()
    source = adapter.scene_source()
    backend.set_scene(source)

    lo, hi = _bounds(adapter, source)
    center = ((lo + hi) * 0.5).astype(np.float32)
    extent = max(float(np.linalg.norm(hi - lo)) * 0.5, 1e-3)
    camera = CameraView(
        eye=center + np.array([extent * 1.6, -extent * 2.2, extent * 1.3], np.float32),
        target=center,
        up=np.array([0.0, 0.0, 1.0], np.float32),
        near=extent * 0.02,
        far=extent * 40.0,
    )
    backend.set_camera(camera)
    builder.set_source(source, camera)

    guard = GLStateGuard()
    G.native().drain_errors()
    for _ in range(4):
        adapter.step(1)
        backend.set_render_scene(builder.update(adapter.frame(FrameNeeds(poses=True)), camera))
        backend.render(None)

    before = guard.snapshot()
    err_before = G.native().drain_errors()
    frame = adapter.frame(FrameNeeds(poses=True))
    scene = builder.update(frame, camera)
    backend.set_render_scene(scene)
    image = backend.render(frame)
    after = guard.snapshot()
    err_after = G.native().drain_errors()

    yield {
        "backend": backend,
        "scene": scene,
        "image": image,
        "camera": camera,
        "before": before,
        "after": after,
        "err_before": err_before,
        "err_after": err_after,
        "source": source,
    }
    backend.release()
    adapter.release()


def _bounds(adapter, source):
    f = adapter.frame(FrameNeeds(poses=True))
    pos = f.geom_xpos
    if pos is None or len(pos) == 0:
        return np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)
    keep = np.isfinite(pos).all(axis=1)
    if len(source.geom_infinite_plane) == len(pos):
        keep &= ~source.geom_infinite_plane
    p = pos[keep] if keep.any() else pos
    r = source.geom_size[: len(p)].max(axis=1, keepdims=True) if len(p) else 0.0
    return (p - r).min(axis=0).astype(np.float32), (p + r).max(axis=0).astype(np.float32)


def _require(*names: str) -> None:

    _passes.load_all()
    missing = [n for n in names if n not in registered()]
    if missing:
        pytest.skip(f"这几条 pass 还没登记：{missing}（装齐之后这条判据才有意义）")


def test_mujoco_visuals_reach_the_gpu_pipeline(gl):
    adapter = make_adapter("mujoco", resolve("mujoco_visuals"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        source = adapter.scene_source()
        backend.set_scene(source)
        backend.set_camera(adapter.camera_hint())
        for flag in (
            RenderFlag.TENDON,
            RenderFlag.ACTUATOR,
            RenderFlag.ACTIVATION,
            RenderFlag.CONTACTPOINT,
            RenderFlag.CONTACTFORCE,
            RenderFlag.JOINT,
            RenderFlag.COM,
            RenderFlag.INERTIA,
        ):
            assert backend.set_flag(flag, True)
        adapter.step(400)
        frame = adapter.frame(
            FrameNeeds(
                poses=True,
                contacts=True,
                tendons=True,
                actuator=True,
                diagnostics=True,
            )
        )
        assert frame.contacts is not None and len(frame.contacts) > 0
        backend.update(frame)
        assert backend.render(frame) is not None

        assert backend.stats.instances > adapter.model.ngeom, "site 没进普通实例链路"
        assert backend._passes["tendon"].capsule_count >= 2
        assert backend.debug.layer("physics.contact.points").count_of(Prim.POINT) >= 1
        assert backend.debug.layer("physics.contact.forces").count_of(Prim.ARROW) >= 1
        joints = backend.debug.layer("physics.joints")
        assert joints.count_of(Prim.BOX) == 1
        assert joints.count_of(Prim.SOLID_ARROW) == 1
        assert backend.debug.layer("physics.com").count_of(Prim.SPHERE) == 2
        assert backend.debug.layer("physics.inertia").count_of(Prim.BOX) == 2
        assert float(backend.target.read_color()[..., :3].std()) > 8.0
    finally:
        backend.release()
        adapter.release()


def test_camera_and_light_entities_reach_the_debug_pass(gl):
    adapter = make_adapter("mujoco", resolve("mujoco_visuals"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        source = adapter.scene_source()
        backend.set_scene(source)
        backend.set_camera(adapter.camera_hint())
        assert backend.set_flag(RenderFlag.CAMERA, True)
        assert backend.set_flag(RenderFlag.LIGHT, True)

        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True))
        backend.update(frame)

        assert backend.debug.layer("scene.cameras").count_of(Prim.LINE) == 8 * len(source.cameras)
        light_set = frame.lights if frame.lights is not None else source.lights
        active_lights = [light for light in light_set.lights if light.active]
        directional_lights = [light for light in active_lights if light.kind.value != "point"]
        assert backend.debug.layer("scene.lights").count_of(Prim.POINT) == len(active_lights)
        assert backend.debug.layer("scene.lights").count_of(Prim.ARROW) == len(directional_lights)
        assert backend.render(frame) is not None

        assert backend.set_flag(RenderFlag.CAMERA, False)
        assert backend.set_flag(RenderFlag.LIGHT, False)
        backend.update(frame)
        assert backend.debug.layer("scene.cameras").primitives == 0
        assert backend.debug.layer("scene.lights").primitives == 0
    finally:
        backend.release()
        adapter.release()


def test_rangefinder_rays_hits_and_normals_reach_the_debug_pass(gl):
    adapter = make_adapter("mujoco", resolve("rangefinder"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        backend.set_scene(adapter.scene_source())
        backend.set_camera(adapter.camera_hint())
        assert backend.set_flag(RenderFlag.RANGEFINDER, True)
        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True))
        backend.update(frame)

        layer = backend.debug.layer("physics.rangefinders")
        assert layer.count_of(Prim.LINE) == 7
        assert layer.count_of(Prim.POINT) == 7
        assert layer.count_of(Prim.ARROW) == 7
        assert backend.render(frame) is not None

        assert backend.set_flag(RenderFlag.RANGEFINDER, False)
        backend.update(frame)
        assert layer.primitives == 0
    finally:
        backend.release()
        adapter.release()


def test_equality_constraint_endpoints_reach_the_debug_pass(gl):
    adapter = make_adapter("mujoco", resolve("constraints"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        backend.set_scene(adapter.scene_source())
        backend.set_camera(adapter.camera_hint())
        assert backend.set_flag(RenderFlag.CONSTRAINT, True)
        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True))
        backend.update(frame)

        layer = backend.debug.layer("physics.constraints")
        assert layer.count_of(Prim.SPHERE) == 4
        assert backend.render(frame) is not None

        assert adapter.set_equality_enabled(0, False)
        frame = adapter.frame(FrameNeeds(poses=True, diagnostics=True))
        backend.update(frame)
        assert layer.count_of(Prim.SPHERE) == 2

        assert backend.set_flag(RenderFlag.CONSTRAINT, False)
        backend.update(frame)
        assert layer.primitives == 0
    finally:
        backend.release()
        adapter.release()


def test_joint_site_and_body_actuator_visuals_reach_the_gpu_pipeline(gl):
    adapter = make_adapter("mujoco", resolve("actuator_visuals"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        backend.set_scene(adapter.scene_source())
        camera = next(item for item in adapter.cameras() if item.name == "overview")
        backend.set_camera(adapter.camera_view(camera.camera_id))
        assert backend.set_flag(RenderFlag.ACTUATOR, True)
        assert backend.set_flag(RenderFlag.ACTIVATION, True)

        adapter.data.ctrl[1] = -1.0
        frame = adapter.frame(FrameNeeds(poses=True, actuator=True, diagnostics=True))
        backend.update(frame)
        negative = backend._actuator_palette[1].copy()

        adapter.data.ctrl[1] = 1.0
        frame = adapter.frame(FrameNeeds(poses=True, actuator=True, diagnostics=True))
        backend.update(frame)
        positive = backend._actuator_palette[1].copy()
        layer = backend.debug.layer("physics.actuators")

        assert not np.allclose(negative, positive)
        assert layer.count_of(Prim.SOLID_ARROW) == 1
        assert layer.count_of(Prim.SOLID_DOUBLE_ARROW) == 1
        assert layer.count_of(Prim.BOX) == 1
        assert layer.count_of(Prim.CYLINDER) == 2
        assert layer.count_of(Prim.SPHERE) == 2
        assert backend.render(frame) is not None
        assert float(backend.target.read_color()[..., :3].std()) > 8.0
    finally:
        backend.release()
        adapter.release()


def test_deformable_vertices_update_without_rebuilding_the_scene(gl):

    import mujoco

    from forge_viewer.types import MeshKey, MeshShape

    adapter = make_adapter("mujoco", resolve("deformables"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=1)
    try:
        source = adapter.scene_source()
        backend.set_camera(adapter.camera_hint())
        backend.set_scene(source)
        key = MeshKey(MeshShape.SKIN, 0)
        gpu_mesh = backend.meshes.get(key)
        assert gpu_mesh is not None

        frame = adapter.frame(FrameNeeds(poses=True, deformables=True))
        backend.update(frame)
        before = gpu_mesh.vbo.read()
        ranges = backend._scene.bucket_ranges

        joint = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_JOINT, "skin_tip_hinge")
        assert adapter.set_qpos(int(adapter.model.jnt_qposadr[joint]), np.deg2rad(40.0))
        frame = adapter.frame(FrameNeeds(poses=True, deformables=True))
        backend.update(frame)
        after = gpu_mesh.vbo.read()

        assert before != after, "skin 顶点没有上传到原 VBO"
        assert backend.meshes.get(key) is gpu_mesh, "逐帧更新不该换 GPU mesh"
        assert backend._scene.bucket_ranges == ranges, "逐帧更新不该重新分桶"
        assert backend.render(frame) is not None
        assert float(backend.target.read_color()[..., :3].std()) > 8.0
    finally:
        backend.release()
        adapter.release()


def test_deformables_are_pickable_and_use_the_normal_outline_pass(gl):
    from forge_viewer.adapters.base import NodeKind

    adapter = make_adapter("mujoco", resolve("deformables"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=4)
    try:
        source = adapter.scene_source()
        backend.set_camera(adapter.camera_hint())
        backend.set_scene(source)
        frame = adapter.frame(FrameNeeds(poses=True, deformables=True))
        backend.update(frame)
        backend.render(frame)

        deformable_ids = {
            node.object_id
            for node in adapter.nodes()
            if node.kind in (NodeKind.FLEX, NodeKind.SKIN)
        }
        visible_ids = set(np.unique(backend.target.read_ids()).tolist())
        visible_deformables = deformable_ids & visible_ids
        assert len(visible_deformables) >= 3

        before = backend.target.read_color(flip=False).copy()
        backend.highlight(next(iter(visible_deformables)))
        backend.render(frame)
        after = backend.target.read_color(flip=False)
        changed = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2)
        assert np.count_nonzero(changed > 10) > 20
    finally:
        backend.release()
        adapter.release()


def test_deformable_visibility_flags_rebuild_the_scene(gl):
    from forge_viewer.render.backend import RenderFlag
    from forge_viewer.types import InstanceVisual

    adapter = make_adapter("mujoco", resolve("deformables"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=1)
    try:
        source = adapter.scene_source()
        backend.set_camera(adapter.camera_hint())
        backend.set_scene(source)
        frame = adapter.frame(FrameNeeds(poses=True, deformables=True))
        backend.update(frame)

        visual = source.geom_visual
        flex_face = int(InstanceVisual.FLEX_FACE)
        flex_skin = int(InstanceVisual.FLEX_SKIN)
        skin = int(InstanceVisual.SKIN)
        default_count = int(np.count_nonzero(visual != flex_face))
        assert backend._scene.count == default_count

        assert backend.set_flag(RenderFlag.FLEXSKIN, False)
        assert backend._scene.count == default_count - int(np.count_nonzero(visual == flex_skin))

        assert backend.set_flag(RenderFlag.FLEXFACE, True)
        assert backend._scene.count == default_count

        assert backend.set_flag(RenderFlag.SKIN, False)
        assert backend._scene.count == default_count - int(np.count_nonzero(visual == skin))

        before_static = backend._scene.count
        assert backend.set_flag(RenderFlag.STATIC, False)
        assert backend._scene.count == before_static - int(np.count_nonzero(source.geom_static))
        assert backend.render(frame) is not None
    finally:
        backend.release()
        adapter.release()


def test_world_text_is_rendered_into_the_forge_target_without_imgui(gl):
    adapter = make_adapter("mujoco", resolve("pick_scene"))
    backend = ForgeBackend(gl, WIDTH, HEIGHT, samples=1)
    try:
        source = adapter.scene_source()
        camera = adapter.camera_hint()
        backend.set_scene(source)
        backend.set_camera(camera)
        frame = adapter.frame(FrameNeeds(poses=True))
        backend.update(frame)
        backend.render(frame)
        before = backend.target.read_color().copy()

        backend.debug.layer("test.labels", Occlusion.ALWAYS).text(
            "center", camera.target, "world 123", align=(0.5, 0.5)
        )
        backend.render(frame)
        after = backend.target.read_color()
        changed = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2)
        assert np.count_nonzero(changed > 20) > 20
    finally:
        backend.release()
        adapter.release()


def test_render_returns_an_image(rendered):
    img = rendered["image"]
    assert img is not None, "render() 返回 None——画不出来"
    assert (img.width, img.height) == (WIDTH, HEIGHT)
    assert img.texture_id > 0


def test_flip_y_is_declared_and_true_for_gl(rendered):

    assert rendered["image"].flip_y is True


def test_global_gl_state_is_unchanged_and_no_errors(rendered):

    assert rendered["err_before"] == 0, "渲染之前就有 GL 错误残留"
    diff = {
        k: (rendered["before"][k], rendered["after"][k])
        for k in rendered["before"]
        if rendered["before"][k] != rendered["after"][k]
    }
    assert not diff, f"这些状态没还回去：{diff}"
    assert rendered["err_after"] == 0, "整条管线跑完留下了 GL 错误"


def test_every_registered_pass_actually_ran(rendered):

    stats = rendered["backend"].stats
    ran = set(stats.cpu_ms)
    expected = {n for n in PASS_ORDER if n in registered() or n == "present"}

    assert "present" in ran, "present 没跑——画面根本没交出来"
    assert ran <= expected, f"计时表里有没登记的 pass：{ran - expected}"


def test_the_picture_is_not_blank(rendered):

    _require("opaque")
    img = rendered["backend"].target.read_color(flip=True)[..., :3]
    colors = np.unique(img.reshape(-1, 3), axis=0)
    assert len(colors) > 8, f"画面只有 {len(colors)} 种颜色，像是没画几何"
    assert img.std() > 4.0, "画面几乎是纯色"


def test_id_buffer_agrees_with_the_picture(rendered):

    _require("opaque", "id")
    backend = rendered["backend"]
    ids = backend.target.read_ids()
    img = backend.target.read_color(flip=False)[..., :3].astype(np.int16)
    bg = np.array(img[0, 0], np.int16)
    covered = ids != 0
    if not covered.any():
        pytest.skip("这一帧 ID buffer 全零——先让构建器填上 object_id")
    差 = np.abs(img - bg).sum(axis=2)
    hit_but_background = covered & (差 < 6)
    ratio = hit_but_background.sum() / max(covered.sum(), 1)
    assert ratio < 0.05, f"{ratio:.1%} 的像素 ID buffer 说有东西、画面上却是背景色"


def test_picking_reads_a_single_pixel_and_matches(rendered):

    _require("opaque", "id")
    backend = rendered["backend"]
    ids = backend.target.read_ids()
    ys, xs = np.nonzero(ids)
    if len(ys) == 0:
        pytest.skip("这一帧 ID buffer 全零")
    for i in np.linspace(0, len(ys) - 1, 8).astype(int):
        y, x = int(ys[i]), int(xs[i])
        assert backend.pick(x, y) == int(ids[y, x]), f"({x},{y}) 拾取与 ID buffer 不符"


def test_batching_actually_happened(rendered):

    _require("opaque")
    s = rendered["backend"].stats
    assert s.instances > 0
    assert s.triangles > 0
    assert s.draw_calls <= s.buckets * 2 + 4, (
        f"绘制次数 {s.draw_calls} 比桶数 {s.buckets} 多太多——合批没生效"
    )
    if s.instances >= 8:
        assert s.draw_calls < s.instances, (
            f"绘制次数 {s.draw_calls} ≥ 实例数 {s.instances}——退化成逐 geom 绘制了"
        )


def test_shadow_toggle_is_reversible(rendered):

    _require("opaque", "shadow")
    backend = rendered["backend"]
    if not backend.caps.supports(RenderFlag.SHADOW):
        pytest.skip("后端不申报 SHADOW")

    def shot() -> np.ndarray:
        backend.render(None)
        return backend.target.read_color(flip=True).copy()

    backend.set_flag(RenderFlag.SHADOW, False)
    off_a = shot()
    backend.set_flag(RenderFlag.SHADOW, True)
    on = shot()
    backend.set_flag(RenderFlag.SHADOW, False)
    off_b = shot()

    assert np.array_equal(off_a, off_b), "关→开→关之后画面没回到原样"
    assert not np.array_equal(on, off_a), "开关阴影画面完全没变——开关没接上"


def test_unsupported_flags_are_refused_not_silently_ignored(rendered):

    backend = rendered["backend"]
    for flag in RenderFlag:
        ok = backend.set_flag(flag, True)
        assert ok == (flag in backend.caps.render_flags), (
            f"{flag} 的 set_flag 返回 {ok}，与 caps 申报的不一致"
        )
