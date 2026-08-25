from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer import math3d as M  # noqa: E402
from forge_viewer.render.forge import gl_native as G  # noqa: E402
from forge_viewer.render.forge.instances import InstanceStore, Strategy  # noqa: E402
from forge_viewer.render.forge.programs import ProgramCache, ProgramSpec, UniformCache  # noqa: E402
from forge_viewer.render.forge.resources import MeshStore  # noqa: E402
from forge_viewer.render.forge.state_guard import (  # noqa: E402
    GLStateGuard,
    bind_default_framebuffer,
)
from forge_viewer.render.forge.targets import IdLayout, RenderTarget, probe_id_layout  # noqa: E402
from forge_viewer.render.forge.timing import FrameTiming  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
)


@pytest.fixture(autouse=True, scope="module")
def _forge_backend_only(require_forge):
    pass


@pytest.fixture(scope="module")
def gl():
    if not glfw.init():
        pytest.skip("GLFW initialization failed")
    for k, v in (
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        (glfw.OPENGL_FORWARD_COMPAT, True),
        (glfw.VISIBLE, False),
    ):
        glfw.window_hint(k, v)
    win = glfw.create_window(256, 192, "forge tests", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("OpenGL 3.3 core context unavailable")
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    G.native().drain_errors()
    yield ctx
    glfw.terminate()


@pytest.fixture(autouse=True)
def _gl_baseline(gl):

    G.native().drain_errors()
    gl.wireframe = False
    gl.front_face = "ccw"
    gl.cull_face = "back"
    gl.depth_func = "<"
    gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    gl.enable_only(moderngl.NOTHING)
    gl.multisample = True
    G.native().depth_mask(True)
    yield


def test_state_guard_restores_every_item(gl):

    guard = GLStateGuard()
    if not guard.available:
        pytest.skip("native glGet* entry points unavailable")
    n = G.native()

    gl.enable(moderngl.BLEND)
    gl.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    gl.blend_func = (moderngl.ONE, moderngl.ONE)
    gl.wireframe = True
    gl.front_face = "cw"
    gl.viewport = (3, 5, 77, 55)
    gl.scissor = (1, 2, 30, 40)
    n.set_enabled(G.GL_SCISSOR_TEST, True)
    n.set_enabled(G.GL_MULTISAMPLE, False)
    n.depth_mask(False)

    before = guard.snapshot()
    assert len(before) == 15

    with guard:
        gl.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        gl.disable(moderngl.BLEND)
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        gl.wireframe = False
        gl.front_face = "ccw"
        gl.viewport = (0, 0, 256, 192)
        gl.multisample = True
        n.depth_mask(True)
        n.set_enabled(G.GL_SCISSOR_TEST, False)
        fbo = gl.simple_framebuffer((32, 32))
        fbo.use()
        bind_default_framebuffer(gl)
        fbo.release()

    after = guard.snapshot()
    diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    assert not diff
    assert n.drain_errors() == 0


def test_bind_default_framebuffer_does_not_touch_viewport(gl):

    fbo = gl.simple_framebuffer((32, 32))
    fbo.use()
    gl.viewport = (7, 11, 64, 48)
    before = gl.viewport
    bind_default_framebuffer(gl)
    assert gl.viewport == before
    assert gl.fbo is gl.screen
    fbo.release()


def test_integer_attachment_clear_is_exact(gl):

    n = G.native()
    tgt = RenderTarget(gl, 64, 48, samples=min(4, max(gl.info.get("GL_MAX_SAMPLES", 4), 1)))
    try:
        tex_i = gl.texture((8, 8), 1, dtype="u4")
        fbo_i = gl.framebuffer([tex_i])
        try:
            fbo_i.use()
            fbo_i.clear(0.13, 0.13, 0.13, 1.0)
            got = int(np.frombuffer(fbo_i.read(components=1, dtype="u4"), np.uint32)[0])
            n.drain_errors()
            assert got == int(np.float32(0.13).view(np.uint32))
        finally:
            tgt.use_main()
            fbo_i.release()
            tex_i.release()
            n.drain_errors()

        for value in (0, 1, 77, 16_777_217, 4_000_000_000):
            tgt.clear_main((0.1, 0.1, 0.1, 1.0))
            tgt.clear_id(value)
            uniq = np.unique(tgt.read_ids())
            assert len(uniq) == 1 and uniq[0] == value
            assert tgt.read_id(17, 11) == value
        assert n.drain_errors() == 0
    finally:
        tgt.release()


def test_samples_one_is_normalised_to_no_msaa(gl):

    n = G.native()
    for requested in (0, 1):
        tgt = RenderTarget(gl, 64, 48, samples=requested)
        try:
            assert tgt.samples == 0
            tgt.clear_main((0.2, 0.3, 0.4, 1.0))
            tgt.clear_id(9)
            tgt.resolve()
            px = tgt.read_color(flip=False)[10, 10]
            assert abs(int(px[0]) - 51) <= 2 and abs(int(px[2]) - 102) <= 2
            assert list(np.unique(tgt.read_ids())) == [9]
            assert n.drain_errors() == 0
        finally:
            tgt.release()


def test_id_layout_is_probed_not_assumed(gl):

    for samples in (1, 4):
        if samples > max(gl.info.get("GL_MAX_SAMPLES", 1), 1):
            continue
        layout = probe_id_layout(gl, samples)
        tgt = RenderTarget(gl, 32, 32, samples=samples, id_layout=layout)
        try:
            assert tgt.id_layout is layout
            if layout is IdLayout.SHARED:
                assert tgt.id_fbo is tgt.fbo

                assert tgt.id_samples == tgt.samples
            else:
                assert tgt.id_fbo is not tgt.fbo
                assert tgt.id_samples == 0
            tgt.clear_main((0, 0, 0, 1))
            tgt.clear_id(0)
        finally:
            tgt.release()
    assert G.native().drain_errors() == 0


def test_depth_mask_replay_is_defused(gl):

    tgt = RenderTarget(gl, 64, 48, samples=1)
    try:
        prog = gl.program(
            vertex_shader="#version 330 core\nin vec3 p;void main(){gl_Position=vec4(p,1);}",
            fragment_shader="#version 330 core\nout vec4 c;uniform vec4 k;void main(){c=k;}",
        )
        vbo = gl.buffer(np.array([[-1, -1, 0], [3, -1, 0], [-1, 3, 0]], "f4").tobytes())
        vao = gl.vertex_array(prog, [(vbo, "3f", "p")])
        gl.enable(moderngl.DEPTH_TEST)
        gl.depth_func = "<"

        tgt.clear_main((0, 0, 0, 1))
        prog["k"].value = (1, 0, 0, 1)
        vao.render()
        tgt.fbo.depth_mask = False
        tgt.clear_main((0, 0, 0, 1))
        prog["k"].value = (0, 1, 0, 1)
        vao.render()

        tgt.resolve()
        px = tgt.read_color(flip=False)[24, 32]
        assert px[1] > 200
    finally:
        tgt.release()


def _quad() -> MeshData:
    p = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
    n = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return MeshData(p, n, uv, np.array([0, 1, 2, 0, 2, 3], np.uint32))


def _tri() -> MeshData:
    p = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], np.float32)
    n = np.tile(np.array([0, 0, 1], np.float32), (3, 1))
    return MeshData(p, n, np.zeros((3, 2), np.float32), np.array([0, 1, 2], np.uint32))


_VS = """#version 330 core
in vec3 in_position; in vec3 in_normal; in vec2 in_uv;
in vec4 in_model0; in vec4 in_model1; in vec4 in_model2; in vec4 in_model3;
in vec4 in_color; in vec4 in_material; in vec4 in_texcoef;
uniform mat4 u_view_proj;
out vec4 v_color;
void main() {
    mat4 m = mat4(in_model0, in_model1, in_model2, in_model3);
    v_color = in_color + vec4(in_material.x, in_texcoef.x - 1.0, in_normal.z - 1.0, 0.0)
            + vec4(in_uv, 0.0, 0.0) * 0.0;
    gl_Position = u_view_proj * m * vec4(in_position, 1.0);
}
"""
_FS = "#version 330 core\nin vec4 v_color; out vec4 o;\nvoid main(){ o = v_color; }\n"

_COLORS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (0.5, 0, 0),
    (0, 0.5, 0),
    (0, 0, 0.5),
]


def _build_scene():
    b = SceneBuilder()
    m0 = b.material_id(Material(name="a"))
    m1 = b.material_id(Material(name="b"))
    plan = [
        (MeshShape.BOX, m0),
        (MeshShape.BOX, m0),
        (MeshShape.BOX, m0),
        (MeshShape.SPHERE, m0),
        (MeshShape.SPHERE, m0),
        (MeshShape.BOX, m1),
        (MeshShape.BOX, m1),
        (MeshShape.SPHERE, m1),
        (MeshShape.SPHERE, m1),
    ]
    for i, (shape, mat) in enumerate(plan):
        x = np.linspace(-0.8, 0.8, 9)[i]
        b.add(
            MeshKey(shape),
            mat,
            M.compose([x, 0.0, 0.0], np.eye(3), [0.07, 0.07, 0.07]),
            [*_COLORS[i], 1.0],
            [0, 0.5, 0.5, 0],
            object_id=i + 1,
        )
    cam = CameraView(
        orthographic=True,
        ortho_height=2.0,
        eye=np.array([0, 0, 3], np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0, 1, 0], np.float32),
    )
    return b.build(cam, LightSet(), 1.0, np.zeros(3))


@pytest.mark.parametrize("strategy", [Strategy.SHARED, Strategy.PER_BUCKET])
def test_both_instance_strategies_draw_the_same_thing(gl, strategy):

    scene = _build_scene()
    store = MeshStore(gl)
    store.sync({MeshKey(MeshShape.BOX): _quad(), MeshKey(MeshShape.SPHERE): _tri()})
    bucket_meshes = [store.get(k) for k, _ in scene.bucket_keys]
    prog = gl.program(vertex_shader=_VS, fragment_shader=_FS)
    prog["u_view_proj"].value = tuple(
        M.to_gl(scene.camera.proj_matrix() @ scene.camera.view_matrix()).ravel()
    )
    fbo = gl.simple_framebuffer((256, 256))
    inst = InstanceStore(gl)
    inst.strategy = strategy
    if strategy is Strategy.SHARED and not G.native().has_attrib_pointer:
        pytest.skip("native glVertexAttribPointer unavailable")
    try:
        inst.rebuild(scene, prog, bucket_meshes, generation=0)
        inst.upload(scene)
        fbo.use()
        fbo.clear(0, 0, 0, 1)
        gl.enable_only(moderngl.NOTHING)
        drawn = sum(inst.draw(b) for b in range(scene.bucket_count()))
        img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(256, 256, 3)

        seen = {tuple(int(v) for v in c) for c in img.reshape(-1, 3)} - {(0, 0, 0)}
        expect = {tuple(round(v * 255) for v in c) for c in _COLORS}
        assert drawn == 9
        assert inst.draw_calls == 4
        # UNORM conversion of an interpolated 0.5 may land on either adjacent
        # 8-bit value across drivers; the instance payload itself is exact.
        assert len(seen) == len(expect)
        assert all(
            any(max(abs(got[i] - want[i]) for i in range(3)) <= 1 for got in seen)
            for want in expect
        )
        assert G.native().drain_errors() == 0
    finally:
        inst.release()
        store.release()
        fbo.release()


def test_capacity_grows_by_doubling(gl):

    inst = InstanceStore(gl)
    try:
        inst._ensure_capacity(10)
        assert inst.capacity == 64
        first = inst.buffer
        inst._ensure_capacity(50)
        assert inst.buffer is first
        inst._ensure_capacity(65)
        assert inst.capacity == 128
    finally:
        inst.release()


def test_object_id_survives_packing_as_an_exact_uint32(gl):

    scene = _build_scene()
    scene.object_id = np.array(
        [1, 2, 3, 16_777_216, 16_777_217, 4_000_000_000, 4_294_967_295, 12345, 7],
        np.uint32,
    )
    inst = InstanceStore(moderngl.get_context())
    inst._ensure_capacity(scene.count)
    raw = inst.pack(scene)
    assert raw.dtype == np.uint32
    assert np.array_equal(raw[:, 32], scene.object_id)
    inst.release()


@pytest.mark.parametrize("strategy", [Strategy.SHARED, Strategy.PER_BUCKET])
def test_object_id_reaches_the_shader_exactly(gl, strategy):

    from forge_viewer.render.forge.instances import INSTANCE_ATTRIBUTES

    ids = np.array(
        [1, 2, 3, 16_777_216, 16_777_217, 4_000_000_000, 4_294_967_295, 12345, 7], np.uint32
    )
    scene = _build_scene()
    scene.object_id = ids

    prog = gl.program(
        vertex_shader="""#version 330 core
in vec3 in_position;
in vec4 in_model0; in vec4 in_model1; in vec4 in_model2; in vec4 in_model3;
in uint in_object_id;
uniform mat4 u_view_proj;
flat out uint v_id;
void main() {
    mat4 m = mat4(in_model0, in_model1, in_model2, in_model3);
    v_id = in_object_id;
    gl_Position = u_view_proj * m * vec4(in_position, 1.0);
}
""",
        fragment_shader="""#version 330 core
flat in uint v_id;
layout(location = 0) out uint o_id;
void main() { o_id = v_id; }
""",
    )
    assert "in_object_id" in prog
    assert any(a[0] == "in_object_id" and a[5] != 0x1406 for a in INSTANCE_ATTRIBUTES)
    prog["u_view_proj"].value = tuple(
        M.to_gl(scene.camera.proj_matrix() @ scene.camera.view_matrix()).ravel()
    )

    store = MeshStore(gl)
    store.sync({MeshKey(MeshShape.BOX): _quad(), MeshKey(MeshShape.SPHERE): _tri()})
    inst = InstanceStore(gl)
    inst.strategy = strategy
    if strategy is Strategy.SHARED and not G.native().has_attrib_pointer:
        pytest.skip("native glVertexAttribPointer unavailable")
    id_tex = gl.texture((256, 256), 1, dtype="u4")
    fbo = gl.framebuffer([id_tex])
    try:
        inst.rebuild(scene, prog, [store.get(k) for k, _ in scene.bucket_keys], generation=0)
        inst.upload(scene)
        fbo.use()
        gl.viewport = (0, 0, 256, 256)
        gl.enable_only(moderngl.NOTHING)

        gl.clear(0.0, 0.0, 0.0, 0.0)
        for b in range(scene.bucket_count()):
            inst.draw(b)
        got = np.frombuffer(fbo.read(components=1, dtype="u4"), np.uint32)
        seen = set(np.unique(got).tolist()) - {0}
        assert seen == set(ids.tolist())
        assert G.native().drain_errors() == 0
    finally:
        inst.release()
        store.release()
        fbo.release()
        id_tex.release()


def test_pack_transposes_row_major_to_column_major(gl):

    scene = _build_scene()
    inst = InstanceStore(moderngl.get_context())
    inst._ensure_capacity(scene.count)

    packed = inst.pack(scene).view(np.float32)
    for i in range(scene.count):
        expect = M.to_gl(scene.transforms[i]).ravel()
        assert np.allclose(packed[i, :16], expect)

    assert np.allclose(packed[:, 12], scene.transforms[:, 0, 3])
    inst.release()


def test_shader_compile_failure_keeps_last_good_program(gl, tmp_path):

    (tmp_path / "a.vert").write_text("#version 330 core\nvoid main(){gl_Position=vec4(0,0,0,1);}")
    (tmp_path / "a.frag").write_text("#version 330 core\nout vec4 c;void main(){c=vec4(1);}")
    cache = ProgramCache(gl, tmp_path)
    spec = ProgramSpec(name="t", vertex="a.vert", fragment="a.frag")
    good = cache.get(spec)
    gen = cache.generation

    import time

    time.sleep(0.01)
    (tmp_path / "a.frag").write_text("#version 330 core\nthis is not GLSL")
    changed = cache.reload_changed()
    assert changed == []
    assert cache.get(spec) is good
    assert cache.generation == gen
    assert cache.last_error

    time.sleep(0.01)
    (tmp_path / "a.frag").write_text("#version 330 core\nout vec4 c;void main(){c=vec4(0.5);}")
    assert cache.reload_changed() == ["t"]
    assert cache.generation == gen + 1
    cache.release()


def test_program_cache_fork_uses_the_parent_shader_snapshot(gl, tmp_path):
    vertex = "#version 330 core\nvoid main(){gl_Position=vec4(0,0,0,1);}"
    original = "#version 330 core\nout vec4 c;void main(){c=vec4(1);}"
    changed = "#version 330 core\nout vec4 c;void main(){c=vec4(0);}"
    (tmp_path / "a.vert").write_text(vertex, encoding="utf-8")
    (tmp_path / "a.frag").write_text(original, encoding="utf-8")
    cache = ProgramCache(gl, tmp_path)
    spec = ProgramSpec(name="snapshot", vertex="a.vert", fragment="a.frag")
    cache.get(spec)
    (tmp_path / "a.frag").write_text(changed, encoding="utf-8")
    peer = cache.fork()
    try:
        peer.get(spec)
        sources, _deps = peer._compiled_sources[spec.key()]
        assert sources["fragment_shader"] == original
    finally:
        peer.release()
        cache.release()


def test_uniform_cache_skips_unchanged_writes(gl):

    prog = gl.program(
        vertex_shader="#version 330 core\nuniform float u_a;void main(){gl_Position=vec4(u_a);}",
        fragment_shader="#version 330 core\nout vec4 c;void main(){c=vec4(1);}",
    )
    cache = UniformCache(prog)
    writes = 0
    member = prog["u_a"]

    class Counting:
        @property
        def value(self):
            return member.value

        @value.setter
        def value(self, v):
            nonlocal writes
            writes += 1
            member.value = v

    cache._program = type("P", (), {"__getitem__": lambda _s, _k: Counting()})()
    for _ in range(10):
        cache.set("u_a", 1.0)
    assert writes == 1
    cache.set("u_a", 2.0)
    assert writes == 2
    cache.set("u_nonexistent", 1.0)
    assert writes == 2


def test_gpu_timing_degrades_to_empty_table(gl):

    t = FrameTiming(gl, enabled=True)
    with t.scope("opaque"):
        gl.clear()
    t.collect()
    assert "opaque" in t.cpu_table()
    if t.gpu_available:
        assert "opaque" in t.gpu_table()
    else:
        assert t.gpu_table() == {}

    off = FrameTiming(gl, enabled=False)
    with off.scope("opaque"):
        gl.clear()
    off.collect()
    assert off.gpu_table() == {}
    assert "opaque" in off.cpu_table()
