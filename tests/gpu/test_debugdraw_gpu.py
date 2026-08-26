from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

moderngl = pytest.importorskip("moderngl")

from forge_viewer.math3d import look_at, perspective  # noqa: E402
from forge_viewer.render.debugdraw import Occlusion  # noqa: E402
from forge_viewer.render.forge import gl_native as G  # noqa: E402
from forge_viewer.render.forge.instances import InstanceStore  # noqa: E402
from forge_viewer.render.forge.passes.base import PassContext  # noqa: E402
from forge_viewer.render.forge.passes.debug import DebugPass  # noqa: E402
from forge_viewer.render.forge.programs import ProgramCache  # noqa: E402
from forge_viewer.render.forge.resources import TextureStore  # noqa: E402
from forge_viewer.render.forge.targets import RenderTarget  # noqa: E402
from forge_viewer.render.forge.timing import FrameTiming  # noqa: E402
from forge_viewer.render.scene import RenderScene  # noqa: E402
from forge_viewer.types import CameraView  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _forge_backend_only(require_forge):
    pass


WIDTH, HEIGHT = 400, 300
BACKGROUND = (0.0, 0.0, 0.0, 1.0)
WALL_RGBA = (0.10, 0.12, 0.14, 1.0)
LINE_RGBA = (1.0, 0.35, 0.10, 1.0)


_WALL_VS = """#version 330 core
in vec3 in_pos;
uniform mat4 u_view_proj;
void main() { gl_Position = u_view_proj * vec4(in_pos, 1.0); }
"""
_WALL_FS = """#version 330 core
uniform vec4 u_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = u_color; }
"""


class Rig:
    def __init__(self, ctx) -> None:
        self.ctx = ctx

        self.target = RenderTarget(ctx, WIDTH, HEIGHT, samples=0)
        self.programs = ProgramCache(ctx)
        self.instances = InstanceStore(ctx)
        self.textures = TextureStore(ctx)
        self.timing = FrameTiming(ctx, enabled=False)
        self.pass_ = DebugPass()
        self.draw = self.pass_.draw
        self.camera = CameraView(
            eye=np.array([0.0, -4.0, 0.0], np.float32),
            target=np.zeros(3, np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            aspect=WIDTH / HEIGHT,
        )
        self._wall_prog = ctx.program(vertex_shader=_WALL_VS, fragment_shader=_WALL_FS)
        self._wall_vbo = ctx.buffer(reserve=6 * 3 * 4)
        self._wall_vao = ctx.vertex_array(self._wall_prog, [(self._wall_vbo, "3f", "in_pos")])

    # ------------------------------------------------------------------
    def context(self) -> PassContext:
        view = look_at(self.camera.eye, self.camera.target, self.camera.up)
        proj = perspective(self.camera.fov_y, WIDTH / HEIGHT, self.camera.near, self.camera.far)
        return PassContext(
            ctx=self.ctx,
            target=self.target,
            scene=RenderScene(),
            camera=self.camera,
            view=view,
            proj=proj,
            view_proj=(proj @ view).astype(np.float32),
            instances=self.instances,
            programs=self.programs,
            textures=self.textures,
            meshes=[],
            timing=self.timing,
            flags={},
        )

    def render(self, wall: tuple[float, float] | None = None) -> None:

        ctx = self.context()
        self.target.clear_main(BACKGROUND)
        self.target.use_main()
        if wall is not None:
            self._draw_wall(ctx, *wall)
        if self.pass_.prepare(ctx):
            self.pass_.execute(ctx)

    def _draw_wall(self, ctx: PassContext, x0: float, x1: float, depth: float = -1.0) -> None:

        quad = np.array(
            [
                [x0, depth, -3.0],
                [x1, depth, -3.0],
                [x0, depth, 3.0],
                [x1, depth, -3.0],
                [x1, depth, 3.0],
                [x0, depth, 3.0],
            ],
            np.float32,
        )
        self._wall_vbo.write(quad)
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.target.fbo.depth_mask = True
        self._wall_prog["u_view_proj"].write(np.ascontiguousarray(ctx.view_proj.T))
        self._wall_prog["u_color"].value = WALL_RGBA
        self._wall_vao.render(moderngl.TRIANGLES, vertices=6)

    def pixels(self) -> np.ndarray:

        self.target.resolve()
        return self.target.read_color(flip=False)

    def release(self) -> None:
        self.pass_.release()
        self._wall_vao.release()
        self._wall_vbo.release()
        self._wall_prog.release()
        self.timing.release() if hasattr(self.timing, "release") else None
        self.instances.release()
        self.textures.release()
        self.programs.release()
        self.target.release()


@pytest.fixture
def rig(gl_ctx):
    r = Rig(gl_ctx)
    yield r
    r.release()
    assert G.native().drain_errors() == 0


def _rgb(px: np.ndarray, x: int, y: int) -> np.ndarray:
    return px[y, x, :3].astype(np.float64)


def _mix_fraction(sample: np.ndarray, base: np.ndarray, full: np.ndarray) -> float:

    d = full - base
    k = int(np.argmax(np.abs(d)))
    return float((sample[k] - base[k]) / d[k])


@pytest.mark.parametrize("occlusion", [Occlusion.DEPTH, Occlusion.ALWAYS, Occlusion.GHOST])
def test_three_occlusion_modes(rig, occlusion):

    layer = rig.draw.layer("ray", occlusion)
    layer.line("ray", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), LINE_RGBA, 9.0)
    rig.render(wall=(-3.0, 0.0))
    px = rig.pixels()

    mid_y = HEIGHT // 2
    left, right = WIDTH // 4, WIDTH * 3 // 4
    hidden = _rgb(px, left, mid_y)
    visible = _rgb(px, right, mid_y)
    wall_only = _rgb(px, left, HEIGHT // 6)
    background = _rgb(px, right, HEIGHT // 6)

    assert wall_only.sum() > 0
    assert background.sum() == 0
    assert visible[0] > 200

    if occlusion is Occlusion.DEPTH:
        assert np.allclose(hidden, wall_only)
    elif occlusion is Occlusion.ALWAYS:
        assert np.allclose(hidden, visible)
    else:
        t = _mix_fraction(hidden, wall_only, visible)
        print(f"\n[metric] ghost occlusion blend t={t:.3f}")
        assert not np.allclose(hidden, wall_only)
        assert not np.allclose(hidden, visible)
        assert 0.05 < t < 0.6


def test_ghost_draws_twice_and_depth_stays_untouched(rig):

    layer = rig.draw.layer("ray", Occlusion.GHOST)
    layer.line("ray", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), LINE_RGBA, 9.0)
    rig.render(wall=(-3.0, 0.0))
    assert rig.pass_.draw_calls == 2

    rig.draw.layer("ray", Occlusion.GHOST).line(
        "ray", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), LINE_RGBA, 9.0
    )
    ctx = rig.context()
    rig.target.clear_main(BACKGROUND)
    rig.target.use_main()
    if rig.pass_.prepare(ctx):
        rig.pass_.execute(ctx)
    rig._draw_wall(ctx, -3.0, 3.0, depth=1.0)
    px = rig.pixels()
    wall_after = _rgb(px, WIDTH // 2, HEIGHT // 2)
    assert np.allclose(wall_after, np.array(WALL_RGBA[:3]) * 255, atol=2)


@pytest.mark.parametrize("width_px", [3.0, 9.0])
def test_line_width_is_constant_on_screen_across_depths(rig, width_px):

    layer = rig.draw.layer("width", Occlusion.ALWAYS)
    layer.line("near", (-1.0, 0.0, -0.5), (1.0, 0.0, -0.5), LINE_RGBA, width_px)
    layer.line("far", (-1.0, 3.0, 0.5), (1.0, 3.0, 0.5), LINE_RGBA, width_px)
    rig.render()
    px = rig.pixels()

    column = px[:, WIDTH // 2, 0] > 60
    runs = _runs(column)
    assert len(runs) == 2
    near_len, far_len = (
        sorted(runs, key=lambda r: -r[0])[0][1],
        sorted(runs, key=lambda r: -r[0])[1][1],
    )
    print(f"\n[metric] width {width_px:g} px: near={near_len}, far={far_len}")

    for got in (near_len, far_len):
        assert abs(got - width_px) <= 1.0
    assert abs(near_len - far_len) <= 1.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:

    out: list[tuple[int, int]] = []
    start = -1
    for i, on in enumerate(mask):
        if on and start < 0:
            start = i
        elif not on and start >= 0:
            out.append((start, i - start))
            start = -1
    if start >= 0:
        out.append((start, len(mask) - start))
    return out


def test_closed_stroke_has_one_continuous_core_and_outer_outline(rig):
    points = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32)
    layer = rig.draw.layer("stroke", Occlusion.ALWAYS)
    layer.polyline("outer", points, (0.52, 0.54, 0.58, 1.0), 9.0, closed=True)
    layer.polyline("core", points, (1.0, 1.0, 1.0, 1.0), 3.0, closed=True)
    rig.render()
    pixels = rig.pixels()

    ctx = rig.context()
    clip = np.column_stack((points, np.ones(len(points)))) @ ctx.view_proj.T
    ndc = clip[:, :2] / clip[:, 3:4]
    screen = np.column_stack(((ndc[:, 0] * 0.5 + 0.5) * WIDTH, (ndc[:, 1] * 0.5 + 0.5) * HEIGHT))
    for x, y in np.rint(screen).astype(int):
        assert pixels[y, x, :3].min() > 240

    top_mid = np.rint((screen[2] + screen[3]) * 0.5).astype(int)
    gray = pixels[top_mid[1] + 3, top_mid[0], :3]
    assert 100 < gray.min() < 220


def test_drag_link_is_one_compound_shape(rig):
    """The ring, connector, and target share one outline with no internal seam."""
    start = np.array([-0.8, 0.0, 0.0], np.float32)
    target = np.array([0.8, 0.0, 0.0], np.float32)
    rig.draw.layer("drag", Occlusion.ALWAYS).drag_link(
        "drag",
        start,
        target,
        (1.0, 1.0, 1.0, 1.0),
        (0.55, 0.58, 0.63, 1.0),
        width_px=4.0,
        radius_px=10.0,
        edge_px=2.0,
    )
    rig.render()
    pixels = rig.pixels()

    ctx = rig.context()
    points = np.column_stack((np.stack((start, target)), np.ones(2)))
    clip = points @ ctx.view_proj.T
    screen = (clip[:, :2] / clip[:, 3:4] * 0.5 + 0.5) * np.array((WIDTH, HEIGHT))
    (sx, sy), (tx, _ty) = np.rint(screen).astype(int)

    assert pixels[sy, sx, :3].max() == 0, "the start anchor must remain hollow"
    assert pixels[sy, sx + 3, :3].max() == 0, "the connector must not enter the ring cavity"
    assert pixels[sy, tx, :3].min() > 240, "the target must be filled"
    core = pixels[sy, sx + 8 : tx + 1, :3]
    assert core.min() > 235, "the connector must merge into both endpoints without a gray seam"
    edge = pixels[sy + 3, (sx + tx) // 2, :3]
    assert 100 < edge.min() < 220, "the compound shape must expose one thin contrast outline"


def test_arrow_head_shape_is_identical_when_depth_direction_is_reversed(rig):

    ctx = rig.context()
    inv_view = np.linalg.inv(ctx.view)
    p00, p11 = float(ctx.proj[0, 0]), float(ctx.proj[1, 1])

    def world(ndc_x: float, ndc_y: float, depth: float) -> np.ndarray:
        view = np.array([ndc_x * depth / p00, ndc_y * depth / p11, -depth, 1.0])
        return (inv_view @ view)[:3].astype(np.float32)

    layer = rig.draw.layer("arrows", Occlusion.ALWAYS)
    layer.arrow("far", world(-0.35, -0.35, 3.0), world(0.35, -0.35, 6.0), LINE_RGBA, 4.4)
    layer.arrow("near", world(-0.35, 0.35, 6.0), world(0.35, 0.35, 3.0), LINE_RGBA, 4.4)
    rig.render()
    mask = rig.pixels()[:, :, 0] > 100

    x0, x1 = int(WIDTH * 0.325) - 10, int(WIDTH * 0.675) + 10
    half_h = 18
    rows = [int(HEIGHT * (0.5 + y * 0.5)) for y in (-0.35, 0.35)]
    shapes = [mask[row - half_h : row + half_h + 1, x0:x1] for row in rows]
    assert np.array_equal(shapes[0], shapes[1])


def test_faded_arrow_is_one_continuous_silhouette(rig):

    a = np.array([-1.5, 0.0, 0.0], np.float32)
    b = np.array([1.5, 0.0, 0.0], np.float32)
    rig.draw.layer("arrow", Occlusion.ALWAYS).arrow("a", a, b, (1.0, 0.3, 0.1, 0.35), 4.4)
    rig.render()
    pixels = rig.pixels()

    ctx = rig.context()
    points = np.column_stack((np.stack((a, b)), np.ones(2)))
    clip = points @ ctx.view_proj.T
    ndc = clip[:, :2] / clip[:, 3:4]
    screen = (ndc * 0.5 + 0.5) * np.array((WIDTH, HEIGHT))
    x0, x1 = np.rint(screen[:, 0]).astype(int)
    y = round(float(screen[0, 1]))
    core = pixels[y, x0 + 3 : x1 - 3, 0]
    assert core.min() > 70
    assert int(core.max()) - int(core.min()) <= 3


def test_invisible_center_shell_masks_only_the_axis(rig):
    layer = rig.draw.layer("readonly-gizmo", Occlusion.ALWAYS)
    layer.arrow("x", (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), LINE_RGBA, 4.4, start_mask_px=10.0)
    layer.point("center", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), 6.5)
    rig.render()
    pixels = rig.pixels()
    x, y = WIDTH // 2, HEIGHT // 2

    assert pixels[y, x, :3].min() > 240
    assert pixels[y, x + 8, :3].max() == 0
    assert pixels[y, x + 12, 0] > 200


def test_ten_thousand_lines_hold_the_frame_budget(rig):

    n = 10_000
    rng = np.random.default_rng(11)
    pts_a = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    pts_b = pts_a + rng.normal(scale=0.05, size=(n, 3)).astype(np.float32)
    rig.draw.layer("bulk", Occlusion.DEPTH).lines("bulk", pts_a, pts_b, LINE_RGBA, 2.0)

    for _ in range(5):
        rig.render()
    rig.ctx.finish()

    frames = 30
    t0 = time.perf_counter()
    for _ in range(frames):
        rig.render()
    rig.ctx.finish()
    ms = (time.perf_counter() - t0) * 1000.0 / frames

    print(f"\n[metric] 10k lines: {ms:.3f} ms/frame ({frames} frames, including finish)")
    assert rig.draw.stats().primitives == n
    assert rig.pass_.draw_calls == 1
    assert ms < 16.6


def test_hundred_frames_grow_neither_primitives_nor_draw_calls(rig):

    layers = [rig.draw.layer(f"l{i}", Occlusion.ALWAYS) for i in range(3)]
    seen = set()
    for frame in range(100):
        for i, layer in enumerate(layers):
            layer.line(
                f"l{i}", (-1.0, 0.0, i * 0.2), (1.0, 0.0 + frame * 0.001, i * 0.2), LINE_RGBA, 3.0
            )
        rig.render()
        seen.add((rig.draw.stats().primitives, rig.pass_.draw_calls))
    assert seen == {(3, 1)}


def test_every_primitive_kind_reaches_the_screen(rig):

    m = np.eye(4, dtype=np.float32)
    m[:3, :3] *= 0.35
    for i, primitive_name in enumerate(("box", "sphere")):
        t = m.copy()
        t[:3, 3] = (-1.0 + 2.0 * i, 0.0, -1.0)
        getattr(rig.draw.layer("solids", Occlusion.DEPTH), primitive_name)(
            primitive_name, t, (0.3, 0.8, 0.4, 1.0)
        )

    layer = rig.draw.layer("marks", Occlusion.ALWAYS)
    layer.line("l", (-1.5, 0.0, 1.0), (1.5, 0.0, 1.0), LINE_RGBA, 4.0)
    layer.arrow("a", (-1.5, 0.0, 0.6), (1.5, 0.0, 0.6), LINE_RGBA, 4.0)
    layer.point("p", (0.0, 0.0, 1.4), (1.0, 1.0, 1.0, 1.0), 10.0)
    frame_m = np.eye(4, dtype=np.float32)
    frame_m[:3, 3] = (-1.6, 0.0, -1.6)
    layer.frame("f", frame_m, 0.6)
    layer.sector("s", (1.2, 0.0, -1.6), (1.2, -1.0, -1.6), (1.8, 0.0, -1.6), (0.9, 0.8, 0.2, 0.9))

    rig.render()
    px = rig.pixels()
    lit = int(np.count_nonzero(px[:, :, :3].max(axis=2) > 20))
    print(f"\n[metric] seven primitive types cover {lit} pixels")
    assert rig.draw.stats().dropped == 0
    assert lit > 2000

    assert rig.pass_.draw_calls == 5


class _NoAttribPointer:
    has_attrib_pointer = False

    def rebind_instance_attributes(self, *_args) -> bool:
        return False


def test_the_no_native_symbol_fallback_draws_the_same_picture(rig):

    def paint() -> None:
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] *= 0.4
        for i, primitive_name in enumerate(("box", "sphere")):
            t = m.copy()
            t[:3, 3] = (-1.0 + 2.0 * i, 0.0, -0.8)
            getattr(rig.draw.layer("solids", Occlusion.GHOST), primitive_name)(
                primitive_name, t, (0.3, 0.8, 0.4, 1.0)
            )
        rig.draw.layer("marks", Occlusion.GHOST).line(
            "l", (-1.5, 0.0, 1.0), (1.5, 0.0, 1.0), LINE_RGBA, 5.0
        )

    paint()
    rig.render(wall=(-4.0, 4.0))
    native = rig.pixels().copy()
    wall_px = np.array(WALL_RGBA[:3]) * 255
    marked = int(np.count_nonzero(np.abs(native[:, :, :3] - wall_px).max(axis=2) > 4))
    assert marked > 500

    saved, rig.pass_._gl = rig.pass_._gl, _NoAttribPointer()
    try:
        paint()
        rig.render(wall=(-4.0, 4.0))
        fallback = rig.pixels().copy()
    finally:
        rig.pass_._gl = saved

    diff = int(np.count_nonzero(np.abs(native.astype(int) - fallback.astype(int)).max(axis=2) > 1))
    assert diff == 0
