"""Backend-neutral debug-draw and world-text tests.

Runs under both backends via the ``backend_name`` fixture (conftest.py); the
forge pass-level tests live in test_debugdraw_gpu.py.  Geometry mirrors that
file: the camera sits at -Y looking at the origin with +Z up, debug lines span
X across the screen center, and occlusion tests insert a thin wall box halfway
between camera and line covering the left half of the screen.

Unlike the forge rig the occluder here is a real scene instance, so both
backends run their full pipeline (scene pass plus debug pass) end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer import math3d as M  # noqa: E402
from forge_viewer.adapters.base import SceneSource  # noqa: E402
from forge_viewer.render.debugdraw import Occlusion  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    LightSet,
    Material,
    MeshKey,
    MeshShape,
)

W, H = 400, 300
BACKGROUND = (0.0, 0.0, 0.0, 1.0)
WALL_RGBA = (0.40, 0.45, 0.50, 1.0)
LINE_RGBA = (1.0, 0.35, 0.10, 1.0)

BOX = MeshKey(MeshShape.BOX)
# Half-extents placing the wall at y in [-1.05, -0.95], x in [-3, 0], z in [-3, 3].
WALL_POS = np.array([-1.5, -1.0, 0.0], np.float32)
WALL_SCALE = np.array([1.5, 0.05, 3.0], np.float32)

MATERIAL = np.array([0.0, 0.0, 0.5, 0.0], np.float32)
AMBIENT = np.full(3, 0.3, np.float32)


def _camera() -> CameraView:
    return CameraView(
        eye=np.array([0.0, -4.0, 0.0], np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0.0, 0.0, 1.0], np.float32),
        aspect=W / H,
    )


class Rig:
    def __init__(self, backend) -> None:
        self.backend = backend
        backend.set_background(BACKGROUND)
        backend.set_scene(SceneSource(meshes={}, textures={}, skybox=None))
        self.camera = _camera()
        backend.set_camera(self.camera)

    def draw(self, wall: bool = False) -> np.ndarray:
        sb = SceneBuilder()
        if wall:
            matid = sb.material_id(Material())
            sb.add(
                BOX,
                matid,
                M.compose(WALL_POS, np.eye(3), WALL_SCALE),
                np.asarray(WALL_RGBA, np.float32),
                MATERIAL,
                object_id=1,
            )
        scene = sb.build(self.camera, LightSet(ambient=AMBIENT), 2.0, np.zeros(3, np.float32))
        self.backend.set_render_scene(scene)
        # The wgpu backend renders offscreen only and returns no ViewportImage.
        if self.backend.caps.name == "webgpu":
            self.backend.render(None)
        else:
            assert self.backend.render(None) is not None
        return self.backend.target.read_color(flip=True)

    def debug_pass(self):
        if self.backend.caps.name == "webgpu":
            return self.backend._debug
        return self.backend._passes["debug"]


def _make_backend(backend_name: str, request, samples: int = 4):
    """Build the backend selected by FORGE_VIEWER_BACKEND; GL stays lazy."""
    if backend_name == "wgpu":
        from forge_viewer.render.webgpu.backend import WgpuBackend

        return WgpuBackend(W, H, samples=samples)
    from forge_viewer.render.forge import passes
    from forge_viewer.render.forge.backend import ForgeBackend

    passes.load_all()
    return ForgeBackend(request.getfixturevalue("gl_ctx"), W, H, samples=samples)


@pytest.fixture
def rig(backend_name, request):
    backend = _make_backend(backend_name, request)
    yield Rig(backend)
    backend.release()


def _rgb(px: np.ndarray, x: int, y: int) -> np.ndarray:
    return px[y, x, :3].astype(np.float64)


def _mix_fraction(sample: np.ndarray, base: np.ndarray, full: np.ndarray) -> float:
    d = full - base
    k = int(np.argmax(np.abs(d)))
    return float((sample[k] - base[k]) / d[k])


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


@pytest.mark.parametrize("occlusion", [Occlusion.DEPTH, Occlusion.ALWAYS, Occlusion.GHOST])
def test_three_occlusion_modes(rig, occlusion):
    layer = rig.backend.debug.layer("ray", occlusion)
    layer.line("ray", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), LINE_RGBA, 9.0)
    px = rig.draw(wall=True)

    mid_y = H // 2
    left, right = W // 4, W * 3 // 4
    hidden = _rgb(px, left, mid_y)
    visible = _rgb(px, right, mid_y)
    wall_only = _rgb(px, left, H // 6)
    background = _rgb(px, right, H // 6)

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


def test_ghost_draws_twice(rig):
    layer = rig.backend.debug.layer("ray", Occlusion.GHOST)
    layer.line("ray", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), LINE_RGBA, 9.0)
    rig.draw(wall=True)
    assert rig.debug_pass().draw_calls == 2


@pytest.mark.parametrize("width_px", [3.0, 9.0])
def test_line_width_is_constant_on_screen_across_depths(rig, width_px):
    layer = rig.backend.debug.layer("width", Occlusion.ALWAYS)
    layer.line("near", (-1.0, 0.0, -0.5), (1.0, 0.0, -0.5), LINE_RGBA, width_px)
    layer.line("far", (-1.0, 3.0, 0.5), (1.0, 3.0, 0.5), LINE_RGBA, width_px)
    px = rig.draw()

    column = px[:, W // 2, 0] > 60
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


def test_every_primitive_kind_reaches_the_screen(rig):
    draw = rig.backend.debug
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] *= 0.35
    for i, kind in enumerate(("box", "sphere")):
        t = m.copy()
        t[:3, 3] = (-1.0 + 2.0 * i, 0.0, -1.0)
        getattr(draw.layer("solids", Occlusion.DEPTH), kind)(kind, t, (0.3, 0.8, 0.4, 1.0))

    layer = draw.layer("marks", Occlusion.ALWAYS)
    layer.line("l", (-1.5, 0.0, 1.0), (1.5, 0.0, 1.0), LINE_RGBA, 4.0)
    layer.arrow("a", (-1.5, 0.0, 0.6), (1.5, 0.0, 0.6), LINE_RGBA, 4.0)
    layer.point("p", (0.0, 0.0, 1.4), (1.0, 1.0, 1.0, 1.0), 10.0)
    frame_m = np.eye(4, dtype=np.float32)
    frame_m[:3, 3] = (-1.6, 0.0, -1.6)
    layer.frame("f", frame_m, 0.6)
    layer.sector("s", (1.2, 0.0, -1.6), (1.2, -1.0, -1.6), (1.8, 0.0, -1.6), (0.9, 0.8, 0.2, 0.9))

    px = rig.draw()
    lit = int(np.count_nonzero(px[:, :, :3].max(axis=2) > 20))
    print(f"\n[metric] seven primitive kinds cover {lit} pixels")
    assert draw.stats().dropped == 0
    assert lit > 2000

    assert rig.debug_pass().draw_calls == 5


def test_world_text_renders_over_the_scene(rig):
    rig.backend.configure_text(size_px=16.0)
    layer = rig.backend.debug.layer("labels", Occlusion.ALWAYS)
    layer.text("label", (0.0, 0.0, 1.0), "M7 overlay", (1.0, 1.0, 1.0, 1.0))
    with_text = rig.draw()

    rig.backend.debug.clear()
    without_text = rig.draw()

    diff = np.abs(with_text.astype(int) - without_text.astype(int)).max(axis=2)
    changed = int(np.count_nonzero(diff > 20))
    print(f"\n[metric] world text changed {changed} pixels")
    assert changed > 20


def test_ten_thousand_lines_render_without_dropping(rig):
    n = 10_000
    rng = np.random.default_rng(11)
    pts_a = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
    pts_b = pts_a + rng.normal(scale=0.05, size=(n, 3)).astype(np.float32)
    rig.backend.debug.layer("bulk", Occlusion.DEPTH).lines("bulk", pts_a, pts_b, LINE_RGBA, 2.0)

    for _ in range(5):
        rig.draw()
    for _ in range(30):
        rig.draw()

    # No per-frame budget assertion here (unlike the forge-only rig): the two
    # backends time frames differently; this checks correctness of the path.
    assert rig.backend.debug.stats().primitives == n
    assert rig.debug_pass().draw_calls == 1
