"""Backend-neutral native-gizmo render tests.

Runs under both backends via the ``backend_name`` fixture (conftest.py); the
CPU-side gizmo geometry/hit-test coverage lives in tests/test_gizmo.py.  The
scene is a single box at the origin with the gizmo on top of it, which also
proves the near-plane depth pinning draws handles over scene geometry.

Expected screen positions come from the shared ``forge_viewer.gizmo``
projection helpers (its ``project`` matches both backends in xy; only clip z
conventions differ), so both backends assert identical geometry.  Tolerances
only cover MSAA and shading rounding (the gizmo shader multiplies the handle
color by a 0.72..1.0 facing term).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer import math3d as M  # noqa: E402
from forge_viewer.adapters.base import SceneSource  # noqa: E402
from forge_viewer.gizmo import (  # noqa: E402
    AXIS_COLORS,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_PT,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    GizmoFrame,
    GizmoMode,
    project,
    world_scale,
)
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    LightSet,
    Material,
    MeshKey,
    MeshShape,
)

W, H = 400, 300
BACKGROUND = (0.05, 0.05, 0.08, 1.0)
BOX_RGBA = (0.35, 0.38, 0.42, 1.0)

BOX = MeshKey(MeshShape.BOX)
MATERIAL = np.array([0.0, 0.0, 0.5, 0.0], np.float32)
AMBIENT = np.full(3, 0.3, np.float32)

ORIGIN = np.zeros(3, np.float32)
RECT = (0.0, 0.0, float(W), float(H))
AXIS_U8 = np.rint(AXIS_COLORS[:, :3] * 255.0)  # X red, Y green, Z blue

# Gaze direction shared by all cameras: every axis and plane handle is fully
# facing (alpha 1.0).
_DIRECTION = np.array((3.0, -4.0, 2.2), np.float64)
_DIRECTION /= np.linalg.norm(_DIRECTION)


def _camera(distance: float = 5.0) -> CameraView:
    return CameraView(
        eye=np.asarray(ORIGIN + _DIRECTION * distance, np.float32),
        target=ORIGIN.copy(),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        near=0.1,
        far=50.0,
        aspect=W / H,
    )


def _frame(mode: GizmoMode = GizmoMode.TRANSLATE) -> GizmoFrame:
    return GizmoFrame(mode=mode, position=ORIGIN.copy(), rotation=np.eye(3, dtype=np.float32))


def _project(cam: CameraView, point) -> tuple[int, int]:
    x, y, _w = project(cam, [point], RECT)[0]
    return round(x), round(y)


def _axis_mask(img: np.ndarray, axis: int) -> np.ndarray:
    """Pixels dominated by the axis color (shade 0.72..1.0, alpha 1)."""
    rgb = img[..., :3].astype(np.int16)
    target = AXIS_U8[axis].astype(np.int16)
    dom = int(np.argmax(target))
    rest = [i for i in range(3) if i != dom]
    return (
        (rgb[..., dom] > target[dom] * 0.6)
        & (rgb[..., dom] - rgb[..., rest[0]] > 40)
        & (rgb[..., dom] - rgb[..., rest[1]] > 40)
    )


class Rig:
    def __init__(self, backend) -> None:
        self.backend = backend
        backend.set_background(BACKGROUND)
        backend.set_scene(SceneSource(meshes={}, textures={}, skybox=None))

    def draw(self, frame: GizmoFrame | None, camera: CameraView, box: bool = True) -> np.ndarray:
        self.backend.set_camera(camera)
        sb = SceneBuilder()
        if box:
            matid = sb.material_id(Material())
            sb.add(
                BOX,
                matid,
                M.compose(ORIGIN, np.eye(3), np.full(3, 0.3, np.float32)),
                np.asarray(BOX_RGBA, np.float32),
                MATERIAL,
                object_id=1,
            )
        scene = sb.build(camera, LightSet(ambient=AMBIENT), 2.0, np.zeros(3, np.float32))
        self.backend.set_render_scene(scene)
        assert self.backend.set_gizmo(frame)
        assert self.backend.render(None) is not None
        return self.backend.target.read_color(flip=True)


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


def test_translate_gizmo_draws_axis_handles_over_the_box(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    img = rig.draw(_frame(), cam)
    scale = world_scale(cam, ORIGIN, H)
    for axis in range(3):
        direction = np.eye(3)[axis]
        # Mid-shaft is past the plane handles and the center hole.
        x, y = _project(cam, ORIGIN + direction * (0.55 * scale))
        patch = img[y - 2 : y + 3, x - 2 : x + 3]
        assert _axis_mask(patch, axis).any(), f"axis {axis} mid-shaft missing"
        assert int(_axis_mask(img, axis).sum()) > 100
        # Just past the center hole the shaft sits inside the box; the
        # near-plane depth pinning must still draw the handle over it.
        x, y = _project(cam, ORIGIN + direction * min(0.25, 0.2 * scale))
        patch = img[y - 1 : y + 2, x - 1 : x + 2]
        assert _axis_mask(patch, axis).any(), f"axis {axis} lost to the box"


def test_gizmo_screen_size_is_constant_across_distances(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    tips = []
    expected = []
    for distance in (2.5, 7.5):
        cam = _camera(distance)
        img = rig.draw(_frame(), cam, box=False)
        cx, cy = _project(cam, ORIGIN)
        scale = world_scale(cam, ORIGIN, H)
        got, want = [], []
        for axis in range(3):
            ys, xs = np.nonzero(_axis_mask(img, axis))
            got.append(float(np.hypot(xs - cx, ys - cy).max()))
            # The arrow tip is axis * scale away in 3D; its projected pixel
            # distance foreshortens with the axis tilt but must not change
            # with camera distance (screen-constant size).
            tip = _project(cam, ORIGIN + np.eye(3)[axis] * scale)
            want.append(float(np.hypot(tip[0] - cx, tip[1] - cy)))
        tips.append(got)
        expected.append(want)
    print(f"\n[metric] axis tip distance px: near={tips[0]}, far={tips[1]}")
    for axis in range(3):
        assert abs(tips[0][axis] - tips[1][axis]) <= 2.0
        assert abs(expected[0][axis] - expected[1][axis]) <= 1.0
        for d in range(2):
            assert abs(tips[d][axis] - expected[d][axis]) <= 4.0


def test_center_shell_hole_exposes_the_scene(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    # Head-on camera: the X shaft projects without foreshortening, so the
    # hole (local CENTER_SHELL_RADIUS, a 3D sphere) maps to exactly
    # CENTER_SHELL_RADIUS * SIZE_PT px along the shaft centerline.
    cam = CameraView(
        eye=np.array((0.0, -5.0, 0.0), np.float32),
        target=ORIGIN.copy(),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        near=0.1,
        far=50.0,
        aspect=W / H,
    )
    with_gizmo = rig.draw(_frame(), cam)
    baseline = rig.draw(None, cam)  # also exercises set_gizmo(None)
    cx, cy = _project(cam, ORIGIN)
    r_sphere = (CENTER_RADIUS + CONTRAST_EDGE_PT / SIZE_PT) * SIZE_PT  # contrast edge sphere
    r_hole = CENTER_SHELL_RADIUS * SIZE_PT
    r_in, r_out = 8.0, 11.0
    assert r_sphere < r_in < r_hole < r_out
    # Only the +X side carries a shaft (handles point along +axis): inside
    # the hole the shaft is discarded and the scene shows through untouched.
    x = round(cx + r_in)
    diff = np.abs(with_gizmo[cy, x, :3].astype(int) - baseline[cy, x, :3].astype(int)).max()
    assert diff <= 6, f"hole pixel ({x}, {cy}) changed by {diff}"
    # Past the hole the shaft renders again.
    x = round(cx + r_out)
    assert _axis_mask(with_gizmo[cy - 1 : cy + 2, x - 1 : x + 2], 0).any()


def test_rotate_mode_draws_axis_and_screen_rings(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    img = rig.draw(_frame(GizmoMode.ROTATE), cam, box=False)
    for axis in range(3):
        assert int(_axis_mask(img, axis).sum()) > 150
    # The screen ring is a grayish circle at SCREEN_RING_RADIUS * SIZE_PT px.
    cx, cy = _project(cam, ORIGIN)
    radius = SCREEN_RING_RADIUS * SIZE_PT
    hits = 0
    for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        x = round(cx + radius * np.cos(angle))
        y = round(cy + radius * np.sin(angle))
        rgb = img[y, x, :3].astype(int)
        if rgb.min() > 110 and rgb.max() - rgb.min() < 60:
            hits += 1
    print(f"\n[metric] screen ring hits: {hits}/16")
    assert hits >= 12


def test_clearing_the_gizmo_removes_all_handles(rig):
    cam = _camera()
    img = rig.draw(_frame(), cam)
    assert sum(int(_axis_mask(img, a).sum()) for a in range(3)) > 300
    cleared = rig.draw(None, cam)
    assert sum(int(_axis_mask(cleared, a).sum()) for a in range(3)) == 0
    # The cleared frame matches the never-gizmo baseline exactly (both
    # backends are deterministic for a static scene).
    baseline = rig.draw(None, cam)
    assert np.array_equal(cleared, baseline)
