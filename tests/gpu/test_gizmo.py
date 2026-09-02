"""Backend-neutral native-gizmo render tests.

Runs under both backends via the ``backend_name`` fixture (conftest.py); the
CPU-side gizmo geometry/hit-test coverage lives in tests/test_gizmo.py.  The
scene is a single box at the origin with the gizmo on top of it, which also
proves the near-plane depth pinning draws handles over scene geometry.

Expected screen positions come from the shared ``mojive.gizmo``
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

from mojive import math3d as M  # noqa: E402
from mojive.adapters.base import SceneSource  # noqa: E402
from mojive.gizmo import (  # noqa: E402
    ACTIVE_HANDLE_COLOR,
    AXIS_COLORS,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_PT,
    HOVER_COLOR,
    JOINT_HANDLE_COLOR,
    JOINT_OUTLINE_COLOR,
    RING_RADIUS,
    SCREEN_RING_RADIUS,
    SIZE_PT,
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    axis_hover_color,
    handle_mask,
    project,
    world_scale,
)
from mojive.render.scene import SceneBuilder  # noqa: E402
from mojive.types import (  # noqa: E402
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
HOVER_U8 = np.rint(HOVER_COLOR[:3] * 255.0)
JOINT_U8 = np.rint(JOINT_HANDLE_COLOR[:3] * 255.0)
JOINT_HOVER_U8 = np.rint(np.asarray(axis_hover_color(JOINT_HANDLE_COLOR)[:3]) * 255.0)
ACTIVE_HANDLE_U8 = np.rint(ACTIVE_HANDLE_COLOR[:3] * 255.0)

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


def _hover_mask(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3].astype(np.int16)
    return np.max(np.abs(rgb - HOVER_U8.astype(np.int16)), axis=-1) < 85


def _joint_mask(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3].astype(np.float64)
    peak = np.max(rgb, axis=-1)
    chroma = rgb / np.maximum(peak[..., None], 1.0)
    target_chroma = JOINT_U8.astype(np.float64) / float(np.max(JOINT_U8))
    return (peak > 64.0) & (np.max(np.abs(chroma - target_chroma), axis=-1) < 0.12)


def _tint_mask(img: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgb = img[..., :3].astype(np.float64)
    peak = np.max(rgb, axis=-1)
    chroma = rgb / np.maximum(peak[..., None], 1.0)
    target_chroma = target.astype(np.float64) / float(np.max(target))
    return (peak > 64.0) & (np.max(np.abs(chroma - target_chroma), axis=-1) < 0.12)


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
    """Build the backend selected by MOJIVE_BACKEND; GL stays lazy."""
    if backend_name == "wgpu":
        from mojive.render.webgpu.backend import WgpuBackend

        return WgpuBackend(W, H, samples=samples)
    from mojive.render.opengl import passes
    from mojive.render.opengl.backend import OpenGLBackend

    passes.load_all()
    return OpenGLBackend(request.getfixturevalue("gl_ctx"), W, H, samples=samples)


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


def test_scalar_joint_color_override_renders_the_primary_palette(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    frame = _frame()
    frame.handle_mask = handle_mask(GizmoHandle.Z)
    frame.handle_color = JOINT_HANDLE_COLOR
    img = rig.draw(frame, cam, box=False)
    scale = world_scale(cam, ORIGIN, H)
    x, y = _project(cam, ORIGIN + np.array((0.0, 0.0, 0.55 * scale)))
    patch = img[y - 3 : y + 4, x - 3 : x + 4]

    joint_pixels = patch[_joint_mask(patch), :3].astype(np.float64)
    assert len(joint_pixels) > 0
    median = np.median(joint_pixels, axis=0)
    assert median[1] > median[0] > median[2]


@pytest.mark.parametrize(
    ("mode", "handle"),
    (
        (GizmoMode.TRANSLATE, GizmoHandle.Z),
        (GizmoMode.ROTATE, GizmoHandle.ROTATE_Z),
    ),
)
def test_scalar_joint_outline_adds_a_thin_neutral_silhouette(rig, mode, handle):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    frame = _frame(mode)
    frame.handle_mask = handle_mask(handle)
    frame.handle_color = JOINT_HANDLE_COLOR
    without = rig.draw(frame, _camera(), box=False)

    frame.outline_color = JOINT_OUTLINE_COLOR
    outlined = rig.draw(frame, _camera(), box=False)

    def neutral_pixels(image):
        rgb = image[..., :3].astype(np.int16)
        return (rgb.min(axis=-1) > 100) & (np.ptp(rgb, axis=-1) < 20)

    added = neutral_pixels(outlined) & ~neutral_pixels(without)
    assert int(added.sum()) > 12
    assert _joint_mask(outlined).any()


def test_scalar_joint_handle_uses_primary_palette_for_hover_and_press(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    frame = _frame()
    frame.handle_mask = handle_mask(GizmoHandle.Z)
    frame.handle_color = JOINT_HANDLE_COLOR
    frame.hovered = GizmoHandle.Z
    hovered = rig.draw(frame, cam, box=False)

    frame.active = GizmoHandle.Z
    active = rig.draw(frame, cam, box=False)

    assert np.allclose(ACTIVE_HANDLE_COLOR, HOVER_COLOR)
    assert _tint_mask(hovered, JOINT_HOVER_U8).any()
    assert _tint_mask(active, ACTIVE_HANDLE_U8).any()


@pytest.mark.parametrize(
    ("mode", "handle", "eye", "up"),
    (
        (GizmoMode.TRANSLATE, GizmoHandle.Z, (0.0, 0.0, 5.0), (0.0, 1.0, 0.0)),
        (GizmoMode.ROTATE, GizmoHandle.ROTATE_Z, (5.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ),
)
def test_active_joint_handle_disappears_when_its_projection_degenerates(
    rig,
    mode: GizmoMode,
    handle: GizmoHandle,
    eye: tuple[float, float, float],
    up: tuple[float, float, float],
):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = CameraView(
        eye=np.asarray(eye, np.float32),
        target=ORIGIN.copy(),
        up=np.asarray(up, np.float32),
        near=0.1,
        far=50.0,
        aspect=W / H,
    )
    frame = _frame(mode)
    frame.active = handle
    frame.handle_mask = handle_mask(handle)
    frame.handle_color = JOINT_HANDLE_COLOR
    frame.active_projection_fade = True

    img = rig.draw(frame, cam, box=False)

    assert not _joint_mask(img).any()


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


def test_rotate_mode_draws_the_trackball_fill_behind_axis_rings(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    frame = _frame(GizmoMode.ROTATE)
    frame.hovered = GizmoHandle.ROTATE_TRACKBALL
    with_trackball = rig.draw(frame, cam, box=False)

    frame.handle_mask &= ~(1 << int(GizmoHandle.ROTATE_TRACKBALL))
    frame.hovered = GizmoHandle.NONE
    without_trackball = rig.draw(frame, cam, box=False)

    cx, cy = _project(cam, ORIGIN)
    difference = np.abs(
        with_trackball[cy - 2 : cy + 3, cx - 2 : cx + 3, :3].astype(int)
        - without_trackball[cy - 2 : cy + 3, cx - 2 : cx + 3, :3].astype(int)
    )
    assert difference.max() >= 8


def test_idle_single_axis_rotation_draws_a_full_ring(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _camera()
    frame = _frame(GizmoMode.ROTATE)
    frame.handle_mask = handle_mask(GizmoHandle.ROTATE_Z)
    img = rig.draw(frame, cam, box=False)

    hits = 0
    radius = RING_RADIUS * world_scale(cam, ORIGIN, H, SIZE_PT)
    for angle in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
        point = ORIGIN + radius * np.array((np.cos(angle), np.sin(angle), 0.0))
        x, y = np.rint(project(cam, (point,), RECT)[0, :2]).astype(int)
        patch = img[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
        hits += bool(_axis_mask(patch, 2).any())
    print(f"\n[metric] idle single-axis ring hits: {hits}/32")
    assert hits >= 26


def test_edge_on_active_rotation_ring_has_a_rounded_silhouette(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = CameraView(
        eye=np.array((0.0, -5.0, 0.0), np.float32),
        target=ORIGIN.copy(),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        near=0.1,
        far=50.0,
        aspect=W / H,
    )
    frame = _frame(GizmoMode.ROTATE)
    frame.active = GizmoHandle.ROTATE_Z
    img = rig.draw(frame, cam, box=False)
    ys, xs = np.nonzero(_axis_mask(img, 2))
    assert len(xs) > 100
    expected_span = 2.0 * RING_RADIUS * SIZE_PT
    assert np.ptp(xs) == pytest.approx(expected_span, abs=6.0)
    assert np.ptp(ys) >= 3

    for x in (int(xs.min()) + 2, int(xs.max()) - 2):
        cap_y = ys[xs == x]
        assert len(cap_y) >= 2
        assert np.ptp(cap_y) >= 1


def _head_on_camera() -> CameraView:
    """Nearly head-on along +Y: the Y handle foreshortens to a faint stub."""
    return CameraView(
        eye=np.array((0.0, -5.0, 1.2), np.float32),
        target=ORIGIN.copy(),
        up=np.array((0.0, 0.0, 1.0), np.float32),
        near=0.1,
        far=50.0,
        aspect=W / H,
    )


def _dominance(img: np.ndarray, x: int, y: int, channel: int, radius: int = 6) -> bool:
    patch = img[y - radius : y + radius + 1, x - radius : x + radius + 1, :3].astype(np.int16)
    rest = [i for i in range(3) if i != channel]
    margin = patch[..., channel] - np.maximum(patch[..., rest[0]], patch[..., rest[1]])
    return bool((margin > 25).any())


def test_foreshortened_axis_leans_toward_its_projected_tip(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _head_on_camera()
    img = rig.draw(_frame(), cam, box=False)
    scale = world_scale(cam, ORIGIN, H)
    cx, cy = _project(cam, ORIGIN)

    # +Y points at the camera: faded and squashed, but the stub must lean
    # toward its projected tip (up-screen here) — never the mirror side.
    tx, ty = _project(cam, ORIGIN + np.array((0.0, 1.0, 0.0)) * scale)
    assert abs(tx - cx) <= 1 and ty < cy
    assert _dominance(img, tx, ty, 1), "foreshortened Y stub missing at its tip"
    assert not _dominance(img, 2 * cx - tx, 2 * cy - ty, 1), "Y stub flipped to the far side"

    # The perpendicular handles keep full length and point along +axis only.
    mx, my = _project(cam, ORIGIN + np.array((1.0, 0.0, 0.0)) * (0.55 * scale))
    assert _dominance(img, mx, my, 0, radius=3), "X mid-shaft missing"
    assert not _dominance(img, 2 * cx - mx, my, 0, radius=3), "X handle leaked to -X"
    mx, my = _project(cam, ORIGIN + np.array((0.0, 0.0, 1.0)) * (0.55 * scale))
    assert _dominance(img, mx, my, 2, radius=3), "Z mid-shaft missing"


def test_handles_follow_the_body_frame(rig):
    if not rig.backend.caps.gizmo:
        pytest.skip("gizmo unsupported by this backend")
    cam = _head_on_camera()
    yaw90 = M.rotvec_to_mat3(np.array((0.0, 0.0, np.pi / 2)))
    frame = GizmoFrame(position=ORIGIN.copy(), rotation=np.asarray(yaw90, np.float32))
    img = rig.draw(frame, cam, box=False)
    scale = world_scale(cam, ORIGIN, H)
    cx = _project(cam, ORIGIN)[0]

    # Body frame yaw +90°: the X handle now lies along world +Y (faded stub
    # leaning up-screen) and the Y handle along world -X (full length left).
    tx, ty = _project(cam, ORIGIN + yaw90[:, 0] * scale)
    assert _dominance(img, tx, ty, 0), "rotated X stub missing at its tip"
    mx, my = _project(cam, ORIGIN + yaw90[:, 1] * (0.55 * scale))
    assert mx < cx and _dominance(img, mx, my, 1, radius=3), "rotated Y mid-shaft missing"
    assert not _dominance(img, 2 * cx - mx, my, 1, radius=3), "rotated Y leaked to +X"
    mx, my = _project(cam, ORIGIN + yaw90[:, 2] * (0.55 * scale))
    assert _dominance(img, mx, my, 2, radius=3), "rotated Z mid-shaft missing"


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
