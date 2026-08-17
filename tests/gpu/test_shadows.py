from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer.render.backend import RenderFlag  # noqa: E402
from forge_viewer.render.forge import gl_native as G  # noqa: E402
from forge_viewer.render.forge import passes as _passes  # noqa: E402
from forge_viewer.render.forge.backend import ForgeBackend, registered  # noqa: E402
from forge_viewer.render.mesh import builtin_mesh  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshKey,
    MeshShape,
)

W, H = 480, 360
GROUND = MeshKey(MeshShape.PLANE)
BOX = MeshKey(MeshShape.BOX)

SUN = np.array([0.45, 0.35, -1.0], np.float32)

GRAZE_SUN = np.array([0.92, 0.30, -0.26], np.float32)


BOX_CENTER = np.array([0.0, 0.0, 1.2], np.float64)
BOX_HALF = 0.5
EXTENT = 8.0

VIEW = CameraView(
    eye=np.array([5.0, -6.0, 4.5], np.float32),
    target=np.array([0.0, 0.0, 0.6], np.float32),
    near=0.1,
    far=60.0,
)
GRAZE_VIEW = CameraView(
    eye=np.array([-1.2, -7.0, 0.42], np.float32),
    target=np.array([0.4, 1.0, 0.02], np.float32),
    near=0.05,
    far=80.0,
)
SPOT_POS = np.array([2.5, -3.0, 5.0], np.float32)
SPOT_DIR = np.array([-2.5, 3.0, -5.0], np.float32)
POINT_POS = np.array([2.5, -3.0, 4.0], np.float32)


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
    win = glfw.create_window(W, H, "forge shadows", None, None)
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


def _placed(sx, sy, sz, pos) -> np.ndarray:

    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    m[:3, 3] = pos
    return m


def _scene(
    camera: CameraView,
    sun_dir=SUN,
    cast: bool = True,
    extent: float = EXTENT,
    box_alpha: float = 1.0,
    with_box: bool = True,
):
    b = SceneBuilder()
    mid = b.material_id(Material(name="test"))
    material = np.array([0.0, 0.0, 0.3, 0.0], np.float32)
    b.add(
        GROUND,
        mid,
        _placed(6, 6, 1, (0, 0, 0)),
        np.array([0.7, 0.7, 0.72, 1.0], np.float32),
        material,
        1,
    )
    if with_box:
        b.add(
            BOX,
            mid,
            _placed(BOX_HALF, BOX_HALF, BOX_HALF, BOX_CENTER),
            np.array([0.8, 0.25, 0.2, box_alpha], np.float32),
            material,
            2,
        )
    sun = Light(
        kind=LightKind.DIRECTIONAL,
        direction=sun_dir,
        diffuse=np.full(3, 0.85, np.float32),
        specular=np.zeros(3, np.float32),
        cast_shadow=cast,
    )

    fill = Light(
        kind=LightKind.DIRECTIONAL,
        direction=np.array([-0.6, -0.4, -1.0], np.float32),
        diffuse=np.full(3, 0.15, np.float32),
        specular=np.zeros(3, np.float32),
        cast_shadow=False,
    )
    lights = LightSet(lights=(sun, fill), headlight=None, ambient=np.full(3, 0.12, np.float32))
    return b.build(camera, lights, extent, np.zeros(3, np.float32))


def _spot_scene(camera: CameraView, cast: bool = True):
    b = SceneBuilder()
    mid = b.material_id(Material(name="spot-test"))
    material = np.array([0.0, 0.0, 0.3, 0.0], np.float32)
    b.add(
        GROUND,
        mid,
        _placed(6, 6, 1, (0, 0, 0)),
        np.array([0.7, 0.7, 0.72, 1.0], np.float32),
        material,
        1,
    )
    b.add(
        BOX,
        mid,
        _placed(BOX_HALF, BOX_HALF, BOX_HALF, BOX_CENTER),
        np.array([0.8, 0.25, 0.2, 1.0], np.float32),
        material,
        2,
    )
    spot = Light(
        kind=LightKind.SPOT,
        position=SPOT_POS,
        direction=SPOT_DIR,
        diffuse=np.full(3, 0.9, np.float32),
        specular=np.zeros(3, np.float32),
        cutoff=38.0,
        exponent=2.0,
        cast_shadow=cast,
    )
    lights = LightSet(lights=(spot,), headlight=None, ambient=np.full(3, 0.08, np.float32))
    return b.build(camera, lights, EXTENT, np.zeros(3, np.float32))


def _point_scene(
    camera: CameraView,
    cast: bool = True,
    light_range: float = 0.0,
    active: bool = True,
    kind: LightKind = LightKind.POINT,
    area_radius: float = 0.0,
):
    b = SceneBuilder()
    mid = b.material_id(Material(name="point-test"))
    material = np.array([0.0, 0.0, 0.3, 0.0], np.float32)
    b.add(
        GROUND,
        mid,
        _placed(6, 6, 1, (0, 0, 0)),
        np.array([0.7, 0.7, 0.72, 1.0], np.float32),
        material,
        1,
    )
    b.add(
        BOX,
        mid,
        _placed(BOX_HALF, BOX_HALF, BOX_HALF, BOX_CENTER),
        np.array([0.8, 0.25, 0.2, 1.0], np.float32),
        material,
        2,
    )
    point = Light(
        kind=kind,
        position=POINT_POS,
        diffuse=np.full(3, 0.9, np.float32),
        specular=np.zeros(3, np.float32),
        attenuation=np.array([1.0, 0.02, 0.01], np.float32),
        range=light_range,
        area_radius=area_radius,
        cast_shadow=cast,
        active=active,
    )
    lights = LightSet(lights=(point,), headlight=None, ambient=np.full(3, 0.08, np.float32))
    return b.build(camera, lights, EXTENT, np.zeros(3, np.float32))


@pytest.fixture(scope="module")
def backend(gl):

    _passes.load_all()
    if "shadow" not in registered():
        pytest.skip(f"shadow pass unavailable: {_passes.failed().get('shadow', 'unregistered')}")
    be = ForgeBackend(gl, W, H, samples=4)
    if not be.caps.shadows:
        be.release()
        pytest.skip("backend does not support shadows")
    be.meshes.sync({GROUND: builtin_mesh(GROUND), BOX: builtin_mesh(BOX)})
    be.textures.sync({}, None)
    yield be
    be.release()


def _render(backend, camera=VIEW, shadow=True, **kw) -> np.ndarray:

    backend.set_flag(RenderFlag.SHADOW, bool(shadow))
    backend.set_camera(camera)
    backend.set_render_scene(_scene(camera, **kw))
    backend.render()
    return backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def _render_spot(backend, shadow=True, cast=True) -> np.ndarray:
    backend.set_flag(RenderFlag.SHADOW, bool(shadow))
    backend.set_camera(VIEW)
    backend.set_render_scene(_spot_scene(VIEW, cast=cast))
    backend.render()
    return backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def _render_point(
    backend,
    shadow=True,
    cast=True,
    light_range=0.0,
    active=True,
    kind=LightKind.POINT,
    area_radius=0.0,
) -> np.ndarray:
    backend.set_flag(RenderFlag.SHADOW, bool(shadow))
    backend.set_camera(VIEW)
    backend.set_render_scene(
        _point_scene(
            VIEW,
            cast=cast,
            light_range=light_range,
            active=active,
            kind=kind,
            area_radius=area_radius,
        )
    )
    backend.render()
    return backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def _darkened(lit: np.ndarray, shadowed: np.ndarray, threshold: int = 12) -> np.ndarray:

    return (lit - shadowed).sum(axis=2) > threshold


def _to_pixel(camera: CameraView, world) -> tuple[float, float]:

    cam = camera.with_aspect(W / H)
    vp = (cam.proj_matrix() @ cam.view_matrix()).astype(np.float64)
    clip = vp @ np.array([world[0], world[1], world[2], 1.0])
    ndc = clip[:3] / clip[3]
    return (ndc[0] * 0.5 + 0.5) * W, (1.0 - (ndc[1] * 0.5 + 0.5)) * H


def _luma(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def _high_frequency(y: np.ndarray) -> float:

    dx = np.diff(y, axis=1)
    dy = np.diff(y, axis=0)
    rms = np.sqrt((np.sum(dx**2) + np.sum(dy**2)) / (dx.size + dy.size))
    return float(rms / max(y.mean(), 1e-6))


def test_shadow_pass_actually_ran(backend):

    _render(backend, shadow=True)
    order = list(backend.stats.cpu_ms)
    assert "shadow" in order
    assert order.index("shadow") < order.index("opaque")


def test_ground_darkens_where_the_light_points(backend):

    lit = _render(backend, shadow=False)
    shadowed = _render(backend, shadow=True)
    mask = _darkened(lit, shadowed)
    assert mask.sum() > 500

    d = SUN / np.linalg.norm(SUN)
    hit = BOX_CENTER + d * (BOX_CENTER[2] / -d[2])
    px, py = _to_pixel(VIEW, hit)
    ys, xs = np.nonzero(mask)
    dist = float(np.hypot(xs.mean() - px, ys.mean() - py))
    assert dist < 8.0

    assert int(((shadowed - lit).sum(axis=2) > 12).sum()) == 0


def test_no_acne_at_grazing_incidence(backend):

    img = _render(backend, camera=GRAZE_VIEW, sun_dir=GRAZE_SUN, shadow=True)
    y = _luma(img)
    bottom, top = y[H // 2 :], y[: H // 2]
    assert bottom.mean() > 2.0 * top.mean()
    assert bottom.std() < 0.5 * bottom.mean()

    energy = _high_frequency(bottom)
    assert energy < 0.05


def test_cast_shadow_false_removes_the_shadow_and_nothing_else(backend):

    no_flag = _render(backend, shadow=False)
    with_shadow = _render(backend, shadow=True)
    no_cast = _render(backend, shadow=True, cast=False)

    assert _darkened(no_flag, with_shadow).sum() > 500
    assert np.array_equal(no_cast, no_flag)


def test_shadow_flag_is_pixel_reversible(backend):

    before = _render(backend, shadow=False)
    during = _render(backend, shadow=True)
    after = _render(backend, shadow=False)
    assert not np.array_equal(before, during)
    assert np.array_equal(before, after)


def test_transparent_geometry_does_not_cast(backend):

    opaque_lit = _render(backend, shadow=False)
    opaque_shadowed = _render(backend, shadow=True)
    assert _darkened(opaque_lit, opaque_shadowed).sum() > 500

    glass_lit = _render(backend, shadow=False, box_alpha=0.5)
    glass_shadowed = _render(backend, shadow=True, box_alpha=0.5)
    dark = int(_darkened(glass_lit, glass_shadowed).sum())
    assert dark == 0


def test_every_cascade_addresses_its_own_tile_and_texel(backend):

    d = SUN / np.linalg.norm(SUN)
    hit = BOX_CENTER + d * (BOX_CENTER[2] / -d[2])
    px, py = _to_pixel(VIEW, hit)

    areas = []
    for extent in (10.0, 30.0, 90.0):
        lit = _render(backend, shadow=False, extent=extent)
        shadowed = _render(backend, shadow=True, extent=extent)
        mask = _darkened(lit, shadowed)
        assert mask.sum() > 500
        ys, xs = np.nonzero(mask)
        dist = float(np.hypot(xs.mean() - px, ys.mean() - py))
        assert dist < 8.0
        areas.append(int(mask.sum()))

    spread = (max(areas) - min(areas)) / max(min(areas), 1)
    assert spread < 0.05


def test_render_leaves_no_gl_error(backend):

    G.native().drain_errors()
    _render(backend, shadow=True)
    assert backend.ctx.error == "GL_NO_ERROR"


def test_spot_shadow_uses_its_perspective_distance_map(backend):

    lit = _render_spot(backend, shadow=False)
    shadowed = _render_spot(backend, shadow=True)
    no_cast = _render_spot(backend, shadow=True, cast=False)
    mask = _darkened(lit, shadowed)

    assert mask.sum() > 250
    assert np.array_equal(no_cast, lit)

    direction = SPOT_DIR / np.linalg.norm(SPOT_DIR)
    hit = BOX_CENTER + direction * (BOX_CENTER[2] / -direction[2])
    px, py = _to_pixel(VIEW, hit)
    ys, xs = np.nonzero(mask)
    assert float(np.hypot(xs.mean() - px, ys.mean() - py)) < 16.0


def test_point_shadow_uses_all_six_cube_faces(backend):

    lit = _render_point(backend, shadow=False)
    shadowed = _render_point(backend, shadow=True)
    no_cast = _render_point(backend, shadow=True, cast=False)
    mask = _darkened(lit, shadowed)

    assert mask.sum() > 250
    assert np.array_equal(no_cast, lit)

    direction = BOX_CENTER - POINT_POS
    direction /= np.linalg.norm(direction)
    hit = BOX_CENTER + direction * (BOX_CENTER[2] / -direction[2])
    px, py = _to_pixel(VIEW, hit)
    ys, xs = np.nonzero(mask)
    assert float(np.hypot(xs.mean() - px, ys.mean() - py)) < 18.0


def _many_spot_scene(camera: CameraView, cast_count: int):
    scene = _spot_scene(camera)
    lights = tuple(
        Light(
            kind=LightKind.SPOT,
            position=SPOT_POS,
            direction=SPOT_DIR,
            diffuse=np.full(3, 0.12, np.float32),
            specular=np.zeros(3, np.float32),
            cutoff=38.0,
            exponent=2.0,
            cast_shadow=i < cast_count,
        )
        for i in range(8)
    )
    scene.lights = LightSet(lights=lights, headlight=None, ambient=np.full(3, 0.04, np.float32))
    return scene


def _render_many_spots(backend, cast_count: int, shadow: bool = True) -> np.ndarray:
    backend.set_flag(RenderFlag.SHADOW, shadow)
    backend.set_camera(VIEW)
    backend.set_render_scene(_many_spot_scene(VIEW, cast_count))
    backend.render()
    return backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def test_eight_local_lights_cast_simultaneously(backend):

    lit = _render_many_spots(backend, 0)
    one = _render_many_spots(backend, 1)
    all_eight = _render_many_spots(backend, 8)
    one_delta = np.maximum(_luma(lit) - _luma(one), 0.0).sum()
    all_delta = np.maximum(_luma(lit) - _luma(all_eight), 0.0).sum()

    assert one_delta > 1000.0
    assert all_delta > 5.0 * one_delta

    for _ in range(6):
        _render_many_spots(backend, 8)
    assert backend.stats.cpu_ms["shadow"] < 16.7


def test_light_range_culls_lighting_and_shadow_casters(backend):
    unlimited = _render_point(backend, shadow=True)
    limited = _render_point(backend, shadow=True, light_range=2.0)
    inactive = _render_point(backend, shadow=True, active=False)

    assert not np.array_equal(unlimited, limited)
    assert np.array_equal(limited, inactive)


def test_area_light_soft_shadow_is_reversible_to_hard_shadow(backend):
    lit = _render_point(backend, shadow=False, kind=LightKind.AREA, area_radius=0.5)
    ambient = _render_point(backend, shadow=True, active=False)
    hard_point = _render_point(backend, shadow=True)
    hard_area = _render_point(backend, shadow=True, kind=LightKind.AREA, area_radius=0.0)
    soft = _render_point(backend, shadow=True, kind=LightKind.AREA, area_radius=0.5)

    assert np.array_equal(hard_area, hard_point)
    assert not np.array_equal(soft, hard_area)

    available = np.maximum(_luma(lit) - _luma(ambient), 1.0)
    hard_fraction = np.clip((_luma(lit) - _luma(hard_area)) / available, 0.0, 1.0)
    soft_fraction = np.clip((_luma(lit) - _luma(soft)) / available, 0.0, 1.0)
    hard_partial = np.count_nonzero((hard_fraction > 0.15) & (hard_fraction < 0.85))
    soft_partial = np.count_nonzero((soft_fraction > 0.15) & (soft_fraction < 0.85))
    assert soft_partial > 1.5 * hard_partial


def _render_atmosphere(backend, *, fog: bool, haze: bool) -> np.ndarray:
    scene = _scene(VIEW)
    base = scene.lights
    scene.lights = LightSet(
        lights=base.lights,
        headlight=base.headlight,
        ambient=base.ambient,
        fog_color=np.array([0.12, 0.32, 0.65], np.float32),
        fog_start=2.0,
        fog_end=11.0,
        haze_color=np.array([0.82, 0.62, 0.35], np.float32),
        haze_density=0.18,
    )
    backend.set_flag(RenderFlag.SHADOW, False)
    backend.set_flag(RenderFlag.FOG, fog)
    backend.set_flag(RenderFlag.HAZE, haze)
    backend.set_camera(VIEW)
    backend.set_render_scene(scene)
    backend.render()
    return backend.target.read_color(flip=True)[..., :3].astype(np.int16)


def test_fog_and_haze_are_independent_and_reversible(backend):
    assert RenderFlag.FOG in backend.caps.render_flags
    assert RenderFlag.HAZE in backend.caps.render_flags

    clear = _render_atmosphere(backend, fog=False, haze=False)
    fog = _render_atmosphere(backend, fog=True, haze=False)
    haze = _render_atmosphere(backend, fog=False, haze=True)
    restored = _render_atmosphere(backend, fog=False, haze=False)

    assert np.count_nonzero(np.max(np.abs(fog - clear), axis=2) > 5) > 10_000
    assert np.count_nonzero(np.max(np.abs(haze - clear), axis=2) > 5) > 10_000
    assert not np.array_equal(fog, haze)
    assert np.array_equal(restored, clear)
