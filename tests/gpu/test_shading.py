from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

glfw = pytest.importorskip("glfw")
moderngl = pytest.importorskip("moderngl")

from forge_viewer.adapters.base import SceneSource  # noqa: E402
from forge_viewer.render.backend import DebugView, RenderFlag  # noqa: E402
from forge_viewer.render.forge import color, passes  # noqa: E402
from forge_viewer.render.forge.backend import ForgeBackend  # noqa: E402
from forge_viewer.render.forge.passes import opaque as opaque_pass  # noqa: E402
from forge_viewer.render.forge.passes.base import MAX_SCENE_LIGHTS  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    Light,
    LightSet,
    LightType,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    ShadingModel,
    TextureData,
    TextureType,
)

WIDTH, HEIGHT = 128, 96
QUAD = MeshKey(MeshShape.PLANE, -1)
NO_AMBIENT = np.zeros(3, np.float32)


def _quad() -> MeshData:

    p = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32)
    n = np.tile(np.array([0.0, -1.0, 0.0], np.float32), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return MeshData(p, n, uv, np.array([0, 1, 2, 0, 2, 3], np.uint32))


def _transform(scale: float, y: float) -> np.ndarray:

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[2, 2] = scale
    m[1, 3] = y
    return m


class Rig:
    def __init__(self, backend, textures=None, skybox=None, mesh=None) -> None:
        self.backend = backend
        backend.set_scene(
            SceneSource(meshes={QUAD: mesh or _quad()}, textures=textures or {}, skybox=skybox)
        )
        self.camera = CameraView(
            eye=np.array([0.0, -3.0, 0.0], np.float32),
            target=np.zeros(3, np.float32),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            near=0.1,
            far=20.0,
        )
        backend.set_camera(self.camera)

    def draw(
        self,
        quads,
        ambient=NO_AMBIENT,
        lights=(),
        headlight=None,
        selected: int = 0,
        shading_model: ShadingModel = ShadingModel.LINEAR,
    ) -> np.ndarray:

        sb = SceneBuilder()
        matid = sb.material_id(Material())
        for i, (rgba, mat, scale, y) in enumerate(quads):
            sb.add(
                QUAD,
                matid,
                _transform(scale, y),
                np.asarray(rgba, np.float32),
                np.array([*mat, 0.0], np.float32),
                object_id=i + 1,
            )
        scene = sb.build(
            self.camera,
            LightSet(
                lights=tuple(lights), headlight=headlight, ambient=np.asarray(ambient, np.float32)
            ),
            2.0,
            np.zeros(3, np.float32),
            shading_model=shading_model,
        )
        self.backend.highlight(selected)
        self.backend.set_render_scene(scene)
        assert self.backend.render(None) is not None
        return self.backend.target.read_color(flip=True)

    @staticmethod
    def center(img: np.ndarray) -> np.ndarray:
        return img[HEIGHT // 2, WIDTH // 2].astype(np.int32)

    @staticmethod
    def corner(img: np.ndarray) -> np.ndarray:
        return img[2, 2].astype(np.int32)


def _make_backend(backend_name: str, request, samples: int = 4):
    """Build the backend selected by FORGE_VIEWER_BACKEND; GL stays lazy."""
    if backend_name == "wgpu":
        from forge_viewer.render.webgpu.backend import WgpuBackend

        return WgpuBackend(WIDTH, HEIGHT, samples=samples)
    passes.load_all()
    return ForgeBackend(request.getfixturevalue("gl_ctx"), WIDTH, HEIGHT, samples=samples)


@pytest.fixture
def rig(backend_name, request):
    backend = _make_backend(backend_name, request)
    yield Rig(backend)
    backend.release()


def _dir_light(diffuse: float, specular: float = 0.0) -> Light:

    return Light(
        type=LightType.DIRECTIONAL,
        direction=np.array([0.0, 1.0, 0.0], np.float32),
        diffuse=np.full(3, diffuse, np.float32),
        specular=np.full(3, specular, np.float32),
        ambient=np.zeros(3, np.float32),
        cast_shadow=False,
    )


@pytest.mark.parametrize(
    ("srgb", "ambient"),
    [
        ((0.50, 0.25, 0.75), 0.25),
        ((0.90, 0.60, 0.20), 0.35),
        ((1.00, 0.50, 0.25), 0.50),
    ],
)
def test_flat_ambient_matches_the_cpu_implementation(rig, srgb, ambient):

    albedo = color.srgb_to_linear(np.array(srgb))
    got = Rig.center(
        rig.draw([(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)], ambient=[ambient] * 3)
    )
    want = color.shade_flat(albedo, np.full(3, ambient)).astype(np.int32)

    assert np.abs(got[:3] - want).max() <= 1, f"GPU {got[:3]} vs CPU {want}"
    assert got[3] == 255

    wrong = color.to_u8(
        color.finish(2.0 * color.srgb_to_linear(np.full(3, ambient)) * albedo)
    ).astype(np.int32)
    assert np.abs(got[:3] - wrong).max() >= 5


def test_diffuse_alone_is_monotone_and_matches_the_cpu(rig):

    albedo = color.srgb_to_linear(np.array([0.8, 0.8, 0.8]))
    seen = []
    for level in (0.25, 0.5, 0.75, 1.0):
        img = rig.draw(
            [(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)],
            ambient=NO_AMBIENT,
            lights=[_dir_light(level)],
        )
        got = Rig.center(img)[:3]

        want = color.to_u8(color.finish(color.srgb_to_linear(level) * albedo)).astype(np.int32)
        assert np.abs(got - want).max() <= 1, f"diffuse={level}: GPU {got} vs CPU {want}"
        seen.append(int(got[0]))
    assert seen == sorted(seen) and seen[0] < seen[-1]
    assert seen[0] > 0


def test_mujoco_classic_lighting_combines_terms_in_display_space(rig):
    display_albedo = np.array([0.8, 0.6, 0.4], np.float32)
    albedo = color.srgb_to_linear(display_albedo)
    img = rig.draw(
        [(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)],
        ambient=[0.1] * 3,
        lights=[_dir_light(0.4), _dir_light(0.5)],
        shading_model=ShadingModel.MUJOCO_CLASSIC,
    )
    got = Rig.center(img)[:3]
    want = np.rint(display_albedo * 255.0).astype(np.int32)

    assert np.abs(got - want).max() <= 2


def test_mujoco_classic_texture_modulates_fixed_function_specular(backend_name, request):
    texel = np.array([64, 128, 192], np.uint8)
    pixels = np.empty((4, 4, 3), np.uint8)
    pixels[:] = texel
    backend = _make_backend(backend_name, request, samples=1)
    try:
        rig = Rig(
            backend,
            textures={"surface": TextureData("surface", TextureType.TWO_D, pixels, srgb=True)},
        )
        sb = SceneBuilder()
        matid = sb.material_id(Material(texture="surface"))
        sb.add(
            QUAD,
            matid,
            _transform(2.0, 0.0),
            np.ones(4, np.float32),
            np.array([0.0, 1.0, 1.0, 0.0], np.float32),
            object_id=1,
            tex_coef=np.array([1.0, 1.0, 0.0, 0.0], np.float32),
        )
        scene = sb.build(
            rig.camera,
            LightSet(lights=(_dir_light(0.0, specular=1.0),), ambient=NO_AMBIENT),
            2.0,
            np.zeros(3, np.float32),
            shading_model=ShadingModel.MUJOCO_CLASSIC,
        )
        backend.set_render_scene(scene)
        assert backend.render(None) is not None
        got = Rig.center(backend.target.read_color(flip=True))[:3]

        assert np.abs(got - texel.astype(np.int32)).max() <= 3
    finally:
        backend.release()


def test_mujoco_classic_specular_is_not_scaled_by_diffuse_cosine(rig):
    light = Light(
        type=LightType.DIRECTIONAL,
        direction=np.array([0.0, 0.5, np.sqrt(0.75)], np.float32),
        diffuse=np.zeros(3, np.float32),
        specular=np.full(3, 0.5, np.float32),
        ambient=np.zeros(3, np.float32),
        cast_shadow=False,
    )
    img = rig.draw(
        [(np.ones(4, np.float32), (0.0, 1.0, 0.0), 2.0, 0.0)],
        lights=(light,),
        shading_model=ShadingModel.MUJOCO_CLASSIC,
    )

    assert Rig.center(img)[0] == pytest.approx(128, abs=3)


def test_light_color_is_decoded_from_the_display_domain(rig):

    albedo = color.srgb_to_linear(np.array([1.0, 1.0, 1.0]))
    full = Rig.center(
        rig.draw([(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)], lights=[_dir_light(1.0)])
    )
    half = Rig.center(
        rig.draw([(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)], lights=[_dir_light(0.5)])
    )
    ratio = half[0] / max(full[0], 1)
    assert ratio == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("count", (17, MAX_SCENE_LIGHTS))
def test_all_mujoco_scene_light_slots_reach_the_shader(rig, count):
    dark = [_dir_light(0.0) for _ in range(count - 1)]
    marker = Light(
        type=LightType.DIRECTIONAL,
        direction=np.array([0.0, 1.0, 0.0], np.float32),
        diffuse=np.array([0.55, 0.0, 0.0], np.float32),
        specular=np.zeros(3, np.float32),
        ambient=np.zeros(3, np.float32),
        cast_shadow=False,
    )
    albedo = color.srgb_to_linear(np.ones(3))
    center = Rig.center(
        rig.draw(
            [(np.append(albedo, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)],
            lights=[*dark, marker],
        )
    )

    assert center[0] > 80
    assert center[1] < 3
    assert center[2] < 3


def test_mujoco_classic_ignores_local_light_range(rig):
    light = Light(
        type=LightType.SPOT,
        position=np.array([0.0, -1.0, 0.0], np.float32),
        direction=np.array([0.0, 1.0, 0.0], np.float32),
        diffuse=np.ones(3, np.float32),
        specular=np.zeros(3, np.float32),
        attenuation=np.array([1.0, 0.0, 0.0], np.float32),
        range=0.5,
        cutoff=45.0,
        exponent=0.0,
        cast_shadow=False,
    )
    surface = [(np.ones(4, np.float32), (0.0, 0.0, 0.5), 2.0, 0.0)]

    limited = Rig.center(rig.draw(surface, lights=[light]))
    classic = Rig.center(
        rig.draw(surface, lights=[light], shading_model=ShadingModel.MUJOCO_CLASSIC)
    )

    assert limited[:3].max() == 0
    assert classic[:3].min() > 200


def test_transparent_leaves_depth_mask_on(rig):

    opaque_quad = (np.array([0.2, 0.8, 0.3, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.5)
    glass = (np.array([0.9, 0.2, 0.2, 0.4], np.float32), (0.0, 0.0, 0.5), 1.0, -0.5)

    shots = []
    for _ in range(3):
        img = rig.draw([opaque_quad, glass], ambient=[0.3] * 3)
        if rig.backend.caps.name == "forge":
            assert rig.backend.target.fbo.depth_mask is True

        assert Rig.corner(img)[:3].max() > 60
        shots.append(img.copy())
    assert np.array_equal(shots[0], shots[2])
    assert rig.backend.stats.draw_calls == 2


def test_a_bucket_change_after_the_first_frame_still_draws(rig):

    quad = (np.array([0.2, 0.8, 0.3, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.5)
    glass = (np.array([0.9, 0.2, 0.2, 0.4], np.float32), (0.0, 0.0, 0.5), 1.0, -0.5)
    first = Rig.center(rig.draw([quad], ambient=[0.3] * 3))
    second = Rig.corner(rig.draw([quad, glass], ambient=[0.3] * 3))
    assert first[:3].max() > 60
    assert np.abs(second - first).max() <= 2
    assert rig.backend.stats.triangles == 4


def test_blending_happens_after_gamma(rig):

    ambient = [0.3] * 3
    back_rgb = np.array([0.2, 0.8, 0.3], np.float32)
    front_rgb = np.array([0.9, 0.2, 0.2], np.float32)
    alpha = 0.4

    back_only = Rig.center(
        rig.draw([(np.append(back_rgb, 1.0), (0.0, 0.0, 0.5), 2.0, 0.5)], ambient=ambient)
    )
    both = Rig.center(
        rig.draw(
            [
                (np.append(back_rgb, 1.0), (0.0, 0.0, 0.5), 2.0, 0.5),
                (np.append(front_rgb, alpha), (0.0, 0.0, 0.5), 1.0, -0.5),
            ],
            ambient=ambient,
        )
    )
    front_solid = color.to_u8(color.finish(color.ambient_linear(ambient) * front_rgb)).astype(
        np.int32
    )
    want = back_only[:3] * (1.0 - alpha) + front_solid * alpha
    assert np.abs(both[:3] - want).max() <= 2


def test_additive_transparency_matches_mujoco_blending(rig):
    ambient = [0.3] * 3
    back_rgb = np.array([0.2, 0.5, 0.3], np.float32)
    front_rgb = np.array([0.7, 0.2, 0.1], np.float32)
    alpha = 0.4
    quads = [
        (np.append(back_rgb, 1.0), (0.0, 0.0, 0.5), 2.0, 0.5),
        (np.append(front_rgb, alpha), (0.0, 0.0, 0.5), 1.0, -0.5),
    ]
    back_only = Rig.center(rig.draw(quads[:1], ambient=ambient))
    front_solid = color.to_u8(color.finish(color.ambient_linear(ambient) * front_rgb)).astype(
        np.int32
    )

    rig.backend.set_flag(RenderFlag.ADDITIVE, False)
    standard = Rig.center(rig.draw(quads, ambient=ambient))
    rig.backend.set_flag(RenderFlag.ADDITIVE, True)
    additive = Rig.center(rig.draw(quads, ambient=ambient))
    rig.backend.set_flag(RenderFlag.ADDITIVE, False)
    restored = Rig.center(rig.draw(quads, ambient=ambient))

    assert np.array_equal(restored, standard)
    assert np.all(additive[:3] >= standard[:3])
    want = np.clip(back_only[:3] + front_solid * alpha, 0, 255)
    assert np.abs(additive[:3] - want).max() <= 2


def test_skybox_only_shows_up_when_it_is_on(backend_name, request):

    sky = np.zeros((6, 4, 4, 4), np.uint8)
    sky[..., :3] = np.array([40, 90, 200], np.uint8)
    sky[..., 3] = 255
    backend = _make_backend(backend_name, request)
    try:
        rig = Rig(
            backend,
            textures={"sky": TextureData("sky", TextureType.SKYBOX, sky)},
            skybox="sky",
        )
        quad = (np.array([0.5, 0.5, 0.5, 1.0], np.float32), (0.0, 0.0, 0.5), 0.3, 0.0)

        backend.set_flag(RenderFlag.SKYBOX, False)
        off = Rig.corner(rig.draw([quad], ambient=[0.3] * 3))
        backend.set_flag(RenderFlag.SKYBOX, True)
        on = Rig.corner(rig.draw([quad], ambient=[0.3] * 3))

        clear = np.array([round(c * 255) for c in backend._background])
        assert np.abs(off - clear).max() <= 1
        assert np.abs(on - off).max() > 20

        assert not np.array_equal(Rig.center(rig.draw([quad], ambient=[0.3] * 3)), on)
    finally:
        backend.release()


def test_image_light_uses_cube_radiance_and_mujoco_intensity_scale(backend_name, request):
    value = 128
    cube = np.full((6, 8, 8, 3), value, np.uint8)
    backend = _make_backend(backend_name, request)
    try:
        rig = Rig(
            backend,
            textures={"studio": TextureData("studio", TextureType.CUBE, cube)},
        )
        surface = [(np.ones(4, np.float32), (0.0, 0.0, 0.5), 2.0, 0.0)]
        off = Rig.center(rig.draw(surface))
        on = Rig.center(
            rig.draw(
                surface,
                lights=[
                    Light(
                        type=LightType.IMAGE,
                        texture="studio",
                        intensity=opaque_pass.IMAGE_LIGHT_REFERENCE_INTENSITY,
                    )
                ],
            )
        )

        radiance = color.srgb_to_linear(value / 255.0)
        expected = color.to_u8(color.finish(np.full(3, radiance))).astype(np.int32)
        assert np.max(np.abs(on[:3] - expected)) <= 2
        assert np.max(off[:3]) == 0
    finally:
        backend.release()


def test_cube_texture_reaches_an_ordinary_material(backend_name, request):
    cube = np.empty((6, 8, 8, 3), np.uint8)
    cube[:] = [230, 30, 50]
    backend = _make_backend(backend_name, request)
    try:
        rig = Rig(backend, textures={"body": TextureData("body", TextureType.CUBE, cube)})
        sb = SceneBuilder()
        matid = sb.material_id(Material(texture="body", tex_uniform=True))
        sb.add(
            QUAD,
            matid,
            _transform(2.0, 0.0),
            np.ones(4, np.float32),
            np.array([0.0, 0.0, 0.5, 0.0], np.float32),
            object_id=1,
            cube_coef=np.ones(4, np.float32),
        )
        scene = sb.build(
            rig.camera,
            LightSet(ambient=np.ones(3, np.float32)),
            2.0,
            np.zeros(3, np.float32),
        )
        backend.set_render_scene(scene)
        assert backend.render(None) is not None
        center = Rig.center(backend.target.read_color(flip=True))

        assert center[0] > 180
        assert center[1] < 90
        assert center[2] < 110
    finally:
        backend.release()


def test_normals_survive_non_uniform_scale(backend_name, request):

    n_local = np.array([0.0, -1.0, 1.0], np.float32) / np.sqrt(2.0)
    slanted = MeshData(
        np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32),
        np.tile(n_local, (4, 1)),
        np.zeros((4, 2), np.float32),
        np.array([0, 1, 2, 0, 2, 3], np.uint32),
    )
    backend = _make_backend(backend_name, request)
    try:
        rig = Rig(backend, mesh=slanted)
        backend.set_debug_view(DebugView.NORMAL)
        scale = np.diag([4.0, 1.0, 4.0]).astype(np.float32)
        got = Rig.center(
            rig.draw([(np.array([1, 1, 1, 1], np.float32), (0.0, 0.0, 0.5), 4.0, 0.0)])
        )

        right = np.linalg.inv(scale).T @ n_local
        right = right / np.linalg.norm(right)
        wrong = scale @ n_local
        wrong = wrong / np.linalg.norm(wrong)
        want = np.rint((right * 0.5 + 0.5) * 255).astype(int)
        bad = np.rint((wrong * 0.5 + 0.5) * 255).astype(int)

        assert np.abs(got[:3] - want).max() <= 2
        assert np.abs(got[:3] - bad).max() > 20
    finally:
        backend.release()


def test_shared_id_layout_is_not_polluted_by_shading_passes(require_forge, gl_ctx):

    passes.load_all()
    backend = ForgeBackend(gl_ctx, WIDTH, HEIGHT, samples=0)
    try:
        from forge_viewer.render.forge.targets import IdLayout

        if backend.target.id_layout is not IdLayout.SHARED:
            pytest.skip("integer attachments are unavailable even without MSAA")
        rig = Rig(backend)
        img = rig.draw(
            [
                (np.array([0.2, 0.8, 0.3, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.5),
                (np.array([0.9, 0.2, 0.2, 0.4], np.float32), (0.0, 0.0, 0.5), 1.0, -0.5),
            ],
            ambient=[0.3] * 3,
        )
        ids = backend.target.read_ids()
        assert int(ids[HEIGHT // 2, WIDTH // 2]) == 1
        assert Rig.center(img)[:3].max() > 20
    finally:
        backend.release()


def test_highlight_needs_the_emission_term(rig, monkeypatch):

    quad = (np.array([0.55, 0.45, 0.10, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.0)
    dark = [0.12] * 3

    plain = float(Rig.center(rig.draw([quad], ambient=dark))[:3].mean())
    lit = float(Rig.center(rig.draw([quad], ambient=dark, selected=1))[:3].mean())

    if rig.backend.caps.name == "wgpu":
        from forge_viewer.render.webgpu import backend as webgpu_backend

        monkeypatch.setattr(webgpu_backend, "HIGHLIGHT_EMISSION", 0.0)
    else:
        monkeypatch.setattr(opaque_pass, "HIGHLIGHT_EMISSION", 0.0)
    mix_only = float(Rig.center(rig.draw([quad], ambient=dark, selected=1))[:3].mean())

    assert lit - plain > 40.0
    assert (lit - plain) > 4.0 * abs(mix_only - plain)


def test_albedo_view_is_the_unlit_base_color(rig):

    rgb = np.array([0.6, 0.3, 0.15], np.float32)
    rig.backend.set_debug_view(DebugView.ALBEDO)
    got = Rig.center(
        rig.draw([(np.append(rgb, 1.0), (0.0, 0.0, 0.5), 2.0, 0.0)], ambient=[0.9] * 3)
    )
    want = color.to_u8(color.gamma_encode(rgb)).astype(np.int32)
    assert np.abs(got[:3] - want).max() <= 1


def test_overdraw_clears_to_zero_and_counts_layers(rig):

    if DebugView.OVERDRAW not in rig.backend.caps.debug_views:
        pytest.skip("overdraw debug view unsupported by this backend")
    rig.backend.set_debug_view(DebugView.OVERDRAW)
    img = rig.draw(
        [
            (np.array([0.5, 0.5, 0.5, 1.0], np.float32), (0.0, 0.0, 0.5), 0.6, 0.5),
            (np.array([0.5, 0.5, 0.5, 1.0], np.float32), (0.0, 0.0, 0.5), 0.4, -0.5),
        ],
        ambient=[0.3] * 3,
    )
    assert Rig.corner(img)[0] == 0

    assert round(int(Rig.center(img)[0]) / 16) == 2


def test_wireframe_is_another_program_and_still_shades(rig):

    if DebugView.WIREFRAME not in rig.backend.caps.debug_views:
        pytest.skip("wireframe debug view unsupported by this backend")
    quad = (np.array([0.7, 0.7, 0.7, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.0)
    inside = (HEIGHT // 4, WIDTH // 2)

    shaded = rig.draw([quad], ambient=[0.35] * 3)
    rig.backend.set_debug_view(DebugView.WIREFRAME)
    wire = rig.draw([quad], ambient=[0.35] * 3)
    rig.backend.set_debug_view(DebugView.SHADED)
    back = rig.draw([quad], ambient=[0.35] * 3)

    assert np.array_equal(Rig.center(shaded), Rig.center(back))
    assert np.abs(wire[inside].astype(int) - shaded[inside].astype(int)).max() <= 2

    face = int(Rig.center(shaded)[0])
    line = int(Rig.center(wire)[0])
    assert face > 20
    assert line < face * 0.75
