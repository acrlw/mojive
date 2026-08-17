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
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    TextureData,
    TextureKind,
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
    def __init__(self, backend: ForgeBackend, textures=None, skybox=None, mesh=None) -> None:
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


@pytest.fixture
def rig(gl_ctx):
    passes.load_all()
    backend = ForgeBackend(gl_ctx, WIDTH, HEIGHT, samples=4)
    yield Rig(backend)
    backend.release()


def _dir_light(diffuse: float, specular: float = 0.0) -> Light:

    return Light(
        kind=LightKind.DIRECTIONAL,
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


def test_transparent_leaves_depth_mask_on(rig):

    opaque_quad = (np.array([0.2, 0.8, 0.3, 1.0], np.float32), (0.0, 0.0, 0.5), 2.0, 0.5)
    glass = (np.array([0.9, 0.2, 0.2, 0.4], np.float32), (0.0, 0.0, 0.5), 1.0, -0.5)

    shots = []
    for _ in range(3):
        img = rig.draw([opaque_quad, glass], ambient=[0.3] * 3)
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


def test_skybox_only_shows_up_when_it_is_on(gl_ctx):

    passes.load_all()
    sky = np.zeros((6, 4, 4, 4), np.uint8)
    sky[..., :3] = np.array([40, 90, 200], np.uint8)
    sky[..., 3] = 255
    backend = ForgeBackend(gl_ctx, WIDTH, HEIGHT, samples=4)
    try:
        rig = Rig(
            backend,
            textures={"sky": TextureData("sky", TextureKind.SKYBOX, sky)},
            skybox="sky",
        )
        quad = (np.array([0.5, 0.5, 0.5, 1.0], np.float32), (0.0, 0.0, 0.5), 0.3, 0.0)

        backend.set_flag(RenderFlag.SKYBOX, False)
        off = Rig.corner(rig.draw([quad], ambient=[0.3] * 3))
        backend.set_flag(RenderFlag.SKYBOX, True)
        on = Rig.corner(rig.draw([quad], ambient=[0.3] * 3))

        clear = np.array([round(c * 255) for c in backend._make_context(backend._scene).background])
        assert np.abs(off - clear).max() <= 1
        assert np.abs(on - off).max() > 20

        assert not np.array_equal(Rig.center(rig.draw([quad], ambient=[0.3] * 3)), on)
    finally:
        backend.release()


def test_normals_survive_non_uniform_scale(gl_ctx):

    passes.load_all()
    n_local = np.array([0.0, -1.0, 1.0], np.float32) / np.sqrt(2.0)
    slanted = MeshData(
        np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32),
        np.tile(n_local, (4, 1)),
        np.zeros((4, 2), np.float32),
        np.array([0, 1, 2, 0, 2, 3], np.uint32),
    )
    backend = ForgeBackend(gl_ctx, WIDTH, HEIGHT, samples=4)
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


def test_shared_id_layout_is_not_polluted_by_shading_passes(gl_ctx):

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
