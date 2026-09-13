from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

moderngl = pytest.importorskip("moderngl")

from mojive import math3d as M  # noqa: E402
from mojive.render.backend import RenderFlag  # noqa: E402
from mojive.render.mesh import all_builtin  # noqa: E402
from mojive.render.opengl import gl_native as G  # noqa: E402
from mojive.render.opengl.instances import InstanceStore, Strategy  # noqa: E402
from mojive.render.opengl.passes.base import PassContext, state_opaque  # noqa: E402
from mojive.render.opengl.passes.idbuffer import IdBufferPass  # noqa: E402
from mojive.render.opengl.passes.outline import OUTLINE_RADIUS, OutlinePass  # noqa: E402
from mojive.render.opengl.programs import ProgramCache  # noqa: E402
from mojive.render.opengl.resources import MeshStore, TextureStore  # noqa: E402
from mojive.render.opengl.targets import IdLayout, RenderTarget, probe_id_layout  # noqa: E402
from mojive.render.opengl.timing import FrameTiming  # noqa: E402
from mojive.render.scene import SceneBuilder  # noqa: E402
from mojive.types import CameraView, LightSet, Material, MeshKey, MeshShape  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _opengl_backend_only(require_opengl):
    pass


W, H = 256, 192
ASPECT = W / H
ORTHO = 2.0


BACKGROUND = (0.05, 0.05, 0.08, 1.0)
OUTLINE_COLOR = (1.0, 0.0, 1.0, 1.0)


MATERIAL = np.array([0.0, 0.5, 0.5, 0.0], np.float32)


def u8(color) -> tuple[int, int, int]:
    return tuple(round(c * 255.0) for c in color[:3])


def col_to_world(col: float) -> float:
    return ((col / W) * 2.0 - 1.0) * ORTHO * ASPECT * 0.5


def world_to_col(x: float) -> float:
    return (x / (ORTHO * ASPECT * 0.5) * 0.5 + 0.5) * W


def world_to_row(y: float) -> float:
    return (y / (ORTHO * 0.5) * 0.5 + 0.5) * H


_REF_VS = """#version 330 core
in vec3 in_position;
in vec4 in_model0; in vec4 in_model1; in vec4 in_model2; in vec4 in_model3;
in vec4 in_color;
uniform mat4 u_view_proj;
out vec4 v_color;
void main() {
    mat4 m = mat4(in_model0, in_model1, in_model2, in_model3);
    v_color = in_color;
    gl_Position = u_view_proj * m * vec4(in_position, 1.0);
}
"""
_REF_FS = """#version 330 core
in vec4 v_color;
layout(location = 0) out vec4 o_color;
void main() { o_color = v_color; }
"""


class Rig:
    def __init__(self, gl, samples: int) -> None:
        self.gl = gl
        self.target = RenderTarget(gl, W, H, samples=samples)
        self.programs = ProgramCache(gl)
        self.meshes = MeshStore(gl)
        self.meshes.sync(all_builtin())
        self.textures = TextureStore(gl)
        self.instances = InstanceStore(gl)
        self.timing = FrameTiming(gl, enabled=False)
        self.ref = gl.program(vertex_shader=_REF_VS, fragment_shader=_REF_FS)
        self.id_pass = IdBufferPass()
        self.outline = OutlinePass()
        self.outline.color = OUTLINE_COLOR
        self.scene = None
        self.bucket_meshes: list = []

    # ------------------------------------------------------------------
    def set_scene(self, scene, strategy: Strategy | None = None) -> None:
        self.scene = scene
        if strategy is not None:
            self.instances.strategy = strategy
        self.bucket_meshes = [self.meshes.get(k) for k, _ in scene.bucket_keys]
        self.instances.rebuild(scene, self.ref, self.bucket_meshes, generation=0)
        self.instances.upload(scene)

    def context(self, selected: int = 0, outline: bool = True) -> PassContext:
        cam = self.scene.camera.with_aspect(ASPECT)
        view, proj = cam.view_matrix(), cam.proj_matrix()
        return PassContext(
            ctx=self.gl,
            target=self.target,
            scene=self.scene,
            camera=cam,
            view=view,
            proj=proj,
            view_proj=(proj @ view).astype(np.float32),
            instances=self.instances,
            programs=self.programs,
            textures=self.textures,
            meshes=self.bucket_meshes,
            timing=self.timing,
            flags={RenderFlag.OUTLINE: outline},
            selected_id=selected,
            background=BACKGROUND,
        )

    def frame(self, selected: int = 0, outline: bool = True) -> PassContext:

        ctx = self.context(selected, outline)
        self._reference_color(ctx)
        for p in (self.id_pass, self.outline):
            if p.prepare(ctx):
                p.execute(ctx)
        assert G.native().drain_errors() == 0
        return ctx

    def _reference_color(self, ctx: PassContext) -> None:

        t = self.target
        t.clear_main(BACKGROUND)
        if t.id_layout is IdLayout.SHARED:
            t.fbo.color_mask = ((True, True, True, True), (False, False, False, False))
        t.use_main()
        state_opaque(self.gl)
        self.ref["u_view_proj"].write(M.to_gl(ctx.view_proj))
        for b in ctx.scene.opaque_buckets:
            self.instances.draw(b)
        if t.id_layout is IdLayout.SHARED:
            t.fbo.color_mask = ((True, True, True, True), (True, True, True, True))

    # ------------------------------------------------------------------
    def picture(self) -> np.ndarray:

        t = self.target
        if t.samples > 1:
            t.resolve()
            raw = t.resolve_fbo.read(components=4, dtype="f1")
        else:
            raw = t.fbo.read(components=4, dtype="f1", attachment=0)
        return np.frombuffer(raw, np.uint8).reshape(H, W, 4)[..., :3]

    def ids(self) -> np.ndarray:
        return self.target.read_ids()

    def outline_mask(self) -> np.ndarray:
        return np.all(self.picture() == np.array(u8(OUTLINE_COLOR), np.uint8), axis=-1)

    def outline_coverage(self) -> np.ndarray:
        fbo = self.outline._mask_fbo
        assert fbo is not None
        raw = fbo.read(components=1, dtype="f1")
        return np.frombuffer(raw, np.uint8).reshape(H, W).astype(np.float32) / 255.0

    def release(self) -> None:
        self.outline.release()
        self.id_pass.release()
        self.instances.release()
        self.meshes.release()
        self.textures.release()
        self.programs.release()
        self.ref.release()
        self.target.release()


@pytest.fixture
def rig(gl_ctx, request):
    samples = getattr(request, "param", 4)
    r = Rig(gl_ctx, samples)
    yield r
    r.release()


def _camera() -> CameraView:
    return CameraView(
        orthographic=True,
        ortho_height=ORTHO,
        eye=np.array([0.0, 0.0, 4.0], np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0.0, 1.0, 0.0], np.float32),
        near=0.1,
        far=10.0,
        aspect=ASPECT,
    )


class Scene:
    def __init__(self) -> None:
        self.b = SceneBuilder()
        self.mat = Material(name="t")
        self.matid = self.b.material_id(self.mat)

    def add(self, shape, pos, scale, color, object_id, rot=None):
        self.b.add(
            MeshKey(shape),
            self.matid,
            M.compose(pos, np.eye(3) if rot is None else rot, scale),
            np.asarray(color, np.float32),
            MATERIAL,
            object_id=object_id,
        )
        return self

    def build(self):
        return self.b.build(_camera(), LightSet(), 2.0, np.zeros(3, np.float32))


# ================================================================ ID buffer

ROD_ID, SPHERE_ID = 11, 22
ROD_COLOR = (1.0, 0.0, 0.0, 1.0)
SPHERE_COLOR = (0.0, 1.0, 0.0, 1.0)


def _rod_and_sphere():

    s = Scene()
    s.add(MeshShape.SPHERE, (0.15, 0.0, -0.5), (0.4, 0.4, 0.4), SPHERE_COLOR, SPHERE_ID)
    s.add(
        MeshShape.CYLINDER,
        (0.0, 0.0, 0.5),
        (0.03, 0.03, 0.9),
        ROD_COLOR,
        ROD_ID,
        rot=M.axis_angle_to_mat3((1.0, 0.0, 0.0), np.pi / 2),
    )
    return s.build()


@pytest.mark.parametrize("rig", [0, 4], indirect=True)
def test_id_buffer_matches_the_picture_pixel_by_pixel(rig):

    rig.set_scene(_rod_and_sphere())
    rig.frame()
    assert rig.target.id_layout is probe_id_layout(rig.gl, rig.target.samples)
    _check_id_matches_picture(rig)


def _check_id_matches_picture(rig) -> None:
    ids = rig.ids()
    img = rig.picture()
    owner = np.full(ids.shape, -1, np.int64)
    for color, oid in ((BACKGROUND, 0), (ROD_COLOR, ROD_ID), (SPHERE_COLOR, SPHERE_ID)):
        owner[np.all(img == np.array(u8(color), np.uint8), axis=-1)] = oid

    known = owner >= 0
    assert known.mean() > 0.9
    assert np.count_nonzero(owner == ROD_ID) > 300
    assert np.count_nonzero(owner == SPHERE_ID) > 1500
    assert set(np.unique(ids).tolist()) == {0, ROD_ID, SPHERE_ID}

    bad = np.count_nonzero(ids[known] != owner[known])
    assert bad == 0


def test_id_buffer_works_with_the_per_bucket_fallback(rig):

    rig.set_scene(_rod_and_sphere(), strategy=Strategy.PER_BUCKET)
    rig.frame()
    assert rig.instances.strategy is Strategy.PER_BUCKET
    _check_id_matches_picture(rig)


def test_object_id_is_exact_beyond_2_to_the_24(rig):

    ids = [1, 2, 16_777_216, 16_777_217, 4_000_000_000, 4_294_967_295]
    s = Scene()
    xs = np.linspace(-0.9, 0.9, len(ids))
    for x, oid in zip(xs, ids, strict=True):
        s.add(MeshShape.BOX, (float(x), 0.0, 0.0), (0.1, 0.3, 0.1), (0.6, 0.6, 0.6, 1.0), oid)
    rig.set_scene(s.build())
    rig.frame()

    row = int(world_to_row(0.0))
    for x, oid in zip(xs, ids, strict=True):
        col = int(world_to_col(float(x)))
        assert rig.target.read_id(col, row) == oid


def test_transparent_geometry_does_not_write_ids(rig):

    s = Scene()
    s.add(MeshShape.BOX, (0.0, 0.0, -0.4), (0.3, 0.3, 0.1), (0.2, 0.2, 0.9, 1.0), 5)
    s.add(MeshShape.BOX, (0.0, 0.0, 0.4), (0.5, 0.5, 0.1), (0.9, 0.9, 0.2, 0.4), 6)
    scene = s.build()
    assert scene.transparent_buckets
    rig.set_scene(scene)
    rig.frame()

    ids = rig.ids()
    assert 6 not in set(np.unique(ids).tolist())
    center = int(world_to_row(0.0)), int(world_to_col(0.0))
    assert ids[center] == 5


SEL = 7


def _selected_box(center_x=0.0, half=(0.35, 0.35, 0.1), occluder=False):
    s = Scene()
    s.add(MeshShape.BOX, (center_x, 0.0, 0.0), half, (0.2, 0.6, 0.3, 1.0), SEL)
    if occluder:
        s.add(MeshShape.BOX, (center_x, 0.0, 0.6), (0.6, 0.6, 0.1), (0.8, 0.2, 0.2, 1.0), 8)
    return s.build()


def test_outline_is_solid_through_an_occluder(rig):

    rig.set_scene(_selected_box())
    rig.frame(selected=SEL)
    alone = int(rig.outline_mask().sum())

    rig.set_scene(_selected_box(occluder=True))
    rig.frame(selected=SEL)
    occluded_ids = rig.ids()
    behind = int(rig.outline_mask().sum())

    center = int(world_to_row(0.0)), int(world_to_col(0.0))
    assert occluded_ids[center] == 8
    assert alone > 200
    assert abs(alone - behind) <= 2


def test_one_outline_per_link(rig):

    seam_x = 0.15

    def build(second_id: int):
        s = Scene()
        s.add(MeshShape.BOX, (-0.15, 0.0, 0.0), (0.3, 0.35, 0.1), (0.2, 0.6, 0.3, 1.0), SEL)
        s.add(MeshShape.BOX, (0.35, 0.0, 0.0), (0.3, 0.35, 0.1), (0.2, 0.6, 0.3, 1.0), second_id)
        return s.build()

    seam = int(world_to_col(seam_x))
    rows = slice(int(world_to_row(-0.2)), int(world_to_row(0.2)))
    strip = slice(seam - 2, seam + 3)

    rig.set_scene(build(SEL))
    rig.frame(selected=SEL)
    together = rig.outline_mask()
    assert together.sum() > 200
    assert together[rows, strip].sum() == 0

    rig.set_scene(build(8))
    rig.frame(selected=SEL)
    split = rig.outline_mask()
    assert split[rows, strip].sum() > 0


def test_corner_is_round_not_square(rig):

    rig.set_scene(_selected_box())
    rig.frame(selected=SEL)
    out = rig.outline_mask()
    rows, cols = np.nonzero(rig.outline_coverage() >= 1.0)
    r1, c1 = int(rows.max()), int(cols.max())

    r = OUTLINE_RADIUS
    assert out[r1, c1 + r]
    assert out[r1 + r, c1]
    assert out[r1 + 2, c1 + 2]
    assert not out[r1 + r, c1 + r]


def test_outline_outer_edge_is_antialiased(rig):

    rig.set_scene(_selected_box())
    rig.frame(selected=SEL)
    ids, picture = rig.ids(), rig.picture()
    rows, cols = np.nonzero(ids == SEL)
    pixel = picture[int(rows.max()) + 2, int(cols.max()) + 3]

    assert not np.array_equal(pixel, np.array(u8(OUTLINE_COLOR), np.uint8))
    assert not np.array_equal(pixel, np.array(u8(BACKGROUND), np.uint8))


def test_outline_mask_keeps_subpixel_triangle_coverage(rig):

    s = Scene()
    rot = M.axis_angle_to_mat3((0.0, 0.0, 1.0), np.deg2rad(23.0))
    s.add(
        MeshShape.BOX,
        (0.0, 0.0, 0.0),
        (0.42, 0.22, 0.1),
        (0.2, 0.6, 0.3, 1.0),
        SEL,
        rot=rot,
    )
    rig.set_scene(s.build())
    rig.frame(selected=SEL)

    coverage = rig.outline_coverage()
    partial = coverage[(coverage > 0.0) & (coverage < 1.0)]
    assert len(partial) > 50
    assert len(np.unique(partial)) >= 3


def test_outline_works_for_a_huge_object_id(rig):

    big = 4_294_967_295
    s = Scene()
    s.add(MeshShape.BOX, (0.0, 0.0, 0.0), (0.3, 0.3, 0.1), (0.2, 0.6, 0.3, 1.0), big)
    rig.set_scene(s.build())
    rig.frame(selected=big)
    assert rig.outline_mask().sum() > 200


def test_no_outline_smeared_along_the_viewport_border(rig):

    rig.set_scene(_selected_box(half=(0.3, 0.3, 0.1)))
    rig.frame(selected=SEL)
    out = rig.outline_mask()
    assert out.sum() > 200
    assert out[0].sum() == 0 and out[-1].sum() == 0
    assert out[:, 0].sum() == 0 and out[:, -1].sum() == 0


def test_outline_hugs_the_viewport_edge_when_clipped(rig):

    rig.set_scene(_selected_box(center_x=-1.1, half=(0.5, 0.4, 0.1)))
    rig.frame(selected=SEL)
    ids, out = rig.ids(), rig.outline_mask()
    assert (ids[:, 0] == SEL).any()

    rows = np.nonzero((ids == SEL).any(axis=1))[0]
    inner = rows[OUTLINE_RADIUS:-OUTLINE_RADIUS]
    band = out[:, :OUTLINE_RADIUS].any(axis=1)
    missing = [int(r) for r in inner if not band[r]]
    assert not missing


def test_object_near_the_border_still_gets_an_outer_outline(rig):

    half_x = 0.5
    left_edge = col_to_world(2.0)
    rig.set_scene(_selected_box(center_x=left_edge + half_x, half=(half_x, 0.4, 0.1)))
    rig.frame(selected=SEL)
    ids, out = rig.ids(), rig.outline_mask()

    assert (ids[:, 2] == SEL).any()
    assert not (ids[:, 0] == SEL).any() and not (ids[:, 1] == SEL).any()
    assert out[:, :2].sum() > 50
    inside = int((out & (ids == SEL)).sum())
    assert inside == 0


def test_outline_pass_is_skipped_when_there_is_nothing_to_outline(rig):

    s = Scene()
    s.add(MeshShape.BOX, (0.0, 0.0, 0.0), (0.35, 0.35, 0.1), (0.2, 0.6, 0.3, 1.0), SEL)
    s.add(MeshShape.BOX, (0.9, 0.0, 0.0), (0.2, 0.2, 0.1), (0.5, 0.5, 0.5, 1.0), 0)
    rig.set_scene(s.build())
    assert rig.outline.prepare(rig.context(selected=0)) is False
    assert rig.outline.prepare(rig.context(selected=SEL, outline=False)) is False
    assert rig.outline.prepare(rig.context(selected=999)) is False
    assert rig.outline.prepare(rig.context(selected=SEL)) is True

    rig.frame(selected=0)
    assert rig.outline_mask().sum() == 0
