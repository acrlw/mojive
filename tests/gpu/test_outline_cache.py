"""Complete outline-alpha reuse, invalidation, and failure recovery on OpenGL."""

from __future__ import annotations

import numpy as np
import pytest
from tests.gpu.test_id_outline import SEL, H, Scene, W, _selected_box
from tests.gpu.test_id_outline import rig as rig

from mojive.render.backend import RenderFlag
from mojive.render.opengl import gl_native as G
from mojive.render.opengl.passes.outline import OutlinePass
from mojive.render.opengl.targets import RenderTarget
from mojive.types import MeshShape

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _opengl_only(require_opengl):
    pass


def _managed(scene):
    scene.structure_revision = scene.pose_revision = 1
    scene.visual_revision = scene.identity_revision = 1
    return scene


def _draw(rig, ctx=None, outline=None):
    ctx = rig.context(selected=SEL) if ctx is None else ctx
    outline = rig.outline if outline is None else outline
    rig.instances.upload(rig.scene)
    rig.gl.multisample = bool(ctx.flag(RenderFlag.MSAA))
    rig._reference_color(ctx)
    assert outline.prepare(ctx)
    outline.execute(ctx)
    rig.target.resolve()
    assert G.native().drain_errors() == 0
    return rig.target.read_color(flip=False).copy()


def _assert_fresh_picture(rig, ctx, expected):
    fresh = OutlinePass()
    fresh.color, fresh.xray = rig.outline.color, rig.outline.xray
    try:
        np.testing.assert_array_equal(_draw(rig, ctx, fresh), expected)
    finally:
        fresh.release()


@pytest.mark.parametrize("rig", [0, 4], indirect=True)
@pytest.mark.parametrize("xray", [False, True])
def test_complete_alpha_reused_with_identical_edges_and_live_color(rig, monkeypatch, xray):
    # Include both a viewport-clipped edge and fractional multisample coverage.
    scene = _managed(_selected_box(center_x=-1.1, half=(0.503, 0.407, 0.1)))
    rig.set_scene(scene)
    rig.outline.xray = xray
    first = _draw(rig)
    assert rig.outline.cache_status == "rendered"
    assert rig.outline._alpha_tex is None
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "primed"

    def unexpected_render(_ctx):
        pytest.fail("An unchanged outline rebuilt its complete alpha product")

    with monkeypatch.context() as patch:
        patch.setattr(rig.outline, "_render_alpha", unexpected_render)
        patch.setattr(rig.outline, "_render_mask", unexpected_render)
        np.testing.assert_array_equal(_draw(rig), first)
        assert rig.outline.cache_status == "reused"
        rig.outline.color = (0.2, 0.7, 1.0, 0.65)
        recolored = _draw(rig)
        assert rig.outline.cache_status == "reused"
        assert np.any(recolored != first)
    _assert_fresh_picture(rig, rig.context(selected=SEL), recolored)


@pytest.mark.parametrize(
    "change",
    [
        "camera",
        "pose",
        "visual",
        "identity",
        "structure",
        "mesh",
        "program",
        "size",
        "samples",
        "msaa_flag",
        "scene",
        "selection",
        "outline_id",
        "xray",
    ],
)
def test_outline_alpha_tracks_render_dependencies(rig, change):
    scene = Scene()
    scene.add(MeshShape.BOX, (-0.4, 0.0, 0.0), (0.25, 0.35, 0.1), (0.2, 0.6, 0.3, 1), SEL)
    scene.add(MeshShape.SPHERE, (0.4, 0.0, 0.0), (0.25, 0.25, 0.25), (0.6, 0.2, 0.3, 1), 8)
    rig.set_scene(_managed(scene.build()))
    _draw(rig)
    _draw(rig)
    assert rig.outline.cache_status == "primed"
    _draw(rig)
    assert rig.outline.cache_status == "reused"

    scene = rig.scene
    if change == "camera":
        scene.camera.eye[0] += 0.3
    elif change == "pose":
        scene.transforms[0, 0, 3] += 0.2
        scene.pose_revision += 1
    elif change == "visual":
        scene.colors[0, 3] = -1.35  # Ordered coverage participates in the selected mask.
        scene.visual_revision += 1
    elif change == "identity":
        scene.object_id[:] = scene.object_id[::-1]
        scene.identity_revision += 1
    elif change == "structure":
        scene.structure_revision += 1
    elif change == "program":
        rig.programs.generation += 1
    elif change in {"size", "samples"}:
        rig.target.release()
        rig.target = RenderTarget(
            rig.gl,
            W + (16 if change == "size" else 0),
            H,
            samples=0 if change == "samples" else 4,
        )
    elif change == "scene":
        rig.set_scene(_managed(_selected_box(center_x=0.4)))
    elif change == "xray":
        rig.outline.xray = True

    ctx = rig.context(selected=8 if change == "selection" else SEL)
    if change == "outline_id":
        ctx.outline_id = 8
    if change == "mesh":
        ctx.mesh_revision += 1
    if change == "msaa_flag":
        ctx.flags[RenderFlag.MSAA] = False
    changed = _draw(rig, ctx)
    assert rig.outline.cache_status == "rendered"
    np.testing.assert_array_equal(_draw(rig, ctx), changed)
    assert rig.outline.cache_status == "primed"
    np.testing.assert_array_equal(_draw(rig, ctx), changed)
    assert rig.outline.cache_status == "reused"
    _assert_fresh_picture(rig, ctx, changed)


def test_revision_zero_outline_sources_redraw_even_without_a_frame_serial(rig, monkeypatch):
    rig.set_scene(_selected_box())

    def unexpected_alpha(_ctx):
        pytest.fail("A legacy source built an alpha product that it cannot reuse")

    monkeypatch.setattr(rig.outline, "_render_alpha", unexpected_alpha)
    monkeypatch.setattr(rig.outline, "_ensure_alpha", unexpected_alpha)
    first = _draw(rig)
    _draw(rig)
    assert rig.outline.cache_status == "rendered"
    rig.scene.transforms[0, 0, 3] += 0.3
    changed = _draw(rig)
    assert rig.outline.cache_status == "rendered"
    assert np.any(changed != first)
    assert rig.outline._alpha_tex is None
    _assert_fresh_picture(rig, rig.context(selected=SEL), changed)


@pytest.mark.parametrize("motion", ["pose", "camera"])
def test_continuously_changing_outline_uses_direct_rendering(rig, monkeypatch, motion):
    rig.set_scene(_managed(_selected_box()))

    def unexpected_alpha(_ctx):
        pytest.fail("A changing selection paid for an unused alpha product")

    with monkeypatch.context() as patch:
        patch.setattr(rig.outline, "_render_alpha", unexpected_alpha)
        patch.setattr(rig.outline, "_ensure_alpha", unexpected_alpha)
        for _ in range(4):
            if motion == "pose":
                rig.scene.transforms[0, 0, 3] += 0.05
                rig.scene.pose_revision += 1
            else:
                rig.scene.camera.eye[0] += 0.05
            picture = _draw(rig)
            assert rig.outline.cache_status == "rendered"
            assert rig.outline._alpha_tex is None
    np.testing.assert_array_equal(_draw(rig), picture)
    assert rig.outline.cache_status == "primed"
    np.testing.assert_array_equal(_draw(rig), picture)
    assert rig.outline.cache_status == "reused"


def test_failed_alpha_refresh_cannot_revive_the_previous_cached_product(rig, monkeypatch):
    rig.set_scene(_managed(_selected_box()))
    first = _draw(rig)
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "primed"
    render = rig.outline._render_alpha

    def fail_after_overwrite(ctx):
        render(ctx)
        raise RuntimeError("Injected alpha refresh failure")

    rig.outline.xray = True
    _draw(rig)
    assert rig.outline.cache_status == "rendered"
    with monkeypatch.context() as patch:
        patch.setattr(rig.outline, "_render_alpha", fail_after_overwrite)
        with pytest.raises(RuntimeError, match="alpha refresh failure"):
            _draw(rig)
    rig.outline.xray = False
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "rendered"
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "primed"
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "reused"


def test_released_outline_rebuilds_its_alpha_and_program_bindings(rig):
    rig.set_scene(_managed(_selected_box()))
    first = _draw(rig)
    _draw(rig)
    assert rig.outline._alpha_tex is not None
    rig.outline.release()
    np.testing.assert_array_equal(_draw(rig), first)
    assert rig.outline.cache_status == "rendered"
    assert rig.outline._alpha_tex is None
