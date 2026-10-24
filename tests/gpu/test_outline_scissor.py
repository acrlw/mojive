"""Exact color parity between bounded and full-frame outline rasterization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from tests.gpu.test_id_outline import SEL, H, Scene, W
from tests.gpu.test_id_outline import rig as rig
from tests.gpu.test_outline_cache import _draw, _managed

from mojive.render.backend import RenderFlag
from mojive.render.opengl.passes.outline import OutlinePass
from mojive.render.opengl.targets import RenderTarget
from mojive.types import MeshShape

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _opengl_only(require_opengl):
    pass


def _full_frame_picture(rig, ctx):
    reference = OutlinePass()
    reference.color, reference.xray = rig.outline.color, rig.outline.xray
    reference._selected_scissor = lambda _ctx, _selected: None
    try:
        return _draw(rig, ctx, reference)
    finally:
        reference.release()


@pytest.mark.parametrize("rig", [0, 4], indirect=True)
@pytest.mark.parametrize("msaa", [False, True])
@pytest.mark.parametrize("xray", [False, True])
def test_scissor_preserves_viewport_edges_and_cached_alpha(rig, msaa, xray):
    for x, y in ((0, 0), (-1.25, 0), (1.25, 0), (0, -0.95), (0, 0.95)):
        scene = Scene()
        scene.add(MeshShape.BOX, (x, y, 0), (0.253, 0.207, 0.1), (0.2, 0.6, 0.3, -1.35), SEL)
        rig.set_scene(_managed(scene.build()))
        rig.outline.xray = xray
        ctx = rig.context(selected=SEL)
        ctx.flags[RenderFlag.MSAA] = msaa
        expected = _full_frame_picture(rig, ctx)
        for status in ("rendered", "primed", "reused"):
            actual = _draw(rig, ctx)
            np.testing.assert_array_equal(actual, expected)
            assert rig.outline.cache_status == status
        if rig.target.samples == 4 and msaa and not xray and (x, y) in ((0, 0), (-1.25, 0)):
            output = Path("output/outline-scissor")
            output.mkdir(parents=True, exist_ok=True)
            for name, pixels in (("bounded", actual), ("full-frame", expected)):
                Image.fromarray(np.flipud(pixels)).save(output / f"{x}-{y}-{name}.png")
        _, _, width, height = rig.outline._scissor
        assert width * height < rig.target.width * rig.target.height / 2


@pytest.mark.parametrize("rig", [0, 4], indirect=True)
def test_scissor_tracks_selection_resize_deformation_and_legacy_motion(rig, monkeypatch):
    scene = Scene()
    scene.add(MeshShape.BOX, (-0.65, 0, 0), (0.2, 0.25, 0.1), (0.2, 0.6, 0.3, 1), SEL)
    scene.add(MeshShape.SPHERE, (0.65, 0, 0), (0.2, 0.2, 0.2), (0.6, 0.2, 0.3, 1), 8)
    rig.set_scene(_managed(scene.build()))
    ctx = rig.context(selected=SEL)
    for change in ("initial", "rigid", "selection", "resize", "deform", "legacy"):
        if change == "rigid":
            rig.scene.transforms[:, 1, 3] += 0.15
            rig.scene.pose_revision += 1
        elif change == "selection":
            ctx.outline_id = 8
        elif change == "resize":
            rig.target.release()
            rig.target = RenderTarget(rig.gl, W + 32, H + 16, samples=rig.target.samples)
            ctx = rig.context(selected=SEL)
        elif change == "deform":
            bucket = next(
                b for b, key in enumerate(rig.scene.bucket_keys) if key[0].shape is MeshShape.BOX
            )
            mesh = rig.bucket_meshes[bucket]
            positions = mesh._vertices[:, :3].copy()
            positions[:, 0] += 1.5
            mesh.update(positions, mesh._vertices[:, 3:6])
            ctx.mesh_revision += 1

            def unexpected_bounds(_mesh):
                pytest.fail("A selected dynamic mesh computed outline bounds")

            monkeypatch.setattr(type(mesh), "local_bounds", property(unexpected_bounds))
        elif change == "legacy":
            rig.scene.pose_revision = 0
            rig.scene.transforms[:, 1, 3] += 0.35
        expected = _full_frame_picture(rig, ctx)
        for _ in range(3):
            np.testing.assert_array_equal(_draw(rig, ctx), expected)
        if change in ("deform", "legacy"):
            assert rig.outline._scissor is None
        else:
            assert rig.outline._scissor is not None


@pytest.mark.parametrize("rig", [0, 4], indirect=True)
def test_near_plane_crossing_uses_the_full_frame_path(rig):
    scene = Scene()
    scene.add(MeshShape.BOX, (0, 0, 3.85), (0.04, 0.04, 0.1), (0.2, 0.6, 0.3, 1), SEL)
    scene = _managed(scene.build())
    scene.camera = replace(scene.camera, orthographic=False)
    rig.set_scene(scene)
    ctx = rig.context(selected=SEL)
    expected = _full_frame_picture(rig, ctx)
    for _ in range(3):
        np.testing.assert_array_equal(_draw(rig, ctx), expected)
        assert rig.outline._scissor is None
