"""Conservative outline raster bounds and GPU-mesh bound ownership."""

from types import SimpleNamespace

import numpy as np
import pytest

from mojive.render.opengl.instances import GpuMesh
from mojive.render.opengl.passes.outline import OutlinePass


def _context(*, bounds=None, transforms=None, ids=(7,)):
    if bounds is None:
        bounds = np.array([[-0.25] * 3, [0.25] * 3])
    if transforms is None:
        transforms = np.repeat(np.eye(4)[None], len(ids), axis=0)
    return SimpleNamespace(
        target=SimpleNamespace(width=200, height=100),
        meshes=[SimpleNamespace(local_bounds=bounds)],
        scene=SimpleNamespace(
            object_id=np.array(ids), transforms=transforms, bucket_ranges=((0, len(ids)),)
        ),
        view_proj=np.eye(4),
    )


def _bounds(ctx):
    outline = OutlinePass()
    outline._buckets = (0,)
    return outline._selected_scissor(ctx, 7)


def test_selected_bounds_include_kernel_and_raster_padding():
    assert _bounds(_context()) == (70, 32, 60, 36)


def test_bounds_union_every_selected_instance_and_ignore_other_objects():
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[0, 0, 3], transforms[1, 0, 3] = -0.5, 0.5
    transforms[2, :3, :3] *= 100
    assert _bounds(_context(transforms=transforms, ids=(7, 7, 8))) == (20, 32, 160, 36)


@pytest.mark.parametrize("invalid", ["near", "eye", "nonfinite", "unknown"])
def test_uncertain_bounds_keep_full_viewport(invalid):
    ctx = _context()
    if invalid == "near":
        ctx.scene.transforms[0, 2, 3] = -1
    elif invalid == "eye":
        ctx.view_proj[3, 3] = 0
    elif invalid == "nonfinite":
        ctx.scene.transforms[0, 0, 3] = np.nan
    else:
        ctx.meshes[0].local_bounds = None
    assert _bounds(ctx) is None


def test_offscreen_bounds_clamp_to_an_empty_raster_region():
    ctx = _context()
    ctx.scene.transforms[0, 0, 3] = 4
    assert _bounds(ctx) == (200, 32, 0, 36)


def test_asymmetric_float32_extents_keep_the_uploaded_upper_endpoint():
    bounds = np.array([[-100000, -0.25, -0.25], [0.001, 0.25, 0.25]], dtype=np.float32)
    ctx = _context(bounds=bounds)
    ctx.view_proj[0, 0] = 10000
    assert _bounds(ctx) == (0, 32, 200, 36)


def test_gpu_mesh_recomputes_bounds_after_deformation():
    class Buffer:
        def write(self, _data):
            pass

    ctx = SimpleNamespace(buffer=lambda _data: Buffer())
    positions = np.array([[-1, -2, -3], [4, 5, 6]], dtype=np.float32)
    mesh = GpuMesh(ctx, positions, np.zeros_like(positions), np.zeros((2, 2)), [0, 1, 0])
    assert not mesh.has_vertex_updates
    assert mesh._local_bounds is None
    np.testing.assert_array_equal(mesh.local_bounds, positions)
    cached = mesh.local_bounds
    assert mesh.local_bounds is cached
    changed = positions * -2
    mesh.update(changed, np.zeros_like(changed))
    assert mesh.has_vertex_updates
    assert mesh._local_bounds is None
    np.testing.assert_array_equal(mesh.local_bounds, [changed[1], changed[0]])
    assert mesh.local_bounds is not cached


def test_failed_upload_does_not_mark_a_mesh_as_dynamically_updated():
    class Buffer:
        def write(self, _data):
            raise RuntimeError("upload failed")

    ctx = SimpleNamespace(buffer=lambda _data: Buffer())
    positions = np.zeros((3, 3), dtype=np.float32)
    mesh = GpuMesh(ctx, positions, positions, np.zeros((3, 2)), [0, 1, 2])
    with pytest.raises(RuntimeError, match="upload failed"):
        mesh.update(positions, positions)
    assert not mesh.has_vertex_updates


def test_dynamically_updated_mesh_never_scans_outline_bounds():
    class DynamicMesh:
        has_vertex_updates = True

        @property
        def local_bounds(self):
            pytest.fail("A deformable outline scanned uploaded vertices")

    ctx = _context()
    ctx.meshes[0] = DynamicMesh()
    assert _bounds(ctx) is None


@pytest.mark.parametrize("fail", [False, True])
def test_scissor_restored_and_alpha_cleared_without_clipping(fail):
    outline = OutlinePass()
    outline._scissor = (20, 30, 40, 50)
    gl = SimpleNamespace(scissor=(1, 2, 300, 400))
    calls = []

    def clear(*_args):
        calls.append(("clear", gl.scissor))

    def render(*_args, **_kwargs):
        calls.append(("render", gl.scissor))
        if fail:
            raise RuntimeError("draw failed")

    vao, fbo = SimpleNamespace(render=render), SimpleNamespace(clear=clear)
    if fail:
        with pytest.raises(RuntimeError, match="draw failed"):
            outline._draw_fullscreen(gl, vao, clear=fbo)
    else:
        outline._draw_fullscreen(gl, vao, clear=fbo)
    assert calls == [("clear", None), ("render", outline._scissor)]
    assert gl.scissor == (1, 2, 300, 400)


def test_outline_intersects_an_existing_scissor_without_broadening_it():
    outline = OutlinePass()
    outline._scissor = (20, 30, 40, 50)
    gl = SimpleNamespace(scissor=(10, 20, 30, 40))
    seen = []
    vao = SimpleNamespace(render=lambda *_args, **_kwargs: seen.append(gl.scissor))
    outline._draw_fullscreen(gl, vao)
    assert seen == [(20, 30, 20, 30)]
    assert gl.scissor == (10, 20, 30, 40)
