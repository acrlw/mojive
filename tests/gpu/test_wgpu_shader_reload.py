"""WGSL hot-reload coverage for the wgpu backend's scene shader module.

wgpu-side counterpart of forge's ProgramCache reload test in
test_forge_core.py; runs only under ``FORGE_VIEWER_BACKEND=wgpu`` (see
Makefile ``gpu-wgpu``).
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("wgpu")

from forge_viewer.adapters.base import SceneSource  # noqa: E402
from forge_viewer.render.scene import SceneBuilder  # noqa: E402
from forge_viewer.render.webgpu import programs  # noqa: E402
from forge_viewer.render.webgpu.backend import WgpuBackend  # noqa: E402
from forge_viewer.types import (  # noqa: E402
    CameraView,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
)

WIDTH, HEIGHT = 128, 96
QUAD = MeshKey(MeshShape.PLANE, -1)

# fs_scene's display-domain return; swapping the channels is a visible,
# still-compilable edit of the scene fragment stage.
_FS_SCENE_RETURN = "return vec4f(rgb, base.a);"
_FS_SCENE_SWAPPED = "return vec4f(rgb.bgr, base.a);"


def _quad() -> MeshData:

    p = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], np.float32)
    n = np.tile(np.array([0.0, -1.0, 0.0], np.float32), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    return MeshData(p, n, uv, np.array([0, 1, 2, 0, 2, 3], np.uint32))


def _lit_quad_backend() -> WgpuBackend:
    """One ambient-lit quad; mirrors the SceneBuilder rig in test_shading."""
    backend = WgpuBackend(WIDTH, HEIGHT, samples=1)
    backend.set_scene(SceneSource(meshes={QUAD: _quad()}, textures={}, skybox=None))
    camera = CameraView(
        eye=np.array([0.0, -3.0, 0.0], np.float32),
        target=np.zeros(3, np.float32),
        up=np.array([0.0, 0.0, 1.0], np.float32),
        near=0.1,
        far=20.0,
    )
    backend.set_camera(camera)
    sb = SceneBuilder()
    matid = sb.material_id(Material())
    sb.add(
        QUAD,
        matid,
        np.eye(4, dtype=np.float32),
        np.array([0.8, 0.2, 0.3, 1.0], np.float32),
        np.zeros(4, np.float32),
        object_id=1,
    )
    ambient = np.full(3, 0.5, np.float32)
    backend.set_render_scene(
        sb.build(camera, LightSet(lights=(), headlight=None, ambient=ambient), 2.0, ambient)
    )
    return backend


def _write(path, text: str) -> None:
    # Keep mtimes distinct from the previous edit on coarse-mtime filesystems.
    time.sleep(0.01)
    path.write_text(text, encoding="utf-8")


def test_scene_shader_hot_reload_keeps_last_good_module(backend_name, tmp_path, monkeypatch):
    if backend_name != "wgpu":
        pytest.skip("wgpu shader hot reload; run with FORGE_VIEWER_BACKEND=wgpu")
    shader_copy = tmp_path / "shaders"
    shutil.copytree(programs._SHADER_DIR, shader_copy)
    monkeypatch.setattr(programs, "_SHADER_DIR", shader_copy)

    backend = _lit_quad_backend()
    try:
        backend.enable_hot_reload()
        assert backend.render(None) is not None
        base = backend.target.read_color(flip=True).copy()

        # A compilable edit takes effect on the next render.
        scene_wgsl = shader_copy / "scene.wgsl"
        original = scene_wgsl.read_text(encoding="utf-8")
        assert original.count(_FS_SCENE_RETURN) == 1
        _write(scene_wgsl, original.replace(_FS_SCENE_RETURN, _FS_SCENE_SWAPPED))
        assert backend.render(None) is not None
        swapped = backend.target.read_color(flip=True)
        changed = np.max(np.abs(swapped.astype(np.int16) - base.astype(np.int16)), axis=-1)
        assert np.count_nonzero(changed > 10) > 100

        # Broken WGSL keeps the last good module and pipelines.
        _write(scene_wgsl, "this is not WGSL\n")
        assert backend.render(None) is not None
        assert np.array_equal(backend.target.read_color(flip=True), swapped)
        assert backend._shader_reload_error

        # A fixed source reloads again and restores the original output.
        _write(scene_wgsl, original)
        assert backend.render(None) is not None
        assert np.array_equal(backend.target.read_color(flip=True), base)
        assert not backend._shader_reload_error
    finally:
        backend.release()
