from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..adapters.base import FrameNeeds
from ..render.forge.backend import ForgeBackend
from ..types import CameraView


@dataclass
class HarnessStats:
    draw_calls: int = 0
    instances: int = 0
    triangles: int = 0
    buckets: int = 0
    frame_ms: float = 0.0
    cpu_ms: dict | None = None
    gpu_ms: dict | None = None


class OffscreenHarness:
    def __init__(
        self,
        asset: Path,
        width: int = 1280,
        height: int = 800,
        samples: int = 4,
        backend: str = "mujoco",
    ) -> None:
        import glfw

        self._glfw = glfw
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        for k, v in (
            (glfw.CONTEXT_VERSION_MAJOR, 3),
            (glfw.CONTEXT_VERSION_MINOR, 3),
            (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
            (glfw.OPENGL_FORWARD_COMPAT, True),
            (glfw.VISIBLE, False),
        ):
            glfw.window_hint(k, v)
        self.window = glfw.create_window(width, height, "forge harness", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create an OpenGL 3.3 core context")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        from ..backends import make_adapter
        from ..render.builder import SceneSourceBuilder

        self.adapter = make_adapter(backend, asset)
        self.backend = ForgeBackend(None, width, height, samples)
        self.builder = SceneSourceBuilder()
        self.source = self.adapter.scene_source()
        self.backend.set_scene(self.source)
        self.camera = self._frame_camera()
        self.builder.set_source(self.source, self.camera)
        self.backend.set_camera(self.camera)
        self.needs = FrameNeeds(poses=True)

    # ------------------------------------------------------------------
    def _frame_camera(self) -> CameraView:

        hint = self.adapter.camera_hint()
        if hint is not None:
            return hint
        frame = self.adapter.frame(FrameNeeds(poses=True))
        pos = frame.geom_xpos
        if pos is None or len(pos) == 0:
            lo, hi = np.full(3, -0.5, np.float32), np.full(3, 0.5, np.float32)
        else:
            finite = np.isfinite(pos).all(axis=1)
            if len(self.source.geom_infinite_plane) == len(pos):
                finite &= ~self.source.geom_infinite_plane
            p = pos[finite] if finite.any() else pos
            r = self.source.geom_size[: len(p)].max(axis=1, keepdims=True) if len(p) else 0.0
            lo, hi = (p - r).min(axis=0), (p + r).max(axis=0)
        center = ((lo + hi) * 0.5).astype(np.float32)
        extent = float(np.linalg.norm(hi - lo)) * 0.5 or 1.0
        dist = extent * 2.6
        eye = center + np.array([dist * 0.75, -dist, dist * 0.55], np.float32)
        return CameraView(
            eye=eye,
            target=center,
            up=np.array([0.0, 0.0, 1.0], np.float32),
            near=max(extent * 0.01, 1e-3),
            far=extent * 40.0,
        )

    def step_and_render(self, steps: int = 1):
        if steps:
            self.adapter.step(steps)
        frame = self.adapter.frame(self.needs)
        self.backend.update(frame)
        return self.backend.render(frame)

    def warmup(self, frames: int = 4) -> None:

        for _ in range(frames):
            self.step_and_render(1)

    def stats(self) -> HarnessStats:
        s = self.backend.stats
        return HarnessStats(
            draw_calls=s.draw_calls,
            instances=s.instances,
            triangles=s.triangles,
            buckets=s.buckets,
            frame_ms=s.frame_cpu_ms,
            cpu_ms=dict(s.cpu_ms),
            gpu_ms=dict(s.gpu_ms),
        )

    def save_png(self, path: Path) -> None:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        img = self.backend.target.read_color(flip=True)
        Image.fromarray(img, "RGBA").convert("RGB").save(path)

    def release(self) -> None:
        with contextlib.suppress(Exception):
            self.backend.release()
        with contextlib.suppress(Exception):
            self.adapter.release()
        with contextlib.suppress(Exception):
            self._glfw.terminate()

    def __enter__(self) -> OffscreenHarness:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
