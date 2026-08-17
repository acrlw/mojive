from __future__ import annotations

import moderngl
import numpy as np

from .... import math3d
from ....log import get_logger
from ....types import MeshShape
from ...backend import RenderFlag
from ..registry import register_pass
from .base import PassContext, state_opaque, state_transparent
from .opaque import OpaquePass, draw_buckets

log = get_logger("reflect")

GL_CLIP_DISTANCE0 = 0x3000


class ReflectPass(OpaquePass):
    name = "reflect"
    linear_out = True

    def __init__(self) -> None:
        super().__init__()
        self.color: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self._size: tuple[int, int] = (0, 0)
        self._mirror = np.eye(4, dtype=np.float32)
        self._view = np.eye(4, dtype=np.float32)
        self._view_proj = np.eye(4, dtype=np.float32)
        self._eye = np.zeros(3, np.float32)
        self._plane = (0.0, 0.0, 1.0, 0.0)
        self._skip_bucket = -1

    def view_matrices(self, ctx: PassContext):
        return self._view, self._view_proj, self._eye

    def clip_plane(self, ctx: PassContext) -> tuple[float, float, float, float]:
        return self._plane

    def _ensure_target(self, ctx: PassContext) -> bool:
        w, h = int(ctx.target.width), int(ctx.target.height)
        if w <= 0 or h <= 0:
            return False
        if self._size == (w, h) and self.fbo is not None:
            return True
        self.release()
        gl = ctx.ctx

        self.color = gl.texture((w, h), 3, dtype="f2")
        self.color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.color.repeat_x = self.color.repeat_y = False
        self.depth = gl.depth_texture((w, h))
        self.fbo = gl.framebuffer([self.color], self.depth)
        self._size = (w, h)
        return True

    @staticmethod
    def find_plane(scene) -> tuple[int, tuple[float, float, float, float]] | None:

        if scene.count == 0 or len(scene.material) == 0 or not scene.bucket_keys:
            return None
        refl = np.asarray(scene.material[:, 3])
        planes = np.fromiter(
            (
                0 <= int(bucket) < len(scene.bucket_keys)
                and scene.bucket_keys[int(bucket)][0].shape is MeshShape.PLANE
                for bucket in scene.bucket
            ),
            dtype=bool,
            count=scene.count,
        )
        candidates = np.flatnonzero(planes & (refl > 0.0))
        if not len(candidates):
            return None
        idx = int(candidates[np.argmax(refl[candidates])])
        m = np.asarray(scene.transforms[idx], np.float64)
        basis = m[:3, :3]
        try:
            normal = np.linalg.inv(basis).T @ np.array([0.0, 0.0, 1.0])
        except np.linalg.LinAlgError:
            return None
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            return None
        normal = normal / length
        point = m[:3, 3]
        d = -float(np.dot(normal, point))
        return idx, (float(normal[0]), float(normal[1]), float(normal[2]), d)

    def prepare(self, ctx: PassContext) -> bool:
        ctx.reflection = None
        if not ctx.flag(RenderFlag.REFLECTION):
            return False
        found = self.find_plane(ctx.scene)
        if found is None:
            return False
        index, plane = found
        if not self._ensure_target(ctx):
            return False

        eye = np.asarray(ctx.camera.eye, np.float64)
        if float(np.dot(plane[:3], eye) + plane[3]) <= 1e-4:
            return False

        n = np.array(plane[:3], np.float64)
        point = -np.array(plane[:3], np.float64) * plane[3]
        self._mirror = math3d.mirror(point, n)

        self._view = (np.asarray(ctx.view, np.float64) @ self._mirror).astype(np.float32)
        self._view_proj = (np.asarray(ctx.proj, np.float64) @ self._view).astype(np.float32)
        self._eye = (self._mirror @ np.append(eye, 1.0))[:3].astype(np.float32)
        self._plane = plane

        self._skip_bucket = int(ctx.scene.bucket[index]) if index < len(ctx.scene.bucket) else -1

        if not super().prepare(ctx):
            return False

        ctx.scene_program = None
        return True

    def execute(self, ctx: PassContext) -> None:
        if self.fbo is None:
            return
        gl = ctx.ctx
        self.fbo.use()

        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

        state_opaque(gl)

        gl.front_face = "cw"
        gl.enable_direct(GL_CLIP_DISTANCE0)
        try:
            buckets = tuple(b for b in ctx.scene.opaque_buckets if b != self._skip_bucket)
            draw_buckets(ctx, buckets)
            if ctx.scene.transparent_buckets and ctx.flag(RenderFlag.TRANSPARENT):
                state_transparent(gl, additive=ctx.flag(RenderFlag.ADDITIVE, False))
                gl.front_face = "cw"
                gl.enable_direct(GL_CLIP_DISTANCE0)
                self.fbo.depth_mask = False
                transparent = tuple(
                    b for b in ctx.scene.transparent_draw_order(self._eye) if b != self._skip_bucket
                )
                draw_buckets(ctx, transparent)
        finally:
            self.fbo.depth_mask = True
            gl.disable_direct(GL_CLIP_DISTANCE0)
            gl.front_face = "ccw"
        ctx.reflection = self.color

    def release(self) -> None:
        super().release()
        for obj in (self.fbo, self.color, self.depth):
            try:
                if obj is not None:
                    obj.release()
            except Exception as e:
                log.debug("Failed to release reflection targets: {}", e)
        self.fbo = self.color = self.depth = None
        self._size = (0, 0)


register_pass("reflect", ReflectPass)
