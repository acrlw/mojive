"""Planar reflection render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from .... import math3d
from ....log import get_logger
from ....types import MeshShape
from ...backend import RenderFlag
from ...dependencies import camera_key, flags_key, lifecycle_key, lights_key
from ..registry import register_pass
from .base import PassContext, state_opaque, state_transparent
from .opaque import OpaquePass, draw_buckets

log = get_logger("reflect")

GL_CLIP_DISTANCE0 = 0x3000
MAX_REFLECTION_PLANES = 4


class _PlaneGroup:
    def __init__(self, plane, indices, buckets) -> None:
        self.plane = plane
        self.indices = indices
        self.buckets = buckets


class ReflectPass(OpaquePass):
    name = "reflect"
    linear_out = True

    def __init__(self) -> None:
        super().__init__()
        self.colors: list[moderngl.Texture] = []
        self.depth: moderngl.Texture | None = None
        self.fbos: list[moderngl.Framebuffer] = []
        self._size: tuple[int, int] = (0, 0)
        self._mirror = np.eye(4, dtype=np.float32)
        self._view = np.eye(4, dtype=np.float32)
        self._view_proj = np.eye(4, dtype=np.float32)
        self._eye = np.zeros(3, np.float32)
        self._plane = (0.0, 0.0, 1.0, 0.0)
        self._groups: tuple[_PlaneGroup, ...] = ()
        self.reflection_info = np.zeros(0, np.uint32)
        self.cache_status = "off"
        self._cache_scene = None
        self._cache_key: tuple | None = None
        self._pending_key: tuple | None = None

    def view_matrices(self, ctx: PassContext):
        return self._view, self._view_proj, self._eye

    def clip_plane(self, ctx: PassContext) -> tuple[float, float, float, float]:
        return self._plane

    def _ensure_target(self, ctx: PassContext, count: int) -> bool:
        w, h = int(ctx.target.width), int(ctx.target.height)
        if w <= 0 or h <= 0:
            return False
        if self._size == (w, h) and len(self.fbos) == count:
            return True
        self.release()
        gl = ctx.ctx

        self.depth = gl.depth_texture((w, h))
        for _ in range(count):
            color = gl.texture((w, h), 3, dtype="f2")
            color.filter = (moderngl.LINEAR, moderngl.LINEAR)
            color.repeat_x = color.repeat_y = False
            self.colors.append(color)
            self.fbos.append(gl.framebuffer([color], self.depth))
        self._size = (w, h)
        return True

    @staticmethod
    def find_plane(scene) -> tuple[int, tuple[float, float, float, float]] | None:
        if scene.count == 0 or len(scene.material) == 0 or not scene.bucket_keys:
            return None
        refl = np.asarray(scene.material[:, 3])
        planar = np.fromiter(
            (
                0 <= int(bucket) < len(scene.bucket_keys)
                and scene.bucket_keys[int(bucket)][0].shape in (MeshShape.PLANE, MeshShape.BOX)
                for bucket in scene.bucket
            ),
            dtype=bool,
            count=scene.count,
        )
        candidates = np.flatnonzero(planar & (refl > 0.0))
        if not len(candidates):
            return None
        idx = int(candidates[np.argmax(refl[candidates])])
        plane = ReflectPass._plane_equation(scene, idx)
        return (idx, plane) if plane is not None else None

    @classmethod
    def find_planes(cls, scene) -> tuple[_PlaneGroup, ...]:
        if scene.count == 0 or len(scene.material) == 0 or not scene.bucket_keys:
            return ()
        reflectance = np.asarray(scene.material[:, 3])
        candidates = [
            int(index)
            for index in np.argsort(-reflectance)
            if reflectance[index] > 0.0
            and 0 <= int(scene.bucket[index]) < len(scene.bucket_keys)
            and scene.bucket_keys[int(scene.bucket[index])][0].shape
            in (MeshShape.PLANE, MeshShape.BOX)
        ]
        groups: list[_PlaneGroup] = []
        for index in candidates:
            plane = cls._plane_equation(scene, index)
            if plane is None:
                continue
            group = next(
                (
                    item
                    for item in groups
                    if np.allclose(item.plane, plane, atol=1e-5)
                    or np.allclose(item.plane, -np.asarray(plane), atol=1e-5)
                ),
                None,
            )
            if group is not None:
                group.indices.append(index)
                group.buckets.add(int(scene.bucket[index]))
            elif len(groups) < MAX_REFLECTION_PLANES:
                groups.append(_PlaneGroup(plane, [index], {int(scene.bucket[index])}))
        return tuple(groups)

    @staticmethod
    def _plane_equation(scene, index: int) -> tuple[float, float, float, float] | None:
        transform = np.asarray(scene.transforms[index], np.float64)
        try:
            normal = np.linalg.inv(transform[:3, :3]).T @ np.array([0.0, 0.0, 1.0])
        except np.linalg.LinAlgError:
            return None
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            return None
        normal /= length
        shape = scene.bucket_keys[int(scene.bucket[index])][0].shape
        point = transform[:3, 3]
        if shape is MeshShape.BOX:
            point = point + transform[:3, 2]
        d = -float(np.dot(normal, point))
        return (float(normal[0]), float(normal[1]), float(normal[2]), d)

    def _build_reflection_info(self, scene, groups: tuple[_PlaneGroup, ...]) -> None:
        if len(self.reflection_info) != scene.count:
            self.reflection_info = np.zeros(scene.count, np.uint32)
        else:
            self.reflection_info.fill(0)
        for layer, group in enumerate(groups):
            for index in group.indices:
                shape = scene.bucket_keys[int(scene.bucket[index])][0].shape
                self.reflection_info[index] = np.uint32(
                    (layer + 1) | (8 if shape is MeshShape.BOX else 0)
                )

    def _set_plane(self, ctx: PassContext, plane) -> None:
        eye = np.asarray(ctx.camera.eye, np.float64)
        normal = np.array(plane[:3], np.float64)
        point = -normal * plane[3]
        self._mirror = math3d.mirror(point, normal)
        self._view = (np.asarray(ctx.view, np.float64) @ self._mirror).astype(np.float32)
        self._view_proj = (np.asarray(ctx.proj, np.float64) @ self._view).astype(np.float32)
        self._eye = (self._mirror @ np.append(eye, 1.0))[:3].astype(np.float32)
        self._plane = plane

    def prepare_instances(self, ctx: PassContext) -> None:
        """Resolve planar-reflection instance metadata before GPU upload."""

        ctx.reflection = ()
        self._groups = ()
        self.cache_status = "off"
        self._pending_key = None
        if not ctx.flag(RenderFlag.REFLECTION):
            self._build_reflection_info(ctx.scene, ())
            return
        eye = np.asarray(ctx.camera.eye, np.float64)
        groups = tuple(
            group
            for group in self.find_planes(ctx.scene)
            if float(np.dot(group.plane[:3], eye) + group.plane[3]) > 1e-4
        )
        if not groups:
            self._build_reflection_info(ctx.scene, ())
            return
        if not self._ensure_target(ctx, len(groups)):
            self._build_reflection_info(ctx.scene, ())
            return
        self._groups = groups
        self._build_reflection_info(ctx.scene, groups)

    def prepare(self, ctx: PassContext) -> bool:
        if not self._groups:
            return False
        key = self._dependency_key(ctx)
        if self._cache_scene is ctx.scene and self._cache_key == key and self.colors:
            ctx.reflection = tuple(self.colors)
            self.cache_status = "reused"
            return False

        self._cache_scene = None
        self._cache_key = None
        self._pending_key = key
        self._set_plane(ctx, self._groups[0].plane)
        if not super().prepare(ctx):
            self._pending_key = None
            return False

        ctx.scene_program = None
        return True

    def _dependency_key(self, ctx: PassContext) -> tuple:
        scene = ctx.scene
        return (
            lifecycle_key(scene, ctx.frame_serial, visual=True, identity=True),
            int(ctx.mesh_revision),
            int(ctx.texture_revision),
            camera_key(ctx.camera),
            lights_key(scene.lights),
            flags_key(ctx.flags),
            ctx.debug_view.value,
            int(ctx.selected_id),
            scene.shading_model.value,
            self._size,
            int(ctx.programs.generation),
        )

    def execute(self, ctx: PassContext) -> None:
        if not self.fbos or self.program is None:
            return
        gl = ctx.ctx
        excluded = set().union(*(group.buckets for group in self._groups))
        use_shadow = bool(self._spec_used and self._spec_used.defines.get("USE_SHADOW"))
        for fbo, group in zip(self.fbos, self._groups, strict=True):
            self._set_plane(ctx, group.plane)
            self._frame_uniforms(ctx, self.program, False, use_shadow)
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            state_opaque(gl)
            gl.front_face = "cw"
            gl.enable_direct(GL_CLIP_DISTANCE0)
            try:
                draw_buckets(ctx, tuple(b for b in ctx.scene.opaque_buckets if b not in excluded))
                if ctx.scene.transparent_buckets and ctx.flag(RenderFlag.TRANSPARENT):
                    state_transparent(gl, additive=ctx.flag(RenderFlag.ADDITIVE, False))
                    gl.front_face = "cw"
                    gl.enable_direct(GL_CLIP_DISTANCE0)
                    fbo.depth_mask = False
                    draw_buckets(
                        ctx,
                        tuple(
                            b
                            for b in ctx.scene.transparent_draw_order(self._eye)
                            if b not in excluded
                        ),
                    )
            finally:
                fbo.depth_mask = True
                gl.disable_direct(GL_CLIP_DISTANCE0)
                gl.front_face = "ccw"
        ctx.reflection = tuple(self.colors)
        self._cache_scene = ctx.scene
        self._cache_key = self._pending_key
        self._pending_key = None
        self.cache_status = "rendered"

    def release(self) -> None:
        super().release()
        for obj in (*self.fbos, *self.colors, self.depth):
            try:
                if obj is not None:
                    obj.release()
            except Exception as e:
                log.debug("Failed to release reflection targets: {}", e)
        self.fbos.clear()
        self.colors.clear()
        self.depth = None
        self._size = (0, 0)
        self._groups = ()
        self.reflection_info = np.zeros(0, np.uint32)
        self._cache_scene = None
        self._cache_key = None
        self._pending_key = None
        self.cache_status = "off"


register_pass("reflect", ReflectPass)
