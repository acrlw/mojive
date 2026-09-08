"""Native 3D gizmo render pass."""

from __future__ import annotations

import moderngl
import numpy as np

from ....gizmo import (
    ACTIVE_HANDLE_COLOR,
    AXIS_COLORS,
    AXIS_HANDLES,
    CENTER_COLOR,
    CENTER_RADIUS,
    CENTER_SHELL_RADIUS,
    CONTRAST_EDGE_COLOR,
    CONTRAST_EDGE_PT,
    HOVER_COLOR,
    PLANE_ACTIVE_ALPHA,
    PLANE_ALPHA,
    PLANE_HANDLES,
    ROTATE_AXIS_HANDLES,
    SIZE_PT,
    TRACKBALL_RADIUS,
    GizmoHandle,
    GizmoMode,
    axis_hover_color,
    axis_rotation,
    display_handles,
    handle_projection_alpha,
    rotation_half_basis,
    rotation_handle_color,
    rotation_ring_is_full,
    screen_rotation_basis,
    trackball_color,
)
from ....log import get_logger
from ....types import MeshKey, MeshShape
from ...backend import RenderFlag
from ..backend import register_pass
from ..instances import GpuMesh
from ..programs import ProgramSpec
from ..targets import IdLayout
from .base import BasePass, PassContext, state_overlay

log = get_logger("gizmo")

_SPEC = ProgramSpec("gizmo", "gizmo.vert", "gizmo.frag")
_MESH_LAYOUT = "3f 3f 8x"


class GizmoPass(BasePass):
    name = "gizmo"

    def __init__(self) -> None:
        self._program: moderngl.Program | None = None
        self._generation = -1
        self._meshes: dict[str, GpuMesh] = {}
        self._vaos: dict[str, moderngl.VertexArray] = {}
        self._model = np.eye(4, dtype=np.float32)
        self._upload = np.eye(4, dtype=np.float32)
        self._view = np.eye(4, dtype=np.float32)
        self._view_proj = np.eye(4, dtype=np.float32)
        self._broken = ""

    def prepare(self, ctx: PassContext) -> bool:
        return ctx.gizmo is not None and not self._broken

    def execute(self, ctx: PassContext) -> None:
        frame = ctx.gizmo
        if frame is None or not self._sync(ctx):
            return

        origin4 = np.ones(4, np.float64)
        origin4[:3] = frame.position
        clip = ctx.proj @ (ctx.view @ origin4)
        if clip[3] <= 0.0:
            return
        scale = float(frame.size_px) * ctx.px_scale * float(clip[3])
        if scale <= 0.0:
            return

        target = ctx.target
        target.use_main()
        shared_id = target.id_layout is IdLayout.SHARED and target.id_fbo is target.fbo
        if shared_id:
            target.fbo.color_mask = ((True, True, True, True), (False, False, False, False))

        state_overlay(ctx.ctx, depth_test=True)
        ctx.ctx.multisample = bool(ctx.flag(RenderFlag.MSAA))
        ctx.ctx.wireframe = False
        target.fbo.depth_mask = True
        self._set_common(ctx)
        try:
            if frame.mode is GizmoMode.TRANSLATE:
                self._draw_translate(ctx, frame, scale)
            else:
                self._draw_rotate(ctx, frame, scale)
        finally:
            target.fbo.depth_mask = True
            if shared_id:
                target.fbo.color_mask = ((True, True, True, True), (True, True, True, True))

    def _sync(self, ctx: PassContext) -> bool:
        if self._program is not None and self._generation == ctx.programs.generation:
            return True
        try:
            self._program = ctx.programs.get(_SPEC)
            if not self._meshes:
                from ...mesh import builtin_mesh, gizmo_mesh

                data = {
                    "arrow": gizmo_mesh("arrow"),
                    "arrow_edge": gizmo_mesh("arrow_edge"),
                    "plane": gizmo_mesh("plane"),
                    "ring": gizmo_mesh("ring"),
                    "half_ring": gizmo_mesh("half_ring"),
                    "ring_edge": gizmo_mesh("ring_edge"),
                    "half_ring_edge": gizmo_mesh("half_ring_edge"),
                    "screen_ring": gizmo_mesh("screen_ring"),
                    "screen_ring_edge": gizmo_mesh("screen_ring_edge"),
                    "trackball": builtin_mesh(MeshKey(MeshShape.DISK)),
                    "center": builtin_mesh(MeshKey(MeshShape.SPHERE)),
                }
                self._meshes = {
                    name: GpuMesh(ctx.ctx, m.positions, m.normals, m.uvs, m.indices)
                    for name, m in data.items()
                }
            for vao in self._vaos.values():
                vao.release()
            self._vaos = {
                name: ctx.ctx.vertex_array(
                    self._program,
                    [(mesh.vbo, _MESH_LAYOUT, "in_position", "in_normal")],
                    mesh.ibo,
                    index_element_size=4,
                )
                for name, mesh in self._meshes.items()
            }
        except Exception as exc:
            self._broken = str(exc)
            log.error("Native gizmo initialization failed: {}", exc)
            return False
        self._generation = ctx.programs.generation
        return True

    def _set_common(self, ctx: PassContext) -> None:
        assert self._program is not None
        np.copyto(self._view, ctx.view.T)
        np.copyto(self._view_proj, ctx.view_proj.T)
        self._program["u_view"].write(self._view)
        self._program["u_view_proj"].write(self._view_proj)

    def _draw_translate(self, ctx: PassContext, frame, scale: float) -> None:
        visible = display_handles(frame)
        ctx.target.fbo.depth_mask = False
        for axis, handle in enumerate(PLANE_HANDLES):
            if handle in visible:
                alpha = handle_projection_alpha(
                    frame, handle, ctx.camera, frame.position, frame.rotation[:, axis]
                )
                if alpha <= 0.0:
                    continue
                opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
                self._draw(
                    ctx,
                    "plane",
                    frame.position,
                    axis_rotation(frame.rotation, axis),
                    scale,
                    self._color(frame, handle, axis, alpha=opacity),
                )
        ctx.target.fbo.depth_mask = True
        ctx.ctx.enable(moderngl.CULL_FACE)
        ctx.ctx.front_face = "ccw"
        ctx.ctx.cull_face = "back"
        try:
            for axis, handle in enumerate(AXIS_HANDLES):
                if handle in visible:
                    alpha = handle_projection_alpha(
                        frame, handle, ctx.camera, frame.position, frame.rotation[:, axis]
                    )
                    if alpha <= 0.0:
                        continue
                    if frame.outline_color is not None:
                        ctx.target.fbo.depth_mask = False
                        edge_color = np.asarray(frame.outline_color, np.float32).copy()
                        edge_color[3] *= alpha
                        self._draw(
                            ctx,
                            "arrow_edge",
                            frame.position,
                            axis_rotation(frame.rotation, axis),
                            scale,
                            edge_color,
                            mask_radius=CENTER_SHELL_RADIUS,
                        )
                        ctx.target.fbo.depth_mask = True
                    self._draw(
                        ctx,
                        "arrow",
                        frame.position,
                        axis_rotation(frame.rotation, axis),
                        scale,
                        self._color(frame, handle, axis, alpha),
                        mask_radius=CENTER_SHELL_RADIUS,
                    )
        finally:
            ctx.ctx.disable(moderngl.CULL_FACE)
        if GizmoHandle.SCREEN in visible:
            ctx.ctx.disable(moderngl.DEPTH_TEST)
            ctx.ctx.enable(moderngl.CULL_FACE)
            ctx.target.fbo.depth_mask = False
            try:
                self._draw(
                    ctx,
                    "center",
                    frame.position,
                    np.eye(3),
                    scale * (CENTER_RADIUS + CONTRAST_EDGE_PT / SIZE_PT),
                    CONTRAST_EDGE_COLOR,
                )
                self._draw(
                    ctx,
                    "center",
                    frame.position,
                    np.eye(3),
                    scale * CENTER_RADIUS,
                    HOVER_COLOR if self._hot(frame, GizmoHandle.SCREEN) else CENTER_COLOR,
                )
            finally:
                ctx.target.fbo.depth_mask = True
                ctx.ctx.disable(moderngl.CULL_FACE)
                ctx.ctx.enable(moderngl.DEPTH_TEST)

    def _draw_rotate(self, ctx: PassContext, frame, scale: float) -> None:
        visible = display_handles(frame)
        ctx.ctx.enable(moderngl.CULL_FACE)
        ctx.ctx.front_face = "ccw"
        ctx.ctx.cull_face = "back"
        try:
            if GizmoHandle.ROTATE_TRACKBALL in visible:
                ctx.target.fbo.depth_mask = False
                ctx.ctx.disable(moderngl.DEPTH_TEST)
                try:
                    self._draw(
                        ctx,
                        "trackball",
                        frame.position,
                        screen_rotation_basis(ctx.camera),
                        scale * TRACKBALL_RADIUS,
                        trackball_color(frame),
                    )
                finally:
                    ctx.ctx.enable(moderngl.DEPTH_TEST)
                    ctx.target.fbo.depth_mask = True
            for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
                if handle not in visible:
                    continue
                if frame.active_rotation_overlay and frame.active is handle:
                    continue
                full = rotation_ring_is_full(frame, handle)
                alpha = handle_projection_alpha(
                    frame, handle, ctx.camera, frame.position, frame.rotation[:, axis]
                )
                if alpha <= 0.0:
                    continue
                rotation = (
                    axis_rotation(frame.rotation, axis)
                    if full
                    else rotation_half_basis(ctx.camera, frame.position, frame.rotation, axis)
                )
                if frame.outline_color is not None:
                    ctx.target.fbo.depth_mask = False
                    edge_color = np.asarray(frame.outline_color, np.float32).copy()
                    edge_color[3] *= alpha
                    self._draw(
                        ctx,
                        "ring_edge" if full else "half_ring_edge",
                        frame.position,
                        rotation,
                        scale,
                        edge_color,
                    )
                    ctx.target.fbo.depth_mask = True
                self._draw(
                    ctx,
                    "ring" if full else "half_ring",
                    frame.position,
                    rotation,
                    scale,
                    rotation_handle_color(frame, handle, axis, alpha),
                )
            if GizmoHandle.ROTATE_SCREEN in visible and not (
                frame.active_rotation_overlay and frame.active is GizmoHandle.ROTATE_SCREEN
            ):
                ctx.ctx.disable(moderngl.DEPTH_TEST)
                try:
                    basis = screen_rotation_basis(ctx.camera)
                    self._draw(
                        ctx,
                        "screen_ring_edge",
                        frame.position,
                        basis,
                        scale,
                        CONTRAST_EDGE_COLOR,
                    )
                    self._draw(
                        ctx,
                        "screen_ring",
                        frame.position,
                        basis,
                        scale,
                        HOVER_COLOR
                        if self._hot(frame, GizmoHandle.ROTATE_SCREEN)
                        else CENTER_COLOR,
                    )
                finally:
                    ctx.ctx.enable(moderngl.DEPTH_TEST)
        finally:
            ctx.ctx.disable(moderngl.CULL_FACE)

    @staticmethod
    def _hot(frame, handle: GizmoHandle) -> bool:
        return frame.active is handle or frame.hovered is handle

    def _color(self, frame, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        base = AXIS_COLORS[axis] if frame.handle_color is None else frame.handle_color
        if frame.handle_color is not None and frame.active is handle:
            color = ACTIVE_HANDLE_COLOR.copy()
        elif frame.handle_color is not None and frame.hovered is handle:
            color = np.asarray(axis_hover_color(base), np.float32)
        else:
            color = HOVER_COLOR.copy() if self._hot(frame, handle) else np.asarray(base).copy()
        color[3] = alpha
        return color

    def _draw(
        self,
        ctx,
        mesh_name: str,
        position,
        rotation,
        scale: float,
        color,
        *,
        mask_radius: float = 0.0,
    ) -> None:
        assert self._program is not None
        self._model[:] = 0.0
        self._model[:3, :3] = np.asarray(rotation, np.float32) * float(scale)
        self._model[:3, 3] = position
        self._model[3, 3] = 1.0
        np.copyto(self._upload, self._model.T)
        self._program["u_model"].write(self._upload)
        self._program["u_color"].value = tuple(float(x) for x in color)
        self._program["u_mask_radius"].value = float(mask_radius)
        mesh = self._meshes[mesh_name]
        self._vaos[mesh_name].render(moderngl.TRIANGLES, vertices=mesh.index_count)
        ctx.draw_calls += 1

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        for mesh in self._meshes.values():
            mesh.release()
        self._vaos.clear()
        self._meshes.clear()
        self._program = None


register_pass("gizmo", GizmoPass)
