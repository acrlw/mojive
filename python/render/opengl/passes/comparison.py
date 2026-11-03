"""Composite the nearest visual and collision surfaces without internal overdraw."""

from __future__ import annotations

import moderngl

from ...backend import RenderFlag
from .. import gl_native
from ..programs import ProgramSpec
from .base import PassContext
from .opaque import draw_buckets


class ComparisonLayer:
    def __init__(self) -> None:
        self._key = None
        self._resources = []
        self._program = None
        self._vao = None

    def draw(
        self,
        ctx: PassContext,
        buckets: tuple[int, ...],
        destination: moderngl.Framebuffer,
        program: moderngl.Program,
        *,
        reflected: bool = False,
        occluders: tuple[int, ...] | None = None,
    ) -> None:
        if not buckets:
            return
        gl = ctx.ctx
        key = (destination.size, destination.depth_attachment.samples, reflected)
        if key != self._key:
            self.release()
            try:
                size, samples, _ = key
                depth = gl.depth_texture(size, samples=samples)
                self._resources.append(depth)
                self._frames, self._images, self._resolved = [], [], []
                for _ in range(2):
                    # Reflection radiance can exceed one; displayed layers are premultiplied RGBA8.
                    dtype = "f2" if reflected else "f1"
                    color = gl.texture(size, 4, samples=samples, dtype=dtype)
                    self._resources.append(color)
                    frame = gl.framebuffer([color], depth)
                    self._resources.append(frame)
                    image, resolved = color, None
                    if samples:
                        image = gl.texture(size, 4, dtype=dtype)
                        self._resources.append(image)
                        resolved = gl.framebuffer([image])
                        self._resources.append(resolved)
                    image.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    self._frames.append(frame)
                    self._images.append(image)
                    self._resolved.append(resolved)
                self._key = key
            except Exception:
                self.release()
                raise
        composite = ctx.programs.get(
            ProgramSpec(name="comparison", vertex="fullscreen.vert", fragment="comparison.frag")
        )
        if composite is not self._program:
            if self._vao is not None:
                self._vao.release()
            self._program = composite
            self._vao = gl.vertex_array(composite, [])
        scene = ctx.scene
        if occluders is None:
            occluders = tuple(b for b in scene.opaque_buckets if b not in scene.comparison_buckets)
        gl.multisample = bool(ctx.flag(RenderFlag.MSAA))
        gl.front_face = "cw" if reflected else "ccw"
        gl.enable(moderngl.DEPTH_TEST)
        gl.disable(moderngl.BLEND)
        gl.depth_func = "<"
        native = gl_native.native()
        try:
            for role, (frame, resolved) in enumerate(
                zip(self._frames, self._resolved, strict=True)
            ):
                frame.use()
                frame.depth_mask = True
                frame.color_mask = (True, True, True, True)
                native.clear_color_float(0, (0.0, 0.0, 0.0, 0.0))
                native.clear_depth_only(1.0)
                # Opaque scene geometry occludes both layers. Reuse the exact color vertex
                # transform and skip lighting in this depth-only draw.
                frame.color_mask = (False, False, False, False)
                if ctx.flag(RenderFlag.CULL_FACE):
                    gl.enable(moderngl.CULL_FACE)
                else:
                    gl.disable(moderngl.CULL_FACE)
                program["u_comparison"].value = (2.0, 0.0, 0.0, 0.0)
                draw_buckets(ctx, occluders)
                frame.color_mask = (True, True, True, True)
                program["u_comparison"].value = (1.0, 0.0, 0.0, 0.0)
                role_buckets = tuple(
                    b
                    for b in buckets
                    if (scene.colors[scene.bucket_ranges[b][0], 3] < -2.5) == bool(role)
                )
                # Depth writes replace farther fragments, including overlapping internal
                # parts. Each role contributes exactly one surface per sample.
                draw_buckets(ctx, role_buckets)
                if resolved is not None:
                    gl.copy_framebuffer(resolved, frame)
        finally:
            program["u_comparison"].value = (0.0, 0.0, 0.0, 0.0)
        destination.use()
        masks = destination.color_mask
        if len(destination.color_attachments) > 1:
            destination.color_mask = ((True, True, True, True), (False, False, False, False))
        destination.depth_mask = False
        gl.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        if reflected:
            gl.disable_direct(0x3000)
        gl.enable(moderngl.BLEND)
        gl.blend_equation = moderngl.FUNC_ADD
        gl.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ZERO,
            moderngl.ONE,
        )
        for unit, name in enumerate(("u_visual", "u_collision")):
            self._images[unit].use(unit)
            composite[name].value = unit
        self._vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.draw_calls += 1
        destination.depth_mask = True
        destination.color_mask = masks
        if reflected:
            gl.enable_direct(0x3000)
        gl.front_face = "ccw"
        gl.depth_func = "<"

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
        self._vao = self._program = None
        for resource in reversed(self._resources):
            resource.release()
        self._resources.clear()
        self._key = None
