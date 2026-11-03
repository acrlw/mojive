"""Sorted transparent geometry render pass."""

from __future__ import annotations

import moderngl

from ...backend import DebugView, RenderFlag
from ..registry import register_pass
from ..targets import IdLayout
from .base import BasePass, PassContext, state_overdraw, state_transparent
from .comparison import ComparisonLayer
from .opaque import draw_buckets

_MASK_ON = (True, True, True, True)
_MASK_OFF = (False, False, False, False)


class TransparentPass(BasePass):
    name = "transparent"

    def __init__(self) -> None:
        self._order: tuple[int, ...] = ()
        self._comparison = ComparisonLayer()

    def prepare(self, ctx: PassContext) -> bool:
        if ctx.scene_program is None:
            return False
        self._order = ctx.scene.transparent_draw_order() if ctx.flag(RenderFlag.TRANSPARENT) else ()
        return bool(self._order or ctx.scene.comparison_buckets)

    def execute(self, ctx: PassContext) -> None:
        target, gl = ctx.target, ctx.ctx
        target.use_main()
        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_OFF)

        if ctx.debug_view is DebugView.OVERDRAW and not ctx.flag(RenderFlag.WIREFRAME, False):
            state_overdraw(gl)
        else:
            state_transparent(gl, additive=ctx.flag(RenderFlag.ADDITIVE, False))
        if not ctx.flag(RenderFlag.CULL_FACE):
            gl.disable(moderngl.CULL_FACE)
        gl.multisample = bool(ctx.flag(RenderFlag.MSAA))

        target.fbo.depth_mask = False
        try:
            draw_buckets(ctx, self._order)
        finally:
            target.fbo.depth_mask = True
            if target.id_layout is IdLayout.SHARED:
                target.fbo.color_mask = (_MASK_ON, _MASK_ON)
        if ctx.debug_view is not DebugView.OVERDRAW or ctx.flag(RenderFlag.WIREFRAME):
            self._comparison.draw(ctx, ctx.scene.comparison_buckets, target.fbo, ctx.scene_program)

    def release(self) -> None:
        self._order = ()
        self._comparison.release()


register_pass("transparent", TransparentPass)
