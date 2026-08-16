from __future__ import annotations

import moderngl

from ....types import ViewportImage
from ...backend import DebugView
from ..programs import ProgramSpec
from .base import BasePass, PassContext

_MODE = {DebugView.SEGMENT: 1, DebugView.IDCOLOR: 2}


class PresentPass(BasePass):
    name = "present"

    def __init__(self) -> None:
        self._vao: moderngl.VertexArray | None = None
        self._generation = -1
        self.image: ViewportImage | None = None

    def _program(self, ctx: PassContext):
        spec = ProgramSpec(
            name="present",
            vertex="fullscreen.vert",
            fragment="present.frag",
            defines={"ID_MULTISAMPLE": 1} if ctx.target.id_multisample else {},
        )
        prog = ctx.programs.get(spec)

        if self._vao is None or self._generation != ctx.programs.generation:
            if self._vao is not None:
                self._vao.release()
            self._vao = ctx.ctx.vertex_array(prog, [])
            self._generation = ctx.programs.generation
        return prog

    def prepare(self, ctx: PassContext) -> bool:
        self.image = None
        return True

    def execute(self, ctx: PassContext) -> None:
        target = ctx.target
        mode = _MODE.get(ctx.debug_view, 0)

        if mode == 0:
            target.resolve()
        else:
            prog = self._program(ctx)
            target.resolve()
            target.resolve_fbo.use()
            ctx.ctx.viewport = (0, 0, target.width, target.height)
            ctx.ctx.enable_only(moderngl.NOTHING)
            target.resolve_tex.use(0)
            target.id_tex.use(1)
            prog["u_color"].value = 0
            prog["u_ids"].value = 1
            prog["u_size"].value = (target.width, target.height)
            prog["u_mode"].value = mode
            prog["u_selected"].value = int(ctx.selected_id)
            assert self._vao is not None
            self._vao.render(moderngl.TRIANGLES, vertices=3)

        self.image = ViewportImage(
            texture_id=target.resolve_tex.glo,
            width=target.width,
            height=target.height,
            flip_y=True,
        )

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
