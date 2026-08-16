from __future__ import annotations

import moderngl
import numpy as np

from ....log import get_logger
from ...backend import DebugView, RenderFlag
from .. import color
from ..programs import ProgramSpec, UniformCache
from ..registry import register_pass
from ..targets import IdLayout
from .base import BasePass, PassContext

log = get_logger("skybox")

_MASK_ON = (True, True, True, True)
_MASK_OFF = (False, False, False, False)


class SkyboxPass(BasePass):
    name = "skybox"

    def __init__(self) -> None:
        self._program: moderngl.Program | None = None
        self._vao: moderngl.VertexArray | None = None
        self._uniforms: UniformCache | None = None
        self._generation = -1
        self._texture: moderngl.TextureCube | None = None
        self._inv = np.zeros((4, 4), np.float32)

    # ------------------------------------------------------------------
    def prepare(self, ctx: PassContext) -> bool:

        self._texture = ctx.textures.skybox
        if self._texture is None or not ctx.flag(RenderFlag.SKYBOX):
            return False
        if ctx.debug_view is not DebugView.SHADED:
            return False

        try:
            prog = ctx.programs.get(ProgramSpec("skybox", "skybox.vert", "skybox.frag"))
        except Exception as e:
            log.error("Skybox shader compilation failed; the sky is skipped this frame: {}", e)
            return False

        if self._vao is None or self._program is not prog:
            if self._vao is not None:
                self._vao.release()
            self._vao = ctx.ctx.vertex_array(prog, [])
            self._program = prog
        if self._uniforms is None:
            self._uniforms = UniformCache(prog, ctx.programs.generation)
        else:
            self._uniforms.rebind(prog, ctx.programs.generation)

        np.copyto(self._inv, np.linalg.inv(ctx.view_proj.astype(np.float64)).T)
        prog["u_inv_view_proj"].write(self._inv)
        self._uniforms.set("u_skybox", 0)
        self._uniforms.set("u_exposure", color.EXPOSURE)
        self._uniforms.set("u_tonemap", 1 if ctx.flag(RenderFlag.TONEMAP) else 0)
        self._generation = ctx.programs.generation
        return True

    # ------------------------------------------------------------------
    def execute(self, ctx: PassContext) -> None:
        target, gl = ctx.target, ctx.ctx
        assert self._vao is not None and self._texture is not None

        target.use_main()
        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_OFF)

        gl.enable_only(moderngl.DEPTH_TEST)
        gl.depth_func = "<="
        gl.wireframe = False
        gl.multisample = bool(ctx.flag(RenderFlag.MSAA))

        target.fbo.depth_mask = False

        self._texture.use(0)
        self._vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.draw_calls += 1

        target.fbo.depth_mask = True
        gl.depth_func = "<"
        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_ON)

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        self._program = None
        self._uniforms = None


register_pass("skybox", SkyboxPass)
