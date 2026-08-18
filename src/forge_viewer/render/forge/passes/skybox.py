"""Environment skybox render pass."""

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
        self._haze_program: moderngl.Program | None = None
        self._haze_buffer: moderngl.Buffer | None = None
        self._haze_vao: moderngl.VertexArray | None = None
        self._haze_uniforms: UniformCache | None = None
        self._generation = -1
        self._texture: moderngl.TextureCube | None = None
        self._inv = np.zeros((4, 4), np.float32)
        self._haze_vertex_count = 0
        self._haze_enabled = False

    def prepare(self, ctx: PassContext) -> bool:
        self._texture = ctx.textures.skybox
        if self._texture is None or not ctx.flag(RenderFlag.SKYBOX):
            return False
        if ctx.debug_view is not DebugView.SHADED:
            return False

        try:
            prog = ctx.programs.get(ProgramSpec("skybox", "skybox.vert", "skybox.frag"))
            haze_prog = ctx.programs.get(ProgramSpec("horizon_haze", "haze.vert", "haze.frag"))
        except Exception as e:
            log.error("Skybox shader compilation failed; the sky is skipped this frame: {}", e)
            return False

        if self._vao is None or self._program is not prog:
            if self._vao is not None:
                self._vao.release()
            self._vao = ctx.ctx.vertex_array(prog, [])
            self._program = prog
        if self._haze_buffer is None:
            vertices = self._haze_vertices()
            self._haze_buffer = ctx.ctx.buffer(vertices.tobytes())
            self._haze_vertex_count = len(vertices)
        if self._haze_vao is None or self._haze_program is not haze_prog:
            if self._haze_vao is not None:
                self._haze_vao.release()
            self._haze_vao = ctx.ctx.vertex_array(
                haze_prog,
                [(self._haze_buffer, "3f", "in_haze")],
            )
            self._haze_program = haze_prog
        if self._uniforms is None:
            self._uniforms = UniformCache(prog, ctx.programs.generation)
        else:
            self._uniforms.rebind(prog, ctx.programs.generation)
        if self._haze_uniforms is None:
            self._haze_uniforms = UniformCache(haze_prog, ctx.programs.generation)
        else:
            self._haze_uniforms.rebind(haze_prog, ctx.programs.generation)

        np.copyto(self._inv, np.linalg.inv(ctx.view_proj.astype(np.float64)).T)
        prog["u_inv_view_proj"].write(self._inv)
        self._uniforms.set("u_skybox", 0)
        self._uniforms.set("u_exposure", color.EXPOSURE)
        self._uniforms.set("u_tonemap", 1 if ctx.flag(RenderFlag.TONEMAP) else 0)
        self._prepare_horizon_haze(ctx)
        self._generation = ctx.programs.generation
        return True

    @staticmethod
    def _haze_vertices(slices: int = 64) -> np.ndarray:
        vertices: list[tuple[float, float, float]] = []
        for layer0, layer1 in ((0.0, 1.0), (1.0, 2.0)):
            for index in range(slices):
                angle0 = 2.0 * np.pi * index / slices
                angle1 = 2.0 * np.pi * (index + 1) / slices
                p00 = (np.cos(angle0), np.sin(angle0), layer0)
                p01 = (np.cos(angle1), np.sin(angle1), layer0)
                p11 = (np.cos(angle1), np.sin(angle1), layer1)
                p10 = (np.cos(angle0), np.sin(angle0), layer1)
                vertices.extend((p00, p01, p11, p00, p11, p10))
        return np.asarray(vertices, np.float32)

    def _prepare_horizon_haze(self, ctx: PassContext) -> None:
        assert self._haze_uniforms is not None and self._haze_program is not None
        lights = ctx.scene.lights
        self._haze_enabled = (
            ctx.flag(RenderFlag.HAZE, False)
            and lights.horizon_haze
            and bool(ctx.scene.infinite_planes)
            and lights.haze_density > 0.0
        )
        if not self._haze_enabled:
            return

        transform = ctx.scene.transforms[ctx.scene.infinite_planes[0]]
        normal = np.asarray(transform[:3, 2], np.float64)
        normal /= np.linalg.norm(normal)
        elevation = float(np.dot(ctx.camera.eye - transform[:3, 3], normal))
        if elevation < 0.0:
            self._haze_enabled = False
            return

        radius = float(lights.haze_density)
        alpha = np.arctan2(1.0, radius)
        beta = 0.75 * np.pi - alpha
        transition = float(np.sqrt(0.5) * radius * np.sin(alpha) / np.sin(beta))
        distance = float(ctx.camera.far * 0.70)

        self._haze_program["u_view_proj"].write(
            np.ascontiguousarray(ctx.view_proj.T, dtype=np.float32)
        )
        self._haze_uniforms.set("u_eye", tuple(float(value) for value in ctx.camera.eye))
        self._haze_uniforms.set(
            "u_basis_x",
            tuple(float(value) for value in transform[:3, 0] / np.linalg.norm(transform[:3, 0])),
        )
        self._haze_uniforms.set(
            "u_basis_y",
            tuple(float(value) for value in transform[:3, 1] / np.linalg.norm(transform[:3, 1])),
        )
        self._haze_uniforms.set("u_normal", tuple(float(value) for value in normal))
        self._haze_uniforms.set("u_geometry", (distance, elevation, radius, transition))
        self._haze_uniforms.set("u_color", tuple(float(value) for value in lights.haze_color))
        self._haze_uniforms.set("u_exposure", color.EXPOSURE)
        self._haze_uniforms.set("u_tonemap", 1 if ctx.flag(RenderFlag.TONEMAP) else 0)

    def _render_horizon_haze(self, ctx: PassContext) -> None:
        if not self._haze_enabled or self._haze_vao is None or self._haze_vertex_count == 0:
            return
        gl = ctx.ctx
        gl.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
        gl.depth_func = "<"
        gl.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.target.fbo.depth_mask = False
        self._haze_vao.render(moderngl.TRIANGLES, vertices=self._haze_vertex_count)
        ctx.target.fbo.depth_mask = True
        ctx.draw_calls += 1

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

        self._render_horizon_haze(ctx)

        target.fbo.depth_mask = True
        gl.depth_func = "<"
        if target.id_layout is IdLayout.SHARED:
            target.fbo.color_mask = (_MASK_ON, _MASK_ON)

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._haze_vao is not None:
            self._haze_vao.release()
            self._haze_vao = None
        if self._haze_buffer is not None:
            self._haze_buffer.release()
            self._haze_buffer = None
        self._program = None
        self._uniforms = None
        self._haze_program = None
        self._haze_uniforms = None


register_pass("skybox", SkyboxPass)
