"""Render-pass contracts, shared context, and pipeline states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import moderngl
import numpy as np

from ...backend import DebugView, RenderFlag
from ...lighting import (
    LOCAL_SHADOW_SLOTS as LOCAL_SHADOW_SLOTS,
)
from ...lighting import (
    MAX_SCENE_LIGHTS as MAX_SCENE_LIGHTS,
)
from ...lighting import (
    LightSchedule as LightSchedule,
)
from ...lighting import (
    schedule_lights as schedule_lights,
)
from ...scene import RenderScene

if TYPE_CHECKING:
    from mojive.interaction.gizmo import GizmoFrame

    from ....types import CameraView
    from ...debugdraw import DebugDraw
    from ..instances import GpuMesh, InstanceStore
    from ..programs import ProgramCache
    from ..resources import TextureStore
    from ..targets import RenderTarget
    from ..timing import FrameTiming


@dataclass
class ShadowResult:
    atlas: moderngl.Texture | None = None

    matrices: np.ndarray = field(default_factory=lambda: np.zeros((3, 4, 4), np.float32))

    splits: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    texel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    tile_uv: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), np.float32))

    cascade_count: int = 0
    quality: int = 1
    enabled: bool = False

    local_count: int = 0
    local_light_indices: np.ndarray = field(
        default_factory=lambda: np.full(LOCAL_SHADOW_SLOTS, -1, np.int32)
    )
    local_kinds: np.ndarray = field(default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.int32))
    local_matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((LOCAL_SHADOW_SLOTS, 4, 4), np.float32)
    )
    local_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((LOCAL_SHADOW_SLOTS, 4), np.float32)
    )

    local_texel: np.ndarray = field(
        default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.float32)
    )

    local_radius: np.ndarray = field(
        default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.float32)
    )

    local_layers: np.ndarray = field(default_factory=lambda: np.zeros(LOCAL_SHADOW_SLOTS, np.int32))

    local_tex: Any = None


@dataclass
class PassContext:
    ctx: moderngl.Context
    target: RenderTarget
    scene: RenderScene
    camera: CameraView
    view: np.ndarray
    proj: np.ndarray
    view_proj: np.ndarray
    instances: InstanceStore
    programs: ProgramCache
    textures: TextureStore
    meshes: list[GpuMesh | None]
    timing: FrameTiming
    flags: dict[RenderFlag, bool]
    debug_view: DebugView = DebugView.SHADED
    debug: DebugDraw | None = None
    selected_id: int = 0
    outline_id: int | None = None
    include_transparent_ids: bool = False
    opaque_depth_ready: bool = False
    gizmo: GizmoFrame | None = None
    ui_scale: float = 1.0
    time: float = 0.0
    background: tuple[float, float, float, float] = (0.13, 0.14, 0.16, 1.0)
    frame_serial: int = 0
    mesh_revision: int = 0
    texture_revision: int = 0

    shadow: ShadowResult = field(default_factory=ShadowResult)
    scene_program: moderngl.Program | None = None

    reflection: Any = None

    draw_calls: int = 0
    instance_count: int = 0
    triangle_count: int = 0

    def flag(self, f: RenderFlag, default: bool = True) -> bool:
        return self.flags.get(f, default)

    @property
    def px_scale(self) -> float:
        p11 = float(self.proj[1, 1])
        return 2.0 / (p11 * max(self.target.height, 1)) if abs(p11) > 1e-9 else 0.0


class RenderPass(Protocol):
    name: str

    def prepare_instances(self, ctx: PassContext) -> None: ...

    def prepare(self, ctx: PassContext) -> bool: ...

    def execute(self, ctx: PassContext) -> None: ...

    def release(self) -> None: ...


class BasePass:
    name = "pass"

    def prepare_instances(self, ctx: PassContext) -> None:
        """Apply pass-owned instance variants before the frame upload."""

        del ctx

    def prepare(self, ctx: PassContext) -> bool:
        return True

    def execute(self, ctx: PassContext) -> None: ...

    def release(self) -> None: ...


def state_opaque(ctx: moderngl.Context) -> None:
    ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.depth_func = "<"
    ctx.front_face = "ccw"
    ctx.cull_face = "back"
    ctx.wireframe = False


def state_transparent(ctx: moderngl.Context, *, additive: bool = False) -> None:
    ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
    ctx.depth_func = "<"
    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE if additive else moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ZERO,
        moderngl.ONE,
    )
    ctx.front_face = "ccw"
    ctx.cull_face = "back"


def state_overdraw(ctx: moderngl.Context) -> None:
    ctx.enable_only(moderngl.BLEND | moderngl.CULL_FACE)
    ctx.blend_func = (moderngl.ONE, moderngl.ONE, moderngl.ZERO, moderngl.ONE)
    ctx.front_face = "ccw"
    ctx.cull_face = "back"


def state_overlay(ctx: moderngl.Context, depth_test: bool) -> None:
    flags = moderngl.BLEND | (moderngl.DEPTH_TEST if depth_test else 0)
    ctx.enable_only(flags)
    ctx.depth_func = "<"
    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
        moderngl.ZERO,
        moderngl.ONE,
    )
