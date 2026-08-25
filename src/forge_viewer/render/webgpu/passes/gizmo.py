"""Native 3D gizmo rendering for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ....gizmo import (
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
    GizmoFrame,
    GizmoHandle,
    GizmoMode,
    axis_handle_alpha,
    axis_rotation,
    display_handles,
    plane_handle_alpha,
    rotation_half_basis,
    rotation_ring_alpha,
    screen_rotation_basis,
)
from ....types import CameraView, MeshKey, MeshShape
from ...mesh import builtin_mesh, gizmo_mesh
from ..meshes import GpuMesh
from ..programs import load_wgsl

_SLOT_DTYPE = np.dtype(
    [
        ("view_proj", "(4,4)f4"),
        ("view", "(4,4)f4"),
        ("model", "(4,4)f4"),
        ("color", "(4,)f4"),
        ("params", "(4,)f4"),  # x: mask_radius
    ]
)
# One dynamic-offset window per handle draw; translate mode draws the most
# (3 planes + 3 arrows + 2 center spheres).
_SLOTS = 8
_SLOT_BYTES = 256

_ALPHA_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha"},
}

# Standard GpuMesh stream; only position/normal are read (forge "3f 3f 8x").
_VERTEX_LAYOUT = {
    "array_stride": 32,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
    ],
}

# Pipeline variants (compare, depth write, cull) per handle group.
_PIPE_PLANE = ("less", False, "none")
_PIPE_HANDLE = ("less", True, "back")
_PIPE_OVERLAY = ("always", False, "back")

_MESHES = ("arrow", "plane", "ring", "half_ring", "screen_ring", "screen_ring_edge", "center")


class GizmoPass:
    name = "gizmo"

    def __init__(self, device: wgpu.GPUDevice, samples: int) -> None:
        self._device = device
        self._samples = samples
        self._module = device.create_shader_module(code=load_wgsl("gizmo.wgsl"))
        self._uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": True,
                        "min_binding_size": _SLOT_DTYPE.itemsize,
                    },
                }
            ]
        )
        self._pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._uniform_layout]
        )
        self._pipelines: dict[tuple[str, bool, str], wgpu.GPURenderPipeline] = {}
        self._uniforms = device.create_buffer(
            size=_SLOT_BYTES * _SLOTS,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # The uniform buffer is persistent, so the bind group is built once;
        # the per-handle slot is the dynamic offset.
        self._uniform_group = device.create_bind_group(
            layout=self._uniform_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniforms,
                        "offset": 0,
                        "size": _SLOT_DTYPE.itemsize,
                    },
                }
            ],
        )
        self._block = np.zeros((), _SLOT_DTYPE)
        data = {name: gizmo_mesh(name) for name in _MESHES[:-1]}
        data["center"] = builtin_mesh(MeshKey(MeshShape.SPHERE))
        self._meshes = {name: GpuMesh(device, m) for name, m in data.items()}
        # Planned draws: (mesh name, pipeline key) in forge's order.
        self._draws: list[tuple[str, tuple[str, bool, str]]] = []

    def prepare(
        self,
        frame: GizmoFrame | None,
        camera: CameraView,
        view: np.ndarray,
        proj: np.ndarray,
        view_proj: np.ndarray,
        height: int,
    ) -> None:
        """CPU half of forge's execute: pick handles and upload their slots."""
        self._draws = []
        if frame is None:
            return
        origin4 = np.ones(4, np.float64)
        origin4[:3] = frame.position
        clip = np.asarray(view_proj, np.float64) @ origin4
        if clip[3] <= 0.0:
            return
        # forge PassContext.px_scale: 2 / (proj[1,1] * height).
        p11 = float(np.asarray(proj)[1, 1])
        px_scale = 2.0 / (p11 * max(height, 1)) if abs(p11) > 1e-9 else 0.0
        scale = float(frame.size_px) * px_scale * float(clip[3])
        if scale <= 0.0:
            return
        plans = (
            self._plan_translate(frame, camera, scale)
            if frame.mode is GizmoMode.TRANSLATE
            else self._plan_rotate(frame, camera, scale)
        )
        assert len(plans) <= _SLOTS
        block = self._block
        block["view_proj"][:] = np.asarray(view_proj, np.float32).T
        block["view"][:] = np.asarray(view, np.float32).T
        for slot, (mesh_name, key, model, color, mask_radius) in enumerate(plans):
            block["model"][:] = model.T
            block["color"][:] = color
            block["params"][:] = (mask_radius, 0.0, 0.0, 0.0)
            self._device.queue.write_buffer(self._uniforms, slot * _SLOT_BYTES, block.tobytes())
            self._draws.append((mesh_name, key))

    def _plan_translate(self, frame: GizmoFrame, camera: CameraView, scale: float) -> list:
        visible = display_handles(frame)
        plans = []
        for axis, handle in enumerate(PLANE_HANDLES):
            if handle in visible:
                alpha = (
                    1.0
                    if frame.active is handle
                    else plane_handle_alpha(camera, frame.position, frame.rotation[:, axis])
                )
                if alpha <= 0.0:
                    continue
                opacity = PLANE_ACTIVE_ALPHA if frame.active is handle else PLANE_ALPHA * alpha
                plans.append(
                    (
                        "plane",
                        _PIPE_PLANE,
                        _model(frame.position, axis_rotation(frame.rotation, axis), scale),
                        self._color(frame, handle, axis, alpha=opacity),
                        0.0,
                    )
                )
        for axis, handle in enumerate(AXIS_HANDLES):
            if handle in visible:
                alpha = (
                    1.0
                    if frame.active is handle
                    else axis_handle_alpha(camera, frame.position, frame.rotation[:, axis])
                )
                if alpha <= 0.0:
                    continue
                plans.append(
                    (
                        "arrow",
                        _PIPE_HANDLE,
                        _model(frame.position, axis_rotation(frame.rotation, axis), scale),
                        self._color(frame, handle, axis, alpha),
                        CENTER_SHELL_RADIUS,
                    )
                )
        if GizmoHandle.SCREEN in visible:
            plans.append(
                (
                    "center",
                    _PIPE_OVERLAY,
                    _model(
                        frame.position,
                        np.eye(3),
                        scale * (CENTER_RADIUS + CONTRAST_EDGE_PT / SIZE_PT),
                    ),
                    CONTRAST_EDGE_COLOR,
                    0.0,
                )
            )
            plans.append(
                (
                    "center",
                    _PIPE_OVERLAY,
                    _model(frame.position, np.eye(3), scale * CENTER_RADIUS),
                    HOVER_COLOR if self._hot(frame, GizmoHandle.SCREEN) else CENTER_COLOR,
                    0.0,
                )
            )
        return plans

    def _plan_rotate(self, frame: GizmoFrame, camera: CameraView, scale: float) -> list:
        visible = display_handles(frame)
        plans = []
        for axis, handle in enumerate(ROTATE_AXIS_HANDLES):
            if handle not in visible:
                continue
            if frame.active_rotation_overlay and frame.active is handle:
                continue
            full = frame.active is handle
            alpha = (
                1.0
                if full
                else rotation_ring_alpha(camera, frame.position, frame.rotation[:, axis])
            )
            if alpha <= 0.0:
                continue
            plans.append(
                (
                    "ring" if full else "half_ring",
                    _PIPE_HANDLE,
                    _model(
                        frame.position,
                        axis_rotation(frame.rotation, axis)
                        if full
                        else rotation_half_basis(camera, frame.position, frame.rotation, axis),
                        scale,
                    ),
                    self._color(frame, handle, axis, alpha),
                    0.0,
                )
            )
        if GizmoHandle.ROTATE_SCREEN in visible and not (
            frame.active_rotation_overlay and frame.active is GizmoHandle.ROTATE_SCREEN
        ):
            basis = screen_rotation_basis(camera)
            plans.append(
                (
                    "screen_ring_edge",
                    _PIPE_OVERLAY,
                    _model(frame.position, basis, scale),
                    CONTRAST_EDGE_COLOR,
                    0.0,
                )
            )
            plans.append(
                (
                    "screen_ring",
                    _PIPE_OVERLAY,
                    _model(frame.position, basis, scale),
                    HOVER_COLOR if self._hot(frame, GizmoHandle.ROTATE_SCREEN) else CENTER_COLOR,
                    0.0,
                )
            )
        return plans

    @staticmethod
    def _hot(frame: GizmoFrame, handle: GizmoHandle) -> bool:
        return frame.active is handle or frame.hovered is handle

    def _color(self, frame: GizmoFrame, handle: GizmoHandle, axis: int, alpha: float = 1.0):
        color = HOVER_COLOR.copy() if self._hot(frame, handle) else AXIS_COLORS[axis].copy()
        color[3] = alpha
        return color

    def _pipeline(self, compare: str, write: bool, cull: str) -> wgpu.GPURenderPipeline:
        key = (compare, write, cull)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        pipeline = self._device.create_render_pipeline(
            layout=self._pipeline_layout,
            vertex={
                "module": self._module,
                "entry_point": "vs_gizmo",
                "buffers": [_VERTEX_LAYOUT],
            },
            fragment={
                "module": self._module,
                "entry_point": "fs_gizmo",
                "targets": [{"format": "rgba8unorm", "blend": _ALPHA_BLEND}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": cull},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": write,
                "depth_compare": compare,
            },
            multisample={"count": self._samples},
        )
        self._pipelines[key] = pipeline
        return pipeline

    def execute(self, pass_encoder: wgpu.GPURenderPassEncoder) -> int:
        """Encode the planned handle draws; returns the draw-call count."""
        for slot, (mesh_name, key) in enumerate(self._draws):
            mesh = self._meshes[mesh_name]
            pass_encoder.set_pipeline(self._pipeline(*key))
            pass_encoder.set_bind_group(0, self._uniform_group, [slot * _SLOT_BYTES])
            pass_encoder.set_vertex_buffer(0, mesh.vbo)
            pass_encoder.set_index_buffer(mesh.ibo, "uint32")
            pass_encoder.draw_indexed(mesh.index_count)
        return len(self._draws)

    def set_samples(self, samples: int) -> None:
        if int(samples) == self._samples:
            return
        self._samples = int(samples)
        self._pipelines.clear()

    def release(self) -> None:
        for mesh in self._meshes.values():
            mesh.release()
        self._meshes.clear()
        self._uniforms.destroy()


def _model(position, rotation, scale: float) -> np.ndarray:
    m = np.zeros((4, 4), np.float32)
    m[:3, :3] = np.asarray(rotation, np.float32) * float(scale)
    m[:3, 3] = position
    m[3, 3] = 1.0
    return m
