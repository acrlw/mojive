"""Native 3D gizmo rendering for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ....gizmo import GizmoFrame
from ....types import CameraView, MeshKey, MeshShape
from ...gizmo_plan import _MESHES, GizmoPlanner
from ...mesh import builtin_mesh, gizmo_mesh
from ..blend import ALPHA_BLEND
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
_SLOTS = 16
_SLOT_BYTES = 256

# Standard GpuMesh stream; only position/normal are read (opengl "3f 3f 8x").
_VERTEX_LAYOUT = {
    "array_stride": 32,
    "step_mode": "vertex",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x3", "offset": 12, "shader_location": 1},
    ],
}


class GizmoPass(GizmoPlanner):
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
        data = {name: gizmo_mesh(name) for name in _MESHES if name not in {"trackball", "center"}}
        data["trackball"] = builtin_mesh(MeshKey(MeshShape.DISK))
        data["center"] = builtin_mesh(MeshKey(MeshShape.SPHERE))
        self._meshes = {name: GpuMesh(device, m) for name, m in data.items()}
        # Planned draws: (mesh name, pipeline key) in opengl's order.
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
        """CPU half of opengl's execute: pick handles and upload their slots."""
        self._draws = []
        if frame is None:
            return
        plans = self.plan(frame, camera, view_proj, proj, height)
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
                "targets": [{"format": "rgba8unorm", "blend": ALPHA_BLEND}],
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
