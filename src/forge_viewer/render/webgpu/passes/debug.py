"""Debug primitives and world-space text for wgpu."""

from __future__ import annotations

import numpy as np
import wgpu

from ....log import get_logger
from ....types import MeshKey
from ...debugdraw import RECORD_FLOATS, DebugDraw, Occlusion, PackedFrame, Path
from ...mesh import builtin_mesh
from ...text import RECORD_FLOATS as TEXT_RECORD_FLOATS
from ...text import TextLayout
from ..meshes import GpuMesh
from ..programs import load_wgsl

log = get_logger("debug")

SECTOR_SEGMENTS = 32
STROKE_JOIN_SEGMENTS = 6
GHOST_ALPHA = 0.28

# Frame-constant debug uniforms: slot 0 alpha=1, slot 1 the GHOST occluded
# alpha.  Dynamic offset windows are 256-byte aligned.
_DEBUG_DTYPE = np.dtype(
    [
        ("view_proj", "(4,4)f4"),
        ("view", "(4,4)f4"),
        ("proj", "(4,4)f4"),
        ("params", "(4,)f4"),  # viewport width/height, px_scale, alpha
    ]
)
_UNIFORM_SLOTS = 2
_UNIFORM_SLOT_BYTES = 256

_ALPHA_BLEND = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha"},
}

_VERTICES: dict[Path, int] = {
    Path.SEGMENT: 15,
    Path.STROKE: 6 + 3 * STROKE_JOIN_SEGMENTS,
    Path.POINT: 6,
    Path.DRAG_LINK: 6,
    Path.SECTOR: 3 * SECTOR_SEGMENTS,
}

_ENTRIES: dict[Path, tuple[str, str]] = {
    Path.SEGMENT: ("vs_debug_line", "fs_debug_line"),
    Path.STROKE: ("vs_debug_stroke", "fs_debug_line"),
    Path.POINT: ("vs_debug_point", "fs_debug_point"),
    Path.DRAG_LINK: ("vs_debug_drag_link", "fs_debug_drag_link"),
    Path.SOLID: ("vs_debug_solid", "fs_debug_solid"),
    Path.SECTOR: ("vs_debug_sector", "fs_debug_line"),
}


def _instanced(path: Path, *attrs: tuple[str, int]) -> dict:
    """Instance-step vertex layout for one packed record stream."""
    return {
        "array_stride": RECORD_FLOATS[path] * 4,
        "step_mode": "instance",
        "attributes": [
            {"format": fmt, "offset": offset, "shader_location": loc}
            for loc, (fmt, offset) in enumerate(attrs)
        ],
    }


_LAYOUTS: dict[Path, list[dict]] = {
    Path.SEGMENT: [
        _instanced(
            Path.SEGMENT,
            ("float32x3", 0),
            ("float32x3", 12),
            ("float32x4", 24),
            ("float32", 40),
            ("float32", 44),
            ("float32", 48),
        )
    ],
    Path.STROKE: [
        _instanced(
            Path.STROKE,
            ("float32x3", 0),
            ("float32x3", 12),
            ("float32x3", 24),
            ("float32x4", 36),
            ("float32", 52),
        )
    ],
    Path.POINT: [_instanced(Path.POINT, ("float32x3", 0), ("float32x4", 12), ("float32", 28))],
    Path.DRAG_LINK: [
        _instanced(
            Path.DRAG_LINK,
            ("float32x3", 0),
            ("float32x3", 12),
            ("float32x4", 24),
            ("float32x4", 40),
            ("float32", 56),
            ("float32", 60),
            ("float32", 64),
        )
    ],
    Path.SECTOR: [
        _instanced(
            Path.SECTOR,
            ("float32x3", 0),
            ("float32x3", 12),
            ("float32x3", 24),
            ("float32x4", 36),
            ("float32", 52),
        )
    ],
}

# Solid path: slot 0 is the built-in mesh (position, normal; uv skipped like
# forge's "3f 3f 8x"), slot 1 the instance record (model columns + color).
_SOLID_LAYOUTS = [
    {
        "array_stride": 32,
        "step_mode": "vertex",
        "attributes": [
            {"format": "float32x3", "offset": 0, "shader_location": 0},
            {"format": "float32x3", "offset": 12, "shader_location": 1},
        ],
    },
    {
        "array_stride": RECORD_FLOATS[Path.SOLID] * 4,
        "step_mode": "instance",
        "attributes": [
            {"format": "float32x4", "offset": 0, "shader_location": 2},
            {"format": "float32x4", "offset": 16, "shader_location": 3},
            {"format": "float32x4", "offset": 32, "shader_location": 4},
            {"format": "float32x4", "offset": 48, "shader_location": 5},
            {"format": "float32x4", "offset": 64, "shader_location": 6},
        ],
    },
]

_TEXT_LAYOUT = {
    "array_stride": TEXT_RECORD_FLOATS * 4,
    "step_mode": "instance",
    "attributes": [
        {"format": "float32x3", "offset": 0, "shader_location": 0},
        {"format": "float32x2", "offset": 12, "shader_location": 1},
        {"format": "float32x4", "offset": 20, "shader_location": 2},
        {"format": "float32x4", "offset": 36, "shader_location": 3},
        {"format": "float32x4", "offset": 52, "shader_location": 4},
    ],
}


class DebugPass:
    name = "debug"

    def __init__(self, device: wgpu.GPUDevice, samples: int, draw: DebugDraw) -> None:
        self._device = device
        self._samples = samples
        self.draw = draw
        self.draw_calls = 0
        self._module = device.create_shader_module(
            code=load_wgsl(
                "debug_common.wgsl",
                "debug_line.wgsl",
                "debug_stroke.wgsl",
                "debug_point.wgsl",
                "debug_solid.wgsl",
                "debug_sector.wgsl",
                "debug_drag_link.wgsl",
                "debug_text.wgsl",
            )
        )
        self._uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": True,
                        "min_binding_size": _DEBUG_DTYPE.itemsize,
                    },
                }
            ]
        )
        self._atlas_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        self._pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._uniform_layout]
        )
        self._text_pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._uniform_layout, self._atlas_layout]
        )
        self._uniforms = device.create_buffer(
            size=_UNIFORM_SLOT_BYTES * _UNIFORM_SLOTS,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        # The uniform buffer is persistent, so the bind group is built once;
        # per-draw alpha is the dynamic offset.
        self._uniform_group = device.create_bind_group(
            layout=self._uniform_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniforms,
                        "offset": 0,
                        "size": _DEBUG_DTYPE.itemsize,
                    },
                }
            ],
        )
        self._uniform_block = np.zeros((), _DEBUG_DTYPE)
        self._pipelines: dict[tuple[Path, str], wgpu.GPURenderPipeline] = {}
        self._text_pipelines: dict[str, wgpu.GPURenderPipeline] = {}
        self._buffers: dict[Path, wgpu.GPUBuffer | None] = dict.fromkeys(Path, None)
        self._meshes: dict[MeshKey, GpuMesh | None] = {}
        self._text = TextLayout()
        self._text_buffer: wgpu.GPUBuffer | None = None
        self._atlas: wgpu.GPUTexture | None = None
        self._atlas_group: wgpu.GPUBindGroup | None = None
        self._sampler = device.create_sampler(min_filter="linear", mag_filter="linear")
        self._frame: PackedFrame | None = None
        self._text_ready = False

    def configure_text(
        self,
        primary: str = "",
        primary_index: int = 0,
        fallback: str = "",
        fallback_index: int = 0,
        size_px: float = 14.0,
    ) -> None:
        self._text.configure(primary, primary_index, fallback, fallback_index, size_px)

    def set_samples(self, samples: int) -> None:
        if int(samples) == self._samples:
            return
        self._samples = int(samples)
        self._pipelines.clear()
        self._text_pipelines.clear()

    def prepare(
        self,
        view: np.ndarray,
        proj: np.ndarray,
        view_proj: np.ndarray,
        width: int,
        height: int,
        now: float,
    ) -> None:
        """Build the packed frame and upload it; mirrors DebugPass.execute's CPU half."""
        self._frame = None
        self._text_ready = False
        if self.draw.primitives == 0:
            return
        block = self._uniform_block
        block["view_proj"][:] = np.asarray(view_proj, np.float32).T
        block["view"][:] = np.asarray(view, np.float32).T
        block["proj"][:] = np.asarray(proj, np.float32).T
        # forge PassContext.px_scale: 2 / (proj[1,1] * height).
        p11 = float(proj[1, 1])
        px_scale = 2.0 / (p11 * max(height, 1)) if abs(p11) > 1e-9 else 0.0
        for slot, alpha in ((0, 1.0), (1, GHOST_ALPHA)):
            block["params"][:] = (float(width), float(height), px_scale, alpha)
            self._device.queue.write_buffer(
                self._uniforms, slot * _UNIFORM_SLOT_BYTES, block.tobytes()
            )
        self.draw.render_frame(self._upload, now=now)

    def _upload(self, frame: PackedFrame) -> None:
        for path in Path:
            n = frame.counts[path]
            if n:
                buf = self._ensure_buffer(path, n)
                self._device.queue.write_buffer(buf, 0, frame.stream(path).tobytes())
        self._text_ready = self._text.prepare(frame.texts, frame.text_count)
        if self._text_ready:
            self._sync_text_gpu()
        self._frame = frame

    def _ensure_buffer(self, path: Path, records: int) -> wgpu.GPUBuffer:
        stride = RECORD_FLOATS[path] * 4
        buf = self._buffers[path]
        if buf is not None and buf.size >= records * stride:
            return buf
        need = max(records, (buf.size // stride * 2) if buf is not None else 0, 64)
        if buf is not None:
            buf.destroy()
        buf = self._device.create_buffer(
            size=need * stride, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )
        self._buffers[path] = buf
        return buf

    def _mesh(self, key: MeshKey | None) -> GpuMesh | None:
        if key is None:
            return None
        if key in self._meshes:
            return self._meshes[key]
        gpu: GpuMesh | None = None
        try:
            gpu = GpuMesh(self._device, builtin_mesh(key))
        except Exception as e:
            log.error(
                "Built-in mesh {} is unavailable; solid annotations cannot be drawn: {}", key, e
            )
        self._meshes[key] = gpu
        return gpu

    # -- text -----------------------------------------------------------------

    def _sync_text_gpu(self) -> None:
        if self._text.atlas_dirty:
            w, h = self._text.atlas_size
            if self._atlas is None or self._atlas.size != (w, h, 1):
                if self._atlas is not None:
                    self._atlas.destroy()
                self._atlas = self._device.create_texture(
                    size=(w, h, 1),
                    format="r8unorm",
                    usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                )
                self._atlas_group = self._device.create_bind_group(
                    layout=self._atlas_layout,
                    entries=[
                        {"binding": 0, "resource": self._atlas.create_view()},
                        {"binding": 1, "resource": self._sampler},
                    ],
                )
            self._device.queue.write_texture(
                {"texture": self._atlas},
                self._text.pixels,
                {"bytes_per_row": w, "rows_per_image": h},
                (w, h, 1),
            )
            self._text.mark_uploaded()
        n = self._text.count
        stride = _TEXT_LAYOUT["array_stride"]
        buf = self._text_buffer
        if buf is None or buf.size < n * stride:
            need = max(n, (buf.size // stride * 2) if buf is not None else 0, 64)
            if buf is not None:
                buf.destroy()
            buf = self._device.create_buffer(
                size=need * stride, usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
            )
            self._text_buffer = buf
        self._device.queue.write_buffer(buf, 0, self._text.records[:n].tobytes())

    # -- pipelines ------------------------------------------------------------

    def _pipeline(self, path: Path, compare: str) -> wgpu.GPURenderPipeline:
        key = (path, compare)
        pipeline = self._pipelines.get(key)
        if pipeline is not None:
            return pipeline
        vs, fs = _ENTRIES[path]
        pipeline = self._device.create_render_pipeline(
            layout=self._pipeline_layout,
            vertex={
                "module": self._module,
                "entry_point": vs,
                "buffers": _SOLID_LAYOUTS if path is Path.SOLID else _LAYOUTS[path],
            },
            fragment={
                "module": self._module,
                "entry_point": fs,
                "targets": [{"format": "rgba8unorm", "blend": _ALPHA_BLEND}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": False,
                "depth_compare": compare,
            },
            multisample={"count": self._samples},
        )
        self._pipelines[key] = pipeline
        return pipeline

    def _text_pipeline(self, compare: str) -> wgpu.GPURenderPipeline:
        pipeline = self._text_pipelines.get(compare)
        if pipeline is not None:
            return pipeline
        pipeline = self._device.create_render_pipeline(
            layout=self._text_pipeline_layout,
            vertex={
                "module": self._module,
                "entry_point": "vs_debug_text",
                "buffers": [_TEXT_LAYOUT],
            },
            fragment={
                "module": self._module,
                "entry_point": "fs_debug_text",
                "targets": [{"format": "rgba8unorm", "blend": _ALPHA_BLEND}],
            },
            primitive={"topology": "triangle-list", "front_face": "ccw", "cull_mode": "none"},
            depth_stencil={
                "format": "depth24plus",
                "depth_write_enabled": False,
                "depth_compare": compare,
            },
            multisample={"count": self._samples},
        )
        self._text_pipelines[compare] = pipeline
        return pipeline

    # -- drawing ----------------------------------------------------------------

    def execute(self, pass_encoder: wgpu.GPURenderPassEncoder) -> int:
        """Encode the packed frame into the main color pass; returns draw calls."""
        self.draw_calls = 0
        frame = self._frame
        if frame is None:
            return 0
        i = 0
        while i < frame.batch_count:
            occ = frame.batches[i].occlusion
            j = i
            while j < frame.batch_count and frame.batches[j].occlusion is occ:
                j += 1
            if occ is Occlusion.GHOST:
                self._draw_range(pass_encoder, frame, i, j, "less", 1.0)
                self._draw_range(pass_encoder, frame, i, j, "greater", GHOST_ALPHA)
            else:
                compare = "less" if occ is Occlusion.DEPTH else "always"
                self._draw_range(pass_encoder, frame, i, j, compare, 1.0)
            i = j
        if self._text_ready and self._atlas_group is not None:
            for batch in self._text.batches():
                if batch.occlusion is Occlusion.GHOST:
                    self._draw_text(pass_encoder, batch, "less", 1.0)
                    self._draw_text(pass_encoder, batch, "greater", GHOST_ALPHA)
                else:
                    compare = "less" if batch.occlusion is Occlusion.DEPTH else "always"
                    self._draw_text(pass_encoder, batch, compare, 1.0)
        return self.draw_calls

    def _draw_range(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        frame: PackedFrame,
        i0: int,
        i1: int,
        compare: str,
        alpha: float,
    ) -> None:
        offset = 0 if alpha == 1.0 else _UNIFORM_SLOT_BYTES
        for i in range(i0, i1):
            b = frame.batches[i]
            buf = self._buffers[b.path]
            if buf is None:
                continue
            pass_encoder.set_pipeline(self._pipeline(b.path, compare))
            pass_encoder.set_bind_group(0, self._uniform_group, [offset])
            if b.path is Path.SOLID:
                mesh = self._mesh(b.mesh)
                if mesh is None:
                    self.draw.drop(b.count, f"{b.path} batch is missing a mesh")
                    continue
                pass_encoder.set_vertex_buffer(0, mesh.vbo)
                pass_encoder.set_vertex_buffer(1, buf)
                pass_encoder.set_index_buffer(mesh.ibo, "uint32")
                pass_encoder.draw_indexed(mesh.index_count, b.count, 0, 0, b.start)
            else:
                stride = RECORD_FLOATS[b.path] * 4
                pass_encoder.set_vertex_buffer(0, buf, offset=b.start * stride)
                pass_encoder.draw(_VERTICES[b.path], b.count)
            self.draw_calls += 1

    def _draw_text(
        self,
        pass_encoder: wgpu.GPURenderPassEncoder,
        batch,
        compare: str,
        alpha: float,
    ) -> None:
        if self._text_buffer is None:
            return
        offset = 0 if alpha == 1.0 else _UNIFORM_SLOT_BYTES
        pass_encoder.set_pipeline(self._text_pipeline(compare))
        pass_encoder.set_bind_group(0, self._uniform_group, [offset])
        pass_encoder.set_bind_group(1, self._atlas_group)
        pass_encoder.set_vertex_buffer(
            0, self._text_buffer, offset=batch.start * _TEXT_LAYOUT["array_stride"]
        )
        pass_encoder.draw(6, batch.count)
        self.draw_calls += 1

    def release(self) -> None:
        for buf in self._buffers.values():
            if buf is not None:
                buf.destroy()
        self._buffers = dict.fromkeys(Path, None)
        for mesh in self._meshes.values():
            if mesh is not None:
                mesh.release()
        self._meshes.clear()
        if self._text_buffer is not None:
            self._text_buffer.destroy()
            self._text_buffer = None
        if self._atlas is not None:
            self._atlas.destroy()
            self._atlas = None
        self._atlas_group = None
        self._uniforms.destroy()
