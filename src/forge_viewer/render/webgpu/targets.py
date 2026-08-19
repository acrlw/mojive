"""Render targets, frame uniforms, and readbacks for the webgpu backend.

Target chain (mirrors ``render.forge.targets.RenderTarget`` with WebGPU
constraints applied):

- color: rgba8unorm, MSAA when samples > 1, resolved for readback
- depth: depth24plus, MSAA matching color, test-only (never read back)
- export: single-sampled MRT (r32float GL-compatible depth + r32uint object
  id) re-rasterized by a dedicated pass, because WebGPU cannot resolve
  multisampled integer or depth attachments

Readback convention matches forge: ``flip=True`` returns the image top row
first (WebGPU texture rows are already top-first, so no actual flip happens).
"""

from __future__ import annotations

import numpy as np
import wgpu

from ...types import CameraView

# Frame uniform block, mirrors `struct Frame` in shaders/scene.wgsl.
FRAME_DTYPE = np.dtype(
    [
        ("view_proj", "(4,4)f4"),
        ("view", "(4,4)f4"),
        ("camera_pos", "(4,)f4"),
        ("camera_dir", "(4,)f4"),
        ("ambient", "(4,)f4"),
        ("headlight_diffuse", "(4,)f4"),
        ("headlight_specular", "(4,)f4"),
        ("fog", "(4,)f4"),
        ("fog_color", "(4,)f4"),
        ("haze_color", "(4,)f4"),
        ("highlight_color", "(4,)f4"),
        ("highlight", "(4,)f4"),
        ("shading", "(4,)f4"),  # exposure, tonemap on, near, far
        ("flags", "(4,)f4"),  # x: orthographic
        ("ids", "(4,)u4"),  # x: selected id, y: light count
        ("image_light", "(4,)f4"),  # x: gain, y: max mip level
    ]
)
FRAME_BYTES = FRAME_DTYPE.itemsize
assert FRAME_BYTES == 352


def perspective_wgpu(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed perspective with WebGPU clip conventions (z in [0, 1])."""
    f = 1.0 / np.tan(fov_y * 0.5)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = (far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def perspective_intrinsics_wgpu(focal_length, sensor_size, principal_offset, near, far):
    focal = np.asarray(focal_length, np.float64).reshape(2)
    sensor = np.asarray(sensor_size, np.float64).reshape(2)
    principal = np.asarray(principal_offset, np.float64).reshape(2)
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = 2.0 * focal[0] / sensor[0]
    m[1, 1] = 2.0 * focal[1] / sensor[1]
    m[0, 2] = 2.0 * principal[0] / sensor[0]
    m[1, 2] = -2.0 * principal[1] / sensor[1]
    m[2, 2] = far / (near - far)
    m[2, 3] = (far * near) / (near - far)
    m[3, 2] = -1.0
    return m.astype(np.float32)


def orthographic_wgpu(height: float, aspect: float, near: float, far: float) -> np.ndarray:
    h = max(height, 1e-6) * 0.5
    w = h * max(aspect, 1e-6)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / w
    m[1, 1] = 1.0 / h
    m[2, 2] = 1.0 / (near - far)
    m[2, 3] = near / (near - far)
    return m


def proj_matrix_wgpu(camera: CameraView) -> np.ndarray:
    if camera.orthographic:
        return orthographic_wgpu(camera.ortho_height, camera.aspect, camera.near, camera.far)
    if camera.uses_intrinsics():
        return perspective_intrinsics_wgpu(
            camera.focal_length,
            camera.sensor_size,
            camera.principal_offset,
            camera.near,
            camera.far,
        )
    return perspective_wgpu(camera.fov_y, camera.aspect, camera.near, camera.far)


def _aligned_row_bytes(width: int, bpp: int) -> int:
    raw = width * bpp
    return (raw + 255) // 256 * 256


class RenderTargetWgpu:
    def __init__(self, device: wgpu.GPUDevice, width: int, height: int, samples: int = 4) -> None:
        self._device = device
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.samples = 4 if int(samples) > 1 else 1
        self._build()

    def _build(self) -> None:
        device = self._device
        size = (self.width, self.height, 1)
        color_usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.color = device.create_texture(size=size, format="rgba8unorm", usage=color_usage)
        self.color_ms = (
            device.create_texture(
                size=size,
                format="rgba8unorm",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
                sample_count=self.samples,
            )
            if self.samples > 1
            else None
        )
        self.zbuf = device.create_texture(
            size=size,
            format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.samples,
        )
        export_usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.export_depth = device.create_texture(size=size, format="r32float", usage=export_usage)
        self.export_id = device.create_texture(size=size, format="r32uint", usage=export_usage)
        self.export_zbuf = device.create_texture(
            size=size, format="depth24plus", usage=wgpu.TextureUsage.RENDER_ATTACHMENT
        )
        self.frame_buffer = device.create_buffer(
            size=FRAME_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        for tex in (
            self.color,
            self.color_ms,
            self.zbuf,
            self.export_depth,
            self.export_id,
            self.export_zbuf,
        ):
            if tex is not None:
                tex.destroy()
        self._build()

    def _read_texture(self, texture, dtype, channels: int, flip: bool) -> np.ndarray:
        bpp = np.dtype(dtype).itemsize * channels
        row_bytes = _aligned_row_bytes(self.width, bpp)
        data = self._device.queue.read_texture(
            {"texture": texture, "origin": (0, 0, 0)},
            {"bytes_per_row": row_bytes, "rows_per_image": self.height},
            (self.width, self.height, 1),
        )
        raw = np.frombuffer(data, np.uint8).reshape(self.height, row_bytes)
        trimmed = raw[:, : self.width * bpp]
        image = trimmed.view(dtype).reshape(self.height, self.width, channels)
        if channels == 1:
            image = image[..., 0]
        # WebGPU rows are already top-first; forge's flip=True means top-first.
        if not flip:
            image = image[::-1]
        return np.ascontiguousarray(image)

    def read_color(self, flip: bool = True) -> np.ndarray:
        return self._read_texture(self.color, np.uint8, 4, flip)

    def read_depth(self, flip: bool = True) -> np.ndarray:
        return self._read_texture(self.export_depth, np.float32, 1, flip)

    def read_ids(self, flip: bool = False) -> np.ndarray:
        return self._read_texture(self.export_id, np.uint32, 1, flip)

    def read_id(self, x: int, y: int) -> int:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0
        row_bytes = _aligned_row_bytes(1, 4)
        data = self._device.queue.read_texture(
            {"texture": self.export_id, "origin": (int(x), int(y), 0)},
            {"bytes_per_row": row_bytes, "rows_per_image": 1},
            (1, 1, 1),
        )
        return int(np.frombuffer(data, np.uint32)[0])

    def release(self) -> None:
        for tex in (
            self.color,
            self.color_ms,
            self.zbuf,
            self.export_depth,
            self.export_id,
            self.export_zbuf,
        ):
            if tex is not None:
                tex.destroy()
