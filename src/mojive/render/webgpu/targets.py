"""wgpu render targets, frame uniforms, and pixel readback."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import wgpu

from ...types import CameraView
from .readback import WgpuReadbackQueue, aligned_row_bytes, rgba_to_rgb

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
        ("flags", "(4,)f4"),  # x: orthographic, y: linear out (reflection pass)
        ("ids", "(4,)u4"),  # x: selected id, y: light count
        ("image_light", "(4,)f4"),  # x: gain, y: max mip level
        ("clip_plane", "(4,)f4"),  # reflection clip plane; (0,0,0,1) disables
        ("reflection", "(4,)f4"),  # x/y: reflection target size; x=0 disables
    ]
)
FRAME_BYTES = FRAME_DTYPE.itemsize
assert FRAME_BYTES == 384


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
    return aligned_row_bytes(width, bpp)


class RenderTargetWgpu:
    def __init__(self, device: wgpu.GPUDevice, width: int, height: int, samples: int = 4) -> None:
        self._device = device
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.samples = self._sample_count(samples)
        self._build()
        self._readbacks: WgpuReadbackQueue | None = None
        self.frame_buffer = device.create_buffer(
            size=FRAME_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    @staticmethod
    def _sample_count(samples: int) -> int:
        return 4 if int(samples) > 1 else 1

    def _build(self) -> None:
        device = self._device
        size = (self.width, self.height, 1)
        # TEXTURE_BINDING: the viewer window binds color as the viewport image.
        color_usage = (
            wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.TEXTURE_BINDING
        )
        self.color = device.create_texture(size=size, format="rgba8unorm", usage=color_usage)
        self.color_view = self.color.create_view()
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
        self.color_ms_view = self.color_ms.create_view() if self.color_ms is not None else None
        self.zbuf = device.create_texture(
            size=size,
            format="depth24plus",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.samples,
        )
        self.zbuf_view = self.zbuf.create_view()
        export_usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self.export_depth = device.create_texture(size=size, format="r32float", usage=export_usage)
        self.export_depth_view = self.export_depth.create_view()
        # export_id is additionally sampled by the present pass (SEGMENT/IDCOLOR).
        self.export_id = device.create_texture(
            size=size,
            format="r32uint",
            usage=export_usage | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        self.export_id_view = self.export_id.create_view()
        self.export_segmentation = device.create_texture(
            size=size,
            format="rg32sint",
            usage=export_usage,
        )
        self.export_segmentation_view = self.export_segmentation.create_view()
        self.export_zbuf = device.create_texture(
            size=size, format="depth24plus", usage=wgpu.TextureUsage.RENDER_ATTACHMENT
        )
        self.export_zbuf_view = self.export_zbuf.create_view()

    def resize(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        if self._readbacks is not None:
            self._readbacks.drain()
        self.width, self.height = width, height
        self._release_textures()
        self._build()

    def set_samples(self, samples: int) -> bool:
        samples = self._sample_count(samples)
        if samples == self.samples:
            return False
        if self._readbacks is not None:
            self._readbacks.drain()
        self.samples = samples
        self._release_textures()
        self._build()
        return True

    def _release_textures(self) -> None:
        for tex in (
            self.color,
            self.color_ms,
            self.zbuf,
            self.export_depth,
            self.export_id,
            self.export_segmentation,
            self.export_zbuf,
        ):
            if tex is not None:
                tex.destroy()
        self.color_view = None
        self.color_ms_view = None
        self.zbuf_view = None
        self.export_depth_view = None
        self.export_id_view = None
        self.export_segmentation_view = None
        self.export_zbuf_view = None

    def _readback_queue(self) -> WgpuReadbackQueue:
        if self._readbacks is None:
            self._readbacks = WgpuReadbackQueue(self._device)
        return self._readbacks

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
        # WebGPU rows are already top-first; opengl's flip=True means top-first.
        if not flip:
            image = image[::-1]
        return np.ascontiguousarray(image)

    def read_color(self, flip: bool = True) -> np.ndarray:
        return self._read_texture(self.color, np.uint8, 4, flip)

    def read_rgb(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        """Read packed RGB into a new or caller-owned array."""

        shape = (self.height, self.width, 3)
        if out is None:
            out = np.empty(shape, np.uint8)
        elif out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous uint8 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        return rgba_to_rgb(self.read_color(flip=flip), out)

    def read_rgb_async(
        self, flip: bool = True, out: np.ndarray | None = None
    ) -> Future[np.ndarray]:
        """Queue packed RGB readback without waiting on the render thread."""

        return self._readback_queue().enqueue(
            self.color,
            width=self.width,
            height=self.height,
            dtype=np.uint8,
            storage_channels=4,
            output_channels=3,
            flip=flip,
            out=out,
        )

    def read_depth(self, flip: bool = True) -> np.ndarray:
        return self._read_texture(self.export_depth, np.float32, 1, flip)

    def read_metric_depth(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.height, self.width)
        if out is None:
            return self._read_texture(self.export_depth, np.float32, 1, flip)
        if out.shape != shape or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous float32 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        np.copyto(out, self._read_texture(self.export_depth, np.float32, 1, flip))
        return out

    def read_metric_depth_async(
        self, flip: bool = True, out: np.ndarray | None = None
    ) -> Future[np.ndarray]:
        """Queue metric-depth readback without waiting on the render thread."""

        return self._readback_queue().enqueue(
            self.export_depth,
            width=self.width,
            height=self.height,
            dtype=np.float32,
            storage_channels=1,
            flip=flip,
            out=out,
        )

    def read_ids(self, flip: bool = False) -> np.ndarray:
        return self._read_texture(self.export_id, np.uint32, 1, flip)

    def read_segmentation(self, flip: bool = True, out: np.ndarray | None = None) -> np.ndarray:
        shape = (self.height, self.width, 2)
        if out is None:
            return self._read_texture(self.export_segmentation, np.int32, 2, flip)
        if out.shape != shape or out.dtype != np.int32 or not out.flags.c_contiguous:
            raise ValueError(
                f"Expected C-contiguous int32 destination with shape {shape}, "
                f"got {out.dtype} {out.shape}"
            )
        np.copyto(out, self._read_texture(self.export_segmentation, np.int32, 2, flip))
        return out

    def read_segmentation_async(
        self, flip: bool = True, out: np.ndarray | None = None
    ) -> Future[np.ndarray]:
        """Queue semantic-ID readback without waiting on the render thread."""

        return self._readback_queue().enqueue(
            self.export_segmentation,
            width=self.width,
            height=self.height,
            dtype=np.int32,
            storage_channels=2,
            flip=flip,
            out=out,
        )

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
        if self._readbacks is not None:
            self._readbacks.release()
            self._readbacks = None
        self._release_textures()
        self.frame_buffer.destroy()
