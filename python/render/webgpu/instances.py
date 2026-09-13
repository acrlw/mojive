"""Lifecycle-aware GPU instance streams for the WebGPU backend."""

from __future__ import annotations

import numpy as np
import wgpu

from ..scene import RenderScene

POSE_DTYPE = np.dtype([("model", "(4,4)f4")])
VISUAL_DTYPE = np.dtype(
    [
        ("color", "(4,)f4"),
        ("material", "(4,)f4"),
        ("texcoef", "(4,)f4"),
        ("cubecoef", "(4,)f4"),
    ]
)
IDENTITY_DTYPE = np.dtype(
    [
        ("object_id", "u4"),
        # Zero disables planar reflection. Bits 0..2 store layer+1 and bit 3
        # limits a box reflector to its local +Z face.
        ("reflection_info", "u4"),
        ("segmentation", "(2,)i4"),
    ]
)

POSE_STRIDE = POSE_DTYPE.itemsize
VISUAL_STRIDE = VISUAL_DTYPE.itemsize
IDENTITY_STRIDE = IDENTITY_DTYPE.itemsize
INSTANCE_STRIDE = POSE_STRIDE + VISUAL_STRIDE + IDENTITY_STRIDE
assert (POSE_STRIDE, VISUAL_STRIDE, IDENTITY_STRIDE, INSTANCE_STRIDE) == (64, 64, 16, 144)


def _runs(mask: np.ndarray):
    """Yield contiguous true ranges without allocating one index per row."""

    if not len(mask) or not bool(np.any(mask)):
        return
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    yield from zip(edges[::2], edges[1::2], strict=True)


class InstanceStore:
    def __init__(self, device: wgpu.GPUDevice) -> None:
        self._device = device
        self.pose_buffer: wgpu.GPUBuffer | None = None
        self.visual_buffer: wgpu.GPUBuffer | None = None
        self.identity_buffer: wgpu.GPUBuffer | None = None
        self.capacity = 0
        self.count = 0
        self._pose = np.zeros(0, POSE_DTYPE)
        self._visual = np.zeros(0, VISUAL_DTYPE)
        self._identity = np.zeros(0, IDENTITY_DTYPE)
        self._last_structure = -1
        self._last_scene = None
        self._last_pose = -1
        self._last_visual = -1
        self._last_identity = -1
        self._valid = False
        self.uploaded_bytes = 0
        self.uploaded_streams: dict[str, int] = {}

    @property
    def buffer(self) -> wgpu.GPUBuffer | None:
        """Compatibility alias for transform-only consumers."""

        return self.pose_buffer

    def _new_buffer(self, size: int) -> wgpu.GPUBuffer:
        return self._device.create_buffer(
            size=size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

    def _ensure_capacity(self, count: int) -> bool:
        if count <= self.capacity and self.pose_buffer is not None:
            return False
        new_cap = max(count, self.capacity * 2, 64)
        for buffer in (self.pose_buffer, self.visual_buffer, self.identity_buffer):
            if buffer is not None:
                buffer.destroy()
        self.pose_buffer = self._new_buffer(new_cap * POSE_STRIDE)
        self.visual_buffer = self._new_buffer(new_cap * VISUAL_STRIDE)
        self.identity_buffer = self._new_buffer(new_cap * IDENTITY_STRIDE)
        self.capacity = new_cap
        self._pose = np.zeros(new_cap, POSE_DTYPE)
        self._visual = np.zeros(new_cap, VISUAL_DTYPE)
        self._identity = np.zeros(new_cap, IDENTITY_DTYPE)
        self._valid = False
        return True

    def invalidate_upload(self) -> None:
        """Force all lifecycle streams to be repacked on the next upload."""

        self._valid = False

    def _write_ranges(self, name, buffer, staging, changed, stride) -> None:
        written = 0
        for start, stop in _runs(changed):
            payload = staging[start:stop]
            self._device.queue.write_buffer(buffer, int(start) * stride, payload)
            written += payload.nbytes
        if written:
            self.uploaded_streams[name] = written
            self.uploaded_bytes += written

    def upload(self, scene: RenderScene, reflection_info: np.ndarray | None = None) -> None:
        n = scene.count
        grew = self._ensure_capacity(max(n, 1))
        self.count = n
        self.uploaded_bytes = 0
        self.uploaded_streams = {}
        if n == 0:
            self._last_scene = scene
            self._last_structure = scene.structure_revision
            self._last_pose = scene.pose_revision
            self._last_visual = scene.visual_revision
            self._last_identity = scene.identity_revision
            self._valid = True
            return

        full = (
            grew
            or not self._valid
            or scene is not self._last_scene
            or scene.structure_revision != self._last_structure
        )

        pose_dirty = full or not scene.pose_revision or scene.pose_revision != self._last_pose
        if pose_dirty:
            source = scene.transforms.transpose(0, 2, 1)
            changed = (
                np.ones(n, bool) if full else np.any(self._pose[:n]["model"] != source, axis=(1, 2))
            )
            self._pose[:n]["model"] = source
            assert self.pose_buffer is not None
            self._write_ranges("pose", self.pose_buffer, self._pose, changed, POSE_STRIDE)

        visual_dirty = (
            full or not scene.visual_revision or scene.visual_revision != self._last_visual
        )
        if visual_dirty:
            changed = np.ones(n, bool) if full else np.zeros(n, bool)
            for field, source in (
                ("color", scene.colors),
                ("material", scene.material),
                ("texcoef", scene.tex_coef),
                ("cubecoef", scene.cube_coef),
            ):
                if not full:
                    changed |= np.any(self._visual[:n][field] != source, axis=1)
                self._visual[:n][field] = source
            assert self.visual_buffer is not None
            self._write_ranges("visual", self.visual_buffer, self._visual, changed, VISUAL_STRIDE)

        reflect = np.asarray(reflection_info, np.uint32) if reflection_info is not None else None
        if reflect is None or reflect.size != n:
            reflect = np.zeros(n, np.uint32)
        else:
            reflect = reflect.reshape(n)
        identity_dirty = (
            full
            or not scene.identity_revision
            or scene.identity_revision != self._last_identity
            or not np.array_equal(self._identity[:n]["reflection_info"], reflect)
        )
        if identity_dirty:
            changed = np.ones(n, bool) if full else self._identity[:n]["reflection_info"] != reflect
            if not full and (
                not scene.identity_revision or scene.identity_revision != self._last_identity
            ):
                changed |= self._identity[:n]["object_id"] != scene.object_id
                segmentation = scene.segmentation if scene.segmentation.shape == (n, 2) else -1
                changed |= np.any(self._identity[:n]["segmentation"] != segmentation, axis=1)
            self._identity[:n]["object_id"] = scene.object_id
            self._identity[:n]["reflection_info"] = reflect
            self._identity[:n]["segmentation"] = (
                scene.segmentation if scene.segmentation.shape == (n, 2) else -1
            )
            assert self.identity_buffer is not None
            self._write_ranges(
                "identity", self.identity_buffer, self._identity, changed, IDENTITY_STRIDE
            )

        self._last_structure = scene.structure_revision
        self._last_scene = scene
        self._last_pose = scene.pose_revision
        self._last_visual = scene.visual_revision
        self._last_identity = scene.identity_revision
        self._valid = True

    def bindings(self) -> tuple[tuple[wgpu.GPUBuffer, int], ...]:
        assert self.pose_buffer is not None
        assert self.visual_buffer is not None
        assert self.identity_buffer is not None
        return (
            (self.pose_buffer, self.capacity * POSE_STRIDE),
            (self.visual_buffer, self.capacity * VISUAL_STRIDE),
            (self.identity_buffer, self.capacity * IDENTITY_STRIDE),
        )

    def release(self) -> None:
        for buffer in (self.pose_buffer, self.visual_buffer, self.identity_buffer):
            if buffer is not None:
                buffer.destroy()
        self.pose_buffer = None
        self.visual_buffer = None
        self.identity_buffer = None
        self.capacity = 0
        self.count = 0
        self._pose = np.zeros(0, POSE_DTYPE)
        self._visual = np.zeros(0, VISUAL_DTYPE)
        self._identity = np.zeros(0, IDENTITY_DTYPE)
        self._valid = False
        self._last_scene = None
        self.uploaded_bytes = 0
        self.uploaded_streams = {}
