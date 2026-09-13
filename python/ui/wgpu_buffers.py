"""Bounded, reusable staging for one merged ImGui vertex/index upload."""

from __future__ import annotations

import ctypes

import numpy as np
import wgpu
from imgui_bundle import imgui


class ImguiBuffers:
    def __init__(self, device, *, max_bytes=64 * 1024 * 1024):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 4:
            raise ValueError("UI buffer budget must be an integer of at least four bytes")
        self.device, self.max_bytes = device, max_bytes
        self.vertices = self.indices = None
        self.vertex_data = np.empty(0, np.uint8)
        self.index_data = np.empty(0, np.uint8)
        self.allocations = self.upload_bytes = 0

    def _reserve(self, vertex_bytes, index_bytes):
        required = (vertex_bytes, index_bytes)
        current = (len(self.vertex_data), len(self.index_data))
        capacity = tuple(max(old, needed) for old, needed in zip(current, required, strict=True))
        limit = self.device.limits["max-buffer-size"]
        if sum(capacity) > self.max_bytes or max(capacity) > limit:
            raise ValueError("UI vertex/index buffer budget exceeded")
        grown = tuple(
            old if old >= needed else min(limit, max(4096, old * 2, needed))
            for old, needed in zip(current, required, strict=True)
        )
        if sum(grown) <= self.max_bytes:
            capacity = grown
        pending = []
        try:
            for index, (old, size) in enumerate(zip(current, capacity, strict=True)):
                if old != size:
                    data = np.empty(size, np.uint8)
                    usage = wgpu.BufferUsage.INDEX if index else wgpu.BufferUsage.VERTEX
                    buffer = self.device.create_buffer(
                        size=size, usage=usage | wgpu.BufferUsage.COPY_DST
                    )
                    pending.append((index, data, buffer))
        except BaseException:
            for _, _, buffer in pending:
                buffer.destroy()
            raise
        for index, data, buffer in pending:
            old = self.indices if index else self.vertices
            if old is not None:
                old.destroy()
            if index:
                self.index_data, self.indices = data, buffer
            else:
                self.vertex_data, self.vertices = data, buffer
            self.allocations += 1

    def upload(self, draw_data):
        """Copy each source once; return offsets into the two shared GPU buffers."""
        batches = [
            (draw_list, len(draw_list.vtx_buffer), len(draw_list.idx_buffer))
            for draw_list in draw_data.cmd_lists
        ]
        if (sum(nv for _, nv, _ in batches), sum(ni for _, _, ni in batches)) != (
            draw_data.total_vtx_count,
            draw_data.total_idx_count,
        ):
            raise ValueError("ImGui draw counts do not match its buffers")
        vertex_bytes = draw_data.total_vtx_count * imgui.VERTEX_SIZE
        index_bytes = draw_data.total_idx_count * imgui.INDEX_SIZE
        # WGPU queue writes require four-byte sizes even for a single uint16 triangle.
        padded_indices = (index_bytes + 3) // 4 * 4
        self._reserve(vertex_bytes, padded_indices)
        vo = io = 0
        offsets = {}
        for draw_list, nv, ni in batches:
            offsets[draw_list] = (vo, io)
            if nv:
                ctypes.memmove(
                    self.vertex_data.ctypes.data + vo * imgui.VERTEX_SIZE,
                    draw_list.vtx_buffer.data_address(),
                    nv * imgui.VERTEX_SIZE,
                )
            if ni:
                ctypes.memmove(
                    self.index_data.ctypes.data + io * imgui.INDEX_SIZE,
                    draw_list.idx_buffer.data_address(),
                    ni * imgui.INDEX_SIZE,
                )
            vo += nv
            io += ni
        self.index_data[index_bytes:padded_indices] = 0
        for buffer, data, count in (
            (self.vertices, self.vertex_data, vertex_bytes),
            (self.indices, self.index_data, padded_indices),
        ):
            if count:
                self.device.queue.write_buffer(buffer, 0, data[:count])
                self.upload_bytes += count
        return offsets

    def close(self):
        for buffer in (self.vertices, self.indices):
            if buffer is not None:
                buffer.destroy()
        self.vertices = self.indices = None
        self.vertex_data = np.empty(0, np.uint8)
        self.index_data = np.empty(0, np.uint8)
