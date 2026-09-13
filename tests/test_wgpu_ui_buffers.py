"""Exact merged uploads, bounded allocations and recovery before native copies."""

from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

pytest.importorskip("wgpu")
from mojive.ui.wgpu_buffers import ImguiBuffers


class Vector:
    def __init__(self, data, count):
        self.data, self.count = data, count

    def __len__(self):
        return self.count

    def data_address(self):
        return self.data.ctypes.data


class DrawList:
    def __init__(self, color, index_size, count=3):
        self.vtx_buffer = Vector(np.full(count * imgui.VERTEX_SIZE, color, np.uint8), count)
        self.idx_buffer = Vector(np.arange(count, dtype=f"uint{index_size * 8}"), count)


def frame(lists):
    return SimpleNamespace(
        cmd_lists=lists,
        total_vtx_count=sum(len(item.vtx_buffer) for item in lists),
        total_idx_count=sum(len(item.idx_buffer) for item in lists),
    )


class Device:
    def __init__(self):
        self.limits = {"max-buffer-size": 1 << 26}
        self.created = []
        self.writes = []
        self.queue = SimpleNamespace(write_buffer=self.write)
        self.fail_at = None

    def create_buffer(self, *, size, usage):
        if len(self.created) == self.fail_at:
            raise RuntimeError("allocation failed")
        buffer = SimpleNamespace(size=size, destroyed=False)
        buffer.destroy = lambda: setattr(buffer, "destroyed", True)
        self.created.append(buffer)
        return buffer

    def write(self, buffer, offset, data):
        assert len(data) % 4 == 0 and offset == 0
        assert len(data) <= buffer.size and not buffer.destroyed
        self.writes.append((buffer, bytes(data)))


@pytest.mark.parametrize("index_size", (2, 4))
def test_upload_copies_all_lists_once_with_offsets_and_padding(monkeypatch, index_size):
    monkeypatch.setattr(imgui, "INDEX_SIZE", index_size)
    lists = [DrawList(40, index_size), DrawList(80, index_size), DrawList(120, index_size)]
    device = Device()
    buffers = ImguiBuffers(device)
    offsets = buffers.upload(frame(lists))
    assert [offsets[item] for item in lists] == [(0, 0), (3, 3), (6, 6)]
    assert len(device.writes) == 2
    assert device.writes[0][1] == b"".join(item.vtx_buffer.data.tobytes() for item in lists)
    indices = b"".join(item.idx_buffer.data.tobytes() for item in lists)
    assert device.writes[1][1] == indices + b"\0" * (-len(indices) % 4)
    allocations = buffers.allocations
    buffers.upload(frame(lists[::-1]))
    assert buffers.allocations == allocations == 2
    buffers.close()
    assert all(buffer.destroyed for buffer in device.created)


def test_inconsistent_totals_fail_before_raw_memory_copy():
    device = Device()
    buffers = ImguiBuffers(device)
    data = frame([DrawList(1, imgui.INDEX_SIZE)])
    data.total_vtx_count -= 1
    with pytest.raises(ValueError, match="do not match"):
        buffers.upload(data)
    assert not device.created and not device.writes


def test_budget_failure_preserves_existing_buffers_and_recovers():
    device = Device()
    buffers = ImguiBuffers(device, max_bytes=128)
    data = frame([DrawList(1, imgui.INDEX_SIZE)])
    buffers.upload(data)
    old = tuple(device.created)
    with pytest.raises(ValueError, match="budget"):
        buffers.upload(frame([DrawList(2, imgui.INDEX_SIZE, count=100)]))
    assert not any(buffer.destroyed for buffer in old)
    buffers.upload(data)
    assert tuple(device.created) == old


def test_partial_gpu_allocation_does_not_replace_live_buffers():
    device = Device()
    buffers = ImguiBuffers(device)
    data = frame([DrawList(1, imgui.INDEX_SIZE)])
    buffers.upload(data)
    old = buffers.vertices, buffers.indices
    device.fail_at = 3
    with pytest.raises(RuntimeError, match="allocation failed"):
        buffers.upload(frame([DrawList(2, imgui.INDEX_SIZE, count=10000)]))
    assert (buffers.vertices, buffers.indices) == old
    assert not any(buffer.destroyed for buffer in old)
    assert device.created[2].destroyed
    buffers.upload(data)


def test_empty_frame_needs_no_buffers():
    device = Device()
    buffers = ImguiBuffers(device)
    assert buffers.upload(frame([])) == {}
    assert not device.created and not device.writes
