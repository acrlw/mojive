"""Wire corner streams stay mesh-owned, lazy, and coherent after deformation."""

from types import SimpleNamespace

import numpy as np
import pytest

from mojive.render.opengl.instances import GpuMesh


class Buffer:
    def __init__(self, data):
        self.data = bytes(data)
        self.writes = 0
        self.released = False
        self.fail = False

    def write(self, data):
        if self.fail:
            raise RuntimeError("upload failed")
        self.data = bytes(data)
        self.writes += 1

    def release(self):
        self.released = True


def mesh_fixture():
    buffers = []

    def allocate(data):
        buffer = Buffer(data)
        buffers.append(buffer)
        return buffer

    ctx = SimpleNamespace(buffer=allocate)
    positions = np.arange(12, dtype=np.float32).reshape(4, 3)
    indices = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    mesh = GpuMesh(ctx, positions, np.zeros_like(positions), np.zeros((4, 2)), indices)
    return ctx, mesh, positions, indices, buffers


def assert_positions(buffer, positions):
    np.testing.assert_array_equal(np.frombuffer(buffer.data, np.float32).reshape(-1, 3), positions)


def test_wire_stream_is_lazy_and_reused_without_frame_uploads():
    ctx, mesh, positions, indices, buffers = mesh_fixture()
    assert len(buffers) == 2
    mesh.update(positions * 2, np.zeros_like(positions))
    assert len(buffers) == 2
    wire = mesh.wire_buffer(ctx)
    assert len(buffers) == 3
    assert_positions(wire, (positions * 2)[indices])
    for _ in range(100):
        assert mesh.wire_buffer(ctx) is wire
    assert wire.writes == 0
    mesh.release()
    assert all(buffer.released for buffer in buffers)


def test_multiple_deformations_upload_only_latest_positions_when_drawn():
    ctx, mesh, positions, indices, buffers = mesh_fixture()
    wire = mesh.wire_buffer(ctx)
    storage = mesh._wire_vertices
    for offset in range(1, 4):
        mesh.update(positions + offset, np.zeros_like(positions))
    assert wire.writes == 0
    assert mesh.wire_buffer(ctx) is wire
    assert mesh._wire_vertices is storage
    assert wire.writes == 1
    assert_positions(wire, (positions + 3)[indices])
    assert len(buffers) == 3
    mesh.release()


def test_lazy_wire_stream_owns_the_uploaded_topology():
    ctx, mesh, positions, indices, _ = mesh_fixture()
    original = indices.copy()
    indices[:] = 0
    assert_positions(mesh.wire_buffer(ctx), positions[original])
    mesh.release()


def test_failed_wire_upload_remains_dirty_until_successful_retry():
    ctx, mesh, positions, indices, _ = mesh_fixture()
    wire = mesh.wire_buffer(ctx)
    mesh.update(positions + 1, np.zeros_like(positions))
    wire.fail = True
    with pytest.raises(RuntimeError, match="upload failed"):
        mesh.wire_buffer(ctx)
    assert_positions(wire, positions[indices])
    wire.fail = False
    assert mesh.wire_buffer(ctx) is wire
    assert_positions(wire, (positions + 1)[indices])
    mesh.release()
