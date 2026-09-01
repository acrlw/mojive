"""CPU-side WebGPU instance packing tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("wgpu")

from mojive.render.scene import RenderScene
from mojive.render.webgpu.instances import InstanceStore


class _Buffer:
    def __init__(self, size: int) -> None:
        self.size = size

    def destroy(self) -> None:
        pass


class _Queue:
    def __init__(self) -> None:
        self.payload_ids: list[int] = []

    def write_buffer(self, _buffer, _offset, payload) -> None:
        self.payload_ids.append(id(payload.base))


class _Device:
    def __init__(self) -> None:
        self.queue = _Queue()

    @staticmethod
    def create_buffer(*, size: int, usage) -> _Buffer:
        del usage
        return _Buffer(size)


def test_instance_upload_reuses_staging_until_capacity_grows() -> None:
    count = 128
    scene = RenderScene(
        count=count,
        transforms=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
        colors=np.ones((count, 4), np.float32),
        material=np.ones((count, 4), np.float32),
        tex_coef=np.ones((count, 4), np.float32),
        cube_coef=np.ones((count, 4), np.float32),
        object_id=np.arange(count, dtype=np.uint32),
    )
    device = _Device()
    store = InstanceStore(device)

    store.upload(scene)
    staging = store._staging
    store.upload(scene)

    assert store._staging is staging
    assert device.queue.payload_ids[-2:] == [id(staging), id(staging)]
    assert store._staging[0]["object_id"] == 0
    assert store._staging[-1]["object_id"] == count - 1
