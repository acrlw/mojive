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
        self.writes: list[tuple[_Buffer, int, int]] = []

    def write_buffer(self, buffer, offset, payload) -> None:
        self.writes.append((buffer, offset, payload.nbytes))


class _Device:
    def __init__(self) -> None:
        self.queue = _Queue()

    @staticmethod
    def create_buffer(*, size: int, usage) -> _Buffer:
        del usage
        return _Buffer(size)


def test_instance_upload_reuses_split_staging_until_capacity_grows() -> None:
    count = 128
    scene = RenderScene(
        count=count,
        transforms=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
        colors=np.ones((count, 4), np.float32),
        material=np.ones((count, 4), np.float32),
        tex_coef=np.ones((count, 4), np.float32),
        cube_coef=np.ones((count, 4), np.float32),
        object_id=np.arange(count, dtype=np.uint32),
        segmentation=np.column_stack(
            (np.arange(count, dtype=np.int32), np.full(count, 5, np.int32))
        ),
    )
    device = _Device()
    store = InstanceStore(device)

    store.upload(scene)
    staging = (store._pose, store._visual, store._identity)
    writes = len(device.queue.writes)
    store.upload(scene)

    assert all(
        current is before
        for current, before in zip(
            (store._pose, store._visual, store._identity), staging, strict=True
        )
    )
    assert len(device.queue.writes) == writes
    assert store._identity[0]["object_id"] == 0
    assert store._identity[count - 1]["object_id"] == count - 1
    assert tuple(store._identity[count - 1]["segmentation"]) == (count - 1, 5)


def test_instance_upload_skips_an_unchanged_revision() -> None:
    count = 2
    scene = RenderScene(
        count=count,
        structure_revision=1,
        pose_revision=1,
        visual_revision=1,
        identity_revision=1,
        transforms=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
        colors=np.ones((count, 4), np.float32),
        material=np.ones((count, 4), np.float32),
        tex_coef=np.ones((count, 4), np.float32),
        cube_coef=np.ones((count, 4), np.float32),
        object_id=np.arange(count, dtype=np.uint32),
        segmentation=np.full((count, 2), -1, np.int32),
    )
    device = _Device()
    store = InstanceStore(device)

    store.upload(scene)
    assert store.uploaded_bytes > 0
    writes = len(device.queue.writes)
    store.upload(scene)

    assert store.uploaded_bytes == 0
    assert len(device.queue.writes) == writes


def test_pose_revision_uploads_only_changed_transform_range() -> None:
    count = 4
    scene = RenderScene(
        count=count,
        structure_revision=1,
        pose_revision=1,
        visual_revision=1,
        identity_revision=1,
        transforms=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
        colors=np.ones((count, 4), np.float32),
        material=np.ones((count, 4), np.float32),
        tex_coef=np.ones((count, 4), np.float32),
        cube_coef=np.ones((count, 4), np.float32),
        object_id=np.arange(count, dtype=np.uint32),
        segmentation=np.full((count, 2), -1, np.int32),
    )
    device = _Device()
    store = InstanceStore(device)
    store.upload(scene)

    scene.transforms[2, 0, 3] = 3.0
    scene.pose_revision += 1
    store.upload(scene)

    assert store.uploaded_streams == {"pose": 64}
    assert store.uploaded_bytes == 64
    assert device.queue.writes[-1][1:] == (2 * 64, 64)


def test_reflection_metadata_is_a_small_identity_stream_update() -> None:
    count = 3
    scene = RenderScene(
        count=count,
        structure_revision=1,
        pose_revision=1,
        visual_revision=1,
        identity_revision=1,
        transforms=np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(),
        colors=np.ones((count, 4), np.float32),
        material=np.ones((count, 4), np.float32),
        tex_coef=np.ones((count, 4), np.float32),
        cube_coef=np.ones((count, 4), np.float32),
        object_id=np.arange(count, dtype=np.uint32),
        segmentation=np.full((count, 2), -1, np.int32),
    )
    device = _Device()
    store = InstanceStore(device)
    store.upload(scene)

    info = np.zeros(count, np.uint32)
    info[1] = 9
    store.upload(scene, info)

    assert store.uploaded_streams == {"identity": 16}
    assert store.uploaded_bytes == 16
    assert device.queue.writes[-1][1:] == (16, 16)


def test_new_scene_object_invalidates_equal_numeric_revisions() -> None:
    def make_scene(color: float) -> RenderScene:
        return RenderScene(
            count=1,
            structure_revision=1,
            pose_revision=1,
            visual_revision=1,
            identity_revision=1,
            transforms=np.eye(4, dtype=np.float32)[None],
            colors=np.full((1, 4), color, np.float32),
            material=np.ones((1, 4), np.float32),
            tex_coef=np.ones((1, 4), np.float32),
            cube_coef=np.ones((1, 4), np.float32),
            object_id=np.ones(1, np.uint32),
            segmentation=np.full((1, 2), -1, np.int32),
        )

    store = InstanceStore(_Device())
    store.upload(make_scene(0.2))
    store.upload(make_scene(0.8))

    assert store.uploaded_bytes == 144
    assert float(store._visual[0]["color"][0]) == pytest.approx(0.8)
