"""Snapshot recordings preserve the exact packets consumed by RemoteSceneAdapter."""

from __future__ import annotations

from dataclasses import replace

import pytest

from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.recording import (
    SNAPSHOT_PREFIX,
    SnapshotWriter,
    read_snapshots,
)
from forge_viewer.remote import RemoteFrame, RemoteStructure, snapshot_structure
from forge_viewer.scene import Scene
from forge_viewer.session import Session

pytestmark = pytest.mark.integration


def test_snapshot_stream_round_trips_structure_frame_and_debug_commands(tmp_path):
    scene = Scene()
    scene.box(name="recorded box")
    session = Session(StaticSceneAdapter(scene))
    structure = snapshot_structure(session)
    frame = RemoteFrame(
        4,
        replace(scene.frame, time=1.25, step=17),
        ({"op": "text", "id": "time", "text": "1.25 s"},),
    )
    path = tmp_path / "trace.fvs"

    with SnapshotWriter(path) as writer:
        writer.write(structure)
        writer.write(frame)
        assert writer.packets == 2

    packets = list(read_snapshots(path))
    assert isinstance(packets[0], RemoteStructure)
    assert packets[0].source.nodes[1].name == "recorded box"
    assert isinstance(packets[1], RemoteFrame)
    assert packets[1].frame.step == 17
    assert packets[1].debug_commands[0]["text"] == "1.25 s"
    session.release()


def test_snapshot_reader_rejects_an_unrelated_file_before_unpickling(tmp_path):
    path = tmp_path / "not-a-trace.fvs"
    path.write_bytes(b"not a forge file")

    with pytest.raises(ValueError, match="not a forge snapshot"):
        list(read_snapshots(path))


def test_snapshot_reader_rejects_future_and_truncated_recordings(tmp_path):
    future = tmp_path / "future.fvs"
    future.write_bytes(SNAPSHOT_PREFIX + b"\x7f")
    with pytest.raises(ValueError, match="unsupported forge snapshot version: 127"):
        list(read_snapshots(future))

    truncated = tmp_path / "truncated.fvs"
    with SnapshotWriter(truncated) as writer:
        writer.write({"frame": 1})
    truncated.write_bytes(truncated.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated forge snapshot packet"):
        list(read_snapshots(truncated))


def test_snapshot_reader_rejects_version_one_recordings(tmp_path):
    legacy = tmp_path / "legacy.fvs"
    legacy.write_bytes(SNAPSHOT_PREFIX + b"\x01")
    with pytest.raises(ValueError, match="unsupported forge snapshot version: 1"):
        list(read_snapshots(legacy))
