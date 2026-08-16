"""Snapshot recordings preserve the exact packets consumed by RemoteSceneAdapter."""

from __future__ import annotations

from dataclasses import replace

import pytest

from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.recording import SnapshotWriter, read_snapshots
from forge_viewer.remote import RemoteFrame, RemoteStructure, snapshot_structure
from forge_viewer.scene import Scene
from forge_viewer.session import Session


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
