"""The recording and replay CLI preserve stream identity and read-only capabilities."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mojive import cli
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.recording import SnapshotWriter, read_snapshots
from mojive.remote import RemoteFrame, snapshot_structure
from mojive.scene import Scene
from mojive.session import Session

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("speed", [1.0, 0.004])
def test_replay_preserves_requested_speed_and_read_only_capabilities(tmp_path, monkeypatch, speed):
    clock = [0.0]
    times = []
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    session = Session(StaticSceneAdapter(Scene()))
    structure = snapshot_structure(session)
    structure = replace(
        structure,
        caps=replace(structure.caps, model_properties=True, keyframes=True),
    )
    path = tmp_path / "replay.fvs"
    with SnapshotWriter(path) as writer:
        writer.write(structure)
        writer.write(RemoteFrame(1, session.frame, structure_revision=structure.structure_revision))
        writer.write(
            RemoteFrame(
                2,
                replace(session.frame, time=0.04),
                structure_revision=structure.structure_revision,
            )
        )
    session.release()
    published = []

    class Publisher:
        def __init__(self, *args):
            pass

        def publish_structure(self, structure):
            published.append(structure)

        def publish_frame(self, *args):
            times.append(clock[0])

        def pump_commands(self, handler):
            pass

        def close(self):
            pass

    monkeypatch.setattr("mojive.remote.SnapshotPublisher", Publisher)
    args = SimpleNamespace(snapshot=path, host="localhost", port=0, speed=speed, loop=False)
    assert cli.cmd_replay(args) == 0
    assert times == pytest.approx([0.0, 0.04 / speed])
    caps = published[0].caps
    assert caps.name.startswith("replay:")
    assert not any(
        getattr(caps, name)
        for name in (
            "simulation",
            "write_pose",
            "write_qpos",
            "perturb",
            "asset_loading",
            "reload",
            "scene_authoring",
            "scene_files",
            "edit_history",
            "model_composition",
            "topology_editing",
            "model_properties",
            "model_assets",
            "keyframes",
        )
    )


@pytest.mark.parametrize("hz", [1000.0, 0.25])
def test_serve_records_the_frame_structure_revision_at_requested_rate(tmp_path, monkeypatch, hz):
    clock = [0.0]
    times = []
    command_times = []
    command_arrival = min(0.1, 0.5 / hz)
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    path = tmp_path / "serve.fvs"

    class Publisher:
        def __init__(self, *args):
            self.count = 0

        def pump_commands(self, handler):
            if not command_times and clock[0] >= command_arrival:
                command_times.append(clock[0])
                assert not handler({"op": "unknown"}).ok

        def publish_structure(self, structure):
            pass

        def publish_frame(self, frame):
            self.count += 1
            times.append(clock[0])
            if self.count == 2:
                raise StopIteration("stop recording")
            return self.count

        def close(self):
            pass

    monkeypatch.setattr("mojive.remote.SnapshotPublisher", Publisher)
    monkeypatch.setattr("mojive.backends.make_adapter", lambda *args: ToyPhysicsAdapter())
    monkeypatch.setattr(cli, "_resolve", lambda asset: tmp_path / "toy.xml")
    args = SimpleNamespace(
        asset=None,
        backend="toy",
        host="localhost",
        port=0,
        record_snapshot=path,
        paused=True,
        hz=hz,
    )
    with pytest.raises(StopIteration, match="stop recording"):
        cli.cmd_serve(args)
    structure, frame = list(read_snapshots(path))
    assert frame.structure_revision == structure.structure_revision
    assert times == pytest.approx([0.0, 1.0 / hz])
    assert len(command_times) == 1
    assert command_arrival <= command_times[0] <= command_arrival + 0.011
