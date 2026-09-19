"""Compact recording validation and local FK use the recorded model and world order."""

import json

import numpy as np
import pytest

from mojive.adapters.joint_replay import JointReplayAdapter
from mojive.tools.g1_replay import generate

pytestmark = pytest.mark.physics


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr("mojive.tools.g1_worlds.JOINT_NAMES", ("hinge",))
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><geom type="plane" size="0 0 1"/>'
        '<body pos="0 0 1"><freejoint name="floating_base_joint"/>'
        '<geom type="sphere" size="0.1"/><body pos="0 0 0.2">'
        '<joint name="hinge" axis="0 1 0"/><geom type="capsule" size="0.1 0.2"/>'
        "</body></body></worldbody></mujoco>"
    )
    archive = tmp_path / "archive"
    generate(model, archive, 4, 6, 30)
    return archive


def test_joint_recording_replays_actual_qpos_and_reuses_buffers(archive):
    import mujoco

    replay = JointReplayAdapter(archive, realtime=False)
    assert isinstance(replay.qpos, np.memmap)
    assert replay.qpos.shape == (6, 4, 8)
    frame = replay.update(2, 0.25)
    original = frame.geom_xpos.copy()
    data = mujoco.MjData(replay.model)
    n = len(replay.worlds.pose_indices)
    for i in range(4):
        data.qpos[:] = replay.qpos[2, i]
        mujoco.mj_kinematics(replay.model, data)
        np.testing.assert_allclose(
            frame.geom_xpos[i * n : (i + 1) * n],
            data.geom_xpos[replay.worlds.pose_indices] + replay.worlds.offsets[i],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            frame.geom_xmat[i * n : (i + 1) * n],
            data.geom_xmat[replay.worlds.pose_indices].reshape(-1, 3, 3),
            atol=1e-7,
        )
    assert replay.update(3, 0.5) is frame
    assert frame.time == 0.5 and frame.step == 3
    assert not np.array_equal(frame.geom_xpos, original)
    replay.update(8, 0.75)
    np.testing.assert_array_equal(frame.geom_xpos, original)
    subset = JointReplayAdapter(archive, worlds=2)
    assert subset.qpos.shape == (6, 4, 8)
    assert subset.metadata["shape"] == [6, 4, 8]
    assert subset.worlds.world_count == 2
    bounded = JointReplayAdapter(archive, max_worlds=1)
    assert bounded.world_selection().world_ids == (0,)
    with pytest.raises(ValueError, match="Choose between"):
        JointReplayAdapter(archive, worlds=2, max_worlds=1)
    subset_frame = subset.update(2, 0.25)
    np.testing.assert_allclose(
        subset_frame.geom_xpos[:n] - subset.worlds.offsets[0],
        original[:n] - replay.worlds.offsets[0],
        atol=1e-6,
    )
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["joint_names"] = ["wrong"]
    (archive / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="joint layout"):
        JointReplayAdapter(archive)
    manifest["model_sha256"] = "wrong"
    (archive / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        JointReplayAdapter(archive)


def test_selection_filters_before_fk_and_keeps_original_identity(archive, monkeypatch):
    import mujoco

    replay = JointReplayAdapter(archive, world_ids=(3, 1), max_worlds=2, realtime=False)
    calls = []
    original = mujoco.mj_kinematics

    def counted(model, data):
        calls.append(data.qpos.copy())
        original(model, data)

    monkeypatch.setattr(mujoco, "mj_kinematics", counted)
    frame = replay.update(2, 0.25)
    np.testing.assert_array_equal(calls, replay.qpos[2, [3, 1]])
    assert replay.positions.shape[0] == 2
    assert replay.world_selection().world_ids == (3, 1)
    assert [n.name for n in replay.nodes()[1:]] == ["World 3", "World 1"]
    assert replay.nodes()[0].children == [4, 2]
    assert set(replay.scene_source().geom_object_id) == {0, 4, 2}
    assert set(replay.scene_source().geom_segmentation[:, 0]) == {-1, 3, 1}
    before = frame.geom_xpos.copy()
    revision = replay.structure_revision
    source = replay.scene_source()
    assert not replay.set_world_selection((3, 1))
    assert len(calls) == 2
    for ids in ((), (0, 0), (0, 1, 2), (-1,), (4,), (True,), (1.5,)):
        with pytest.raises(ValueError):
            replay.set_world_selection(ids)
        assert replay.scene_source() is source
        assert replay.structure_revision == revision
        np.testing.assert_array_equal(frame.geom_xpos, before)
    assert replay.set_world_selection((1,))
    assert len(calls) == 3
    assert replay.scene_source().meshes is source.meshes
    assert replay.structure_revision == revision + 1


def test_corrupt_unselected_world_is_not_read_or_evaluated(archive):
    values = np.load(archive / "qpos.npy", mmap_mode="r+")
    values[:, 3] = np.nan
    values.flush()
    replay = JointReplayAdapter(archive, world_ids=(0,), realtime=False)
    frame = replay.update(2, 0.1)
    before = frame.geom_xpos.copy()
    source = replay.scene_source()
    with pytest.raises(ValueError, match="finite"):
        replay.set_world_selection((3,))
    assert replay.world_selection().world_ids == (0,)
    assert replay.scene_source() is source
    np.testing.assert_array_equal(frame.geom_xpos, before)


def test_initial_world_count_cannot_bypass_preview_limit(archive):
    with pytest.raises(ValueError, match="between 1 and 2"):
        JointReplayAdapter(archive, worlds=4, max_worlds=2)


def test_realtime_only_evaluates_new_samples(archive, monkeypatch):
    from mojive.adapters.base import FrameNeeds

    now = [100.0]
    monkeypatch.setattr("mojive.adapters.joint_replay.time.perf_counter", lambda: now[0])
    replay = JointReplayAdapter(archive, worlds=1)
    frame = replay.frame(FrameNeeds())
    before = frame.geom_xpos.copy()
    now[0] += 0.01
    assert replay.frame(FrameNeeds()) is frame
    np.testing.assert_array_equal(frame.geom_xpos, before)
    assert frame.step == 0
    now[0] += 2 / 30
    assert replay.frame(FrameNeeds()).step == 2
    assert not np.array_equal(frame.geom_xpos, before)


def test_world_selection_rpc_preserves_camera_and_document(archive):
    from mojive import commands as cmd
    from mojive.control.operations import OPERATIONS, document_state
    from mojive.control.rpc import ControlService, RpcError
    from mojive.control.schema import Validator

    replay = JointReplayAdapter(archive, world_ids=(3, 1), max_worlds=2, realtime=False)
    service = ControlService(replay)
    try:
        session = service.session
        assert session.submit(cmd.Select(4)).ok
        assert session.submit(cmd.SetVisible(4, False)).ok
        camera = session.camera
        document = document_state(session)
        info = service.dispatch("get_world_selection", {})
        Validator(OPERATIONS["get_world_selection"].output_schema).validate(info)
        assert info == {"total_worlds": 4, "world_ids": [3, 1], "max_worlds": 2}
        result = service.dispatch("set_world_selection", {"world_ids": [1, 3]})
        assert result["ok"]
        assert session.selected_node.name == "World 3"
        assert not session.selected_node.visible
        assert not next(n for n in session.source.nodes if n.object_id == 4).visible
        assert session.camera is camera
        assert document_state(session) == document
        assert not session.can_undo
        for ids in ([0, 1, 2], [1, 1], [4]):
            with pytest.raises(RpcError):
                service.dispatch("set_world_selection", {"world_ids": ids})
            assert session.world_selection.world_ids == (1, 3)
        service.dispatch("set_world_selection", {"world_ids": [1]})
        assert session.selected_node is None
    finally:
        service.close()


def test_pause_seek_speed_and_end_of_clip_do_not_jump_or_run_fk_while_paused(archive, monkeypatch):
    import mujoco

    from mojive.adapters.base import FrameNeeds

    now = [100.0]
    monkeypatch.setattr("mojive.adapters.joint_replay.time.perf_counter", lambda: now[0])
    replay = JointReplayAdapter(archive, worlds=1, paused=True)
    calls = []
    original = mujoco.mj_kinematics

    def counted(model, data):
        calls.append(1)
        original(model, data)

    monkeypatch.setattr(mujoco, "mj_kinematics", counted)
    for _ in range(10):
        now[0] += 10
        replay.frame(FrameNeeds())
    assert not calls
    replay.seek_replay(2)
    assert replay.replay_info().frame_index == 2 and replay.replay_info().paused
    replay.set_replay_playback(paused=False, speed=2, loop=False)
    replay.frame(FrameNeeds())
    now[0] += 1.1 / 30
    assert replay.frame(FrameNeeds()).step == 4
    replay.set_replay_playback(paused=True)
    before = replay.frame(FrameNeeds()).geom_xpos.copy()
    count = len(calls)
    now[0] += 10
    np.testing.assert_array_equal(replay.frame(FrameNeeds()).geom_xpos, before)
    assert len(calls) == count
    replay.set_replay_playback(paused=False)
    replay.frame(FrameNeeds())
    now[0] += 1 / 30
    assert replay.frame(FrameNeeds()).step == 5
    assert replay.replay_info().paused
    replay.set_replay_playback(paused=False)
    assert replay.replay_info().frame_index == 0
    assert not replay.replay_info().paused


def test_corrupt_playback_pauses_at_last_good_pose_and_invalid_seek_is_atomic(archive, monkeypatch):
    from mojive.adapters.base import FrameNeeds

    values = np.load(archive / "qpos.npy", mmap_mode="r+")
    values[1, 0] = np.nan
    values.flush()
    now = [100.0]
    monkeypatch.setattr("mojive.adapters.joint_replay.time.perf_counter", lambda: now[0])
    replay = JointReplayAdapter(archive, world_ids=(0,))
    frame = replay.frame(FrameNeeds())
    before = frame.geom_xpos.copy()
    now[0] += 1.1 / 30
    assert replay.frame(FrameNeeds()) is frame
    assert replay.replay_info().paused and "finite" in replay.replay_info().error
    np.testing.assert_array_equal(frame.geom_xpos, before)
    for index in (-1, 6, 1):
        with pytest.raises(ValueError):
            replay.seek_replay(index)
        np.testing.assert_array_equal(frame.geom_xpos, before)
    replay.seek_replay(2)
    assert not replay.replay_info().error
    assert replay.replay_info().frame_index == 2


def test_rollout_sync_is_manual_bounded_atomic_and_reuses_structure(archive, monkeypatch):
    import threading
    import time

    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.rollout_replay import RolloutReplayAdapter
    from mojive.remote.rollout import RolloutServer, RolloutStore
    from mojive.session import Session

    # Early diagnostic archives omit spacing; the documented default remains compatible.
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest.pop("display_spacing")
    (archive / "manifest.json").write_text(json.dumps(manifest))
    store = RolloutStore.from_archive(archive, max_worlds=2, max_frames=6)
    with RolloutServer(store, port=0) as server:
        adapter = RolloutReplayAdapter(
            f"http://127.0.0.1:{server.address[1]}", world_ids=(3, 0), max_worlds=2, window_frames=4
        )
        session = Session(adapter)

        def finish():
            until = time.monotonic() + 3
            while adapter.rollout_sync_info().pending and time.monotonic() < until:
                session.tick(FrameNeeds())
                time.sleep(0.005)
            assert not adapter.rollout_sync_info().pending

        try:
            assert adapter.replay_info().paused
            source = session.source
            revision = adapter.structure_revision
            camera = session.camera
            assert session.submit(cmd.Select(4)).ok
            assert session.submit(cmd.SetVisible(4, False)).ok
            adapter.seek_replay(2)
            generation = session.structure_generation
            requests = store.window_requests
            poses_sent = store.sent_pose_bytes
            for _ in range(30):
                session.tick(FrameNeeds())
            assert store.window_requests == requests
            adapter.sync_rollout()
            finish()
            assert store.sent_pose_bytes == poses_sent
            assert session.source is source and session.structure_generation == generation
            assert adapter.replay_info().frame_index == 2
            values = np.array(np.load(archive / "qpos.npy"))
            values[..., 0] += 0.3
            store.publish(values, start_step=100)
            for _ in range(10):
                session.tick(FrameNeeds())
            assert store.window_requests == requests + 1
            assert adapter.replay_info().frame_index == 2
            gate = threading.Event()
            entered = threading.Event()
            fetch = adapter._client.fetch

            def slow(*args, **kwargs):
                entered.set()
                assert gate.wait(3)
                return fetch(*args, **kwargs)

            monkeypatch.setattr(adapter._client, "fetch", slow)
            adapter.sync_rollout()
            assert entered.wait(1)
            with pytest.raises(ValueError, match="already"):
                adapter.sync_rollout()
            for _ in range(10):
                session.tick(FrameNeeds())
            assert adapter.rollout_sync_info().pending
            with pytest.raises(ValueError, match="pending rollout sync"):
                adapter.set_world_selection((0,))
            assert adapter.replay_info().frame_index == 2
            gate.set()
            finish()
            assert not adapter.rollout_sync_info().error
            assert adapter.replay_info().paused and adapter.replay_info().frame_index == 0
            assert adapter.rollout_sync_info().start_step == 102
            assert adapter.structure_revision == revision and session.source is source
            assert session.selected_node.name == "World 3" and not session.selected_node.visible
            adapter.sync_rollout((0, 3))
            finish()
            assert session.world_selection.world_ids == (0, 3)
            assert session.selected_node.name == "World 3" and not session.selected_node.visible
            assert session.camera is camera
            before = session.frame.geom_xpos.copy()
            old_source = session.source

            def fail(*args, **kwargs):
                raise TimeoutError("preview source unavailable")

            monkeypatch.setattr(adapter._client, "fetch", fail)
            adapter.sync_rollout((1,))
            finish()
            assert "preview source unavailable" in adapter.rollout_sync_info().error
            assert session.source is old_source
            assert session.world_selection.world_ids == (0, 3)
            np.testing.assert_array_equal(session.frame.geom_xpos, before)
        finally:
            session.release()


def test_rollout_rejects_changed_model_without_installing_or_rebuilding(archive, monkeypatch):
    import time
    from dataclasses import replace

    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.rollout_replay import RolloutReplayAdapter
    from mojive.remote.rollout import RolloutServer, RolloutStore

    store = RolloutStore.from_archive(archive)
    with RolloutServer(store, port=0) as server:
        adapter = RolloutReplayAdapter(
            f"http://127.0.0.1:{server.address[1]}", worlds=1, window_frames=4
        )
        try:
            original = adapter.scene_source()
            good = adapter._client.fetch((0,), 4)
            bad = replace(good, metadata={**good.metadata, "model_sha256": "different"})
            monkeypatch.setattr(adapter._client, "fetch", lambda *args, **kwargs: bad)
            adapter.sync_rollout()
            until = time.monotonic() + 2
            while adapter.rollout_sync_info().pending and time.monotonic() < until:
                adapter.prepare_frame(FrameNeeds())
                time.sleep(0.005)
            assert "model_sha256 changed" in adapter.rollout_sync_info().error
            assert adapter.scene_source() is original
        finally:
            adapter.release()
