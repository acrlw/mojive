"""Bounded pull transport, immutable publication, and original world identities."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojive.remote.rollout import RolloutClient, RolloutServer, RolloutStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    model = b"small-model-placeholder"
    metadata = {
        "version": 1,
        "dtype": "float32",
        "hz": 30,
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "mujoco_version": "test",
        "joint_names": [],
        "shape": [10, 8, 7],
    }
    value = RolloutStore(model, metadata, max_worlds=3, max_frames=10)
    value.publish(np.arange(10 * 8 * 7, dtype=np.float32).reshape(10, 8, 7), start_step=100)
    return value


def test_windows_filter_worlds_and_time_before_transfer_and_keep_model_separate(store):
    with RolloutServer(store, port=0) as server:
        client = RolloutClient(f"http://127.0.0.1:{server.address[1]}")
        info = client.info()
        assert client.model(info["model_sha256"]) == store.model
        window = client.fetch((7, 1), 4)
        assert window.qpos.shape == (4, 2, 7)
        assert window.metadata["world_ids"] == [7, 1]
        assert window.metadata["total_worlds"] == 8
        assert window.metadata["start_step"] == 106
        np.testing.assert_array_equal(window.qpos, store._snapshot().qpos[-4:, [7, 1]])
        assert not window.qpos.flags.writeable
        assert store.sent_pose_bytes == 4 * 2 * 7 * 4
        assert client.fetch((7, 1), 4, after=info["revision"]) is None
        assert store.sent_pose_bytes == 224
        assert store.window_requests == 2


def test_publication_owns_window_and_failed_publication_is_atomic(store):
    values = np.ones((6, 8, 7), np.float32)
    revision = store.publish(values)
    values[:] = 99
    assert np.all(store.window((2,), 2).qpos == 1)
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        store.publish(values)
    assert store.info()["revision"] == revision
    assert np.all(store.window((2,), 2).qpos == 1)


def test_returned_metadata_cannot_change_published_identity(store):
    info = store.info()
    info["shape"][1] = 100
    info["joint_names"].append("injected")
    window = store.window((0,), 2)
    window.metadata["joint_names"].append("another")
    window.metadata["shape"][2] = 999
    assert store.info()["shape"] == [10, 8, 7]
    assert store.info()["joint_names"] == []
    assert store.window((0,), 2).metadata["shape"] == [2, 1, 7]


def test_preselected_producer_batch_retains_original_ids(store):
    values = np.zeros((6, 2, 7), np.float32)
    values[:, 0] = 7
    values[:, 1] = 2
    store.publish(values, world_ids=(7, 2), start_step=500)
    info = store.info()
    assert info["total_worlds"] == 8 and info["world_ids"] == [7, 2]
    result = store.window((2, 7), 3)
    np.testing.assert_array_equal(result.qpos, values[-3:, [1, 0]])
    with pytest.raises(ValueError, match="absent"):
        store.window((0,), 3)


@pytest.mark.parametrize(
    "ids,frames",
    [((), 2), ((0, 0), 2), ((8,), 2), ((-1,), 2), ((0, 1, 2, 3), 2), ((0,), 1), ((0,), 11)],
)
def test_invalid_windows_are_rejected_without_data_transfer(store, ids, frames):
    with RolloutServer(store, port=0) as server:
        client = RolloutClient(f"http://127.0.0.1:{server.address[1]}")
        with pytest.raises(ValueError):
            client.fetch(ids, frames)
        assert store.sent_pose_bytes == 0


def test_server_and_client_enforce_byte_budgets(store):
    store.max_window_bytes = 100
    with pytest.raises(ValueError, match="byte budget"):
        store.window((0,), 4)
    store.max_window_bytes = 10000
    with RolloutServer(store, port=0) as server:
        client = RolloutClient(f"http://127.0.0.1:{server.address[1]}", max_bytes=50)
        with pytest.raises(ValueError, match="payload size"):
            client.fetch((0,), 4)


def test_concurrent_publication_never_mixes_world_or_frame_generations(store):
    def publish():
        for i in range(10):
            store.publish(np.full((6, 8, 7), i, np.float32), start_step=i * 6)

    store.publish(np.zeros((6, 8, 7), np.float32))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(publish)
        for _ in range(30):
            window = store.window((7, 1), 4)
            assert np.unique(window.qpos).size == 1
            assert window.metadata["start_step"] == int(window.qpos[0, 0, 0]) * 6 + 2
        future.result()


def test_archive_source_is_memory_mapped_and_does_not_scan_unrequested_worlds(tmp_path, store):
    metadata = store.info()
    metadata.pop("revision")
    values = np.zeros(metadata["shape"], np.float32)
    values[:, 7] = np.nan
    (tmp_path / "model.mjb").write_bytes(store.model)
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    np.save(tmp_path / "qpos.npy", values)
    # The resident publication budget must not reject a large read-only mmap.
    mapped = RolloutStore.from_archive(tmp_path, max_source_bytes=16)
    assert isinstance(mapped._snapshot().qpos, np.memmap)
    assert mapped.window((0,), 4).qpos.shape == (4, 1, 7)
    with pytest.raises(ValueError, match="finite"):
        mapped.window((7,), 4)
    with pytest.raises(ValueError, match="source byte budget"):
        mapped.publish(np.zeros_like(values))


@pytest.mark.parametrize("status", [200, 400])
def test_slow_response_body_cannot_extend_download_indefinitely(status):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", "100")
            self.end_headers()
            try:
                for _ in range(25):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

    with HTTPServer(("127.0.0.1", 0), SlowHandler) as server:
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            client = RolloutClient(f"http://127.0.0.1:{server.server_port}", timeout=0.12)
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                client.info()
            assert time.monotonic() - started < 0.4
        finally:
            worker.join(timeout=2)


@pytest.mark.physics
def test_model_factory_publishes_without_archive_or_training_state_changes():
    import mujoco

    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.rollout_replay import RolloutReplayAdapter

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    values = np.broadcast_to(model.qpos0, (4, 2, model.nq)).astype(np.float32).copy()
    values[:, 0, 0] = np.arange(4) * 0.1
    store = RolloutStore.from_model(model, total_worlds=32, hz=30, max_worlds=2)
    with pytest.raises(ValueError, match="No completed"):
        store.info()
    store.publish(values, world_ids=(31, 7), start_step=10)
    with RolloutServer(store, port=0) as server:
        adapter = RolloutReplayAdapter(f"http://127.0.0.1:{server.address[1]}", window_frames=4)
        try:
            before = adapter.frame(FrameNeeds()).geom_xpos.copy()
            adapter.seek_replay(3)
            after = adapter.frame(FrameNeeds()).geom_xpos
            assert after[0, 0] - before[0, 0] == pytest.approx(0.3)
            np.testing.assert_array_equal(after[1], before[1])
            np.testing.assert_array_equal(model.qpos0[:3], np.zeros(3))
            assert adapter.world_selection().total_worlds == 32
            assert adapter.world_selection().world_ids == (31, 7)
        finally:
            adapter.release()
