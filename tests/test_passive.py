"""Passive state exchange and recovery without a graphics context."""

import multiprocessing as mp
import threading

import numpy as np
import pytest

from mojive.app.passive import PassiveViewer, _Mailbox

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")


def test_perturbation_preserves_other_forces_and_clears_after_worker_failure():
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    context = mp.get_context("spawn")
    mailbox = _Mailbox(
        context,
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),
        model.nu,
        model.nbody,
    )
    viewer = PassiveViewer.__new__(PassiveViewer)
    viewer._model, viewer._data = model, data
    viewer._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    viewer._mailbox = mailbox
    viewer._state, viewer._ctrl, viewer._ctrl_changed, viewer._force, viewer._force_changed = (
        mailbox.arrays()
    )
    viewer._applied_force = np.zeros(model.nbody, dtype=bool)
    viewer._step = 0
    viewer._step_known = False
    viewer._lock = threading.RLock()
    viewer._closed = False
    viewer._stop = context.Event()
    viewer._process = context.Process()
    viewer._connection, peer = context.Pipe()
    viewer._completion, completion_peer = context.Pipe(duplex=False)
    viewer._events = context.Queue(maxsize=64)
    try:
        data.xfrc_applied[2] = 7
        viewer._force[1] = 3
        viewer._force_changed[1] = 1
        assert viewer._publish()
        np.testing.assert_array_equal(data.xfrc_applied[1], 3)
        np.testing.assert_array_equal(data.xfrc_applied[2], 7)
        viewer._force[1] = 0
        viewer._force_changed[1] = 2
        assert viewer._publish()
        np.testing.assert_array_equal(data.xfrc_applied[1], 0)
        data.xfrc_applied[1] = 5
        assert viewer._publish()
        np.testing.assert_array_equal(data.xfrc_applied[1], 5)
        viewer._force[1] = 3
        viewer._force_changed[1] = 1
        assert viewer._publish()
        # An unpublished drag must not clear an unrelated caller-owned force.
        viewer._force_changed[2] = 1
        # Simulate a worker that died while holding the interprocess lock.
        mailbox.lock.acquire()
        try:
            viewer.close()
        finally:
            mailbox.lock.release()
        np.testing.assert_array_equal(data.xfrc_applied[1], 0)
        np.testing.assert_array_equal(data.xfrc_applied[2], 7)
    finally:
        viewer.close()
        peer.close()
        completion_peer.close()


def _complete_with_large_error(connection):
    connection.send("encoder failure: " + "x" * 200_000)
    connection.close()


def test_close_drains_large_failure_before_joining_and_reports_it():
    from types import SimpleNamespace

    context = mp.get_context("spawn")
    viewer = PassiveViewer.__new__(PassiveViewer)
    viewer._lock = threading.RLock()
    viewer._closed = False
    viewer._stop = context.Event()
    viewer._completion, child = context.Pipe(duplex=False)
    viewer._connection, peer = context.Pipe()
    viewer._events = context.Queue(maxsize=64)
    viewer._data = SimpleNamespace(xfrc_applied=np.zeros((1, 6)))
    viewer._applied_force = np.zeros(1, dtype=bool)
    viewer._process = context.Process(target=_complete_with_large_error, args=(child,))
    viewer._process.start()
    child.close()
    try:
        with pytest.raises(RuntimeError, match="encoder failure"):
            viewer.close()
        assert not viewer._process.is_alive()
        viewer.close()
    finally:
        peer.close()
