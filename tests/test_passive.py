"""Passive state exchange and recovery without a graphics context."""

import multiprocessing as mp
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mojive.app.passive import PassiveViewer, _Mailbox

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")


class _DebugClient:
    def __init__(self):
        self.messages = []

    def send(self, **message):
        self.messages.append(message)


def test_debug_commands_validate_before_asynchronous_publication():
    viewer = PassiveViewer.__new__(PassiveViewer)
    viewer._lock = threading.RLock()
    viewer._closed = False
    viewer._stop = SimpleNamespace(is_set=lambda: False)
    viewer._process = SimpleNamespace(is_alive=lambda: True)
    viewer._debug_client = _DebugClient()
    commands = (
        {
            "op": "arrow",
            "layer": "policy.velocity",
            "id": "target",
            "a": [0.0, 0.0, 1.0],
            "b": [1.0, 0.0, 1.0],
            "color": [0.2, 0.95, 0.3, 1.0],
        },
    )

    viewer.publish_debug_commands(commands)

    assert viewer._debug_client.messages == list(commands)
    with pytest.raises(ValueError, match="finite JSON"):
        viewer.publish_debug_commands((commands[0], {"op": "point", "p": [float("nan")] * 3}))
    assert viewer._debug_client.messages == list(commands)


def test_debug_commands_reject_closed_viewer():
    viewer = PassiveViewer.__new__(PassiveViewer)
    viewer._lock = threading.RLock()
    viewer._closed = True
    viewer._stop = SimpleNamespace(is_set=lambda: True)
    viewer._process = SimpleNamespace(is_alive=lambda: False)
    viewer._debug_client = _DebugClient()

    with pytest.raises(RuntimeError, match="closed"):
        viewer.publish_debug_commands(())


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


@pytest.mark.parametrize("mode", ["translate", "rotate"])
@pytest.mark.parametrize("strength", [1.0, 3.0])
def test_display_grab_point_reaches_caller_and_clears_when_switching_or_released(
    monkeypatch, mode, strength
):
    from mojive import math3d
    from mojive.app.passive import _run_viewer
    from mojive.session import Session
    from mojive.session.rates import StepRate
    from mojive.types import CameraView
    from mojive.ui.perturb import PerturbController

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <body name="first"><freejoint/><geom size=".1" mass="1"/></body>
      <body name="second" pos="1 0 0"><freejoint/><geom size=".1" mass="1"/></body>
      <body name="unrelated" pos="2 0 0"><freejoint/><geom size=".1" mass="1"/></body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    data.xfrc_applied[3] = 7.0
    mujoco.mj_forward(model, data)
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
    viewer._step, viewer._step_known, viewer._step_rate = 0, False, StepRate()
    mujoco.mj_getState(model, data, viewer._state, viewer._state_spec)
    connection, peer = context.Pipe()
    completion_peer, completion = context.Pipe(duplex=False)
    events = context.Queue()
    checked = []

    def exercise_display(adapter, **options):
        session = Session(adapter)
        controller = PerturbController()
        controller.force_scale = controller.torque_scale = strength
        try:
            for name in ("first", "second"):
                node = next(n for n in session.nodes if n.name == name)
                body = model.body(name).id
                point = adapter.data.xpos[body] + [0.03, 0.02, 0]
                controller.begin(session, CameraView(), node, point, mode)
                session.perturb.target_pos += [0, 0, 0.1]
                session.perturb.target_mat = math3d.rotvec_to_mat3([0, 0, 0.2])
                for _ in range(3):
                    controller.apply(session)
                    assert viewer._publish()
                    expected = adapter.data.xfrc_applied[body]
                    assert np.linalg.norm(expected) > 0
                    np.testing.assert_allclose(data.xfrc_applied[body], expected)
                    np.testing.assert_array_equal(data.xfrc_applied[3], 7.0)
                if name == "second":
                    assert not data.xfrc_applied[1].any()
            controller.end(session)
            assert viewer._publish()
            assert not data.xfrc_applied[1:3].any()
            np.testing.assert_array_equal(data.xfrc_applied[3], 7.0)
            checked.append(True)
        finally:
            session.release()
        raise KeyboardInterrupt

    monkeypatch.setattr("mojive.app.composition.build_from_adapter", exercise_display)
    try:
        _run_viewer(model, mailbox, context.Event(), connection, completion, events, 60, False, {})
        error = completion_peer.recv()
        assert error is None, error
        assert checked
    finally:
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
