"""Control-vector atomicity and native measurement semantics."""

import numpy as np
import pytest

from mojive import MuJoCoAdapter
from mojive import commands as cmd
from mojive.control.rpc import ControlService
from mojive.session import Session

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="foot" pos="0 0 .09">
      <joint name="height" type="slide" axis="0 0 1"/>
      <geom name="sole" type="sphere" size=".1" mass="1"/>
      <site name="imu"/>
    </body>
  </worldbody>
  <actuator><motor joint="height" ctrllimited="true" ctrlrange="-2 3"/></actuator>
  <sensor><accelerometer site="imu"/><jointpos joint="height"/></sensor>
</mujoco>
"""


def test_vector_control_is_atomic_clamped_and_does_not_capture_state(monkeypatch):
    adapter = MuJoCoAdapter()
    adapter.load_model(mujoco.MjModel.from_xml_string(XML))
    session = Session(adapter)
    assert session.submit(cmd.SetCtrlVector(np.array([9.0]))).ok
    np.testing.assert_array_equal(adapter.data.ctrl, [3.0])
    for values in ([np.nan], [1.0, 2.0], [[1.0]], [np.inf]):
        assert not session.submit(cmd.SetCtrlVector(np.asarray(values))).ok
        np.testing.assert_array_equal(adapter.data.ctrl, [3.0])
    service = ControlService(session=session)
    monkeypatch.setattr(
        adapter, "capture_state", lambda: pytest.fail("full state copy on control update")
    )
    monkeypatch.setattr(adapter, "set_ctrl", lambda *a: pytest.fail("scalar control dispatch"))
    try:
        result = service.dispatch("set_ctrl", {"values": [-9.0]})
        assert result["ok"] and result["count"] == 1
        np.testing.assert_array_equal(adapter.data.ctrl, [-2.0])
    finally:
        service.close()


def test_observations_preserve_native_sensor_contact_and_force_outputs():
    adapter = MuJoCoAdapter()
    adapter.load_model(mujoco.MjModel.from_xml_string(XML))
    model, data = adapter.model, adapter.data
    adapter.set_ctrl_vector(np.array([0.5]))
    mujoco.mj_forward(model, data)
    before = data.time
    observation = adapter.capture_observation()
    assert len(observation.contacts) > 0
    np.testing.assert_array_equal(observation.sensordata, data.sensordata)
    np.testing.assert_array_equal(observation.actuator_force, data.actuator_force)
    assert observation.sensors[-1].data_adr == 3
    contact = observation.contacts[0]
    assert contact.geom_ids == (0, 1) and contact.body_indices == (0, 1)
    expected = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, expected)
    np.testing.assert_allclose(contact.wrench, expected)
    np.testing.assert_allclose(
        contact.world_wrench.reshape(2, 3), expected.reshape(2, 3) @ contact.frame
    )
    assert data.time == before
    saved = observation.sensordata.copy()
    mujoco.mj_step(model, data)
    np.testing.assert_array_equal(observation.sensordata, saved)
    service = ControlService(adapter)
    try:
        state = service.dispatch("get_state", {})
        assert state["observations"]["sensors"][-1]["data_adr"] == 3
        assert state["observations"]["time"] == state["time"]
        assert service.dispatch("get_state", {"observations": False})["observations"] is None
    finally:
        service.close()
