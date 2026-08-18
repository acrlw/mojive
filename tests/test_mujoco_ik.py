"""MuJoCo inverse-kinematics contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import IkOptions, NodeKind
from forge_viewer.adapters.mujoco_adapter import MuJoCoAdapter
from forge_viewer.session import Session
from forge_viewer.types import CameraView
from forge_viewer.ui.gizmo import ObjectGizmo

MODEL = """
<mujoco model="ik">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="upper">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-170 170"/>
      <geom type="capsule" fromto="0 0 0 1 0 0" size=".04"/>
      <body name="lower" pos="1 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" range="-170 170"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size=".035"/>
        <site name="hand" pos="1 0 0" size=".06"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def ik_adapter(tmp_path):
    path = tmp_path / "ik.xml"
    path.write_text(MODEL)
    adapter = MuJoCoAdapter(path)
    yield adapter
    adapter.release()


def _site(adapter):
    return next(node for node in adapter.nodes() if node.kind is NodeKind.SITE)


def test_site_position_ik_converges_and_respects_joint_ranges(ik_adapter):
    assert _site(ik_adapter).ik_target
    target = np.array([1.0, 1.0, 0.0])
    result = ik_adapter.solve_ik(_site(ik_adapter).node_id, target, np.eye(3), IkOptions())
    assert result.success and result.converged
    assert result.position_error < 1e-3
    assert ik_adapter.data.site_xpos[0] == pytest.approx(target, abs=1e-3)
    assert np.all(np.abs(ik_adapter.data.qpos) <= np.deg2rad(170.0) + 1e-8)


def test_joint_lock_and_weight_options_are_applied(ik_adapter):
    shoulder = float(ik_adapter.data.qpos[0])
    result = ik_adapter.solve_ik(
        _site(ik_adapter).node_id,
        np.array([1.4, 0.4, 0.0]),
        np.eye(3),
        IkOptions(locked_joints=(0,), max_iterations=20),
    )
    assert result.success
    assert ik_adapter.data.qpos[0] == pytest.approx(shoulder)


def test_session_records_one_complete_ik_edit_for_undo(ik_adapter, tmp_path):
    session = Session(ik_adapter, tmp_path / "ik.xml")
    assert session.submit(cmd.Pause()).ok
    site = _site(ik_adapter)
    before = ik_adapter.data.qpos.copy()
    result = session.submit(
        cmd.SolveIk(site.node_id, np.array([1.0, 1.0, 0.0]), np.eye(3), IkOptions())
    )
    assert result.ok and session.ik_result is not None
    assert not np.allclose(ik_adapter.data.qpos, before)
    assert session.submit(cmd.UndoIk()).ok
    assert ik_adapter.data.qpos == pytest.approx(before)


def test_ik_requires_a_paused_simulation(ik_adapter, tmp_path):
    session = Session(ik_adapter, tmp_path / "ik.xml")
    assert session.submit(cmd.Play()).ok
    result = session.submit(
        cmd.SolveIk(_site(ik_adapter).node_id, np.ones(3), np.eye(3), IkOptions())
    )
    assert not result.ok
    assert "pause" in result.message


def test_site_selection_and_gizmo_drag_form_one_undoable_ik_edit(ik_adapter, tmp_path):
    session = Session(ik_adapter, tmp_path / "ik.xml")
    assert session.submit(cmd.Pause()).ok
    site = _site(ik_adapter)
    assert session.submit(cmd.SelectNode(site.node_id)).ok
    assert session.selected == 0
    assert session.selected_node is not None
    assert session.selected_node.node_id == site.node_id

    before = ik_adapter.data.qpos.copy()
    target = np.asarray(ik_adapter.data.site_xpos[site.site_index], np.float32)
    camera = CameraView(
        eye=np.array((4.0, -6.0, 3.0), np.float32),
        target=target,
        up=np.array((0.0, 0.0, 1.0), np.float32),
        aspect=4.0 / 3.0,
    )
    rect = (0.0, 0.0, 800.0, 600.0)
    gizmo = ObjectGizmo()
    cursor = np.array((400.0, 300.0))

    assert gizmo.keyboard_interact(session, camera, rect, tuple(cursor), 1)
    direction = gizmo._axis_screen.copy()
    assert gizmo.keyboard_interact(session, camera, rect, tuple(cursor + direction * 35.0), 1)
    assert gizmo.keyboard_interact(session, camera, rect, tuple(cursor + direction * 70.0), 1)
    assert not np.allclose(ik_adapter.data.qpos, before)

    gizmo.keyboard_interact(session, camera, rect, tuple(cursor), -1)
    assert session.submit(cmd.UndoIk()).ok
    assert ik_adapter.data.qpos == pytest.approx(before)
