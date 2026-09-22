"""Physical dragging must match the native viewer at the selected body point."""

import numpy as np
import pytest

from mojive import math3d
from mojive.adapters.mujoco import MuJoCoAdapter

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")


@pytest.fixture(params=["free", "hinge", "ball"])
def adapter(request, tmp_path):
    joint = (
        "<freejoint/>"
        if request.param == "free"
        else (f'<joint type="{request.param}" armature=".0001" damping=".0005"/>')
    )
    path = tmp_path / "pivot.xml"
    path.write_text(f"""
    <mujoco>
      <option timestep=".01" integrator="implicitfast" gravity="0 0 0"/>
      <worldbody>
        <body name="offset" euler="15 25 35">
          {joint}
          <geom type="box" pos=".019 .019 0" size=".0087 .0087 .0087"
                mass=".00253704"/>
        </body>
      </worldbody>
    </mujoco>
    """)
    result = MuJoCoAdapter(path)
    yield result
    result.release()


def native_perturb(adapter, local_position):
    model, data = adapter.model, adapter.data
    scene = mujoco.MjvScene(model, maxgeom=1)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    mujoco.mjv_updateCamera(model, data, camera, scene)
    perturb = mujoco.MjvPerturb()
    perturb.select = 1
    perturb.localpos[:] = local_position
    mujoco.mjv_initPerturb(model, data, scene, perturb)
    return perturb


@pytest.mark.parametrize("mode", ["translate", "rotate"])
@pytest.mark.parametrize("at_point", [False, True], ids=["body-origin", "grab-point"])
def test_perturb_force_matches_native_viewer(adapter, mode, at_point):
    model, data = adapter.model, adapter.data
    node = next(n for n in adapter.nodes() if n.name == "offset")
    local = np.array([0.019, 0.023, 0.004]) if at_point else np.zeros(3)
    native = native_perturb(adapter, local)
    data.qvel[:] = np.linspace(-0.3, 0.4, model.nv)
    mujoco.mj_forward(model, data)
    reference = mujoco.MjData(model)
    mujoco.mj_copyData(reference, model, data)
    position = data.xpos[1] + [0.01, 0.02, 0.005]
    rotation = math3d.rotvec_to_mat3([0.1, -0.2, 0.3]) @ data.xmat[1].reshape(3, 3)
    native.refselpos[:] = position + rotation @ local
    mujoco.mju_mulQuat(native.refquat, math3d.mat3_to_quat(rotation), model.body_iquat[1])
    native.active = (
        mujoco.mjtPertBit.mjPERT_TRANSLATE
        if mode == "translate"
        else mujoco.mjtPertBit.mjPERT_ROTATE
    )
    mujoco.mjv_applyPerturbForce(model, reference, native)
    if at_point:
        assert adapter.apply_perturb_at_point(node.node_id, position, rotation, mode, local)
    else:
        assert adapter.apply_perturb(node.node_id, position, rotation, mode)
    np.testing.assert_allclose(data.xfrc_applied, reference.xfrc_applied, atol=1e-12, rtol=1e-12)


def test_grab_point_drag_stays_finite_and_moves_joint(adapter):
    node = next(n for n in adapter.nodes() if n.name == "offset")
    initial = adapter.data.qpos.copy()
    rotation = adapter.data.xmat[1].reshape(3, 3).copy()
    for step in range(600):
        position = np.array([0.012 * np.sin(step / 20), 0.009 * np.cos(step / 20), 0.003])
        assert adapter.apply_perturb_at_point(
            node.node_id, position, rotation, "translate", np.array([0.019, 0.019, 0.004])
        )
        adapter.step()
    assert adapter.data.time == pytest.approx(6)
    assert np.isfinite(adapter.data.qacc).all()
    assert np.linalg.norm(adapter.data.qpos - initial) > 0.001
    adapter.clear_perturb()
    assert not adapter.data.xfrc_applied.any()


def test_changing_mode_and_grab_point_replaces_the_previous_force(adapter):
    node = next(n for n in adapter.nodes() if n.name == "offset")
    rotation = adapter.data.xmat[1].reshape(3, 3).copy()
    target = np.array([0.01, 0.02, 0.005])
    assert adapter.apply_perturb(node.node_id, target, rotation, "translate")
    local = np.array([0.019, 0.019, 0])
    assert adapter.apply_perturb_at_point(node.node_id, target, rotation, "translate", local)
    applied = adapter.data.xfrc_applied.copy()
    adapter.clear_perturb()
    assert adapter.apply_perturb_at_point(node.node_id, target, rotation, "translate", local)
    np.testing.assert_allclose(adapter.data.xfrc_applied, applied)
    assert adapter.apply_perturb_at_point(node.node_id, target, rotation, "rotate", local)
    assert not adapter.data.xfrc_applied[:, :3].any()


@pytest.mark.parametrize("mode", ["translate", "rotate"])
@pytest.mark.parametrize("strength", [0.0, 0.5, 2.0, 5.0])
def test_strength_scales_the_complete_wrench_without_accumulating(adapter, mode, strength):
    from mojive.adapters.workspace import WorkspaceAdapter
    from mojive.commands import Perturb
    from mojive.session import Session

    session = Session(WorkspaceAdapter(adapter))
    try:
        node = next(n for n in session.nodes if n.name == "offset")
        adapter.data.qvel[:] = np.linspace(-0.3, 0.4, adapter.model.nv)
        mujoco.mj_forward(adapter.model, adapter.data)
        position = adapter.data.xpos[1] + [0.01, 0.02, 0.005]
        rotation = math3d.rotvec_to_mat3([0.1, -0.2, 0.3]) @ adapter.data.xmat[1].reshape(3, 3)
        local = np.array([0.019, 0.023, 0.004])
        assert session.submit(Perturb(node.node_id, position, rotation, mode, local))
        baseline = adapter.data.xfrc_applied.copy()
        assert np.linalg.norm(baseline) > 0
        for _ in range(3):
            assert session.submit(Perturb(node.node_id, position, rotation, mode, local, strength))
            np.testing.assert_allclose(adapter.data.xfrc_applied, baseline * strength)
        for invalid in (-1, np.inf, np.nan):
            assert not session.submit(
                Perturb(node.node_id, position, rotation, mode, local, invalid)
            )
            np.testing.assert_allclose(adapter.data.xfrc_applied, baseline * strength)
    finally:
        session.release()


@pytest.mark.parametrize("workspace", [False, True])
def test_controller_drag_routes_through_session_and_physics_worker(adapter, workspace):
    import time

    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.adapters.workspace import WorkspaceAdapter
    from mojive.session import Session
    from mojive.types import CameraView
    from mojive.ui.perturb import PerturbController

    session = Session(WorkspaceAdapter(adapter) if workspace else adapter)
    try:
        session.submit(cmd.Pause())
        session.tick(FrameNeeds())
        node = next(n for n in session.nodes if n.name == "offset")
        assert session.set_threaded_physics(True)
        point = adapter.data.xpos[1] + adapter.data.xmat[1].reshape(3, 3) @ [0.019, 0.019, 0.004]
        initial = adapter.data.qpos.copy()
        assert session.submit(cmd.Play())
        controller = PerturbController()
        controller.begin(session, CameraView(), node, point, "translate")
        session.perturb.target_pos += [0.01, 0.02, 0.005]
        deadline = time.monotonic() + 3
        while session.frame.time < 0.2 and time.monotonic() < deadline:
            controller.apply(session)
            session.tick(FrameNeeds(), wall_dt=0.01)
            time.sleep(0.002)
        assert session.frame.time >= 0.2
        assert not session.paused
        assert np.isfinite(adapter.data.qpos).all()
        assert np.linalg.norm(adapter.data.qpos - initial) > 0.001
        controller.end(session)
        assert session.submit(cmd.Pause())
        assert not adapter.data.xfrc_applied.any()
    finally:
        session.release()
