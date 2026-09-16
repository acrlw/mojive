"""World physics options affect live simulation and survive document operations."""

import time

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.session import Session

pytestmark = pytest.mark.physics
mujoco = pytest.importorskip("mujoco")


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "physics.xml"
    path.write_text("""<mujoco>
      <option timestep=".002" integrator="implicitfast"/>
      <worldbody><body name="box" pos="0 0 1"><freejoint/>
        <geom type="box" size=".1 .1 .1" mass="1"/>
      </body></worldbody>
    </mujoco>""")
    return path


@pytest.fixture(params=[False, True], ids=["model", "workspace"])
def session(request, model_path):
    primary = MuJoCoAdapter(model_path)
    session = Session(WorkspaceAdapter(primary) if request.param else primary)
    yield session
    session.release()


def primary(session):
    return getattr(session.adapter, "primary", session.adapter)


def values(session):
    return {option.key: option.value for option in session.physics_options}


def test_exposes_all_native_options_and_available_flag_bits(session):
    options = session.physics_options
    assert {o.key.split(".")[0] for o in options} == {
        n for n in dir(mujoco.MjOption()) if not n.startswith("_")
    }
    assert len({o.key for o in options}) == len(options)
    for enum, prefix, name in (
        (mujoco.mjtDisableBit, "mjDSBL_", "disableflags"),
        (mujoco.mjtEnableBit, "mjENBL_", "enableflags"),
    ):
        expected = {
            f"{name}.{key[len(prefix) :].lower()}"
            for key in enum.__members__
            if key.startswith(prefix)
        }
        assert expected <= set(values(session))
    assert session.physics_options is options


def test_live_update_preserves_state_and_refreshes_acceleration(session):
    a = primary(session)
    model, data = a.model, a.data
    data.time = 1.2
    data.qvel[:3] = [1, 2, 3]
    before = a.capture_state()
    generation = session.structure_generation
    assert session.submit(
        cmd.SetPhysicsOptions(
            {
                "gravity": (0, 0, -3),
                "timestep": 0.01,
                "integrator": int(mujoco.mjtIntegrator.mjINT_RK4),
            }
        )
    )
    assert a.model is model and a.data is data
    assert session.structure_generation == generation
    np.testing.assert_array_equal(data.qpos, before.qpos)
    np.testing.assert_array_equal(data.qvel, before.qvel)
    assert data.time == before.time
    assert data.qacc[2] == pytest.approx(-3)
    assert model.opt.timestep == 0.01
    assert session.dirty and session.can_undo
    assert session.submit(cmd.SetPhysicsOptions({"disableflags.gravity": True}))
    assert data.qacc[2] == 0
    assert session.submit(cmd.SetPhysicsOptions({"disableflags.gravity": False}))
    assert data.qacc[2] == pytest.approx(-3)
    a.step()
    assert data.time == pytest.approx(1.21)


@pytest.mark.parametrize(
    "patch",
    [
        {"timestep": 0},
        {"timestep": -1},
        {"tolerance": float("nan")},
        {"gravity": [0, 1]},
        {"iterations": 1.5},
        {"iterations": 2**35},
        {"solver": 99},
        {"disableflags.gravity": 1},
        {"unknown": 1},
        {"o_friction": [1, -1, 0, 0, 0]},
        {"o_solimp": [0.9, 0.95, -1, 0.5, 2]},
    ],
)
def test_invalid_patch_is_atomic(session, patch):
    a = primary(session)
    before = values(session)
    position, velocity = a.data.qpos.copy(), a.data.qvel.copy()
    revision = session.document_revision
    result = session.submit(cmd.SetPhysicsOptions({"gravity": [0, 0, -1], **patch}))
    assert not result.ok
    assert values(session) == before
    np.testing.assert_array_equal(a.data.qpos, position)
    np.testing.assert_array_equal(a.data.qvel, velocity)
    assert session.document_revision == revision
    assert not session.can_undo


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_forward_failure_rolls_back_model_source_and_full_data(session, monkeypatch, error_type):
    a = primary(session)
    before, xml = values(session), a.scene_model_xml(0)
    position = a.data.qpos.copy()
    native = mujoco.mj_forward

    def fail(model, data):
        native(model, data)
        data.qpos[:] = 123
        raise error_type("injected forward failure")

    monkeypatch.setattr(mujoco, "mj_forward", fail)
    result = session.submit(cmd.SetPhysicsOptions({"gravity": (0, 0, -1)}))
    assert not result.ok and "injected forward failure" in result.message
    assert values(session) == before
    assert a.scene_model_xml(0) == xml
    np.testing.assert_array_equal(a.data.qpos, position)
    assert not session.can_undo


def test_undo_redo_reset_and_model_rebuild_preserve_options(session):
    a = primary(session)
    before = values(session)
    patch = {
        "timestep": 0.005,
        "gravity": (1, 2, -3),
        "solver": int(mujoco.mjtSolver.mjSOL_CG),
        "enableflags.energy": True,
        "disableflags.contact": True,
    }
    assert session.submit(cmd.SetPhysicsOptions(patch))
    assert session.submit(cmd.Undo())
    assert values(session) == before
    assert session.submit(cmd.Redo())
    assert all(values(session)[key] == val for key, val in patch.items())
    assert session.submit(cmd.Reset())
    assert all(values(session)[key] == val for key, val in patch.items())
    state = a.capture_edit_state()
    assert a.restore_edit_state(state)
    assert all(values(session)[key] == val for key, val in patch.items())


@pytest.mark.parametrize("empty_root", [False, True])
def test_composed_options_survive_workspace_and_mjcf_roundtrip(model_path, tmp_path, empty_root):
    a = MuJoCoAdapter(None if empty_root else model_path)
    if empty_root:
        a.new_scene()
        assert a.add_scene_model(model_path, np.zeros(3), np.eye(3)) > 0
    w = WorkspaceAdapter(a)
    session = Session(w)
    try:
        # Explicit engine defaults must override the first attached model too.
        patch = {
            "integrator": int(mujoco.mjtIntegrator.mjINT_EULER),
            "timestep": 0.01,
            "gravity": (0, 0, -9.81),
            "disableflags.gravity": True,
            "enableflags.energy": True,
        }
        assert session.submit(cmd.SetPhysicsOptions(patch))
        assert a.add_scene_model(model_path, np.array([2, 0, 0]), np.eye(3)) > 0
        assert all(values(session)[key] == val for key, val in patch.items())
        path = tmp_path / "world.mojive"
        w.save_scene(path)
        w.new_scene()
        w.open_scene(path)
        assert all(values(session)[key] == val for key, val in patch.items())
        target = tmp_path / "world.xml"
        w.save_scene(target)
        loaded = mujoco.MjModel.from_xml_path(str(target))
        assert loaded.opt.timestep == 0.01
        assert loaded.opt.integrator == int(mujoco.mjtIntegrator.mjINT_EULER)
        assert loaded.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        assert loaded.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    finally:
        session.release()


def test_zero_width_contact_impedance_survives_save_and_open(model_path, tmp_path):
    w = WorkspaceAdapter(MuJoCoAdapter(model_path))
    session = Session(w)
    try:
        value = (0.9, 0.9, 0.0, 0.0, 1.0)
        assert session.submit(cmd.SetPhysicsOptions({"o_solimp": value}))
        path = tmp_path / "constant-impedance.mojive.json"
        w.save_scene(path)
        w.open_scene(path)
        assert values(session)["o_solimp"] == value
    finally:
        session.release()


def test_timestep_change_rebases_running_worker_clock(session):
    assert session.set_threaded_physics(True)
    deadline = time.monotonic() + 2
    while session.frame.time < 0.08 and time.monotonic() < deadline:
        session.tick(FrameNeeds(), wall_dt=0.01)
        time.sleep(0.002)
    assert session.frame.time >= 0.08
    assert session.submit(cmd.SetPhysicsOptions({"timestep": 0.05}))
    start = primary(session).data.time
    deadline = time.monotonic() + 0.4
    while session.frame.time <= start and time.monotonic() < deadline:
        session.tick(FrameNeeds(), wall_dt=0.01)
        time.sleep(0.002)
    assert session.frame.time > start
    assert not session.paused


def test_external_clock_cannot_edit_a_display_copy(model_path):
    a = MuJoCoAdapter(model_path, external_clock=True)
    session = Session(a)
    try:
        assert not session.physics_options
        assert not session.submit(cmd.SetPhysicsOptions({"timestep": 0.01}))
        assert a.model.opt.timestep == 0.002
    finally:
        session.release()


@pytest.mark.parametrize("wrapped", [False, True])
def test_owned_compiled_model_supports_runtime_options_without_source(model_path, wrapped):
    a = MuJoCoAdapter()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    a.load_model(model, data)
    session = Session(WorkspaceAdapter(a) if wrapped else a)
    try:
        assert session.physics_options
        assert not session.adapter.caps.edit_history
        assert session.submit(cmd.SetPhysicsOptions({"gravity": (0, 0, -3)}))
        assert model.opt.gravity[2] == -3
        assert data.qacc[2] == -3
    finally:
        session.release()


def test_invalid_workspace_options_restore_the_previous_document(model_path, tmp_path):
    import json

    a = MuJoCoAdapter(model_path)
    w = WorkspaceAdapter(a)
    session = Session(w)
    try:
        assert session.submit(cmd.SetPhysicsOptions({"timestep": 0.005}))
        path = tmp_path / "invalid.mojive.json"
        w.save_scene(path)
        document = json.loads(path.read_text())
        document["physics_options"]["timestep"] = -1
        path.write_text(json.dumps(document))
        before, revision = values(session), session.document_revision
        state = a.capture_state()
        assert not session.submit(cmd.OpenScene(path))
        assert session.document_revision == revision
        assert values(session) == before
        np.testing.assert_array_equal(a.data.qpos, state.qpos)
        assert session.submit(cmd.Undo())
        assert a.model.opt.timestep == 0.002
    finally:
        session.release()
