"""Runtime physics edits must not compile models or invalidate unrelated GPU resources."""

# ruff: noqa: E402

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.physics

from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.mujoco.updates import CONTROL_FIELDS
from mojive.app.mujoco_viewer.state import StateExchange
from mojive.session import Session

_OPTIONS = {
    "integrator": 1,
    "cone": 1,
    "jacobian": 0,
    "solver": 0,
    "timestep": 0.005,
    "iterations": 60,
    "tolerance": 1e-7,
    "ls_iterations": 20,
    "ls_tolerance": 0.02,
    "noslip_iterations": 2,
    "noslip_tolerance": 1e-5,
    "ccd_iterations": 20,
    "ccd_tolerance": 1e-5,
    "sleep_tolerance": 0.002,
    "sdf_iterations": 5,
    "sdf_initpoints": 20,
    "gravity": (0, 0, -3),
    "wind": (1, 0, 0),
    "magnetic": (0, 1, 0),
    "density": 1.2,
    "viscosity": 0.001,
    "impratio": 2,
    "o_margin": 0.01,
    "o_solimp": (0.8, 0.9, 0.002, 0.5, 2),
    "o_solref": (0.03, 1),
    "o_friction": (0.5, 0.5, 0.01, 0.001, 0.001),
    "disableactuator": 2,
}
_FLAGS = [
    (field, int(bit))
    for field, enum, prefix in (
        ("disableflags", mujoco.mjtDisableBit, "mjDSBL_"),
        ("enableflags", mujoco.mjtEnableBit, "mjENBL_"),
    )
    for name, bit in enum.__members__.items()
    if name.startswith(prefix)
]


@pytest.fixture
def runtime():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <asset><texture name="tex" type="2d" builtin="checker" width="4" height="4"/></asset>
      <worldbody><light name="lamp" pos="0 0 3"/>
        <body pos="0 0 1"><freejoint name="root"/>
          <geom type="box" size=".2 .3 .4"/>
        </body>
      </worldbody>
      <actuator><motor joint="root" ctrllimited="true" ctrlrange="-1 1" group="1"/></actuator>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    data.time = 0.75
    data.qpos[0] = 0.25
    data.qvel[1] = 0.1
    mujoco.mj_forward(model, data)
    exchange = StateExchange(model, data)
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(exchange.display_model, exchange.display_data)
    session = Session(adapter)

    def refresh():
        adapter.refresh_model_fields(exchange.pending_model_fields)
        if not exchange.pending_model_fields.isdisjoint(CONTROL_FIELDS):
            session.refresh_control_metadata()
        exchange.pending_model_fields.clear()
        adapter.refresh_model_visuals()
        session.tick(FrameNeeds(poses=True, diagnostics=True, actuator=True))

    yield SimpleNamespace(
        m=model, d=data, exchange=exchange, adapter=adapter, session=session, refresh=refresh
    )
    session.release()


def test_physics_cases_cover_every_mjoption_property(runtime):
    assert {field.name for field in runtime.exchange.physics_options.fields} == (
        set(_OPTIONS) | {"disableflags", "enableflags"}
    )


@pytest.mark.parametrize("mode", ("managed", "passive"))
@pytest.mark.parametrize("name,value", [*_OPTIONS.items(), *_FLAGS])
def test_every_physics_option_preserves_model_data_and_scene(
    runtime, monkeypatch, mode, name, value
):
    r = runtime
    source, generation = r.session.source, r.session.structure_generation
    dm, dd = r.adapter._m, r.adapter._d
    state = (dd.time, dd.qpos.copy(), dd.qvel.copy())

    def forbidden(*args, **kwargs):
        pytest.fail("A physics option rebuilt a model, constants, or scene source")

    monkeypatch.setattr(r.adapter, "_install", forbidden)
    monkeypatch.setattr(r.adapter, "_compile_composed_model", forbidden)
    monkeypatch.setattr(r.adapter, "_build_source", forbidden)
    monkeypatch.setattr(mujoco, "mj_setConst", forbidden)
    # Managed mode writes the live model; passive mode publishes the caller's model.
    edited = r.m if mode == "passive" else dm
    setattr(edited.opt, name, value)
    if mode == "passive":
        r.exchange.sync()
    else:
        mujoco.mj_forward(dm, dd)
    r.refresh()
    np.testing.assert_array_equal(getattr(dm.opt, name), value)
    assert r.session.source is source and r.session.structure_generation == generation
    assert r.adapter._m is dm and r.adapter._d is dd
    assert dd.time == state[0]
    np.testing.assert_array_equal(dd.qpos, state[1])
    np.testing.assert_array_equal(dd.qvel, state[2])
    assert not r.exchange.camera_fields.changed_fields


@pytest.mark.parametrize("name,value", list(_OPTIONS.items()))
def test_passive_ui_physics_edits_return_even_during_state_only_sync(runtime, name, value):
    r = runtime
    setattr(r.exchange.display_model.opt, name, value)
    r.exchange.sync(state_only=True)
    np.testing.assert_array_equal(getattr(r.m.opt, name), value)
    assert not r.exchange.pending_model_fields


@pytest.mark.parametrize(
    "name,value",
    [
        ("geom_friction", 0.4),
        ("geom_solref", 0.1),
        ("geom_solimp", 0.5),
        ("geom_margin", 0.01),
        ("geom_gap", 0.001),
        ("geom_priority", 1),
        ("dof_damping", 0.2),
        ("dof_frictionloss", 0.1),
        ("jnt_stiffness", 1.0),
        ("actuator_gainprm", 0.5),
        ("actuator_biasprm", 0.01),
    ],
)
def test_nonvisual_model_edits_do_not_reset_camera_or_rebuild_scene(
    runtime, monkeypatch, name, value
):
    r = runtime
    source, generation = r.session.source, r.session.structure_generation
    r.exchange.camera_changed = False
    monkeypatch.setattr(r.adapter, "_build_source", lambda: pytest.fail("Unrelated scene rebuild"))
    getattr(r.m, name).flat[0] = value
    r.exchange.sync()
    r.refresh()
    assert r.session.source is source and r.session.structure_generation == generation
    assert not r.exchange.camera_changed
    np.testing.assert_array_equal(getattr(r.adapter._m, name), getattr(r.m, name))


def test_actuator_group_disable_updates_existing_visibility_without_scene_rebuild(runtime):
    r = runtime
    source = r.session.source
    visibility = source.actuator_visible
    assert visibility.tolist() == [True]
    for disabled in (2, 0, 2):
        r.m.opt.disableactuator = disabled
        r.exchange.sync()
        r.refresh()
        assert r.session.source is source and source.actuator_visible is visibility
        assert visibility.tolist() == [not bool(disabled)]


def test_control_metadata_updates_in_place_without_resource_uploads(runtime):
    r = runtime
    source = r.session.source
    ranges = source.actuator_ctrl_range
    joint, actuator = r.session.joints[0], r.session.actuators[0]
    r.m.dof_damping[0] = 0.7
    r.m.jnt_stiffness[0] = 0.3
    r.m.actuator_ctrlrange[0] = (-3, 4)
    r.m.actuator_gainprm[0, 0] = 2
    r.exchange.sync()
    r.refresh()
    assert r.session.source is source and source.actuator_ctrl_range is ranges
    # Panel row caches retain these objects until scene structure changes.
    assert r.session.joints[0] is joint
    assert r.session.actuators[0] is actuator
    assert r.session.joints[0].damping == pytest.approx(0.7)
    assert r.session.joints[0].stiffness == pytest.approx(0.3)
    assert r.session.actuators[0].gain == 2
    assert r.session.actuators[0].ctrl_range == (-3, 4)
    np.testing.assert_array_equal(ranges[0], (-3, 4))


def test_derived_constants_update_diagnostics_without_recompiling_or_uploading(
    runtime, monkeypatch
):
    r = runtime
    source, generation = r.session.source, r.session.structure_generation
    previous_sizes = source.diagnostics.inertia_sizes.copy()
    r.m.body_mass[1] *= 2
    mujoco.mj_setConst(r.m, r.d)
    mujoco.mj_forward(r.m, r.d)
    r.exchange.sync()
    monkeypatch.setattr(
        r.adapter, "_build_source", lambda: pytest.fail("Unrelated resource rebuild")
    )
    r.refresh()
    assert r.session.source is source and r.session.structure_generation == generation
    assert not np.array_equal(source.diagnostics.inertia_sizes, previous_sizes)
    np.testing.assert_allclose(r.exchange.display_model.body_subtreemass, r.m.body_subtreemass)


@pytest.mark.parametrize("name,value", [("geom_size", 0.5), ("tex_data", 17), ("geom_rgba", 0.2)])
def test_actual_visual_edits_still_update_source_without_recompiling(
    runtime, monkeypatch, name, value
):
    r = runtime
    source = r.session.source
    dm, dd = r.adapter._m, r.adapter._d
    monkeypatch.setattr(r.adapter, "_compile_composed_model", lambda: pytest.fail("Model compile"))
    getattr(r.m, name).flat[0] = value
    r.exchange.sync()
    r.refresh()
    assert r.session.source is not source
    assert r.adapter._m is dm and r.adapter._d is dd


def test_light_properties_refresh_without_rebuilding_scene(runtime):
    r = runtime
    source = r.session.source
    r.m.light_bulbradius[0] = 0.5
    r.exchange.sync()
    r.refresh()
    assert r.session.source is source
    assert r.session.frame.lights.lights[0].area_radius == pytest.approx(0.5)
