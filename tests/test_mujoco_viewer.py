"""MuJoCo viewer migration contracts with real model/state and isolated UI ownership."""

# ruff: noqa: E402

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.physics

from mojive import viewer
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.app.mujoco_viewer import handle as handle_module
from mojive.app.mujoco_viewer import launch as launch_module
from mojive.app.mujoco_viewer.panels import (
    LEFT_PANELS,
    RIGHT_PANELS,
    install_key_callback,
    viewer_config,
)
from mojive.app.mujoco_viewer.state import StateExchange, _Field
from mojive.app.mujoco_viewer.visuals import Presentation, snapshot_geometries
from mojive.config import PanelConfig, ViewerConfig
from mojive.render.backend import NullBackend
from mojive.render.debugdraw import DebugDraw
from mojive.session import Session
from mojive.ui.camera import OrbitCamera
from mojive.ui.panels import PanelManager


@pytest.fixture
def model_data():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><asset><texture name="tex" type="2d" builtin="checker" width="4" height="4"/></asset>
    <worldbody><body><freejoint/><geom type="box" size=".2 .3 .4"/></body></worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_public_launch_parameters_include_upstream_names_and_defaults():
    import mujoco.viewer

    for name in ("launch", "launch_passive", "launch_from_path"):
        ours = inspect.signature(getattr(viewer, name)).parameters
        upstream = inspect.signature(getattr(mujoco.viewer, name)).parameters
        for parameter, expected in upstream.items():
            assert parameter in ours
            assert ours[parameter].kind == expected.kind
            assert ours[parameter].default == expected.default


def test_sync_copies_model_and_state_only_when_called(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    assert exchange.display_model is not m and exchange.display_data is not d
    original = exchange.display_model.geom_rgba.copy()
    m.geom_rgba[0] = (0.1, 0.2, 0.3, 1)
    m.opt.timestep = 0.005
    d.qpos[0] = 2
    assert exchange.display_data.qpos[0] == 0
    exchange.sync(True)
    assert exchange.display_data.qpos[0] == 2
    np.testing.assert_array_equal(exchange.display_model.geom_rgba, original)
    assert exchange.display_model.opt.timestep != 0.005
    exchange.sync()
    np.testing.assert_array_equal(exchange.display_model.geom_rgba, m.geom_rgba)
    assert exchange.display_model.opt.timestep == 0.005


def test_ui_changes_do_not_restore_unchanged_stale_state(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    d.qpos[0] = 2
    exchange.display_data.qpos[1] = 3
    exchange.display_model.opt.gravity[2] = -5
    exchange.sync(True)
    assert d.qpos[0] == 2 and d.qpos[1] == 3
    assert m.opt.gravity[2] == -5
    # UI changes apply once; subsequent caller updates remain authoritative.
    d.qpos[1] = 4
    m.opt.gravity[2] = -2
    exchange.sync()
    assert d.qpos[1] == 4 and exchange.display_data.qpos[1] == 4
    assert m.opt.gravity[2] == -2 and exchange.display_model.opt.gravity[2] == -2


def test_unchanged_nan_does_not_overwrite_a_new_caller_value():
    caller = SimpleNamespace(values=np.array([0.0, np.nan]))
    display = SimpleNamespace(values=caller.values.copy())
    field = _Field(caller, display, "values", caller.values.copy())
    display.values[0] = 5
    caller.values[1] = 9
    field.sync(True)
    np.testing.assert_array_equal(caller.values, [5, 9])
    np.testing.assert_array_equal(display.values, [5, 9])


def test_unchanged_model_arrays_are_not_copied_again(model_data, monkeypatch):
    exchange = StateExchange(*model_data)
    exchange.sync()
    arrays = [
        field.previous
        for field in exchange.model_fields.fields
        if isinstance(field.previous, np.ndarray)
    ]
    original = np.copyto

    def copyto(destination, *args, **kwargs):
        assert not any(destination is previous for previous in arrays)
        return original(destination, *args, **kwargs)

    monkeypatch.setattr(np, "copyto", copyto)
    exchange.sync()
    exchange.sync(True)


def test_physics_option_updates_preserve_scene_structure(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(exchange.display_model, exchange.display_data)
    session = Session(adapter)
    try:
        source, generation = session.source, session.structure_generation
        m.opt.gravity[2] = -3
        m.opt.timestep = 0.01
        exchange.sync()
        adapter.refresh_model_fields(exchange.pending_model_fields)
        exchange.pending_model_fields.clear()
        adapter.refresh_model_visuals()
        session.tick(FrameNeeds(poses=True))
        assert session.source is source and session.structure_generation == generation
        assert exchange.display_model.opt.gravity[2] == -3
        assert exchange.display_model.opt.timestep == 0.01
        # Actual resource changes must still invalidate uploaded scene data.
        m.tex_data[:] = 123
        exchange.sync()
        adapter.refresh_model_fields(exchange.pending_model_fields)
        exchange.pending_model_fields.clear()
        session.tick(FrameNeeds(poses=True))
        assert session.source is not source and session.structure_generation > generation
    finally:
        session.release()


@pytest.mark.parametrize(
    "category, field", [("tex", "tex_data"), ("hfield", "hfield_data"), ("mesh", "mesh_vert")]
)
def test_resource_upload_publishes_only_requested_id(category, field):
    assets = {
        "tex": '<texture name="resource{}" type="2d" builtin="checker" width="4" height="4"/>',
        "hfield": '<hfield name="resource{}" nrow="3" ncol="3" size="1 1 1 .1"/>',
        "mesh": '<mesh name="resource{}" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>',
    }
    assets_xml = assets[category].format(0) + assets[category].format(1)
    m = mujoco.MjModel.from_xml_string(f"<mujoco><asset>{assets_xml}</asset></mujoco>")
    exchange = StateExchange(m, mujoco.MjData(m))
    caller, display = getattr(m, field), getattr(exchange.display_model, field)
    baseline = display.copy()
    caller[:] = 0.5 if category != "tex" else 123
    m.opt.gravity[2] = -3
    exchange.sync_resource(category, 0)
    midpoint = len(display) // 2
    np.testing.assert_array_equal(display[:midpoint], caller[:midpoint])
    np.testing.assert_array_equal(display[midpoint:], baseline[midpoint:])
    assert exchange.display_model.opt.gravity[2] != -3
    exchange.sync()
    np.testing.assert_array_equal(display, caller)


def test_resource_layout_change_is_rejected_before_upload(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    original = exchange.display_model.tex_data.copy()
    m.tex_data[:] = 123
    m.tex_width[0] += 1
    with pytest.raises(ValueError, match="layout changed"):
        exchange.sync_resource("tex", 0)
    np.testing.assert_array_equal(exchange.display_model.tex_data, original)


def test_active_perturbation_reapplies_force_on_each_sync(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    # Simulate a UI drag whose target stays fixed while the caller advances/clears forces.
    pert = exchange.display_perturb
    pert.select = 1
    pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
    pert.localmass = 1
    pert.refselpos[:] = (1, 0, 0)
    exchange.sync(True)
    first = d.xfrc_applied.copy()
    assert first[1, 0] > 0
    d.xfrc_applied[:] = 0
    exchange.sync(True)
    np.testing.assert_array_equal(d.xfrc_applied, first)


def test_model_update_invalidates_only_changed_visual_source(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(exchange.display_model, exchange.display_data)
    source = adapter.scene_source()
    exchange.sync()
    assert not adapter.refresh_model_visuals()
    assert adapter.scene_source() is source
    m.geom_rgba[0] = (1, 0, 0, 1)
    exchange.sync()
    assert adapter.refresh_model_visuals()
    np.testing.assert_array_equal(adapter.scene_source().geom_rgba[0], (1, 0, 0, 1))
    source = adapter.scene_source()
    assert adapter.refresh_model_visuals(force=True)
    assert adapter.scene_source() is not source


@pytest.mark.parametrize("azimuth", [-90, 0, 90, 180])
def test_camera_round_trip_keeps_mujoco_angle_conventions(model_data, azimuth):
    exchange = StateExchange(*model_data)
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(exchange.display_model, exchange.display_data)
    app = SimpleNamespace(camera=OrbitCamera(), _model_camera_id=-1)
    display = SimpleNamespace(
        app=app,
        session=SimpleNamespace(adapter=adapter),
        backend=NullBackend(),
        set_camera=lambda view: app.camera.adopt(view, exact=True),
    )
    presentation = Presentation(exchange, mujoco.MjvScene(exchange.display_model, maxgeom=0))
    exchange.cam.azimuth, exchange.cam.elevation, exchange.cam.distance = azimuth, -20, 4
    exchange.cam.lookat[:] = (1, 2, 3)
    exchange.sync()
    presentation.before_frame(display)
    np.testing.assert_allclose(app.camera.pivot, (1, 2, 3))
    assert app.camera.distance == pytest.approx(4, rel=1e-6)
    app.camera.yaw += 10
    presentation.after_frame(display)
    exchange.sync()
    assert ((exchange.cam.azimuth - azimuth - 10 + 180) % 360) - 180 == pytest.approx(0, abs=1e-5)
    assert exchange.cam.elevation == pytest.approx(-20, abs=1e-5)


def test_user_geometry_snapshot_is_owned_and_rejects_unsupported_types(model_data):
    scene = mujoco.MjvScene(model_data[0], maxgeom=2)
    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [0.1, 0, 0],
        [1, 2, 3],
        np.eye(3).reshape(-1),
        [1, 0, 0, 1],
    )
    scene.ngeom = 1
    snapshot = snapshot_geometries(scene)
    scene.geoms[0].pos[:] = 9
    np.testing.assert_array_equal(snapshot[0].position, (1, 2, 3))
    scene.geoms[0].type = mujoco.mjtGeom.mjGEOM_MESH
    with pytest.raises(NotImplementedError, match="mesh"):
        snapshot_geometries(scene)


def test_invalid_camera_is_rejected_before_publishing_model_or_data(model_data):
    m, d = model_data
    exchange = StateExchange(m, d)
    d.qpos[0] = 2
    m.geom_rgba[0] = (1, 0, 0, 1)
    original = exchange.display_model.geom_rgba.copy()
    exchange.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    exchange.cam.fixedcamid = 100
    with pytest.raises(ValueError, match="fixedcamid"):
        exchange.sync()
    assert exchange.display_data.qpos[0] == 0
    np.testing.assert_array_equal(exchange.display_model.geom_rgba, original)


def test_fixed_camera_selection_survives_ui_readback():
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera pos="2 0 2"/></worldbody></mujoco>'
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    exchange = StateExchange(m, d)
    adapter = MuJoCoAdapter(external_clock=True)
    adapter.load_model(exchange.display_model, exchange.display_data)
    app = SimpleNamespace(camera=OrbitCamera(), _model_camera_id=-1)
    session = SimpleNamespace(adapter=adapter, camera=app.camera.view())

    def set_view(view, *, camera_id=-1):
        app.camera.adopt(view, exact=True)
        app._model_camera_id = camera_id
        session.camera = view

    app.set_viewport_camera = set_view
    display = SimpleNamespace(app=app, session=session, backend=NullBackend(), set_camera=set_view)
    presentation = Presentation(exchange, mujoco.MjvScene(exchange.display_model, maxgeom=0))
    exchange.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    exchange.cam.fixedcamid = 0
    exchange.sync()
    presentation.before_frame(display)
    assert app._model_camera_id == 0
    presentation.after_frame(display)
    exchange.sync()
    assert exchange.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED and exchange.cam.fixedcamid == 0


def test_ui_switches_hide_but_do_not_disable_features_and_tab_restores(monkeypatch):
    import glfw

    defaults = viewer_config(None, True, True)
    assert defaults.builtin_panels is None and not defaults.panels
    config = viewer_config(None, False, False)
    panels = PanelManager(config=config.panels)
    assert all(panel.enabled and not panel.open for panel in panels)
    installed = []
    previous_calls = []
    callback_calls = []

    def install(window, callback):
        installed.append(callback)
        return lambda *args: previous_calls.append(args)

    monkeypatch.setattr(glfw, "set_key_callback", install)
    presets = []
    app = SimpleNamespace(
        localizer=SimpleNamespace(preference=lambda name, default: default),
        set_navigation_preset=lambda name, **kwargs: presets.append((name, kwargs)),
    )
    display = SimpleNamespace(window=SimpleNamespace(_window=object()), panels=panels, app=app)
    install_key_callback(display, callback_calls.append)
    assert presets == [("MuJoCo", {"persist": False})]
    installed[0](None, glfw.KEY_TAB, 0, glfw.PRESS, 0)
    assert all(panels.get(name).open == panels.get(name).default_open for name in LEFT_PANELS)
    assert all(not panels.get(name).open for name in RIGHT_PANELS)
    installed[0](None, glfw.KEY_TAB, 0, glfw.PRESS, glfw.MOD_SHIFT)
    assert all(panel.open == panel.default_open for panel in panels)
    assert callback_calls == [glfw.KEY_TAB, glfw.KEY_TAB]
    assert len(previous_calls) == 2
    panels.open("assets")
    panels.close("hierarchy")
    installed[0](None, glfw.KEY_TAB, 0, glfw.PRESS, 0)
    installed[0](None, glfw.KEY_TAB, 0, glfw.PRESS, 0)
    assert panels.get("assets").open and not panels.get("hierarchy").open
    app.localizer.preference = lambda *args: {"orbit": "middle"}
    install_key_callback(display, None)
    assert len(presets) == 1


@pytest.mark.parametrize("use_loader", [False, True])
@pytest.mark.parametrize("selected_panels", [False, True])
def test_managed_launch_owns_clock_and_preserves_model_data(
    model_data, monkeypatch, use_loader, selected_panels
):
    from mojive.app import composition

    m, d = model_data
    calls = []
    config = (
        ViewerConfig(
            builtin_panels=("inspector", "control", "joints", "camera"),
            panels={"joints": PanelConfig(open=False)},
        )
        if selected_panels
        else None
    )

    class Managed:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def run(self):
            calls.append("run")
            mujoco.mj_step(m, d)

    def build(**kwargs):
        assert kwargs["model"] is m and kwargs["data"] is d
        assert kwargs["external_clock"] is False and kwargs["paused"] is False
        options = kwargs["config"]
        panels = PanelManager(config=options.panels, builtin_ids=options.builtin_panels)
        if selected_panels:
            assert {panel.id for panel in panels} == {"inspector", "control", "joints", "camera"}
            assert not panels.get("joints").open
            assert panels.get("inspector").open and panels.get("camera").open
        else:
            assert options.builtin_panels is None
            assert all(panel.enabled and panel.open == panel.default_open for panel in panels)
        calls.append("build")
        return Managed()

    monkeypatch.setattr(composition, "build", build)
    monkeypatch.setattr(launch_module, "install_key_callback", lambda *args: None)
    if use_loader:
        assert viewer.launch(loader=lambda: (m, d), config=config) is None
    else:
        assert viewer.launch(m, d, config=config) is None
    assert d.time > 0 and calls == ["build", "run", "closed"]


def test_launch_rejects_invalid_combinations_before_window(model_data):
    m, d = model_data
    with pytest.raises(ValueError, match="requires a model"):
        viewer.launch(data=d)
    with pytest.raises(ValueError, match="either"):
        viewer.launch(m, loader=lambda: (m, d))
    with pytest.raises(ValueError, match="callback"):
        viewer.launch_passive(m, d, key_callback=3)
    with pytest.raises(ValueError, match="different"):
        viewer.launch_passive(m, mujoco.MjData(mujoco.MjModel.from_xml_string("<mujoco/>")))


@pytest.mark.parametrize("selected_panels", [False, True])
def test_passive_thread_sync_resource_completion_and_close(
    model_data, monkeypatch, selected_panels
):
    from mojive.app import composition

    # A synthetic display can exercise the threaded lifecycle on every host.
    monkeypatch.setattr(launch_module, "sys", SimpleNamespace(platform="linux"))
    created = []

    class Display:
        def __init__(self, adapter):
            self.session = Session(adapter)
            self.backend = NullBackend()
            self.backend.debug = DebugDraw()
            self.app = SimpleNamespace(
                camera=OrbitCamera(),
                _startup=lambda: None,
                _current_viewport_render_size=lambda: (640, 480),
                _viewport_rect=(10, 20, 640, 480),
                _model_camera_id=-1,
            )
            self.window = SimpleNamespace(
                points_to_pixels=lambda value: value, size_pixels=(800, 600)
            )
            self.frames = 0
            self.closed = False

        def is_running(self):
            return not self.closed

        def set_camera(self, view):
            self.app.camera.adopt(view, exact=True)

        def sync(self):
            self.session.tick(FrameNeeds(poses=True))
            self.frames += 1

        def release(self):
            self.closed = True
            self.session.release()

    def build(adapter, **kwargs):
        config = kwargs["config"]
        panels = PanelManager(config=config.panels, builtin_ids=config.builtin_panels)
        if selected_panels:
            assert {panel.id for panel in panels} == {"inspector", "camera"}
            assert not panels.get("camera").open
            assert panels.get("inspector").open
        else:
            assert config.builtin_panels is None
            assert all(panel.enabled and panel.open == panel.default_open for panel in panels)
        display = Display(adapter)
        created.append(display)
        return display

    monkeypatch.setattr(composition, "build_from_adapter", build)
    monkeypatch.setattr(handle_module, "install_key_callback", lambda *args: None)
    m, d = model_data
    config = (
        ViewerConfig(
            builtin_panels=("inspector", "camera"), panels={"camera": PanelConfig(open=False)}
        )
        if selected_panels
        else None
    )
    with viewer.launch_passive(m, d, config=config) as handle:
        assert handle.m is m and handle.d is d
        assert handle.is_running()
        assert isinstance(handle.cam, mujoco.MjvCamera)
        assert isinstance(handle.opt, mujoco.MjvOption)
        with pytest.raises(RuntimeError, match="Only one"):
            viewer.launch_passive(m, d)
        with handle.lock():
            d.qpos[0] = 2
            handle.sync(True)
            assert handle._exchange.display_data.qpos[0] == 2
        m.tex_data[:] = 0
        m.opt.gravity[2] = -3
        handle.update_texture(0)
        assert created[0].frames > 0 and handle.viewport.width == 640
        assert handle.viewport.left == 10 and handle.viewport.bottom == 100
        with handle.lock():
            assert handle._exchange.display_model.opt.gravity[2] != -3
            np.testing.assert_array_equal(handle._exchange.display_model.tex_data, m.tex_data)
        with pytest.raises(ValueError, match="outside"):
            handle.update_texture(10)
    assert not handle.is_running() and handle.m is None and handle.d is None
    assert created[0].closed
    handle.sync()
    handle.close()
    assert not any(thread.name == "mojive-viewer" for thread in threading.enumerate())


def test_passive_startup_failure_is_reported_and_thread_reaped(model_data, monkeypatch):
    from mojive.app import composition

    monkeypatch.setattr(launch_module, "sys", SimpleNamespace(platform="linux"))

    def fail(*args, **kwargs):
        raise RuntimeError("missing display")

    monkeypatch.setattr(composition, "build_from_adapter", fail)
    with pytest.raises(RuntimeError, match="could not start") as error:
        viewer.launch_passive(*model_data)
    assert "missing display" in str(error.value.__cause__)


def test_passive_macos_requires_the_process_viewer_before_starting_a_thread(
    model_data, monkeypatch
):
    monkeypatch.setattr(launch_module, "sys", SimpleNamespace(platform="darwin"))
    with pytest.raises(NotImplementedError, match="process-based viewing"):
        viewer.launch_passive(*model_data)
    assert not any(thread.name == "mojive-viewer" for thread in threading.enumerate())
