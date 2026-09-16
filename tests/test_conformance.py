"""The reference second backend and reusable SceneAdapter conformance checks."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import (
    AdapterCaps,
    FrameNeeds,
    SceneAdapter,
    SceneAdapterBase,
    SceneInspection,
)
from mojive.adapters.conformance import check_adapter, check_scene_provider
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.app.backends import available_backends, make_adapter
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import Light, LightSet

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("simulation", [False, True])
def test_read_only_inspection_does_not_require_editing_or_simulation_stubs(simulation):
    scene = Scene()
    scene.box()
    source = StaticSceneAdapter(scene)

    class Inspector:
        caps = AdapterCaps(name="inspection", simulation=simulation)

        def __getattr__(self, name):
            if name in {
                "structure_revision",
                "scene_source",
                "frame",
                "release",
                "nodes",
                "joints",
                "actuators",
                "cameras",
                "keyframes",
                "sensors",
                "equality_constraints",
                "camera_view",
                "visual_groups",
                "raycast",
                "camera_hint",
                "timestep",
                "scene_models",
            }:
                return getattr(source, name)
            raise AttributeError(name)

    inspector = Inspector()
    assert isinstance(inspector, SceneInspection)
    assert not isinstance(inspector, SceneAdapter)
    report = check_adapter(inspector)
    if simulation:
        required = next(check for check in report.checks if check.name == "editor runtime methods")
        assert not required.ok
        assert all(name in required.detail for name in ("step", "reset", "set_paused"))
    else:
        assert report.ok, [check for check in report.checks if not check.ok]


def test_toy_is_a_real_available_backend_and_passes_the_shared_contract():
    info = next(item for item in available_backends() if item.name == "toy")
    assert info.available
    adapter = make_adapter("toy")
    try:
        report = check_adapter(adapter)
        assert report.ok, [check for check in report.checks if not check.ok]
    finally:
        adapter.release()


def test_toy_physics_steps_edits_and_resets_through_session_commands():
    adapter = ToyPhysicsAdapter()
    session = Session(adapter)
    start = adapter._positions.copy()

    frame = session.tick(FrameNeeds(), wall_dt=adapter.timestep())
    assert frame.step == 1
    assert not np.allclose(adapter._positions, start)
    assert frame.debug_commands[0]["text"].startswith("toy physics")

    assert session.submit(cmd.Pause())
    node = next(node for node in session.nodes if node.name == "red ball")
    target = np.array([0.5, -0.5, 2.0], np.float32)
    assert session.submit(cmd.SetPose(node.node_id, target, np.eye(3)))
    assert np.allclose(adapter._positions[0], target)

    assert session.submit(cmd.Reset())
    assert np.allclose(adapter._positions, start)
    assert adapter._steps == 0


def test_conformance_report_names_a_broken_instance_column():
    scene = Scene()
    scene.box()
    source = scene.source
    source.geom_rgba = source.geom_rgba[:0]

    report = check_adapter(StaticSceneAdapter(scene))

    failed = {check.name for check in report.checks if not check.ok}
    assert "instance columns" in failed


def test_conformance_rejects_a_cycle_in_node_parents():
    scene = Scene()
    scene.box()
    source = scene.source
    source.nodes[0].parent = source.nodes[-1].node_id
    source.nodes[-1].parent = source.nodes[0].node_id
    report = check_adapter(StaticSceneAdapter(scene))
    graph = next(check for check in report.checks if check.name == "node graph")
    assert not graph.ok
    assert "cycle" in graph.detail


def test_conformance_requires_every_light_to_be_an_entity():
    scene = Scene(lights=LightSet(lights=(Light(),)))
    source = scene.source
    source.nodes = [node for node in source.nodes if node.light_index < 0]

    report = check_adapter(StaticSceneAdapter(scene))

    failed = {check.name for check in report.checks if not check.ok}
    assert "light entities" in failed


def test_conformance_requires_every_camera_to_be_an_entity():
    from mojive.types import CameraView

    scene = Scene(camera=CameraView())
    source = scene.source
    source.nodes = [node for node in source.nodes if node.camera_index < 0]

    report = check_adapter(StaticSceneAdapter(scene))

    failed = {check.name for check in report.checks if not check.ok}
    assert "camera entities" in failed


def test_read_only_provider_needs_no_editor_or_physics_interface():
    scene = Scene()
    scene.box()

    class Provider:
        structure_revision = 0

        def scene_source(self):
            return scene.source

        def frame(self, needs):
            return scene.frame

    report = check_scene_provider(Provider())
    assert report.ok, report.checks


def test_minimal_editor_adapter_can_inherit_all_unadvertised_defaults():
    scene = Scene()
    scene.box()

    class Adapter(SceneAdapterBase):
        def scene_source(self):
            return scene.source

        def frame(self, needs):
            return scene.frame

    report = check_adapter(Adapter())
    assert report.ok, report.checks


def test_claimed_write_capability_reports_inherited_unsupported_methods():

    adapter = StaticSceneAdapter(Scene())
    adapter.caps = replace(adapter.caps, write_qpos=True)
    report = check_adapter(adapter)
    failure = next(check for check in report.checks if check.name == "capability write_qpos")
    assert not failure.ok
    assert "set_qpos_batch" in failure.detail
    assert "set_qpos" in failure.detail


def test_external_clock_does_not_require_internal_stepping():

    adapter = StaticSceneAdapter(Scene())
    adapter.caps = replace(adapter.caps, simulation=True, external_clock=True, clock_control=False)
    adapter.timestep = lambda: 0.01
    report = check_adapter(adapter)
    assert report.ok, report.checks


def test_missing_provider_methods_and_stream_failures_are_named():
    report = check_scene_provider(object())
    assert not report.ok
    assert "scene_source" in report.checks[0].detail

    class Broken(SceneAdapterBase):
        caps = AdapterCaps(name="broken")

        def scene_source(self):
            raise ValueError("bad fixture")

    report = check_adapter(Broken())
    assert not report.ok
    assert any("ValueError: bad fixture" in check.detail for check in report.checks)
