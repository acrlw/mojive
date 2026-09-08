"""The reference second backend and reusable SceneAdapter conformance checks."""

from __future__ import annotations

import numpy as np
import pytest

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.conformance import check_adapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.backends import available_backends, make_adapter
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import Light, LightSet

pytestmark = pytest.mark.integration


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
