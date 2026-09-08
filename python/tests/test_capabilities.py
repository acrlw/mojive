"""Unsupported backend features never enter a mutating command path."""

from dataclasses import replace
from pathlib import Path

import pytest

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import AdapterCaps
from mojive.adapters.static import StaticSceneAdapter
from mojive.command_support import unavailable_reason
from mojive.control_rpc import ControlService
from mojive.control_schema import CAPABILITIES_RESULT, Validator
from mojive.session import Session
from mojive.ui.app import _model_filters, _scene_filters


def test_formats_and_extension_revisions_are_explicit():
    caps = AdapterCaps(
        name="other", asset_loading=True, model_formats=(".urdf",), features=(("other.drive", 2),)
    )
    assert caps.accepts_model("Robot.URDF")
    assert not caps.accepts_model("robot.xml")
    assert not caps.supports("other.drive")
    assert caps.supports("other.drive", 2)
    assert not caps.supports("future.feature")
    assert not replace(caps, features=(("other.drive", True),)).supports("other.drive")
    assert not replace(caps, features=(("write_ctrl", 1),)).supports("write_ctrl")
    assert caps.supports("asset_loading")
    assert not caps.supports("asset_loading", 2)
    assert _model_filters(caps) == ["Supported models", "*.urdf"]
    assert "*.xml" not in " ".join(_scene_filters(caps, saving=True))
    assert not _model_filters(replace(caps, asset_loading=False))
    assert not _model_filters(AdapterCaps(asset_loading=True))


def test_unsupported_command_does_not_fence_physics_or_capture_history(monkeypatch):
    session = Session(StaticSceneAdapter(Scene()))
    try:
        monkeypatch.setattr(
            session,
            "_capture_document_state",
            lambda: pytest.fail("captured unsupported edit history"),
        )
        monkeypatch.setattr(
            session, "_dispatch", lambda command: pytest.fail("dispatched unsupported command")
        )
        for command in (
            cmd.LoadAsset(Path("unknown.xml")),
            cmd.SetModelSource(0, "<mujoco/>"),
            cmd.AddModelComponent(0, "tendon", "spatial", "tendon"),
        ):
            assert not session.submit(command).ok
    finally:
        session.release()


def test_generic_topology_does_not_imply_mjcf_or_component_editing():
    caps = AdapterCaps(name="other", topology_editing=True)
    assert unavailable_reason(caps, cmd.AddModelElement(0, "body", "body")) is None
    assert "mujoco.mjcf" in unavailable_reason(caps, cmd.SetModelSource(0, "<mujoco/>"))
    assert "model.components" in unavailable_reason(
        caps, cmd.AddModelComponent(0, "sensor", "x", "x")
    )


def test_rpc_discovery_and_revision_rejection_precede_mutation():
    service = ControlService(StaticSceneAdapter(Scene()))
    try:
        caps = service.dispatch("get_capabilities", {})
        Validator(CAPABILITIES_RESULT).validate(caps)
        assert caps["method_versions"]["mujoco.set_model_source"] == 1
        assert "mujoco.set_model_source" not in caps["available_methods"]
        result = service.handle(
            {
                "id": 1,
                "version": 1,
                "method": "add_scene_object",
                "operation_version": 2,
                "params": {"shape": "box"},
            }
        )
        assert result["error"]["code"] == "operation_version_mismatch"
        assert service.session.source.instance_count == 0
        assert service.handle(
            {"id": 2, "version": 1, "method": "add_scene_object", "params": {"shape": "box"}}
        )["result"]["ok"]
    finally:
        service.close()


def test_unsupported_edit_invalidates_the_active_transaction():
    session = Session(StaticSceneAdapter(Scene()))
    try:
        assert session.submit(cmd.BeginEditTransaction("Atomic edit"))
        assert session.submit(cmd.AddSceneObject("box"))
        assert not session.submit(cmd.SetModelSource(0, "<mujoco/>"))
        assert not session.submit(cmd.EndEditTransaction())
        assert session.source.instance_count == 0
    finally:
        session.release()


def test_control_writeback_is_independent_of_simulation():
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.operations import OPERATIONS

    session = Session(ToyPhysicsAdapter())
    try:
        assert session.adapter.caps.simulation
        assert not session.submit(cmd.SetCtrl(0, 0.5))
        assert not session.submit(cmd.SetCtrlVector([0.5]))
        assert OPERATIONS["set_ctrl"].unavailable_reason(session)
        session.adapter.caps = replace(
            session.adapter.caps, external_clock=True, clock_control=True
        )
        assert not session.submit(cmd.SetSpeed(2))
        assert session.speed == 1
        assert OPERATIONS["set_speed"].unavailable_reason(session)
    finally:
        session.release()


def test_external_clock_commands_are_rejected_before_the_session_fence(monkeypatch):
    from mojive.adapters.toy import ToyPhysicsAdapter

    session = Session(ToyPhysicsAdapter())
    try:
        monkeypatch.setattr(
            session, "_submit", lambda command: pytest.fail("entered unsupported command fence")
        )
        session.adapter.caps = replace(session.adapter.caps, clock_control=False)
        for command in (cmd.Pause(), cmd.Play(), cmd.Step(), cmd.Reset(), cmd.SetSpeed(2)):
            assert not session.submit(command).ok
        session.adapter.caps = replace(
            session.adapter.caps, clock_control=True, external_clock=True
        )
        assert not session.submit(cmd.SetSpeed(2)).ok
    finally:
        session.release()
