"""Native design controls retain adapter addressing and capability boundaries."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import ActuatorInfo, FrameNeeds
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.toy import ToyPhysicsAdapter
from mojive.session import Session


class ControlAdapter(StaticSceneAdapter):
    def __init__(self, *, writable=True, produced=True):
        super().__init__(Scene())
        self.caps = replace(self.caps, write_ctrl=writable)
        self.values = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        self.produced = produced

    def actuators(self):
        return [
            ActuatorInfo(0, "drive", (-1, 1), True, ctrl_address=2, ctrl_count=2),
            ActuatorInfo(1, "brake", (-1, 1), True, ctrl_address=0),
        ]

    def frame(self, needs):
        frame = super().frame(needs)
        frame.ctrl = self.values.copy() if self.produced else None
        return frame

    def set_ctrl(self, index, value):
        self.values[index] = value
        return True

    def set_ctrl_vector(self, values):
        if values.shape != self.values.shape:
            return False
        self.values[:] = values
        return True


class Controls:
    def __init__(self, disabled):
        self.disabled = disabled
        self.hits = {}
        self.buttons = {}
        self.sliders = {}
        self.clicks = set()
        self.edits = {}

    def icon(self, *args):
        pass

    def text(self, *args, **kwargs):
        pass

    def button(self, key, *args, enabled=True, **kwargs):
        self.hits[key] = (0, 0, 1, 1)
        self.buttons[key] = enabled
        return key in self.clicks and enabled

    def slider(self, key, x, y, width, value, *args):
        enabled = not any(self.disabled)
        self.sliders[key] = (value, enabled)
        return key in self.edits and enabled, self.edits.get(key, value)


def study_controls(monkeypatch, session):
    disabled = []
    monkeypatch.setattr(imgui, "begin_disabled", lambda value: disabled.append(value))
    monkeypatch.setattr(imgui, "end_disabled", lambda: disabled.pop())
    monkeypatch.setattr(imgui, "set_cursor_screen_pos", lambda value: None)
    monkeypatch.setattr(imgui, "dummy", lambda value: None)
    monkeypatch.setattr(imgui, "is_popup_open", lambda value: False)
    study = Study.__new__(Study)
    study.s = 1
    study.ui = Controls(disabled)
    study.preview = SimpleNamespace(
        session=session,
        _toggle_playback=Mock(),
        _reset_playback=Mock(),
        _toggle_state_take_recording=Mock(),
    )
    study.menus = SimpleNamespace(popup=Mock())
    return study


def test_multicomponent_controls_write_addresses_and_reset_the_full_vector(monkeypatch):
    adapter = ControlAdapter()
    session = Session(adapter)
    try:
        study = study_controls(monkeypatch, session)
        study.ui.edits["##ctrl-3"] = 0.9
        study.control_panel(0, 0, 300)
        assert study.ui.sliders == {
            "##ctrl-2": (0.4, True),
            "##ctrl-3": (0.5, True),
            "##ctrl-0": (0.2, True),
        }
        np.testing.assert_array_equal(adapter.values, [0.2, 0.3, 0.4, 0.9, 0.6])
        study.ui.edits.clear()
        study.ui.clicks.add("reset-actuators")
        study.control_panel(0, 0, 300)
        np.testing.assert_array_equal(adapter.values, np.zeros(5))
    finally:
        session.release()


@pytest.mark.parametrize("writable,produced", [(False, True), (True, False)])
def test_unavailable_controls_disable_edits_and_reset(monkeypatch, writable, produced):
    adapter = ControlAdapter(writable=writable, produced=produced)
    session = Session(adapter)
    try:
        study = study_controls(monkeypatch, session)
        study.ui.edits["##ctrl-3"] = 0.9
        study.ui.clicks.add("reset-actuators")
        study.control_panel(0, 0, 300)
        assert all(not enabled for _value, enabled in study.ui.sliders.values())
        assert not study.ui.buttons["reset-actuators"]
        np.testing.assert_array_equal(adapter.values, [0.2, 0.3, 0.4, 0.5, 0.6])
    finally:
        session.release()


@pytest.mark.parametrize("simulation,playing", [(True, False), (True, True), (False, False)])
def test_transport_respects_simulation_state_and_capabilities(monkeypatch, simulation, playing):
    adapter = ToyPhysicsAdapter() if simulation else StaticSceneAdapter(Scene())
    session = Session(adapter)
    try:
        if simulation:
            assert session.submit(cmd.Play() if playing else cmd.Pause()).ok
        study = study_controls(monkeypatch, session)
        study.ui.clicks.update(("previous", "next", "record"))
        study.transport(0, 0, 1000, lambda *args: None)
        assert study.ui.buttons["previous"] == session.can_step_back
        assert study.ui.buttons["next"] == (simulation and not playing)
        assert study.ui.buttons["record"] == simulation
        assert study.preview._toggle_state_take_recording.call_count == int(simulation)
        if simulation and not playing:
            frame = session.tick(FrameNeeds.none(), wall_dt=0)
            assert frame.step == 1
        elif simulation:
            assert session.frame.step == 0
    finally:
        session.release()
