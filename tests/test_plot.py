"""Telemetry must draw only valid samples from the current scene."""

from types import SimpleNamespace

import numpy as np
import pytest

from mojive.ui.panels import plot
from mojive.ui.panels.plot import PlotPanel, Ring


def test_partial_and_wrapped_plot_buffers_exclude_unwritten_samples(monkeypatch):
    observed = []
    monkeypatch.setattr(
        plot.imgui,
        "plot_lines",
        lambda _, values, **kw: observed.append(np.roll(values, -kw["values_offset"]).copy()),
    )
    panel = PlotPanel()
    ring = Ring(4)
    for value in (5, 6, 7, 8, 9):
        ring.push(value)
        panel._curve("qpos", ring, 90, "joint")
    np.testing.assert_array_equal(observed[0], [5])
    np.testing.assert_array_equal(observed[2], [5, 6, 7])
    np.testing.assert_array_equal(observed[-1], [6, 7, 8, 9])
    assert np.shares_memory(ring.values, ring.data)
    ring.clear()
    panel._curve("qpos", ring, 90, "joint")
    assert observed[-1].size == 0


@pytest.mark.parametrize("source", ("joint", "sensor"))
def test_plot_clears_samples_when_structure_replaces_the_tracked_channel(monkeypatch, source):
    panel = PlotPanel()
    panel.source = source
    monkeypatch.setattr(plot, "segmented_control", lambda *a, **kw: int(source == "sensor"))
    monkeypatch.setattr(plot.imgui, "spacing", lambda: None)
    monkeypatch.setattr(panel, "_draw_joint", lambda ctx: None)
    monkeypatch.setattr(panel, "_draw_sensor", lambda ctx: None)
    ctx = SimpleNamespace(session=SimpleNamespace(structure_generation=1), tr=str, theme=None)
    panel.draw(ctx)
    rings = (panel._angle, panel._velocity, panel._contact, panel._sensor)
    for ring in rings:
        ring.push(42)
    panel.draw(ctx)
    assert all(ring.filled == 1 for ring in rings)
    ctx.session.structure_generation += 1
    panel.draw(ctx)
    assert all(ring.filled == 0 for ring in rings)


def test_missing_joint_addresses_do_not_sample_the_last_state_element():
    panel = PlotPanel()
    frame = SimpleNamespace(qpos=np.array([42]), qvel=np.array([84]))
    joint = SimpleNamespace(qpos_adr=-1, qvel_adr=-1)
    panel._sample(SimpleNamespace(session=SimpleNamespace(frame=frame, joints=[joint])))
    assert panel._angle.filled == panel._velocity.filled == 0
