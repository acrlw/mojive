"""Rate measurements follow executed steps, independently of display sampling."""

import pytest

from mojive.session.rates import StepRate


def test_physics_rate_uses_counter_deltas_at_display_frequency():
    rate = StepRate()
    for frame in range(61):
        measured = rate.update(frame * 20, frame * 0.01, now=frame / 50)
    assert measured == pytest.approx(1000)
    # Changing the physics timestep does not change executed steps per wall second.
    assert rate.update(1600, 1.6, now=1.6) == pytest.approx(1000)


def test_physics_rate_handles_pause_and_episode_reset():
    rate = StepRate()
    rate.update(0, 0, now=0)
    assert rate.update(500, 0.5, now=0.5) == pytest.approx(1000)
    assert rate.update(500, 0.5, now=1.0) == 0
    assert rate.update(0, 0, now=1.001) == 0
    assert rate.update(250, 0.5, now=1.501) == pytest.approx(500)


def test_physics_time_reset_restarts_a_counter_interval():
    rate = StepRate()
    rate.update(10, 1.0, now=0)
    rate.update(500, 2.0, now=0.5)
    assert rate.update(501, 0.0, now=0.6) == 0
