"""Recorded sample timestamps, axis labels and replay share one visible time mapping."""

from itertools import pairwise

import pytest

from mojive import commands as cmd
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.timeline import timeline_time_to_x
from mojive.ui.workspace.menu_geometry import native_epsilon
from tests.test_native_timeline_boundaries import production_timeline as production_timeline
from tests.test_ui_design_timeline import timeline as timeline

pytestmark = [
    pytest.mark.physics,
    pytest.mark.parametrize("timeline", (1.0, 1.2, 1.25, 2.25), indirect=True),
]


@pytest.fixture
def readable_timeline(production_timeline, tmp_path):
    state = production_timeline
    runtime = DesignRuntime(SpecificationStore(tmp_path / "timeline-design.json"))
    runtime.observe_components = True
    state.ui.design = DesignPresentation(state.ui, runtime)
    state.runtime = runtime
    state.frame(3)
    return state


def _record(state, count, cadence):
    assert state.session.submit(cmd.StartStateTakeRecording()).ok
    state.frame(count, wall_dt=cadence)
    assert state.session.submit(cmd.StopStateTakeRecording()).ok
    state.frame(3)
    return tuple(state.session.state_take_times)


def _latest(state, key):
    return next(record for record in reversed(state.runtime.observed) if record["key"] == key)


def _view(state, first, last):
    state.panel.editor.set_follow_mode("off")
    assert state.panel.editor.set_view_range(first, last)
    state.frame(3)


@pytest.mark.parametrize("native_fonts", (True,))
def test_real_sample_marks_identify_recorded_times_and_seek_to_the_same_playhead(
    readable_timeline, monkeypatch, native_fonts
):
    state = readable_timeline
    times = _record(state, 8, 0.016)
    _view(state, times[0], times[-1])
    index = 2
    sample = _latest(state, f"timeline-take-frame-{index}")
    bounds = state.panel.last_layout.time_bounds
    expected = timeline_time_to_x(times[index], times[0], times[-1], *bounds)
    assert sample["sampleX"] == pytest.approx(expected)
    assert sample["time"] == times[index]
    assert sample["time"] != sample["cellBounds"][0]
    state.click_at((sample["sampleX"], state.panel.last_layout.lane_y(1)))
    assert state.session.state_take_cursor == index
    assert state.panel.editor.playhead == times[index]
    playhead = _latest(state, "timeline-playhead")
    assert playhead["time"] == times[index]
    assert playhead["playheadX"] == pytest.approx(expected)
    assert _latest(state, f"timeline-take-frame-{index}")["active"]
    tooltips = []
    monkeypatch.setattr(state.ui, "tooltip", lambda text, **_options: tooltips.append(text))
    state.event("add_mouse_pos_event", expected, state.panel.last_layout.lane_y(1))
    assert f"Recorded frame {index + 1}/{len(times)}\n{times[index]:.6f} s" in tooltips


@pytest.mark.parametrize("native_fonts", (True,))
def test_take_hover_keeps_exact_frame_identity_when_dense_and_does_not_invent_deleted_frames(
    readable_timeline, monkeypatch, native_fonts
):
    state = readable_timeline
    times = _record(state, 300, 0.002)
    _view(state, times[0], times[-1])
    assert not any(key.startswith("timeline-take-frame-") for key in state.ui.hits)
    tooltips = []
    monkeypatch.setattr(state.ui, "tooltip", lambda text, **_options: tooltips.append(text))
    index = 150
    x = timeline_time_to_x(times[index], times[0], times[-1], *state.panel.last_layout.time_bounds)
    state.event("add_mouse_pos_event", x, state.panel.last_layout.lane_y(1))
    assert f"Recorded frame {index + 1}/{len(times)}\n{times[index]:.6f} s" in tooltips
    assert state.session.submit(cmd.DeleteStateTakeFrames(index - 4, index + 4)).ok
    state.frame(3)
    tooltips.clear()
    state.event("add_mouse_pos_event", x, state.panel.last_layout.lane_y(1))
    assert not tooltips
    assert tuple(state.session.state_take_times) == times[: index - 4] + times[index + 5 :]


@pytest.mark.parametrize("native_fonts", (True,))
def test_single_recorded_frame_hover_uses_the_visible_marker_width(
    readable_timeline, monkeypatch, native_fonts
):
    state = readable_timeline
    times = _record(state, 0, 0.008)
    assert len(times) == 1
    _view(state, times[0], times[0] + 1)
    tooltips = []
    monkeypatch.setattr(state.ui, "tooltip", lambda text, **_options: tooltips.append(text))
    strip = state.ui.hits["timeline-take-strip"]
    state.event("add_mouse_pos_event", strip[0] + strip[2] / 2, strip[1] + strip[3] / 2)
    assert f"Recorded frame 1/1\n{times[0]:.6f} s" in tooltips


@pytest.mark.parametrize("native_fonts", (True,))
def test_axis_values_are_centered_on_their_ticks_and_never_overlap(readable_timeline, native_fonts):
    state = readable_timeline
    _view(state, 0, 0.112)
    state.runtime.observed.clear()
    state.frame()
    ticks = [record for record in state.runtime.observed if record["kind"] == "timeline_tick"]
    assert len(ticks) >= 3
    left, right = state.panel.last_layout.time_bounds
    centered = 0
    for tick in ticks:
        x, _, width, _ = tick["hit"]
        epsilon = native_epsilon(x, width, tick["tickX"])
        assert left - epsilon <= x and x + width <= right + epsilon
        if left + width / 2 <= tick["tickX"] <= right - width / 2:
            assert abs(x + width / 2 - tick["tickX"]) <= epsilon
            centered += 1
    assert centered >= 2
    assert all(a["hit"][0] + a["hit"][2] < b["hit"][0] for a, b in pairwise(ticks))


@pytest.mark.parametrize("native_fonts", (True,))
def test_loop_handles_identify_endpoint_samples_while_shading_keeps_complete_cells(
    readable_timeline, native_fonts
):
    state = readable_timeline
    times = _record(state, 8, 0.016)
    assert state.session.submit(cmd.SetStateTakeLoop(2, 6)).ok
    _view(state, times[0], times[-1])
    loop = _latest(state, "timeline-loop-range")
    first, last = loop["handleXs"]
    left, right = state.panel.last_layout.time_bounds
    assert loop["endpointTimes"] == [times[2], times[6]]
    assert first == pytest.approx(timeline_time_to_x(times[2], times[0], times[-1], left, right))
    assert last == pytest.approx(timeline_time_to_x(times[6], times[0], times[-1], left, right))
    assert loop["coverageXs"][0] < first < last < loop["coverageXs"][1]
    assert state.session.state_take_range == (2, 6)


@pytest.mark.parametrize("native_fonts", (True,))
def test_current_time_can_precede_an_offscreen_loop_without_changing_its_timestamp(
    readable_timeline, monkeypatch, native_fonts
):
    state = readable_timeline
    times = _record(state, 43, 0.008)
    assert state.session.submit(cmd.SetStateTakeLoop(23, 43)).ok
    assert state.session.submit(cmd.SeekStateTake(4)).ok
    _view(state, 0, 0.112)
    labels, original = [], state.ui.component_text

    def text(part, rect, caption, *args, **options):
        labels.append(caption)
        return original(part, rect, caption, *args, **options)

    monkeypatch.setattr(state.ui, "component_text", text)
    state.frame()
    context = _latest(state, "timeline-range-context")
    playhead = _latest(state, "timeline-playhead")
    assert context["rangeKind"] == "loop"
    assert context["timeRange"] == pytest.approx((0.184, 0.344))
    assert context["viewRange"] == [0, 0.112]
    assert playhead["time"] == pytest.approx(0.032)
    assert playhead["time"] == times[4]
    assert {"Loop start", "Loop end"} <= set(labels)
    assert playhead["playheadX"] == pytest.approx(
        timeline_time_to_x(times[4], 0, 0.112, *state.panel.last_layout.time_bounds)
    )
    state.click_at(
        (
            state.ui.hits["timeline-play-pause"][0] + 4,
            state.ui.hits["timeline-play-pause"][1] + 4,
        )
    )
    assert state.session.state_take_cursor == 23
    assert state.panel.editor.playhead == times[23]


@pytest.mark.parametrize("native_fonts", (True,))
def test_empty_timeline_range_identifies_the_view_instead_of_a_recording(
    readable_timeline, monkeypatch, native_fonts
):
    state = readable_timeline
    _view(state, 0.184, 0.344)
    labels, original = [], state.ui.component_text

    def text(part, rect, caption, *args, **options):
        labels.append(caption)
        return original(part, rect, caption, *args, **options)

    monkeypatch.setattr(state.ui, "component_text", text)
    state.frame()
    context = _latest(state, "timeline-range-context")
    assert context["rangeKind"] == "view"
    assert context["timeRange"] == [0.184, 0.344]
    assert {"View start", "View end"} <= set(labels)
