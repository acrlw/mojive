"""The native timeline child owns breathing room for its header, lanes and details."""

from copy import deepcopy

import pytest

from mojive import commands as cmd
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.workspace.menu_geometry import native_epsilon, native_unit
from tests.test_native_timeline_boundaries import production_timeline as production_timeline
from tests.test_timeline_sample_readability import readable_timeline as readable_timeline
from tests.test_ui_design_timeline import timeline as timeline

pytestmark = [
    pytest.mark.physics,
    pytest.mark.parametrize("timeline", (1.0, 1.2, 1.25, 2.25), indirect=True),
    pytest.mark.parametrize("native_fonts", (True,)),
]


def _latest(state, key):
    return next(record for record in reversed(state.runtime.observed) if record["key"] == key)


def _equal(actual, expected, *coordinates):
    assert abs(actual - expected) <= native_epsilon(actual, expected, *coordinates)


def test_body_keeps_its_padding_and_uses_the_container_radius_independently(
    readable_timeline, native_fonts
):
    state = readable_timeline
    body = _latest(state, "timeline-body")
    window = state.children["Timeline body"]
    scale = state.ui.s
    padding = native_unit(4 * scale)
    assert body["padding"] == [padding] * 4
    assert tuple(window.window_padding) == (padding, padding)
    outer_radius = 6 * scale + native_unit(8 * scale)
    _equal(body["radius"], max(0, outer_radius - native_unit(6 * scale)))
    _equal(body["innerRadius"], max(0, body["radius"] - padding))
    _equal(window.window_rounding, body["radius"])
    frame, content = body["hit"], body["content"]
    _equal(content[0] + body["scroll"][0] - frame[0], padding)
    _equal(content[1] + body["scroll"][1] - frame[1], padding)
    _equal(
        frame[0] + frame[2] - content[0] - content[2] - body["scrollbarGutter"][0],
        padding,
        *frame,
        *content,
    )
    selector = state.ui.hits["timeline-model"]
    _equal(selector[0], content[0])
    header = _latest(state, "timeline-model-group")
    _equal(header["outer"][1], content[1])
    assert selector[1] >= content[1]
    _equal(selector[1] - content[1], (header["outer"][3] - selector[3]) / 2, *selector, *content)
    for index in range(3):
        label = _latest(state, "timeline-lane-label-" + str(index))
        _equal(label["hit"][0], content[0])
        assert label["hit"][0] + label["hit"][2] <= state.panel.last_layout.channel_right
    _equal(state.panel.last_layout.lo[0], content[0])
    _equal(state.panel.last_layout.lo[1], content[1])


def test_authored_axis_padding_and_child_corner_edits_stay_local(readable_timeline, native_fonts):
    state = readable_timeline
    spec = deepcopy(state.runtime.spec)
    spec["parts"]["timeline_body"] = {"paddingX": 12, "paddingY": 10}
    spec["parts"]["select"] = {"radius": 9}
    state.runtime.update_spec(spec)
    state.frame(3)
    body = _latest(state, "timeline-body")
    x_pad = y_pad = native_unit(12 * state.ui.s)
    assert body["padding"] == [x_pad, y_pad, x_pad, y_pad]
    _equal(body["innerRadius"], max(0, body["radius"] - x_pad))
    selector = state.ui.hits["timeline-model"]
    _equal(selector[0] - body["hit"][0], x_pad)
    header = _latest(state, "timeline-model-group")
    _equal(header["outer"][1] - body["hit"][1], y_pad, *body["hit"])
    for radius in (23, 0):
        spec["parts"]["timeline_body"]["radius"] = radius
        state.runtime.update_spec(spec)
        state.frame(3)
        authored = _latest(state, "timeline-body")
        _equal(authored["radius"], radius * state.ui.s)
        assert authored["padding"] == body["padding"]
        assert state.ui.hits["timeline-model"] == selector


@pytest.mark.parametrize("body_radius", (None, 23))
def test_actual_selector_and_selected_lane_paints_use_the_body_corner_budget(
    readable_timeline, native_fonts, monkeypatch, body_radius
):
    state, paints = readable_timeline, []
    if body_radius is not None:
        spec = deepcopy(state.runtime.spec)
        spec["parts"]["timeline_body"] = {"radius": body_radius, "paddingX": 4, "paddingY": 4}
        state.runtime.update_spec(spec)
    rect = state.ui.rect

    def observe(x, y, w, h, color, r=6, **options):
        paints.append(((x, y, w, h), r * state.ui.s))
        return rect(x, y, w, h, color, r, **options)

    monkeypatch.setattr(state.ui, "rect", observe)
    backgrounds = []
    rectangle = ImguiDraw2D.rect_filled

    def background(self, lo, hi, color, **options):
        backgrounds.append(((*lo, hi[0] - lo[0], hi[1] - lo[1]), options.get("rounding", 0)))
        return rectangle(self, lo, hi, color, **options)

    monkeypatch.setattr(ImguiDraw2D, "rect_filled", background)
    state.frame(3)
    body = _latest(state, "timeline-body")
    expected = max(0, body["radius"] - body["padding"][0])
    selector = state.ui.hits["timeline-model"]
    assert expected <= selector[3] / 2 - 1 + native_epsilon(expected, selector[3])
    lane = next(
        record
        for record in reversed(state.runtime.observed)
        if record["key"].startswith("timeline-lane-label-") and record["background"]
    )
    for frame in (selector, tuple(lane["hit"])):
        radii = [radius for painted, radius in paints if painted == frame]
        assert radii
        assert all(abs(radius - expected) <= native_epsilon(radius, expected) for radius in radii)
    for key in ("timeline-sheet-background", "timeline-track-background"):
        track = _latest(state, key)
        radii = [radius for frame, radius in backgrounds if frame == tuple(track["hit"])]
        assert radii
        assert all(abs(radius - expected) <= native_epsilon(radius, expected) for radius in radii)
        _equal(track["radius"], expected)


def test_padded_owner_keeps_recorded_sample_seek_and_inclusive_loop_projection(
    readable_timeline, native_fonts
):
    state = readable_timeline
    assert state.session.submit(cmd.StartStateTakeRecording()).ok
    state.frame(12, wall_dt=0.008)
    assert state.session.submit(cmd.StopStateTakeRecording()).ok
    times = tuple(state.session.state_take_times)
    state.panel.editor.set_follow_mode("off")
    assert state.panel.editor.set_view_range(times[0], times[-1])
    state.frame(3)
    sample = _latest(state, "timeline-take-frame-4")
    state.click_at((sample["sampleX"], state.panel.last_layout.lane_y(1)))
    assert state.session.state_take_cursor == 4
    _equal(_latest(state, "timeline-playhead")["playheadX"], sample["sampleX"])
    first, last = (_latest(state, "timeline-take-frame-" + str(index)) for index in (2, 9))
    state.drag(
        (first["sampleX"], state.panel.last_layout.lane_y(1)),
        (last["sampleX"], state.panel.last_layout.lane_y(1)),
        shift=True,
    )
    assert state.session.state_take_range == (2, 9)
    loop = _latest(state, "timeline-loop-range")
    _equal(loop["handleXs"][0], first["sampleX"])
    _equal(loop["handleXs"][1], last["sampleX"])
    assert loop["coverageXs"][0] < loop["handleXs"][0] < loop["handleXs"][1] < loop["coverageXs"][1]
