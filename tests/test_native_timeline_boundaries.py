"""Production timeline wheel input, range edits and ruler labels respect their bounds."""

import math
from itertools import pairwise
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.ui.keyframe_editor.toolbar import TimelineToolbar
from mojive.ui.workspace.menus import Menus
from mojive.ui.workspace.timeline_panel import WorkspaceTimeline
from tests.test_ui_design_timeline import timeline as timeline

pytestmark = pytest.mark.physics


@pytest.fixture
def production_timeline(timeline):
    state = timeline
    state.study.app = state.study.preview
    state.ui = state.study.ui
    state.study.menus = Menus(state.study, ())
    state.panel = WorkspaceTimeline(state.study)
    state.width, state.height = 1200, 700
    state.frame(3)
    state.panel.editor.set_follow_mode("off")

    def text(key, value):
        io = imgui.get_io()
        modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        state.event("add_key_event", modifier, True)
        x, y, width, height = state.ui.hits[key] if key in state.ui.hits else state.controls[key]
        state.click_at((x + width / 2, y + height / 2))
        state.event("add_key_event", modifier, False)
        assert io.want_text_input
        state.event("add_key_event", modifier, True)
        state.event("add_key_event", imgui.Key.a, True)
        state.event("add_key_event", imgui.Key.a, False)
        state.event("add_key_event", modifier, False)
        state.event("add_input_characters_utf8", value)
        state.event("add_key_event", imgui.Key.enter, True)
        state.event("add_key_event", imgui.Key.enter, False)

    state.text = text
    return state


@pytest.mark.parametrize("appearance", ("native", "classic"))
@pytest.mark.parametrize("take_frames", (0, 1, 2))
def test_range_fields_share_the_view_domain_for_zero_or_one_take_frame(
    production_timeline, monkeypatch, appearance, take_frames
):
    state = production_timeline
    if take_frames:
        assert state.session.submit(cmd.StartStateTakeRecording()).ok
        if take_frames > 1:
            state.session.tick(FrameNeeds.none(), wall_dt=0.01)
        assert state.session.submit(cmd.StopStateTakeRecording()).ok
    times = tuple(state.session.state_take_times)
    assert len(times) == take_frames if take_frames < 2 else len(times) >= 2
    state.frame(3)
    editor = state.panel.editor
    assert editor.set_view_range(0, 10)
    if appearance == "classic":
        toolbar = TimelineToolbar()

        def draw(ctx, x, y, width, _height):
            imgui.set_cursor_screen_pos((x, y))
            layout = toolbar._measure_toolbar(editor, ctx, state.session.state_take_times, width)
            toolbar.draw_take_range(
                editor, ctx, state.session.state_take_times, layout.range_widths
            )

        state.panel = SimpleNamespace(editor=editor, draw=draw)
    incoming, drag_float = {}, imgui.drag_float

    def observe(key, value, *args, **kwargs):
        if key.startswith("##timeline-range-"):
            incoming[key] = value
        return drag_float(key, value, *args, **kwargs)

    monkeypatch.setattr(imgui, "drag_float", observe)
    state.frame(3)
    expected = (0, 10) if len(times) < 2 else (times[0], times[-1])
    assert tuple(incoming["##timeline-range-" + name] for name in ("start", "end")) == expected
    state.text("##timeline-range-end", "7200")
    state.frame(2)
    if len(times) < 2:
        assert (editor.view_start, editor.view_end) == (0, 7200)
        assert incoming["##timeline-range-end"] == 7200
    else:
        assert (editor.view_start, editor.view_end) == (0, 10)
        assert incoming["##timeline-range-end"] == times[-1]
        assert state.session.state_take_loop_enabled
    assert tuple(state.session.state_take_times) == times
    was_looping = state.session.state_take_loop_enabled
    loop = state.ui.hits.get("timeline-loop") or state.controls["##timeline-loop"]
    state.click_at((loop[0] + loop[2] / 2, loop[1] + loop[3] / 2))
    assert state.session.state_take_loop_enabled is (
        not was_looping if len(times) > 1 else was_looping
    )


def test_repeated_real_wheel_events_stop_at_readable_minimum_and_maximum(
    production_timeline, monkeypatch
):
    state = production_timeline
    editor = state.panel.editor
    editor.view_start, editor.view_end, editor.view_needs_fit = -4.99e7, 9.63e7, False
    state.frame(2)
    assert editor.view_end - editor.view_start == pytest.approx(60)
    assert abs((editor.view_start + editor.view_end) / 2) < 1
    layout = state.panel.last_layout
    state.event(
        "add_mouse_pos_event", sum(layout.time_bounds) / 2, layout.lo[1] + layout.ruler_height / 2
    )
    for _ in range(25):
        state.event("add_mouse_wheel_event", 0, -100000)
    assert editor.view_end - editor.view_start == pytest.approx(60)
    for _ in range(35):
        state.event("add_mouse_wheel_event", 0, 100000)
    assert editor.view_end - editor.view_start == pytest.approx(0.001)
    captions, text = [], state.ui.text

    def observe(x, y, label, *args, **kwargs):
        if str(label).endswith(" s"):
            captions.append(str(label))
        return text(x, y, label, *args, **kwargs)

    monkeypatch.setattr(state.ui, "text", observe)
    state.frame(2)
    expected = (f"{editor.view_start:.3f} s", f"{editor.view_end:.3f} s")
    assert expected[0] != expected[1]
    assert set(expected) <= set(captions)


@pytest.mark.parametrize("value", ("1e999", "-1", "0.00000001"))
def test_actual_range_input_rejects_nonfinite_reversed_and_unrepresentable_spans(
    production_timeline, value
):
    state = production_timeline
    editor = state.panel.editor
    assert editor.set_view_range(0, 10)
    state.frame(3)
    state.text("##timeline-range-end", value)
    assert (editor.view_start, editor.view_end) == (0, 10)
    assert editor.manual_view_range == (0, 10)


def test_manual_long_and_submillisecond_ranges_render_without_ruler_overlap(
    production_timeline, monkeypatch
):
    state = production_timeline
    editor = state.panel.editor
    labels, component_text = [], state.ui.component_text

    def observe(part, rect, caption, *args, **kwargs):
        if part == "timeline_ruler":
            labels.append((rect.x, rect.x + imgui.calc_text_size(caption).x, caption))
        return component_text(part, rect, caption, *args, **kwargs)

    monkeypatch.setattr(state.ui, "component_text", observe)
    for start, end in ((0, 7200), (1e8, 1e8 + 0.001), (0, 0.00001)):
        assert editor.set_view_range(start, end)
        state.frame(3)
        labels.clear()
        state.frame()
        assert (editor.view_start, editor.view_end) == (start, end)
        assert len(labels) >= 2
        bounds = state.panel.last_layout.time_bounds
        assert all(bounds[0] <= left < right <= bounds[1] for left, right, _ in labels)
        assert all(a[1] < b[0] for a, b in pairwise(labels))
        assert len({label for _left, _right, label in labels}) == len(labels)
        assert all(math.isfinite(value) for value in (editor.view_start, editor.view_end))
