"""The timeline gutter separates columns in painting, projection and pointer routing."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.keyframe_editor.track import TimelineLayout, draw_dope_sheet
from mojive.ui.timeline import timeline_time_to_x, timeline_x_to_time
from mojive.ui.workspace.timeline_track import measure_layout
from tests.test_component_appearance import appearance as appearance
from tests.test_native_timeline_boundaries import production_timeline as production_timeline
from tests.test_ui_design_timeline import timeline as timeline


@pytest.mark.parametrize("width", (35, 90, 180, 900))
@pytest.mark.parametrize("gap", (0, 12, 24))
def test_native_layout_reserves_a_real_adjustable_gap_inside_its_parent(appearance, width, gap):
    ui, runtime, update, frame = appearance
    runtime.observe_components = True
    update({"timeline_sheet": {"columnGap": gap, "marginLeft": 7, "marginRight": 11}})
    ctx = SimpleNamespace(style_scale=ui.s, session=SimpleNamespace(scene_models=()), tr=str)
    layout = frame(
        lambda: measure_layout(
            SimpleNamespace(model_id=-1),
            ctx,
            (20 * ui.s, 30 * ui.s),
            imgui.ImVec2(width * ui.s, 300 * ui.s),
            ui=ui,
        )
    )
    assert layout.lo[0] <= layout.channel_right <= layout.track_left <= layout.hi[0]
    assert layout.track_left - layout.channel_right == pytest.approx(layout.column_gap)
    assert 0 <= layout.column_gap <= gap * ui.s
    if width >= 180:
        assert layout.column_gap == gap * ui.s
    assert layout.track_left <= layout.time_bounds[0] <= layout.time_bounds[1] <= layout.hi[0]
    record = next(item for item in reversed(runtime.observed) if item["key"] == "timeline-sheet")
    assert record["gutterRect"] == pytest.approx(ui.hits["timeline-gutter"])
    assert record["fieldRegions"]["columnGap"] == pytest.approx(ui.hits["timeline-gutter"])


@pytest.mark.parametrize("scale", (1, 1.25, 2.25))
def test_shared_projection_uses_the_same_gutter_geometry_at_each_scale(scale):
    layout = TimelineLayout.measure((20 * scale, 30 * scale), (900 * scale, 300 * scale), scale)
    assert layout.column_gap == 12 * scale
    assert layout.track_left - layout.channel_right == 12 * scale
    for time in (0, 0.3, 1):
        x = timeline_time_to_x(time, 0, 1, *layout.time_bounds)
        assert layout.track_left <= x <= layout.hi[0]
        assert timeline_x_to_time(x, 0, 1, *layout.time_bounds) == pytest.approx(time)


@pytest.fixture
def gutter_timeline(production_timeline, tmp_path, request):
    state = production_timeline
    runtime = DesignRuntime(SpecificationStore(tmp_path / "gutter-design.json"))
    runtime.observe_components = True
    state.ui.design = DesignPresentation(state.ui, runtime)
    state.runtime = runtime
    gap = request.node.callspec.params["gap"]
    spec = deepcopy(runtime.spec)
    spec["parts"]["timeline_sheet"] = {"columnGap": gap}
    runtime.update_spec(spec)
    editor = state.panel.editor
    if request.node.callspec.params["style"] == "classic":

        def draw(ctx, x, y, width, height):
            imgui.set_cursor_screen_pos((x, y))
            layout = TimelineLayout.measure((x, y), (width, height), state.ui.s)
            state.panel.last_layout = replace(layout, column_gap=gap * state.ui.s)
            draw_dope_sheet(
                editor,
                ctx,
                tuple(state.session.scene_models),
                state.session.state_take_times,
                True,
                layout=state.panel.last_layout,
            )

        state.panel = SimpleNamespace(editor=editor, draw=draw)
    state.frame(3)
    editor.set_follow_mode("off")
    assert editor.set_view_range(0, 2)
    state.frame(2)
    return state


@pytest.mark.physics
@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("style", ("native", "classic"))
@pytest.mark.parametrize("gap", (12, 24))
def test_actual_gutter_is_unpainted_and_does_not_select_scrub_or_zoom(
    gutter_timeline, monkeypatch, style, gap, native_fonts
):
    state = gutter_timeline
    editor = state.panel.editor
    rectangles, clips = [], []
    rect_filled, push_clip = ImguiDraw2D.rect_filled, imgui.push_clip_rect

    def rect(draw, lo, hi, color, **kwargs):
        rectangles.append((tuple(lo), tuple(hi)))
        return rect_filled(draw, lo, hi, color, **kwargs)

    def clip(lo, hi, *args):
        clips.append((tuple(lo), tuple(hi)))
        return push_clip(lo, hi, *args)

    monkeypatch.setattr(ImguiDraw2D, "rect_filled", rect)
    monkeypatch.setattr(imgui, "push_clip_rect", clip)
    state.frame()
    layout = state.panel.last_layout
    assert layout.track_left - layout.channel_right == pytest.approx(gap * state.ui.s)
    assert any(lo == (layout.track_left, layout.lo[1]) and hi == layout.hi for lo, hi in rectangles)
    assert any(lo[0] >= layout.track_left and hi[0] <= layout.hi[0] for lo, hi in clips)
    if style == "native":
        label = next(
            row for row in reversed(state.runtime.observed) if row["key"] == "timeline-lane-label-0"
        )
        assert label["hit"][0] + label["hit"][2] <= layout.channel_right
    x = (layout.channel_right + layout.track_left) / 2
    y = layout.lane_y(1)
    editor.edit_lane = "model"
    before = editor.view_start, editor.view_end, editor.playhead
    state.click_at((x, y))
    assert editor.edit_lane == "model" and not editor.pointer_mode
    state.event("add_mouse_wheel_event", 0, 2)
    assert (editor.view_start, editor.view_end, editor.playhead) == before
    state.click_at((layout.lo[0] + 10 * state.ui.s, y))
    assert editor.edit_lane == "take"


@pytest.mark.physics
@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("style", ("native", "classic"))
@pytest.mark.parametrize("gap", (0, 24))
def test_actual_markers_drag_and_wheel_share_the_shifted_time_mapping(
    gutter_timeline, style, gap, native_fonts
):
    state = gutter_timeline
    editor = state.panel.editor
    assert state.session.submit(cmd.AddModelKeyframe(editor.model_id, "Reach")).ok
    key = state.session.model_keyframes(editor.model_id)[-1]
    props = state.session.keyframe_properties(key.keyframe_id)
    assert state.session.submit(cmd.SetModelKeyframe(**{**vars(props), "time": 0.5})).ok
    state.frame(2)
    key = state.session.model_keyframes(editor.model_id)[-1]
    state.click_at(state.point(key.time))
    assert editor.selected_id == key.keyframe_id
    state.drag(state.point(key.time), state.point(0.8))
    changed = next(
        k
        for k in state.session.model_keyframes(editor.model_id)
        if k.keyframe_id == key.keyframe_id
    )
    assert changed.time == pytest.approx(0.8, abs=0.01)
    layout = state.panel.last_layout
    x = round(layout.time_bounds[0] + 0.7 * (layout.time_bounds[1] - layout.time_bounds[0]))
    y = layout.lo[1] + layout.ruler_height / 2
    anchor = timeline_x_to_time(x, editor.view_start, editor.view_end, *layout.time_bounds)
    state.event("add_mouse_pos_event", x, y)
    state.event("add_mouse_wheel_event", 0, 2)
    assert editor.view_end - editor.view_start < 2
    assert timeline_x_to_time(
        x, editor.view_start, editor.view_end, *layout.time_bounds
    ) == pytest.approx(anchor)
    if gap:
        layout = state.panel.last_layout
        state.drag(
            state.point(changed.time),
            ((layout.channel_right + layout.track_left) / 2, layout.lane_y(0)),
        )
        changed = state.session.model_keyframes(editor.model_id)[-1]
        assert changed.time == pytest.approx(editor.view_start)
