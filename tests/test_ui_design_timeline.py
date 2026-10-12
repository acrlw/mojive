"""Native timeline controls operate the registered editor and real Session data."""

from types import SimpleNamespace

import numpy as np
import pytest
from examples.ui_design.native.timeline_panel import ReferenceTimeline
from examples.ui_design.native.widgets import UI
from imgui_bundle import imgui

from mojive import Scene
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.joint_replay import JointReplayAdapter
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.static import StaticSceneAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.config import RecordingConfig
from mojive.interaction.input import InputClaim
from mojive.render.backend import NullBackend
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.text.sources import resolve_font_sources
from mojive.tools.g1_replay import generate
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.keyframe_editor.toolbar import draw_model_header
from mojive.ui.localization import Language, Localizer
from mojive.ui.panels import PanelContext, PanelManager
from mojive.ui.timeline import take_frame_bounds, timeline_time_to_x

pytestmark = pytest.mark.physics


@pytest.fixture(params=(1.0, 2.0))
def timeline(request, monkeypatch):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (330 * request.param, 360 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    style = imgui.get_style()
    style.window_padding = (10 * request.param, 10 * request.param)
    style.scrollbar_size = 7 * request.param
    if request.node.callspec.params.get("native_fonts", False):
        sources = resolve_font_sources(allow_download=False)
        cjk = sources[1] if len(sources) > 1 else None
        report = SimpleNamespace(
            cjk_path=cjk.path if cjk else None, cjk_index=cjk.index if cjk else 0
        )
        ui = UI(SimpleNamespace(style_scale=request.param, font_report=report))
        style.window_padding = (10 * request.param, 10 * request.param)
        style.scrollbar_size = 7 * request.param
    else:
        ui = UI.__new__(UI)
        ui.s = request.param
        ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
        ui.translate, ui.hits, ui.numeric_edits = str, {}, {}
    style.window_min_size = (request.param, request.param)
    session = Session(WorkspaceAdapter(MuJoCoAdapter(resolve("joint_types")), Scene()))
    assert session.submit(cmd.Pause()).ok
    panels = PanelManager(builtin_ids=("keyframes",))
    study = SimpleNamespace(
        s=ui.s,
        ui=ui,
        preview=SimpleNamespace(panels=panels, session=session),
        collapse_bottom=lambda: collapsed.append(True),
    )
    collapsed, video_calls, controls, children = [], [], {}, {}
    settings = []
    ctx = PanelContext(
        session,
        NullBackend(),
        style_scale=ui.s,
        start_take_video=lambda: video_calls.append("start"),
        stop_recording=lambda: video_calls.append("stop"),
    )
    state = SimpleNamespace(
        panel=ReferenceTimeline(study),
        study=study,
        session=session,
        ctx=ctx,
        controls=controls,
        children=children,
        collapsed=collapsed,
        video_calls=video_calls,
        settings=settings,
        width=330,
        height=360,
    )
    for method in (
        "button",
        "input_text",
        "input_double",
        "input_float",
        "drag_float",
        "begin_combo",
        "combo",
        "selectable",
        "checkbox",
        "invisible_button",
    ):
        original = getattr(imgui, method)

        def observe(identifier, *args, _original=original, **kwargs):
            result = _original(identifier, *args, **kwargs)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            controls[identifier] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
            return result

        monkeypatch.setattr(imgui, method, observe)
    original_child = imgui.begin_child

    def begin_child(identifier, *args, **kwargs):
        result = original_child(identifier, *args, **kwargs)
        children[identifier] = imgui.internal.get_current_window()
        return result

    monkeypatch.setattr(imgui, "begin_child", begin_child)

    def frame(count=1, wall_dt=0):
        for _ in range(count):
            state.session.tick(FrameNeeds(qpos=True), wall_dt=wall_dt)
            io.display_size = (state.width * ui.s, state.height * ui.s)
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            ui.hits.clear()
            controls.clear()
            children.clear()
            flags = (
                imgui.WindowFlags_.no_title_bar
                | imgui.WindowFlags_.no_scrollbar
                | imgui.WindowFlags_.no_scroll_with_mouse
            )
            imgui.begin("Native timeline", flags=flags)
            pos, size = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
            state.panel.draw(ctx, pos.x, pos.y, size.x, size.y)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def center(key):
        x, y, w, h = ui.hits[key] if key in ui.hits else controls[key]
        return x + w / 2, y + h / 2

    def reveal(point):
        if imgui.get_current_context().open_popup_stack:
            return
        for _ in range(20):
            child = children.get("Timeline body")
            if child is None:
                return
            clip, target = child.inner_clip_rect, point()
            if clip.min.y + ui.s <= target[1] <= clip.max.y - ui.s:
                return
            io.add_mouse_pos_event(
                child.pos.x + child.size.x - imgui.get_style().scrollbar_size / 2,
                (clip.min.y + clip.max.y) / 2,
            )
            io.add_mouse_wheel_event(0, -1 if target[1] > clip.max.y else 1)
            frame(2)
        raise AssertionError("Timeline control could not be scrolled into view")

    def click_at(point):
        event("add_mouse_pos_event", *point)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(3)

    def click(key):
        primary = {"collapse-timeline", "timeline-settings", "timeline-play-pause", "timeline-time"}
        if state.panel.panel.toolbar._inline:
            primary.update(ui.hits)
        steps = getattr(state.panel.panel.toolbar, "_visible_transport_steps", 0)
        if steps >= 2:
            primary.update(("timeline-previous", "timeline-next"))
        if steps == 4:
            primary.update(("timeline-first", "timeline-last"))
        if key not in primary:
            reveal(lambda: center(key))
        click_at(center(key))

    def point(time, lane="model"):
        reveal(lambda: project(time, lane))
        return project(time, lane)

    def project(time, lane):
        editor = state.panel.editor
        layout = state.panel.last_layout
        left, right = layout.time_bounds
        px = timeline_time_to_x(time, editor.view_start, editor.view_end, left, right)
        py = (
            layout.lo[1] + layout.ruler_height / 2
            if lane == "ruler"
            else layout.lane_y({"model": 0, "take": 1, "snapshots": 2}[lane])
        )
        return round(max(left + 1, min(px, right - 1))), round(py)

    def drag(start, end, *, button=0, shift=False, cancel=None):
        event("add_mouse_pos_event", *start)
        event("add_key_event", imgui.Key.mod_shift, shift)
        event("add_mouse_button_event", button, True)
        for part in (0.25, 0.5, 0.75, 1):
            event(
                "add_mouse_pos_event",
                start[0] + (end[0] - start[0]) * part,
                start[1] + (end[1] - start[1]) * part,
            )
        if cancel == "escape":
            event("add_key_event", imgui.Key.escape, True)
        elif cancel == "application":
            ctx.input_claim = InputClaim(pointer=True)
            frame()
        event("add_mouse_button_event", button, False)
        event("add_key_event", imgui.Key.escape, False)
        event("add_key_event", imgui.Key.mod_shift, False)
        ctx.input_claim = None
        frame(3)

    def text(key, value):
        numeric = key.startswith("##timeline-") and key in ui.numeric_edits
        if numeric:
            event("add_key_event", imgui.Key.mod_ctrl, True)
        click(key)
        if numeric:
            event("add_key_event", imgui.Key.mod_ctrl, False)
        event("add_key_event", imgui.Key.mod_ctrl, True)
        event("add_key_event", imgui.Key.a, True)
        event("add_key_event", imgui.Key.a, False)
        event("add_key_event", imgui.Key.mod_ctrl, False)
        io.add_input_characters_utf8(value)
        frame()
        event("add_key_event", imgui.Key.enter, True)
        event("add_key_event", imgui.Key.enter, False)

    state.frame, state.event, state.click = frame, event, click
    state.click_at, state.point, state.drag, state.text = click_at, point, drag, text
    frame(3)
    try:
        yield state
    finally:
        if state.session is not session:
            state.session.release()
        session.release()
        imgui.destroy_context(context)


def _set_view(state, start=0, end=2):
    editor = state.panel.editor
    editor.view_start, editor.view_end, editor.view_needs_fit = start, end, False
    editor.set_follow_mode("off")
    state.frame(2)


def _record_take(state, count=30):
    state.click("timeline-record")
    assert state.session.state_take_recording
    state.frame(count, wall_dt=1 / 30)
    state.click("timeline-record")
    assert not state.session.state_take_recording
    assert len(state.session.state_take_times) > 2


def test_registered_editor_empty_start_capture_edit_load_delete_and_undo(timeline):
    state = timeline
    assert state.panel.editor is state.study.preview.panels.get("keyframes").editor
    assert not state.session.model_keyframes(0)
    assert not state.session.state_takes
    assert not state.session.scene_snapshots
    joint = next(item for item in state.session.joints if item.name == "hinge_limited")
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, 0.4)).ok
    state.frame()
    state.click("timeline-add-key")
    key = state.session.model_keyframes(0)[0]
    assert state.panel.editor.selected_id == key.keyframe_id
    state.text("##keyframe-name", "Rest")
    state.click("timeline-detail-apply")
    key = state.session.model_keyframes(0)[0]
    assert key.name == "Rest"
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, -0.3)).ok
    state.frame()
    state.click("timeline-detail-load")
    assert state.session.frame.qpos[joint.qpos_adr] == pytest.approx(0.4)
    state.click("timeline-detail-delete")
    assert not state.session.model_keyframes(0)
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.model_keyframes(0)[0].name == "Rest"


@pytest.mark.parametrize("cancel", (None, "escape", "application"))
def test_real_key_drag_commits_once_or_cancels_and_preserves_undo(timeline, cancel):
    state = timeline
    state.click("timeline-add-key")
    key = state.session.model_keyframes(0)[0]
    props = state.session.keyframe_properties(key.keyframe_id)
    assert state.session.submit(cmd.SetModelKeyframe(**{**vars(props), "time": 0.5})).ok
    state.height = 190
    _set_view(state)
    state.click_at(state.point(1.6))
    width = state.study.ui.hits["timeline-sheet"][2]
    depth = len(state.session._undo_stack)
    state.drag(state.point(0.5), state.point(1.2), cancel=cancel)
    updated = state.session.model_keyframes(0)[0]
    assert state.panel.editor.pointer_mode == ""
    assert updated.time == pytest.approx(0.5 if cancel else 1.2, abs=0.015)
    assert len(state.session._undo_stack) == depth + (cancel is None)
    assert state.study.ui.hits["timeline-sheet"][2] == width
    if cancel is None:
        assert state.session.submit(cmd.Undo()).ok
        assert state.session.model_keyframes(0)[0].time == pytest.approx(0.5)


def test_record_transport_scrub_range_loop_snapshot_and_video_use_session(timeline):
    state = timeline
    _record_take(state)
    state.click("timeline-last")
    assert state.session.state_take_cursor == len(state.session.state_take_times) - 1
    state.click("timeline-first")
    assert state.session.state_take_cursor == 0
    state.click("timeline-next")
    assert state.session.state_take_cursor == 1
    state.click("timeline-previous")
    assert state.session.state_take_cursor == 0
    state.click("timeline-play-pause")
    assert state.session.state_take_playing
    state.frame(5, wall_dt=1 / 30)
    assert state.session.state_take_cursor > 0
    state.click("timeline-play-pause")
    assert not state.session.state_take_playing
    _set_view(state)
    state.click_at(state.point(0.4, "ruler"))
    assert state.session.state_take_times[state.session.state_take_cursor] == pytest.approx(
        0.4, abs=0.04
    )
    state.drag(state.point(0.2, "take"), state.point(0.7, "take"), shift=True)
    assert state.session.state_take_range is not None
    loop = state.session.state_take_loop_enabled
    state.click("timeline-loop")
    assert state.session.state_take_loop_enabled is not loop
    state.click("timeline-snapshot")
    assert len(state.session.scene_snapshots) == 1
    assert state.panel.editor.selected_snapshot == state.session.scene_snapshots[0].snapshot_id
    state.click("timeline-detail-delete")
    assert not state.session.scene_snapshots
    state.click("timeline-video")
    assert state.video_calls == ["start"]
    state.ctx.take_video_active = True
    state.frame()
    state.click("timeline-video")
    assert state.video_calls == ["start", "stop"]
    before = tuple(state.session.model_keyframes(0))
    state.click("timeline-add-key")
    assert tuple(state.session.model_keyframes(0)) == before


def test_real_track_zoom_pan_fit_and_direct_follow_settings(timeline):
    state = timeline
    _set_view(state)
    state.event("add_mouse_pos_event", *state.point(0.8, "ruler"))
    state.event("add_mouse_wheel_event", 0, 2)
    assert state.panel.editor.view_end - state.panel.editor.view_start < 2
    start = state.point(0.8, "ruler")
    before = state.panel.editor.view_start
    state.drag(start, (start[0] + 25 * state.study.s, start[1]), button=1)
    assert state.panel.editor.view_start < before
    state.click("timeline-fit")
    assert (state.panel.editor.view_start, state.panel.editor.view_end) == pytest.approx((0, 0.5))
    state.click("timeline-follow-2")
    assert state.panel.editor.follow_mode == "locked"
    state.click("timeline-settings")
    state.click("timeline-pause-at-end")
    assert not state.session.state_take_pause_at_end


@pytest.mark.parametrize("height", (180, 360))
def test_track_wheel_after_loop_reveal_zooms_without_scrolling_body(timeline, height):
    state = timeline
    _record_take(state)
    state.height = height
    state.ctx.recording_config = RecordingConfig()
    state.frame(3)
    _set_view(state)
    loop = state.session.state_take_loop_enabled
    state.click("timeline-loop")
    assert state.session.state_take_loop_enabled is not loop
    state.event("add_mouse_pos_event", *state.point(0.8, "ruler"))
    child = state.children["Timeline body"]
    before = state.panel.editor.view_end - state.panel.editor.view_start
    scroll = child.scroll.y
    state.event("add_mouse_wheel_event", 0, 2)
    assert state.panel.editor.view_end - state.panel.editor.view_start < before
    assert child.scroll.y == pytest.approx(scroll)


def test_recording_options_preview_and_commit_through_context(timeline):
    state = timeline
    state.ctx.recording_config = RecordingConfig()

    def update(value, *, persist=True):
        state.ctx.recording_config = value
        state.settings.append((value, persist))

    state.ctx.set_recording_config = update
    state.frame(3)
    state.click("timeline-settings")
    state.text("##timeline-countdown", "2.5")
    state.click("timeline-pause-at-end")
    assert state.ctx.recording_config.countdown == pytest.approx(2.5)
    assert any(not persist for _, persist in state.settings)
    assert state.settings[-1][1]
    state.text("##timeline-end_hold", "1.5")
    assert state.ctx.recording_config.end_hold == pytest.approx(1.5)
    assert state.settings[-1] == (state.ctx.recording_config, True)
    assert state.settings[-1][1]


@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_settings_labels_share_left_edge_and_fields_fill_compact_popup(
    timeline, native_fonts, language, monkeypatch
):
    state, ui = timeline, timeline.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    state.ctx.recording_config = RecordingConfig()
    labels, popup = {}, {}
    text, begin_popup = ui.text, imgui.begin_popup

    def observe_text(x, y, caption, *args, **kwargs):
        labels[caption] = (x, y)
        return text(x, y, caption, *args, **kwargs)

    def observe_popup(*args, **kwargs):
        opened = begin_popup(*args, **kwargs)
        if opened and args[0] == "Timeline settings":
            pos, size = imgui.get_window_pos(), imgui.get_window_size()
            popup.update(x=pos.x, y=pos.y, width=size.x, height=size.y)
        return opened

    monkeypatch.setattr(ui, "text", observe_text)
    monkeypatch.setattr(imgui, "begin_popup", observe_popup)
    state.frame(3)
    state.click("timeline-settings")
    scale = ui.s
    left = popup["x"] + 10 * scale
    for caption in ("Pause at last frame", "Start delay (s)", "End hold (s)"):
        assert labels[state.ctx.tr(caption)][0] == pytest.approx(left)
    for key in ("timeline-pause-at-end", "##timeline-countdown", "##timeline-end_hold"):
        x, _, width, _ = ui.hits[key]
        assert x + width == pytest.approx(popup["x"] + popup["width"] - 10 * scale)
    assert popup["width"] / scale < 260
    assert popup["height"] / scale < 140
    before = state.session.state_take_pause_at_end
    state.click("##timeline-pause-label")
    assert state.session.state_take_pause_at_end is not before
    state.ctx.take_video_active = True
    state.frame(2)
    state.click("##timeline-pause-label")
    state.click("timeline-pause-at-end")
    assert state.session.state_take_pause_at_end is not before


def test_snapshot_double_click_restores_real_pose(timeline):
    state = timeline
    joint = next(item for item in state.session.joints if item.name == "hinge_limited")
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, 0.4)).ok
    state.frame()
    state.click("timeline-snapshot")
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, -0.3)).ok
    state.frame(24)
    snapshot = state.session.scene_snapshots[0]
    state.click_at(state.point(snapshot.time, "snapshots"))
    state.click_at(state.point(snapshot.time, "snapshots"))
    assert state.session.frame.qpos[joint.qpos_adr] == pytest.approx(0.4)


def test_take_selection_delete_and_model_select_all_batch_undo(timeline):
    state = timeline
    _record_take(state)
    _set_view(state)
    state.drag(state.point(0.2, "take"), state.point(0.7, "take"))
    assert state.panel.editor.take_selection is not None
    first, last = state.panel.editor.take_selection
    cells = tuple(state.session.state_take_cells)
    before = len(state.session.state_take_times)
    state.event("add_key_event", imgui.Key.delete, True)
    state.event("add_key_event", imgui.Key.delete, False)
    assert len(state.session.state_take_times) < before
    assert len(state.session.state_take_segments) == 2
    left = state.study.ui.hits["timeline-take-segment-0"]
    right = state.study.ui.hits["timeline-take-segment-1"]
    layout, editor = state.panel.last_layout, state.panel.editor

    def project(time):
        return timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)

    assert left[0] + left[2] == pytest.approx(project(cells[first][0]))
    assert right[0] == pytest.approx(project(cells[last][1]))
    assert left[0] + left[2] < right[0]

    assert state.session.submit(cmd.ClearStateTake()).ok
    for name in ("Home", "Reach"):
        assert state.session.submit(cmd.AddModelKeyframe(0, name)).ok
    state.frame()
    _set_view(state)
    state.click_at(state.point(0, "model"))
    state.event("add_key_event", imgui.Key.mod_ctrl, True)
    state.event("add_key_event", imgui.Key.a, True)
    state.event("add_key_event", imgui.Key.a, False)
    state.event("add_key_event", imgui.Key.mod_ctrl, False)
    assert len(state.panel.editor.selected_keyframes) == 2
    depth = len(state.session._undo_stack)
    state.event("add_key_event", imgui.Key.delete, True)
    state.event("add_key_event", imgui.Key.delete, False)
    assert not state.session.model_keyframes(0)
    assert len(state.session._undo_stack) == depth + 1
    assert state.session.submit(cmd.Undo()).ok
    assert {key.name for key in state.session.model_keyframes(0)} == {"Home", "Reach"}


@pytest.mark.parametrize("zoom", (0.05, 0.4, 4.0))
def test_take_is_one_continuous_strip_at_each_zoom(timeline, zoom, monkeypatch):
    state = timeline
    _record_take(state, 60)
    state.width = 1600
    editor = state.panel.editor
    editor.follow_mode = "off"
    editor.view_start, editor.view_end, editor.view_needs_fit = 0, zoom, False
    captions = []
    text = ImguiDraw2D.text

    def observe(draw, pos, color, label, **kwargs):
        if label.startswith("Take "):
            captions.append(label)
        return text(draw, pos, color, label, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)
    state.frame()
    assert len(captions) == 1
    strip = state.study.ui.hits["timeline-take-strip"]
    assert strip[2] > 0
    assert (
        state.study.ui.hits["timeline-record"][1] == state.study.ui.hits["timeline-play-pause"][1]
    )
    assert "##timeline-countdown" not in state.study.ui.hits


def test_single_frame_selection_covers_the_whole_displayed_frame_cell(timeline):
    state = timeline
    _record_take(state, 40)
    state.width, state.height = 1000, 500
    times = state.session.state_take_times
    index = len(times) // 2
    dt = times[index + 1] - times[index]
    editor = state.panel.editor
    editor.view_start, editor.view_end = times[index] - 4 * dt, times[index] + 4 * dt
    editor.follow_mode, editor.view_needs_fit = "off", False
    state.frame(3)
    state.drag(
        state.point(times[index] - dt * 0.2, "take"), state.point(times[index] + dt * 0.2, "take")
    )
    assert editor.take_selection == (index, index)
    layout = state.panel.last_layout
    left, right = (
        timeline_time_to_x(time, editor.view_start, editor.view_end, *layout.time_bounds)
        for time in take_frame_bounds(times, index, index)
    )
    selection = state.study.ui.hits["timeline-take-selection"]
    assert selection[0] == pytest.approx(left)
    assert selection[2] == pytest.approx(right - left)
    assert selection[2] > 20 * state.study.s


def test_static_scene_keeps_local_view_and_blocks_backend_only_actions(timeline):
    state = timeline
    scene = Scene()
    scene.box(name="box")
    state.session = Session(StaticSceneAdapter(scene))
    state.ctx.session = state.study.preview.session = state.session
    state.frame(3)
    _set_view(state, 3, 5)
    state.click("timeline-record")
    state.click("timeline-snapshot")
    state.click("timeline-add-key")
    assert not state.session.state_takes
    assert not state.session.scene_snapshots
    assert not state.session.keyframes
    assert (state.panel.editor.view_start, state.panel.editor.view_end) == pytest.approx((3, 5))
    state.event("add_mouse_pos_event", *state.point(4, "ruler"))
    state.event("add_mouse_wheel_event", 0, 1)
    assert state.panel.editor.view_end - state.panel.editor.view_start < 2


def test_real_replay_clip_uses_registered_local_controls(timeline, tmp_path, monkeypatch):
    monkeypatch.setattr("mojive.tools.g1_worlds.JOINT_NAMES", ("hinge",))
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint name="floating_base_joint"/>'
        '<geom type="sphere" size="0.1"/><body pos="0 0 0.2">'
        '<joint name="hinge" axis="0 1 0"/><geom type="capsule" size="0.1 0.2"/>'
        "</body></body></worldbody></mujoco>"
    )
    archive = tmp_path / "archive"
    generate(model, archive, 1, 6, 30)
    state = timeline
    state.session = Session(JointReplayAdapter(archive, worlds=1, realtime=False, paused=True))
    state.ctx.session = state.study.preview.session = state.session
    state.frame(3)
    assert state.panel.panel._replay is state.study.preview.panels.get("keyframes")._replay
    assert "timeline-sheet" not in state.study.ui.hits
    state.click("Next frame##replay_next")
    assert state.session.replay_info.frame_index == 1
    state.click("Previous frame##replay_previous")
    assert state.session.replay_info.frame_index == 0
    state.click("Play##replay_play")
    assert not state.session.replay_info.paused
    state.click("Pause##replay_play")
    assert state.session.replay_info.paused
    state.click("Loop##replay_loop")
    assert not state.session.replay_info.loop
    state.click("##replay_speed")
    state.click("2×")
    assert state.session.replay_info.speed == 2


def test_main_take_selector_creates_selects_and_clears_real_takes(timeline):
    state = timeline
    _record_take(state, 12)
    first_id = state.session.active_state_take_id
    state.click("##timeline-take")
    state.click("New Take##new-take")
    assert len(state.session.state_takes) == 2
    assert state.session.active_state_take_id != first_id
    assert not state.session.state_take_times
    state.click("##timeline-take")
    first_label = next(key for key in state.controls if key.endswith(f"##take-{first_id}"))
    state.click(first_label)
    assert state.session.active_state_take_id == first_id
    assert state.session.state_take_times
    state.click("##timeline-take")
    state.click("Clear take")
    assert not state.session.state_take_times


@pytest.mark.parametrize("width", (210, 330, 1000))
def test_primary_transport_stays_fixed_above_direct_editor_controls(timeline, width):
    state, ui = timeline, timeline.study.ui
    state.width = width
    state.frame(3)
    toolbar = ui.hits["timeline-toolbar"]
    assert toolbar[3] == pytest.approx(28 * ui.s)
    play = ui.hits["timeline-play-pause"]
    time = ui.hits["timeline-time"]
    record = ui.hits["timeline-record"]
    assert play[2] == 72 * ui.s
    assert record[2] == 88 * ui.s
    assert 40 * ui.s <= time[2] <= 80 * ui.s
    assert play[1] == time[1] == toolbar[1]
    assert record[1] > play[1]
    for key in (
        "timeline-take",
        "timeline-snapshot",
        "timeline-video",
        "timeline-loop",
        "timeline-follow",
        "timeline-fit",
        "timeline-settings",
        "##timeline-range-start",
        "##timeline-range-end",
    ):
        assert key in ui.hits
    assert "timeline-more" not in ui.hits and "timeline-options" not in ui.hits
    state.click("timeline-record")
    state.frame(10, wall_dt=1 / 30)
    np.testing.assert_allclose(ui.hits["timeline-time"], time)
    state.click("timeline-record")
    state.click("timeline-play-pause")
    assert state.session.state_take_playing
    np.testing.assert_allclose(ui.hits["timeline-time"], time)
    state.click("timeline-play-pause")
    assert not state.session.state_take_playing
    assert not imgui.get_current_context().open_popup_stack


@pytest.mark.parametrize("width", (210, 1000))
def test_video_stop_action_updates_visible_caption_and_icon(timeline, width, monkeypatch):
    state, ui = timeline, timeline.study.ui
    state.width = width
    state.frame(3)
    _record_take(state, 12)
    shown = {}
    original = ui.button

    def button(key, *args, **kwargs):
        if key == "timeline-video":
            shown.update(kwargs)
        return original(key, *args, **kwargs)

    monkeypatch.setattr(ui, "button", button)
    state.frame(3)
    assert shown["label"] == ("" if width == 210 else "Video")
    assert shown["icon"] == "video"
    before = ui.hits["timeline-video"]
    state.ctx.take_video_active = True
    state.frame(3)
    assert shown["label"] == ("" if width == 210 else "Stop")
    assert shown["icon"] == "stop"
    np.testing.assert_allclose(ui.hits["timeline-video"], before)
    state.click("timeline-video")
    assert state.video_calls == ["stop"]


def test_long_time_readout_reduces_precision_without_ellipsis(timeline, monkeypatch):
    state, ui = timeline, timeline.study.ui
    original, seen = ui.text, []

    def text(x, y, value, *args, **kwargs):
        time = ui.hits.get("timeline-time")
        if time is not None and (x, y) == (time[0] + 8 * ui.s, time[1] + 7 * ui.s):
            seen.append(value)
        return original(x, y, value, *args, **kwargs)

    monkeypatch.setattr(ui, "text", text)
    for value in (1000.123, 100_000.123, 1e20):
        state.panel.editor.playhead = value
        state.frame()
        assert "…" not in seen[-1]
        assert float(seen[-1][:-1].strip()) == pytest.approx(value, abs=0.5)
    assert ui.hits["timeline-time"][2] == 80 * ui.s


@pytest.mark.parametrize("width", (210, 330, 1000))
def test_toolbar_stays_inside_width_and_fixed_while_properties_scroll(timeline, width):
    state, ui = timeline, timeline.study.ui
    state.click("timeline-add-key")
    state.width, state.height = width, 240 if width == 210 else 165
    state.frame(3)
    toolbar = ui.hits["timeline-toolbar"]
    for key, (x, _y, w, _h) in ui.hits.items():
        if key not in ("timeline-sheet", "timeline-toolbar"):
            assert x >= 0, key
            assert x + w <= width * ui.s, key
    child = state.children["Timeline body"]
    assert child.scroll_max.y > 0
    parent = imgui.internal.find_window_by_name("Native timeline")
    assert parent.scroll_max.y == pytest.approx(0)
    clip = child.inner_clip_rect
    state.event("add_mouse_pos_event", clip.min.x + 5 * ui.s, clip.min.y + 5 * ui.s)
    state.event("add_mouse_wheel_event", 0, -5)
    state.frame(3)
    assert child.scroll.y > 0
    np.testing.assert_allclose(ui.hits["timeline-toolbar"], toolbar)
    state.click("collapse-timeline")
    assert state.collapsed == [True]


@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (210, 420, 1000))
def test_follow_modes_remain_visible_and_use_real_editor_navigation(
    timeline, native_fonts, language, width, monkeypatch
):
    state, ui = timeline, timeline.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    state.width = width
    captions, button = [], ui.button

    def observe(key, *args, **kwargs):
        if key.startswith("timeline-") and kwargs.get("icon") and kwargs.get("label"):
            captions.append(kwargs["align"])
        return button(key, *args, **kwargs)

    monkeypatch.setattr(ui, "button", observe)
    state.frame(3)
    rects = [ui.hits[f"timeline-follow-{index}"] for index in range(3)]
    assert all(rect[3] == pytest.approx(28 * ui.s) for rect in rects)
    assert all(rects[index][0] + rects[index][2] == rects[index + 1][0] for index in (0, 1))
    state.click("timeline-follow-2")
    assert state.panel.editor.follow_mode == "locked"
    assert captions and set(captions) == {"center"}
    toolbar = ui.hits["timeline-tools"]
    items = [
        rect
        for key, rect in ui.hits.items()
        if key.startswith("timeline-")
        and key not in ("timeline-toolbar", "timeline-tools", "timeline-follow")
        and toolbar[1] <= rect[1] < toolbar[1] + toolbar[3]
    ]
    for index, rect in enumerate(items):
        x, y, w, h = rect
        assert toolbar[0] <= x < x + w <= toolbar[0] + toolbar[2]
        assert toolbar[1] <= y < y + h <= toolbar[1] + toolbar[3]
        for other in items[index + 1 :]:
            assert (
                x + w <= other[0]
                or other[0] + other[2] <= x
                or y + h <= other[1]
                or other[1] + other[3] <= y
            )


@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (210, 420, 1000))
def test_model_selector_reserves_caption_arrow_and_add_hit_regions(
    timeline, native_fonts, language, width, monkeypatch
):
    state, ui = timeline, timeline.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    state.width = width
    captions, tooltips, text, tooltip = [], [], ImguiDraw2D.text, imgui.set_item_tooltip

    def observe(draw, position, color, value, **kwargs):
        bounds = ui.hits.get("timeline-model-caption")
        if bounds is not None and position[0] == bounds[0]:
            clip = draw._dl._clip_rect_stack[len(draw._dl._clip_rect_stack) - 1]
            captions.append((value, draw.text_size(value)[0], tuple(clip)))
        return text(draw, position, color, value, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", observe)

    def observe_tooltip(value):
        if imgui.get_item_id() == imgui.get_id("##keyframe-model") and imgui.is_item_hovered():
            tooltips.append(value)
        return tooltip(value)

    monkeypatch.setattr(imgui, "set_item_tooltip", observe_tooltip)
    state.frame(3)
    combo, caption, add = (
        ui.hits[key] for key in ("timeline-model", "timeline-model-caption", "timeline-add-key")
    )
    assert combo[3] == add[3] == pytest.approx(28 * ui.s)
    assert caption[0] == pytest.approx(combo[0] + 8 * ui.s)
    assert caption[0] + caption[2] == pytest.approx(combo[0] + combo[2] - 36 * ui.s)
    assert combo[0] + combo[2] + 5 * ui.s == pytest.approx(add[0])
    assert add[0] + add[2] <= state.panel.last_layout.time_bounds[0] - 10 * ui.s
    shown, measured, clip = captions[-1]
    assert measured <= caption[2]
    assert clip[0] >= caption[0] and clip[2] <= caption[0] + caption[2]
    name = state.session.scene_models[0].name
    assert shown == name if width >= 420 else shown != name
    state.event("add_mouse_pos_event", combo[0] + combo[2] / 2, combo[1] + combo[3] / 2)
    assert tooltips[-1] == name
    state.click("timeline-model")
    assert name + "##keyframe-model-0" in state.controls


@pytest.mark.parametrize("native", (False, True))
def test_model_header_selection_and_add_keep_shared_transactions(
    timeline, native, tmp_path, monkeypatch
):
    state = timeline
    path = tmp_path / "Second model.xml"
    path.write_text(
        '<mujoco><worldbody><body><joint type="slide"/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    assert state.session.submit(cmd.AddSceneModel(path)).ok
    second = state.session.scene_models[-1]
    if not native:

        def header(editor, ctx, layout, models, editable, icon_drawer):
            imgui.set_cursor_screen_pos((layout.lo[0], layout.lo[1]))
            draw_model_header(
                editor, ctx, models, editable, layout.channel_width, icon_drawer=icon_drawer
            )

        monkeypatch.setattr(state.panel.track, "draw_header", header)
    state.frame(3)
    state.click("timeline-model" if native else "##keyframe-model")
    if native:
        state.click(second.name + "##keyframe-model-1")
    else:
        state.event("add_key_event", imgui.Key.down_arrow, True)
        state.event("add_key_event", imgui.Key.down_arrow, False)
        state.event("add_key_event", imgui.Key.enter, True)
        state.event("add_key_event", imgui.Key.enter, False)
    assert state.panel.editor.model_id == second.model_id
    depth = len(state.session._undo_stack)
    state.click("timeline-add-key")
    assert len(state.session.model_keyframes(second.model_id)) == 1
    assert not state.session.model_keyframes(0)
    assert len(state.session._undo_stack) == depth + 1
    assert state.session.submit(cmd.Undo()).ok
    assert not state.session.model_keyframes(second.model_id)


def test_lane_selection_uses_one_background_and_real_track_hits(timeline, monkeypatch):
    state = timeline
    rectangles, rect_filled = [], ImguiDraw2D.rect_filled

    def observe(draw, lo, hi, color, **kwargs):
        rectangles.append((lo, hi))
        return rect_filled(draw, lo, hi, color, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "rect_filled", observe)
    for index, lane in enumerate(("model", "take", "snapshots")):
        layout = state.panel.last_layout
        state.click_at((layout.lo[0] + 30 * state.study.s, layout.lane_y(index)))
        assert state.panel.editor.edit_lane == lane
        rectangles.clear()
        state.frame()
        layout = state.panel.last_layout
        backgrounds = [
            (lo, hi)
            for lo, hi in rectangles
            if lo[0] == layout.lo[0]
            and lo[1] >= layout.ruler_bottom
            and hi[0] < layout.lo[0] + layout.channel_width
        ]
        assert len(backgrounds) == 1
        assert backgrounds[0][1][0] - backgrounds[0][0][0] > 20 * state.study.s


@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (140, 190))
def test_compact_toolbar_reflows_real_actions_without_overlap(
    timeline, native_fonts, language, width
):
    state, ui = timeline, timeline.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    state.width = width
    state.frame(3)
    toolbar = ui.hits["timeline-toolbar"]
    assert toolbar[3] == pytest.approx(28 * ui.s)
    rects = [
        rect
        for key, rect in ui.hits.items()
        if key
        in (
            "timeline-play-pause",
            "timeline-time",
            "collapse-timeline",
        )
    ]
    for index, (x, y, w, h) in enumerate(rects):
        assert toolbar[0] <= x < x + w <= toolbar[0] + toolbar[2]
        assert x + w <= width * ui.s
        for other in rects[index + 1 :]:
            assert (
                x + w <= other[0]
                or other[0] + other[2] <= x
                or y + h <= other[1]
                or other[1] + other[3] <= y
            )
    _record_take(state, 10)
    assert not imgui.get_current_context().open_popup_stack
    tools = ui.hits["timeline-tools"]
    for key in (
        "timeline-first",
        "timeline-last",
        "timeline-take",
        "timeline-loop",
        "timeline-snapshot",
        "timeline-video",
    ):
        x, _, w, _ = ui.hits[key]
        assert tools[0] <= x < x + w <= tools[0] + tools[2]
    state.click("timeline-follow-2")
    assert state.panel.editor.follow_mode == "locked"


@pytest.mark.parametrize("native_fonts", (True,))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
@pytest.mark.parametrize("width", (140, 210, 330, 600, 1000, 1600))
def test_all_main_timeline_controls_are_unique_and_reflow_without_overlap(
    timeline, native_fonts, language, width, monkeypatch
):
    state, ui = timeline, timeline.study.ui
    ui.translate = state.ctx.translate = Localizer(Language(language)).text
    state.ctx.recording_config = RecordingConfig()
    state.ctx.set_recording_config = lambda value, **_kwargs: setattr(
        state.ctx, "recording_config", value
    )
    state.width, state.height = width, 600
    submitted, button = [], ui.button

    def observe(key, *args, **kwargs):
        submitted.append(key)
        return button(key, *args, **kwargs)

    monkeypatch.setattr(ui, "button", observe)
    state.frame(2)
    submitted.clear()
    state.frame()
    keys = (
        "timeline-first",
        "timeline-previous",
        "timeline-play-pause",
        "timeline-next",
        "timeline-last",
        "timeline-record",
        "timeline-take",
        "timeline-snapshot",
        "timeline-video",
        "timeline-loop",
        "timeline-follow-0",
        "timeline-follow-1",
        "timeline-follow-2",
        "timeline-fit",
        "timeline-settings",
        "##timeline-range-start",
        "##timeline-range-end",
        "collapse-timeline",
    )
    assert all(key in ui.hits for key in keys)
    assert len(submitted) == len(set(submitted))
    assert "timeline-more" not in ui.hits and "timeline-options" not in ui.hits
    rectangles = [ui.hits[key] for key in keys]
    for index, (x, y, w, h) in enumerate(rectangles):
        assert 0 <= x < x + w <= width * ui.s
        assert h == pytest.approx(28 * ui.s, abs=1)
        for ox, oy, ow, oh in rectangles[index + 1 :]:
            assert (
                x + w <= ox + 0.5 or ox + ow <= x + 0.5 or y + h <= oy + 0.5 or oy + oh <= y + 0.5
            )
    primary = ui.hits["timeline-toolbar"]
    state.click("timeline-follow-2")
    assert state.panel.editor.follow_mode == "locked"
    assert "timeline-pause-at-end" not in ui.hits
    assert "##timeline-countdown" not in ui.hits
    state.click("timeline-settings")
    state.click("timeline-pause-at-end")
    assert not state.session.state_take_pause_at_end
    state.event("add_key_event", imgui.Key.escape, True)
    state.event("add_key_event", imgui.Key.escape, False)
    np.testing.assert_allclose(ui.hits["timeline-toolbar"], primary)
    assert not imgui.get_current_context().open_popup_stack


def test_direct_range_fields_edit_view_and_actual_take_loop(timeline):
    state = timeline
    _set_view(state)
    state.text("##timeline-range-start", "0.4")
    state.text("##timeline-range-end", "1.8")
    assert (state.panel.editor.view_start, state.panel.editor.view_end) == pytest.approx((0.4, 1.8))
    _record_take(state, 16)
    times = state.session.state_take_times
    state.text("##timeline-range-start", str(times[3]))
    state.text("##timeline-range-end", str(times[-3]))
    assert state.session.state_take_range == (3, len(times) - 3)
    before = state.session.state_take_loop_enabled
    state.click("timeline-loop")
    assert state.session.state_take_loop_enabled is not before


def test_active_selected_and_hovered_key_markers_keep_their_solid_center(timeline, monkeypatch):
    state = timeline
    state.click("timeline-add-key")
    key = state.session.model_keyframes(0)[0]
    assert state.session.submit(cmd.LoadKeyframe(key.keyframe_id)).ok
    assert state.session.active_keyframe == key.keyframe_id
    circles, diamonds = [], []
    circle, fill = ImguiDraw2D.circle_filled, ImguiDraw2D.convex_fill

    def observe_circle(draw, center, radius, color, **kwargs):
        circles.append((tuple(center), color))
        return circle(draw, center, radius, color, **kwargs)

    def observe_fill(draw, points, color):
        diamonds.append((points, color))
        return fill(draw, points, color)

    monkeypatch.setattr(ImguiDraw2D, "circle_filled", observe_circle)
    monkeypatch.setattr(ImguiDraw2D, "convex_fill", observe_fill)
    for selected in (False, True):
        state.panel.editor.selected_id = key.keyframe_id if selected else -1
        state.panel.editor.selected_keyframes = {key.keyframe_id} if selected else set()
        state.event("add_mouse_pos_event", *state.point(key.time))
        circles.clear()
        diamonds.clear()
        state.frame()
        assert diamonds
        assert all(color != state.ctx.theme.bg_window for _center, color in circles)
