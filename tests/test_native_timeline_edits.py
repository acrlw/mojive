"""Native timeline numeric gestures restore previews when Escape cancels them."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.config import RecordingConfig
from mojive.render.backend import NullBackend
from mojive.scene.assets import resolve
from mojive.session import Session
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.timeline_panel import NativeTimelineToolbar, _ToolbarRows
from mojive.ui.workspace.widgets import UI

pytestmark = pytest.mark.physics


@pytest.fixture(params=(1.0, 2.0))
def timeline_edits(request):
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (800 * request.param, 400 * request.param)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    ui = UI(SimpleNamespace(style_scale=request.param, font_report=SimpleNamespace(cjk_path=None)))
    session = Session(MuJoCoAdapter(resolve("joint_gizmo")))
    ctx = PanelContext(session, NullBackend(), style_scale=ui.s)
    ctx.recording_config = RecordingConfig()
    writes = []

    def setting(value, *, persist=True):
        ctx.recording_config = RecordingConfig.from_mapping(asdict(value))
        writes.append((ctx.recording_config, persist))

    ctx.set_recording_config = setting
    toolbar = NativeTimelineToolbar(SimpleNamespace(ui=ui, s=ui.s))
    editor = TimelineEditor()
    state = SimpleNamespace(
        ui=ui, session=session, ctx=ctx, editor=editor, writes=writes, mode="recording"
    )

    def frame(count=1):
        for _ in range(count):
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Timeline numeric edits")
            ui.numeric_edits.clear()
            rows = _ToolbarRows(imgui.get_cursor_screen_pos(), 700 * ui.s, ui.s)
            if state.mode == "recording":
                toolbar._draw_recording_fields(ctx)
            else:
                toolbar._draw_range(editor, ctx, session.state_take_times, rows)
            imgui.end()
            imgui.render()

    def event(name, *args):
        getattr(io, name)(*args)
        frame()

    def drag(key, amount):
        x, y, width, height = ui.hits[key]
        event("add_mouse_pos_event", x + width / 2, y + height / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_pos_event", x + width / 2 + amount * ui.s, y + height / 2)

    def cancel():
        event("add_key_event", imgui.Key.escape, True)
        event("add_key_event", imgui.Key.escape, False)
        event("add_mouse_button_event", 0, False)
        frame(3)

    state.frame, state.event, state.drag, state.cancel = frame, event, drag, cancel
    try:
        frame(3)
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("field", ("countdown", "end_hold"))
def test_recording_drag_escape_restores_without_persisting(timeline_edits, field):
    state = timeline_edits
    original = getattr(state.ctx.recording_config, field)
    state.drag("##timeline-" + field, 50)
    assert getattr(state.ctx.recording_config, field) > original
    assert state.writes and not any(persist for _, persist in state.writes)
    state.cancel()
    assert getattr(state.ctx.recording_config, field) == original
    assert not any(persist for _, persist in state.writes)
    state.frame(30)
    state.drag("##timeline-" + field, 30)
    state.event("add_mouse_button_event", 0, False)
    assert getattr(state.ctx.recording_config, field) > original
    assert sum(persist for _, persist in state.writes) == 1


def test_empty_take_range_escape_restores_view(timeline_edits):
    state = timeline_edits
    state.mode = "range"
    state.editor.view_start, state.editor.view_end, state.editor.view_needs_fit = 0.0, 10.0, True
    state.frame(3)
    state.drag("##timeline-range-start", 50)
    assert state.editor.view_start > 0
    state.cancel()
    assert (state.editor.view_start, state.editor.view_end, state.editor.view_needs_fit) == (
        0.0,
        10.0,
        True,
    )


@pytest.mark.parametrize("selected", (None, (5, 30)))
@pytest.mark.parametrize("loop", (False, True))
def test_take_range_escape_restores_range_and_independent_loop(timeline_edits, selected, loop):
    state = timeline_edits
    assert state.session.submit(cmd.StartStateTakeRecording()).ok
    for _ in range(40):
        state.session.tick(FrameNeeds.none(), wall_dt=0.01)
    assert state.session.submit(cmd.StopStateTakeRecording()).ok
    assert state.session.submit(cmd.SetStateTakeRange(*(selected or (None, None)))).ok
    assert state.session.submit(cmd.SetStateTakeLoopEnabled(loop)).ok
    state.mode = "range"
    state.frame(3)
    state.drag("##timeline-range-start", 10)
    assert state.session.state_take_range != selected
    state.cancel()
    assert state.session.state_take_range == selected
    assert state.session.state_take_loop_enabled is loop
