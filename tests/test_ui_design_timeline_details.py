"""Native timeline details retain real editing, loading and snapshot commands."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.adapters.mujoco import MuJoCoAdapter
from mojive.adapters.workspace import WorkspaceAdapter
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.ui.app import ViewerApp
from mojive.ui.keyframe_editor.controller import TimelineEditor
from mojive.ui.keyframe_editor.properties import draw_selected
from mojive.ui.panels import PanelContext
from mojive.ui.workspace.timeline_details import NativeTimelineDetails
from mojive.ui.workspace.widgets import UI

pytestmark = pytest.mark.physics


@pytest.fixture(params=(1.0, 2.0))
def details(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    path = tmp_path / "keys.xml"
    path.write_text(
        '<mujoco><worldbody><body><joint name="joint" type="slide"/>'
        '<geom size="0.1"/></body></worldbody><keyframe>'
        '<key name="Home" time="1" qpos="0.1"/>'
        '<key name="Reach" time="2" qpos="0.2"/></keyframe></mujoco>'
    )
    session = Session(WorkspaceAdapter(MuJoCoAdapter(path)))
    assert session.submit(cmd.Pause()).ok
    session.tick(FrameNeeds(qpos=True), wall_dt=0)
    context = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.delta_time = 1 / 60
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
    scale = request.param
    style = imgui.get_style()
    style.window_padding = (10 * scale, 10 * scale)
    style.item_spacing = (6 * scale, 6 * scale)
    style.font_size_base = 14 * scale
    ui = UI.__new__(UI)
    ui.s, ui.translate, ui.hits, ui.numeric_edits = scale, str, {}, {}
    ui.body = ui.bold = ui.mono = io.fonts.add_font_default()
    study = SimpleNamespace(ui=ui, s=scale)
    ctx = PanelContext(session, NullBackend(), style_scale=scale)
    editor = TimelineEditor()
    editor.set_model(0)
    keys, _ = editor.keyframes(ctx)
    editor.select_key(
        keys[0].keyframe_id, keys[0].time, additive=False, editable=False, mouse_x=0, marker_x=0
    )
    state = SimpleNamespace(
        session=session,
        editor=editor,
        ui=ui,
        ctx=ctx,
        io=io,
        drawer=NativeTimelineDetails(study),
        width=700,
        editable=True,
        native=True,
    )

    def frame(count=1):
        for _ in range(count):
            session.tick(FrameNeeds(qpos=True), wall_dt=0)
            editor.keyframes(ctx)
            io.display_size = (state.width * scale, 240 * scale)
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(io.display_size)
            imgui.begin("Timeline details", flags=imgui.WindowFlags_.no_title_bar)
            ui.hits.clear()
            ui.numeric_edits.clear()
            pos, avail = imgui.get_cursor_screen_pos(), imgui.get_content_region_avail()
            state.origin, state.available = (pos.x, pos.y), avail.x
            state.height = state.drawer.height(editor, ctx, avail.x)
            if state.native:
                state.drawer.draw(editor, ctx, state.editable)
            else:
                draw_selected(editor, ctx, state.editable)
            imgui.end()
            imgui.render()

    def event(method, *args):
        getattr(io, method)(*args)
        frame()

    def click(key):
        x, y, w, h = ui.hits[key]
        event("add_mouse_pos_event", x + w / 2, y + h / 2)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)
        frame(3)

    def text(key, value):
        numeric = key == "timeline-detail-time"
        modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
        if numeric:
            event("add_key_event", modifier, True)
        click(key)
        if numeric:
            event("add_key_event", modifier, False)
        assert io.want_text_input
        event("add_key_event", modifier, True)
        event("add_key_event", imgui.Key.a, True)
        event("add_key_event", imgui.Key.a, False)
        event("add_key_event", modifier, False)
        io.add_input_characters_utf8(value)
        frame()
        event("add_key_event", imgui.Key.enter, True)
        event("add_key_event", imgui.Key.enter, False)

    state.frame, state.event, state.click, state.text = frame, event, click, text
    frame(3)
    try:
        yield state
    finally:
        session.release()
        imgui.destroy_context(context)


@pytest.mark.parametrize("width", (210, 280, 330, 440, 1000))
def test_native_details_reflow_without_overlapping_fields_or_actions(details, width):
    state = details
    state.width = width
    state.frame(3)
    keys = (
        "timeline-detail-name",
        "timeline-detail-time",
        "timeline-detail-apply",
        "timeline-detail-load",
        "timeline-detail-delete",
    )
    rectangles = [state.ui.hits[key] for key in keys]
    for x, y, w, h in rectangles:
        assert h == pytest.approx(28 * state.ui.s, abs=0.5)
        assert x >= state.origin[0] and x + w <= state.origin[0] + state.available + 0.5
        assert y + h <= state.origin[1] + state.height + 0.5
    for index, (x, y, w, h) in enumerate(rectangles):
        for ox, oy, ow, oh in rectangles[index + 1 :]:
            assert not (x < ox + ow and ox < x + w and y < oy + oh and oy < y + h)
    if width >= 280:
        assert len({round(rect[1], 2) for rect in rectangles}) == 1
        assert state.height == pytest.approx(28 * state.ui.s)
        assert state.ui.hits["timeline-detail-name"][2] >= state.ui.measure(state.editor.name)
        assert state.ui.hits["timeline-detail-delete"][2] == pytest.approx(28 * state.ui.s)
    if 280 <= width < 440:
        assert state.ui.hits["timeline-detail-apply"][2] == pytest.approx(28 * state.ui.s)
        assert state.ui.hits["timeline-detail-load"][2] == pytest.approx(44 * state.ui.s)
        assert state.ui.hits["timeline-detail-time"][2] == pytest.approx(70 * state.ui.s)
    if width == 1000:
        assert state.ui.hits["timeline-detail-name"][2] == pytest.approx(280 * state.ui.s)
        assert state.ui.hits["timeline-detail-time"][0] == pytest.approx(
            sum(state.ui.hits["timeline-detail-name"][::2]) + 6 * state.ui.s
        )
    assert state.ui.hits["timeline-detail-bar"][3] == pytest.approx(state.height, abs=0.5)
    state.editor.capture_snapshot(state.ctx)
    state.frame(3)
    for key in (
        "timeline-detail-name",
        "timeline-detail-time",
        "timeline-detail-load",
        "timeline-detail-delete",
    ):
        x, y, w, h = state.ui.hits[key]
        assert x >= state.origin[0] and x + w <= state.origin[0] + state.available + 0.5
        assert y + h <= state.origin[1] + state.height + 0.5


def test_native_details_keep_drafts_until_apply_and_submit_one_undo(details):
    state = details
    before = state.session.keyframe_properties(state.editor.selected_id)
    state.click("timeline-detail-apply")
    assert not state.session.can_undo
    state.text("timeline-detail-name", "Rest pose")
    state.text("timeline-detail-time", "1.75")
    assert state.editor.name == "Rest pose" and state.editor.time == pytest.approx(1.75)
    assert state.session.keyframe_properties(before.keyframe_id) == before
    state.click("timeline-detail-apply")
    assert state.session.keyframe_properties(before.keyframe_id) == replace(
        before, name="Rest pose", time=1.75
    )
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.keyframe_properties(before.keyframe_id) == before
    assert not state.session.can_undo


def test_native_details_empty_name_pause_gate_and_invalid_time_preserve_atomicity(details):
    state = details
    before = state.session.keyframe_properties(state.editor.selected_id)
    state.text("timeline-detail-name", " ")
    state.click("timeline-detail-apply")
    assert state.session.keyframe_properties(before.keyframe_id) == before
    state.text("timeline-detail-name", "New pose")
    state.editable = False
    state.frame()
    state.click("timeline-detail-apply")
    state.click("timeline-detail-delete")
    assert state.session.keyframe_properties(before.keyframe_id) == before
    state.editable = True
    state.editor.time = float("nan")
    state.frame()
    state.click("timeline-detail-apply")
    assert state.editor.error
    assert state.editor.selected_id == before.keyframe_id
    assert state.session.keyframe_properties(before.keyframe_id) == before
    assert not state.session.can_undo
    assert "timeline-detail-error" in state.ui.hits
    assert "timeline-detail-copy-error" in state.ui.hits


def test_native_details_load_delete_and_undo_use_shared_keyframe_commands(details):
    state = details
    before = state.session.keyframe_properties(state.editor.selected_id)
    joint = next(joint for joint in state.session.joints if joint.name == "joint")
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, 0.6)).ok
    state.frame()
    state.click("timeline-detail-load")
    assert state.session.frame.qpos[joint.qpos_adr] == pytest.approx(before.qpos[0])
    state.click("timeline-detail-delete")
    assert [key.name for key in state.session.model_keyframes(0)] == ["Reach"]
    assert state.editor.selected_id < 0
    assert state.session.submit(cmd.Undo()).ok
    assert state.session.keyframe_properties(before.keyframe_id) == before
    state.frame()
    assert state.height == pytest.approx(28 * state.ui.s)
    assert not any(key in state.ui.hits for key in ("timeline-detail-name", "timeline-detail-load"))


def test_native_snapshot_details_restore_and_remove_real_state(details):
    state = details
    joint = next(joint for joint in state.session.joints if joint.name == "joint")
    original = state.session.frame.qpos.copy()
    state.editor.capture_snapshot(state.ctx)
    snapshot_id = state.editor.selected_snapshot
    assert state.session.submit(cmd.SetQpos(joint.qpos_adr, 0.6)).ok
    state.frame(3)
    assert "timeline-detail-apply" not in state.ui.hits
    state.click("timeline-detail-load")
    np.testing.assert_allclose(state.session.frame.qpos, original)
    assert state.editor.selected_snapshot == snapshot_id
    state.click("timeline-detail-delete")
    assert not state.session.scene_snapshots
    assert state.editor.selected_snapshot == -1


def test_native_snapshot_disabled_restore_preserves_selection_and_scene(details):
    state = details
    state.editor.capture_snapshot(state.ctx)
    snapshot_id = state.editor.selected_snapshot
    assert state.session.submit(cmd.SetQpos(state.session.joints[0].qpos_adr, 0.6)).ok
    state.ctx.take_video_active = True
    state.frame(3)
    original = state.session.frame.qpos.copy()
    state.click("timeline-detail-load")
    assert not state.editor.error
    assert state.editor.selected_snapshot == snapshot_id
    np.testing.assert_allclose(state.session.frame.qpos, original)


def test_native_keyframe_apply_retains_the_application_pending_edit_boundary(details):
    state = details
    app = ViewerApp(state.session, NullBackend())
    state.ctx.queue_model_edit = app._queue_model_edit
    before = state.session.keyframe_properties(state.editor.selected_id)
    state.text("timeline-detail-name", "Pending pose")
    state.click("timeline-detail-apply")
    assert app.model_edits.active
    assert state.session.adapter.keyframe_properties(before.keyframe_id) == before
    assert state.session.keyframe_properties(before.keyframe_id).name == "Pending pose"
    assert not state.session.can_undo
    assert len(app.model_edits.commands) == 1
    assert app.model_edits.commands[0].name == "Pending pose"


def test_default_shared_keyframe_properties_still_draw_without_native_presentation(details):
    state = details
    state.native = False
    state.frame(3)
    assert imgui.get_draw_data().total_vtx_count > 0
    assert state.editor.properties.name == "Home"
    assert not state.session.can_undo
