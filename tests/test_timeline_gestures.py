"""A cancelled key drag must not retime the document when its mouse button releases."""

from types import SimpleNamespace

import pytest

from mojive.adapters.base import KeyframeInfo
from mojive.interaction.input import InputClaim
from mojive.ui.keyframe_editor.controller import TimelineHit
from mojive.ui.panels.keyframes import KeyframesPanel
from mojive.ui.pointer_bindings import PointerChord, PointerFrame


@pytest.mark.parametrize("cancel", ["escape", "application", "none"])
def test_key_drag_cancel_preserves_time_and_selection(monkeypatch, cancel):
    panel = KeyframesPanel()
    key = KeyframeInfo(7, "key", 1.0)
    panel.editor.drag_id = panel.editor.selected_id = 7
    panel.editor.selected_keyframes = {7}
    panel.editor.pointer_mode = "key"
    panel.editor.pointer_chord = PointerChord((0,))
    panel.editor.drag_moved = True
    panel.editor.drag_preview_time = panel.editor.playhead = 2.0
    panel.editor.view_start, panel.editor.view_end = 0, 3
    commits = []
    monkeypatch.setattr(
        panel.editor, "retime_keyframe", lambda _ctx, key_id, time: commits.append((key_id, time))
    )
    ctx = SimpleNamespace(
        style_scale=1,
        session=SimpleNamespace(state_take_range=None),
        input_claim=InputClaim(pointer=cancel == "application"),
    )
    hit = TimelineHit((200, 30), (0, 300), "model", True, 7, -1, {7: 200})
    held = PointerFrame(buttons=frozenset() if cancel == "application" else frozenset({0}))
    panel.editor.update_timeline_drag(
        ctx, (key,), (), hit, held, editable=True, clear_range=False, escape=cancel == "escape"
    )
    panel.editor.update_timeline_drag(
        ctx, (key,), (), hit, PointerFrame(), editable=True, clear_range=False, escape=False
    )
    if cancel == "none":
        assert commits == [(7, 2.0)]
    else:
        assert commits == []
        assert panel.editor.playhead == 1.0
        assert panel.editor.selected_keyframes == {7}
    assert panel.editor.drag_id == -1


@pytest.mark.parametrize("mode", ["range", "pan", "scrub"])
def test_application_pointer_claim_does_not_finish_an_interrupted_gesture(monkeypatch, mode):
    panel = KeyframesPanel()
    panel.editor.pointer_mode = mode
    panel.editor.pointer_chord = PointerChord((0,))
    panel.editor.range_preview = (0, 1)
    panel.editor.scrub_resume = True
    commits = []

    def submit(command):
        commits.append(command)
        return SimpleNamespace(ok=True)

    ctx = SimpleNamespace(
        style_scale=1,
        session=SimpleNamespace(state_take_range=(0, 2)),
        input_claim=InputClaim(pointer=True),
        submit=submit,
    )
    hit = TimelineHit((200, 30), (0, 300), "take", True, -1, -1, {})
    released = panel.editor.update_timeline_drag(
        ctx, (), (0, 1, 2), hit, PointerFrame(), editable=True, clear_range=False, escape=False
    )
    assert released
    assert not panel.editor.finish_timeline_range(ctx, hit, hovered=True)
    assert commits == []
    assert panel.editor.pointer_mode == ""
    assert panel.editor.range_preview is None
    assert not panel.editor.scrub_resume


@pytest.mark.parametrize("cancel", ["escape", "application", "none"])
def test_drawn_panel_cancels_drag_using_real_imgui_input(monkeypatch, cancel, tmp_path):
    from dataclasses import replace

    from imgui_bundle import imgui

    from mojive.adapters.base import SceneModelInfo
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.render.backend import NullBackend
    from mojive.scene import Scene
    from mojive.session import Session
    from mojive.ui.panels import PanelContext

    class Adapter(StaticSceneAdapter):
        caps = replace(
            StaticSceneAdapter.caps, keyframes=True, features=(("model.keyframe_edit", 1),)
        )

        def keyframes(self):
            return [KeyframeInfo(7, "key", 0.5, 0)]

        def scene_models(self):
            return (SceneModelInfo(0, "model", tmp_path / "model", False),)

    context = imgui.create_context()
    session = Session(Adapter(Scene()))
    try:
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (1000, 500)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        panel = KeyframesPanel()
        ctx = PanelContext(session, NullBackend())
        positions = []
        commits = []
        from mojive.ui.keyframe_editor import track

        paint = track.paint_dope_sheet

        def observe(editor, *args):
            positions[:] = [(args[3] + (args[4] - args[3]) * 0.25, args[6])]
            return paint(editor, *args)

        monkeypatch.setattr(track, "paint_dope_sheet", observe)
        monkeypatch.setattr(
            panel.editor, "retime_keyframe", lambda _ctx, key, time: commits.append((key, time))
        )

        def frame():
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((1000, 500))
            imgui.begin("Keyframes")
            panel.draw(ctx)
            imgui.end()
            imgui.render()

        for _ in range(3):
            frame()
        panel.editor.view_start, panel.editor.view_end, panel.editor.view_needs_fit = 0, 2, False
        panel.editor.set_follow_mode("off")
        frame()
        x, y = positions[0]
        io.add_mouse_pos_event(x, y)
        frame()
        io.add_mouse_button_event(0, True)
        frame()
        assert panel.editor.drag_id == 7
        io.add_mouse_pos_event(x + 150, y)
        frame()
        assert panel.editor.drag_moved
        if cancel == "escape":
            io.add_key_event(imgui.Key.escape, True)
        elif cancel == "application":
            ctx.input_claim = InputClaim(pointer=True)
        frame()
        io.add_mouse_button_event(0, False)
        io.add_key_event(imgui.Key.escape, False)
        for _ in range(2):
            frame()
        assert bool(commits) == (cancel == "none")
        assert panel.editor.pointer_mode == ""
        assert panel.editor.selected_keyframes == {7}
    finally:
        session.release()
        imgui.destroy_context(context)
