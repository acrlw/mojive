"""Ruler and loop gestures restore real take poses and leave camera navigation alone."""

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import InputClaim, ViewerConfig, build
from mojive import commands as cmd
from mojive.adapters.base import FrameNeeds
from mojive.scene.assets import resolve
from mojive.tools.keyframe_timeline import (
    capture_instability_recovery,
    capture_range_playback,
    choose_follow,
    drag,
    populate_take,
    show_timeline,
    timeline_point,
)
from mojive.tools.ui_runtime import _click, _item_center, _item_rect, _right_click
from mojive.ui.panels.keyframes import nearest_take_frame

pytestmark = pytest.mark.gpu


def test_load_only_keyframes_accept_double_click_and_load_button(tmp_path, monkeypatch):
    from mojive.app.composition import build_from_adapter
    from mojive.tools.timeline_benchmark import _TimelineAdapter

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    adapter = _TimelineAdapter(40)
    with build_from_adapter(
        adapter, paused=True, show_window=False, vsync=False, width=1600, height=1000
    ) as viewer:
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        panel._set_follow_mode("off")
        assert not adapter.caps.supports("model.keyframe_edit")
        key = viewer.session.keyframes[10]
        point = timeline_point(viewer, key.time, "model")
        _click(viewer, point)
        _click(viewer, point)
        assert viewer.session.active_keyframe == key.keyframe_id
        other = viewer.session.keyframes[20]
        _click(viewer, timeline_point(viewer, other.time, "model"))
        _click(viewer, _item_center(viewer, "button", "Load"))
        assert viewer.session.active_keyframe == other.keyframe_id
        before = tuple((key.keyframe_id, key.time) for key in viewer.session.keyframes)
        drag(
            viewer,
            timeline_point(viewer, other.time, "model"),
            timeline_point(viewer, other.time + 0.4, "model"),
        )
        assert tuple((key.keyframe_id, key.time) for key in viewer.session.keyframes) == before


@pytest.mark.parametrize("scale", (1.0, 2.5))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_ruler_numbers_are_left_aligned_and_ink_centered(tmp_path, monkeypatch, scale, language):
    from mojive.ui.imgui_draw import ImguiDraw2D
    from mojive.ui.panels.keyframes import timeline_channel_width

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    with build(
        resolve("joint_types"),
        paused=True,
        show_window=False,
        vsync=False,
        width=min(3200, round(1800 * scale)),
        height=min(1800, round(1100 * scale)),
    ) as viewer:
        viewer.app.set_language(language)
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        panel._view_start, panel._view_end, panel._view_needs_fit = 0, 0.5, False
        viewer.sync()
        lo, hi = _item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
        actual_scale = viewer.window.style_scale
        left = lo[0] + timeline_channel_width(hi[0] - lo[0], actual_scale)
        center_y = lo[1] + 16 * actual_scale
        texts = []
        original = ImguiDraw2D.text

        def observe(draw, pos, color, text, **kwargs):
            if text in {"0.00", "0.50"} and lo[1] <= pos[1] < lo[1] + 32 * actual_scale:
                ink = draw.text_ink_bounds(text)
                texts.append((text, pos, ink))
            return original(draw, pos, color, text, **kwargs)

        monkeypatch.setattr(ImguiDraw2D, "text", observe)
        viewer.sync()
        assert {text for text, _, _ in texts} == {"0.00", "0.50"}
        for text, pos, ink in texts:
            assert pos[1] + (ink[1] + ink[3]) * 0.5 == pytest.approx(center_y, abs=actual_scale)
            if text == "0.00":
                assert pos[0] == pytest.approx(left + 4 * actual_scale)


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_timeline_recovers_from_qacc_at_high_ui_scale(tmp_path, monkeypatch, language):
    from mojive.app.composition import build_workspace

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "2.5")
    with build_workspace(
        resolve("joint_gizmo"),
        config=ViewerConfig(threaded_physics=False),
        paused=True,
        show_window=False,
        vsync=False,
        width=3200,
        height=1800,
    ) as viewer:
        viewer.app.localizer.set_language(language, persist=False)
        populate_take(viewer, 30)
        show_timeline(viewer)
        capture_instability_recovery(viewer, tmp_path)


@pytest.mark.parametrize("backend", ["empty", "toy", "fake"])
def test_timeline_without_models_preserves_drag_and_view_range(backend, tmp_path, monkeypatch):
    from mojive.adapters.base import SceneAdapterBase, SceneFrame, SceneSource
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.adapters.toy import ToyPhysicsAdapter
    from mojive.app.composition import build_from_adapter
    from mojive.scene import Scene

    class FakeAdapter(SceneAdapterBase):
        def scene_source(self):
            return SceneSource()

        def frame(self, needs):
            return SceneFrame()

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    adapter = {
        "empty": lambda: StaticSceneAdapter(Scene()),
        "toy": ToyPhysicsAdapter,
        "fake": FakeAdapter,
    }[backend]()
    with build_from_adapter(
        adapter, paused=True, show_window=False, vsync=False, width=1600, height=1000
    ) as viewer:
        show_timeline(viewer)
        session, panel = viewer.session, viewer.panels.get("Keyframes")
        assert not session.scene_models and not session.state_take_times
        panel._set_follow_mode("off")
        start, end = panel._view_start, panel._view_end
        span = end - start
        drag(
            viewer,
            timeline_point(viewer, start + span * 0.2),
            timeline_point(viewer, start + span * 0.8),
        )
        assert panel._playhead == pytest.approx(start + span * 0.8, abs=span * 0.002)
        assert not panel._editor.pointer_mode
        chosen_time = panel._playhead
        drag(
            viewer,
            timeline_point(viewer, start + span * 0.6),
            timeline_point(viewer, start + span * 0.4),
            button=1,
        )
        assert panel._view_start != pytest.approx(start)
        view_range = panel._view_start, panel._view_end
        for _ in range(4):
            viewer.sync()
        assert (panel._view_start, panel._view_end) == view_range
        assert panel._playhead == chosen_time
        assert session.paused and session.frame.time == 0
        _click(viewer, _item_center(viewer, "invisible_button", "##take-first"))
        for _ in range(3):
            viewer.sync()
        assert panel._playhead == 0


@pytest.fixture
def viewer(tmp_path, monkeypatch, request):
    from mojive.app.composition import build_workspace

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "1")
    builder = build_workspace if getattr(request, "param", "") == "workspace" else build
    with builder(
        resolve("joint_types"),
        paused=True,
        config=ViewerConfig(threaded_physics=False),
        vsync=False,
        width=1600,
        height=1000,
    ) as viewer:
        populate_take(viewer, 900)
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        panel._view_start, panel._view_end = 0, 30
        panel._view_needs_fit = False
        panel._set_follow_mode("off")
        yield viewer


def test_ruler_click_and_drag_seek_recorded_pose_and_stay_at_the_chosen_frame(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    _click(viewer, timeline_point(viewer, 12))
    expected = nearest_take_frame(session.state_take_times, 12)
    for _ in range(5):
        viewer.sync()
    assert session.state_take_cursor == expected
    assert panel._playhead == session.state_take_times[expected]
    assert session.adapter.data.time == pytest.approx(session.state_take_times[expected])
    drag(viewer, timeline_point(viewer, 8), timeline_point(viewer, 17))
    expected = nearest_take_frame(session.state_take_times, 17)
    assert session.state_take_cursor == expected
    assert not session.state_take_playing
    assert session.submit(cmd.PlayStateTake())
    drag(viewer, timeline_point(viewer, 4), timeline_point(viewer, 7))
    assert session.state_take_playing
    assert 7 - 1e-9 <= session.adapter.data.time < 7.5


def test_take_menu_opens_and_clears_recording_without_a_binding_error(viewer):
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(viewer, _item_center(viewer, "selectable", viewer.app.localizer.text("Clear take")))
    assert not viewer.session.state_take_times


def test_status_changes_between_track_selection_and_ruler_navigation(viewer):
    # Status ownership follows the last clicked panel; hover chooses its row hints.
    _click(viewer, timeline_point(viewer, 5))
    for row, hint_id in (
        ("take", "keyframes.select_all"),
        ("ruler", "keyframes.playhead"),
        ("model", "keyframes.select_all"),
    ):
        imgui.get_io().add_mouse_pos_event(*timeline_point(viewer, 5, row))
        for _ in range(3):
            viewer.sync()
        assert hint_id in {hint.hint_id for hint in viewer.app._panel_status_hints}
        assert "keyframes.select_all" in {hint.hint_id for hint in viewer.app._panel_status_hints}


def _key(viewer, key, *, ctrl=False):
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, ctrl)
    io.add_key_event(key, True)
    viewer.sync()
    io.add_key_event(key, False)
    io.add_key_event(modifier, False)
    viewer.sync()


@pytest.mark.parametrize("name", ("delete", "backspace", "escape"))
@pytest.mark.parametrize("whole_keyboard", (False, True))
def test_timeline_preserves_selection_for_application_owned_keys(viewer, name, whole_keyboard):
    panel = viewer.panels.get("Keyframes")
    drag(viewer, timeline_point(viewer, 10, "take"), timeline_point(viewer, 15, "take"))
    selection = panel._editor.take_selection
    assert selection is not None
    times = tuple(viewer.session.state_take_times)
    claim = InputClaim(keyboard=True) if whole_keyboard else InputClaim(keys={name})
    viewer.set_input_handler(lambda context: claim)
    _key(viewer, getattr(imgui.Key, name))
    assert panel._editor.take_selection == selection
    assert tuple(viewer.session.state_take_times) == times
    viewer.set_input_handler(None)
    _key(viewer, getattr(imgui.Key, name))
    assert panel._editor.take_selection is None
    if name != "escape":
        first, last = selection
        assert tuple(viewer.session.state_take_times) == times[:first] + times[last + 1 :]


@pytest.mark.parametrize("claimed", ("a", "ctrl", "super"))
def test_timeline_select_all_respects_application_key_and_modifier_claims(viewer, claimed):
    panel = viewer.panels.get("Keyframes")
    _click(viewer, timeline_point(viewer, 12, "take"))
    panel._editor.take_selection = None
    viewer.set_input_handler(lambda context: InputClaim(keys={claimed}))
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if claimed == "super" else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    io.add_key_event(imgui.Key.a, True)
    viewer.sync()
    io.add_key_event(imgui.Key.a, False)
    io.add_key_event(modifier, False)
    viewer.sync()
    assert panel._editor.take_selection is None
    viewer.set_input_handler(None)
    _key(viewer, imgui.Key.a, ctrl=True)
    assert panel._editor.take_selection == (0, len(viewer.session.state_take_times) - 1)


def test_take_menu_creates_switches_and_removes_individual_takes(viewer):
    session = viewer.session
    original = tuple(session.state_take_times)
    first_id = session.active_state_take_id
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(viewer, _item_center(viewer, "selectable", "New Take##new-take"))
    second_id = session.active_state_take_id
    assert second_id != first_id and len(session.state_takes) == 2
    assert not session.state_take_times
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert session.state_take_recording and session.active_state_take_id == second_id
    for _ in range(3):
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert not session.state_take_recording and session.state_take_times
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(viewer, _item_center(viewer, "selectable", f"Take {first_id}##take-{first_id}"))
    assert tuple(session.state_take_times) == original
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(viewer, _item_center(viewer, "invisible_button", f"##remove-take-{second_id}"))
    assert session.active_state_take_id == first_id
    assert len(session.state_takes) == 1
    assert tuple(session.state_take_times) == original


@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_take_menu_uses_one_row_and_restarts_names_after_deleting_all_takes(viewer, language):
    session = viewer.session
    viewer.app.localizer.set_language(language, persist=False)
    translate = viewer.app.localizer.text
    first_id = session.active_state_take_id
    assert session.submit(cmd.CreateStateTake())
    assert session.submit(cmd.StartStateTakeRecording(new_take=False))
    session.tick(FrameNeeds.none(), wall_dt=0.1)
    assert session.submit(cmd.StopStateTakeRecording())
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    menu_width = None
    for take in session.state_takes:
        name = viewer.panels.get("Keyframes")._take_name(viewer.app._panel_context(), take)
        row_lo, row_hi = _item_rect(viewer, "selectable", f"{name}##take-{take.take_id}")
        menu_width = row_hi[0] - row_lo[0]
        cross_lo, cross_hi = _item_rect(viewer, "invisible_button", f"##remove-take-{take.take_id}")
        assert row_lo[0] <= cross_lo[0] < cross_hi[0] <= row_hi[0]
        assert row_lo[1] == cross_lo[1] and row_hi[1] == cross_hi[1]
        _click(viewer, ((cross_lo[0] + cross_hi[0]) * 0.5, (cross_lo[1] + cross_hi[1]) * 0.5))
        assert take.take_id not in {item.take_id for item in session.state_takes}
    for _ in range(3):
        viewer.sync()
    new_lo, new_hi = _item_rect(viewer, "selectable", translate("New Take") + "##new-take")
    assert new_hi[0] - new_lo[0] >= menu_width
    _click(viewer, _item_center(viewer, "selectable", translate("New Take") + "##new-take"))
    assert session.state_takes[0].name == "Take 1"
    assert session.active_state_take_id != first_id
    viewer.app.localizer.set_language("zh_CN", persist=False)
    for _ in range(3):
        viewer.sync()
    assert viewer.panels.get("Keyframes")._take_label(viewer.app._panel_context()) == "片段 1"


@pytest.mark.parametrize("viewer", ["workspace"], indirect=True)
def test_track_selection_delete_and_ctrl_a_never_cross_model_and_take_tracks(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    model_id = session.scene_models[0].model_id
    for index in (90, 180, 270):
        assert session.submit(cmd.SeekStateTake(index))
        assert session.submit(cmd.AddModelKeyframe(model_id, f"key{index}"))
    viewer.sync()
    keys = tuple(session.keyframes)
    times = tuple(session.state_take_times)
    drag(viewer, timeline_point(viewer, 10, "take"), timeline_point(viewer, 15, "take"))
    first, last = panel._editor.take_selection
    assert first < last
    _key(viewer, imgui.Key.delete)
    assert tuple(session.state_take_times) == times[:first] + times[last + 1 :]
    assert tuple(session.keyframes) == keys
    assert panel._editor.take_selection is None
    _key(viewer, imgui.Key.a, ctrl=True)
    assert panel._editor.take_selection == (0, len(session.state_take_times) - 1)
    _key(viewer, imgui.Key.delete)
    assert not session.state_take_times and tuple(session.keyframes) == keys
    assert len(session.state_takes) == 1
    _click(viewer, timeline_point(viewer, 12, "model"))
    _key(viewer, imgui.Key.a, ctrl=True)
    assert panel._editor.selected_keyframes == {key.keyframe_id for key in keys}
    _key(viewer, imgui.Key.delete)
    draft = viewer.app.model_edits
    assert len(draft.commands) == len(keys)
    assert tuple(session.keyframes) == keys
    draft.applying = True
    result = session.apply_model_edits(draft.resolve_commands(session))
    assert result.ok, result.message
    draft.clear()
    assert not session.keyframes
    assert session.submit(cmd.Undo())
    assert [key.name for key in session.keyframes] == [key.name for key in keys]
    assert session.submit(cmd.Redo())
    assert not session.keyframes


def test_model_additive_click_and_text_edit_shortcuts_preserve_take_selection_scope(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    model_id = session.scene_models[0].model_id
    for index in (90, 180):
        assert session.submit(cmd.SeekStateTake(index))
        assert session.submit(cmd.AddModelKeyframe(model_id, f"pose{index}"))
    viewer.sync()
    keys = tuple(session.keyframes)
    times = tuple(session.state_take_times)
    _click(viewer, timeline_point(viewer, keys[0].time, "model"))
    io = imgui.get_io()
    modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
    io.add_key_event(modifier, True)
    _click(viewer, timeline_point(viewer, keys[1].time, "model"))
    io.add_key_event(modifier, False)
    viewer.sync()
    assert panel._editor.selected_keyframes == {key.keyframe_id for key in keys}
    _click(viewer, timeline_point(viewer, keys[0].time, "model"))
    _click(viewer, _item_center(viewer, "input_text", "##keyframe-name"))
    _key(viewer, imgui.Key.a, ctrl=True)
    _key(viewer, imgui.Key.delete)
    assert panel._name == ""
    assert tuple(session.keyframes) == keys and tuple(session.state_take_times) == times
    assert not viewer.app.model_edits.active


def test_record_button_overwrites_active_take_from_playhead(viewer):
    session = viewer.session
    take_id = session.active_state_take_id
    original = tuple(session.state_take_times)
    _click(viewer, timeline_point(viewer, 7))
    index = session.state_take_cursor
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert session.state_take_recording
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert not session.state_take_recording
    assert session.active_state_take_id == take_id and len(session.state_takes) == 1
    assert tuple(session.state_take_times[: index + 1]) == original[: index + 1]
    assert 7 <= session.state_take_times[-1] < 10


def test_toolbar_reserves_pending_keyframe_names_and_applies_the_batch(viewer):
    before = tuple(viewer.session.keyframes)
    button = _item_center(viewer, "invisible_button", "##add-model-keyframe")
    _click(viewer, button)
    _click(viewer, button)
    draft = viewer.app.model_edits
    assert draft.active
    assert tuple(viewer.session.keyframes) == before
    assert [command.name for command in draft.commands] == ["key1", "key2"]

    draft.applying = True
    result = viewer.session.apply_model_edits(draft.resolve_commands(viewer.session))
    assert result.ok, result.message
    draft.clear()
    assert [key.name for key in viewer.session.keyframes][-2:] == ["key1", "key2"]


def test_model_selector_uses_the_lane_header_without_scrubbing_the_take(viewer):
    from mojive.tools.ui_runtime import _item_rect
    from mojive.ui.panels.keyframes import timeline_channel_width

    lo, hi = _item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
    model_lo, model_hi = _item_rect(viewer, "combo", "##keyframe-model")
    assert lo[0] <= model_lo[0] < model_hi[0] < lo[0] + timeline_channel_width(hi[0] - lo[0], 1)
    assert lo[1] <= model_lo[1] < model_hi[1] <= lo[1] + 32
    cursor = viewer.session.state_take_cursor
    _click(viewer, ((model_lo[0] + model_hi[0]) * 0.5, (model_lo[1] + model_hi[1]) * 0.5))
    assert imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
    assert viewer.session.state_take_cursor == cursor
    imgui.get_io().add_key_event(imgui.Key.escape, True)
    viewer.sync()
    imgui.get_io().add_key_event(imgui.Key.escape, False)
    viewer.sync()


def test_toolbar_reuses_caption_layouts_and_refreshes_state_and_language(viewer):
    panel = viewer.panels.get("Keyframes")
    before = dict(panel._command_layouts)
    assert len(before) == 3
    for _ in range(3):
        viewer.sync()
    assert all(panel._command_layouts[key] is value for key, value in before.items())
    assert viewer.session.submit(cmd.StartStateTakeRecording())
    viewer.sync()
    assert panel._command_layouts["##take-record"][1] == "Stop Recording"
    viewer.app.localizer.set_language("zh_CN", persist=False)
    for _ in range(3):
        viewer.sync()
    assert panel._command_layouts["##take-record"][1] == viewer.app.localizer.text("Stop Recording")
    assert panel._command_layouts["##capture-snapshot"][1] == viewer.app.localizer.text(
        "Capture Snapshot"
    )
    assert len(panel._command_layouts) == 3
    assert viewer.session.submit(cmd.StopStateTakeRecording())


def test_first_and_last_buttons_seek_endpoints_while_step_buttons_move_one_frame(viewer):
    session = viewer.session
    assert session.submit(cmd.SeekStateTake(400))
    last = len(session.state_take_times) - 1
    for name, expected in (("first", 0), ("next", 1), ("last", last), ("previous", last - 1)):
        _click(viewer, _item_center(viewer, "invisible_button", f"##take-{name}"))
        assert session.state_take_cursor == expected


@pytest.mark.parametrize("scale", (1.0, 2.5))
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_selected_range_navigation_and_repeat_button_remain_independent(
    tmp_path, monkeypatch, scale, language
):
    from mojive.ui.panels import keyframes

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", str(scale))
    buttons = []
    original = keyframes._command_button

    def observe(item_id, kind, tooltip, *args, **kwargs):
        buttons.append((item_id, tooltip, kwargs.get("selected", False)))
        return original(item_id, kind, tooltip, *args, **kwargs)

    monkeypatch.setattr(keyframes, "_command_button", observe)
    with build(
        resolve("joint_types"),
        config=ViewerConfig(threaded_physics=False),
        paused=True,
        show_window=False,
        vsync=False,
        width=min(3200, round(1800 * scale)),
        height=min(1800, round(1100 * scale)),
    ) as viewer:
        viewer.app.localizer.set_language(language, persist=False)
        populate_take(viewer, 30)
        show_timeline(viewer)
        capture_range_playback(viewer, tmp_path)
        translate = viewer.app.localizer.text
        loop = [(label, selected) for name, label, selected in buttons if name == "##timeline-loop"]
        assert {label for label, _ in loop} == {translate("Loop")}
        assert {selected for _, selected in loop} == {False, True}
        assert ("##take-first", translate("Range first frame"), False) in buttons
        assert ("##take-last", translate("Range last frame"), False) in buttons


def test_shift_left_range_drag_and_shift_right_or_escape_clear_without_panning(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    before = session.camera
    view_range = panel._view_start, panel._view_end
    drag(viewer, timeline_point(viewer, 18), timeline_point(viewer, 9), shift=True)
    expected = (
        nearest_take_frame(session.state_take_times, 9),
        nearest_take_frame(session.state_take_times, 18),
    )
    assert session.state_take_loop == expected
    assert session.state_take_loop_enabled
    assert (panel._view_start, panel._view_end) == view_range
    np.testing.assert_allclose(session.camera.eye, before.eye)
    drag(
        viewer,
        timeline_point(viewer, 3),
        timeline_point(viewer, 6),
        shift=True,
        cancel=True,
    )
    assert session.state_take_loop is None
    drag(viewer, timeline_point(viewer, 9), timeline_point(viewer, 18), shift=True)
    assert session.state_take_loop == expected
    io = imgui.get_io()
    io.add_key_event(imgui.Key.mod_shift, True)
    _right_click(viewer, timeline_point(viewer, 12))
    io.add_key_event(imgui.Key.mod_shift, False)
    viewer.sync()
    assert session.state_take_loop is None
    assert not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id)
    assert (panel._view_start, panel._view_end) == view_range


def test_selecting_another_range_reenables_loop_after_manually_disabling_it(viewer):
    session = viewer.session
    drag(viewer, timeline_point(viewer, 9), timeline_point(viewer, 18), shift=True)
    bounds = session.state_take_range
    assert session.state_take_loop_enabled
    _click(viewer, _item_center(viewer, "invisible_button", "##timeline-loop"))
    assert not session.state_take_loop_enabled and session.state_take_range == bounds
    drag(viewer, timeline_point(viewer, 3), timeline_point(viewer, 6), shift=True)
    assert session.state_take_loop_enabled and session.state_take_range != bounds


@pytest.mark.parametrize("start_frame", (None, 300))
def test_record_stop_rewinds_and_selects_only_the_latest_recording_pass(
    viewer, start_frame, monkeypatch
):
    import mojive.ui.app.status as status_module

    session, panel = viewer.session, viewer.panels.get("Keyframes")
    original = status_module.draw_status
    details = []

    def observe(*args, **kwargs):
        details.append(kwargs.get("status", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(status_module, "draw_status", observe)
    if start_frame is None:
        _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
        _click(viewer, _item_center(viewer, "selectable", "New Take##new-take"))
    else:
        assert session.submit(cmd.SeekStateTake(start_frame))
    viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    first = session.state_take_recording_start_frame
    assert first == (start_frame or 0)
    for _ in range(5):
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    viewer.sync()
    assert session.state_take_cursor == first
    assert panel._playhead == session.state_take_times[first]
    assert panel._editor.take_selection == (first, len(session.state_take_times) - 1)
    count = len(session.state_take_times) - first
    assert panel.status_detail(str) == f"Selected frames: {count}"
    assert details[-1] == panel.status_detail(viewer.app.localizer.text)
    _right_click(viewer, timeline_point(viewer, panel._playhead, "take"))
    _click(viewer, _item_center(viewer, "menu_item", viewer.app.localizer.text("Delete selection")))
    assert len(session.state_take_times) == first
    assert panel.status_detail(str) == ""


def test_playhead_keeps_moving_through_deleted_gaps_until_the_final_frame(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.DeleteStateTakeFrames(100, 400))
    assert session.submit(cmd.SeekStateTake(99))
    before = session.state_take_times[99]
    assert session.submit(cmd.PlayStateTake())
    session.tick(FrameNeeds.none(), wall_dt=1.0)
    viewer.sync()
    assert session.state_take_playing and session.state_take_cursor == 99
    assert before + 1 <= panel._playhead < session.state_take_times[100]
    assert session.submit(cmd.PauseStateTake())
    _click(viewer, _item_center(viewer, "invisible_button", "##take-first"))
    viewer.sync()
    assert panel._playhead == session.state_take_times[0]
    assert session.submit(cmd.PlayStateTake())
    session.tick(FrameNeeds.none(), wall_dt=30)
    viewer.sync()
    assert not session.state_take_playing
    assert panel._playhead == session.state_take_times[-1]


def test_loop_survives_panel_close_and_space_resumes_replay(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.SetStateTakeLoop(100, 103))
    assert session.submit(cmd.PlayStateTake())
    panel.open = False
    for _ in range(20):
        viewer.sync()
        assert 100 <= session.state_take_cursor <= 103
        assert session.state_take_playing
    viewer.app._toggle_playback()
    assert not session.state_take_playing
    viewer.app._toggle_playback()
    assert session.state_take_playing and session.paused


def test_viewport_play_starts_simulation_but_timeline_play_replays_take(viewer):
    session = viewer.session
    original = tuple(session.state_take_times)
    _click(viewer, _item_center(viewer, "invisible_button", "##take-play-pause"))
    assert session.state_take_playing and session.playback_source == "take"
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-toggle"))
    assert not session.state_take_playing and session.paused
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-toggle"))
    assert not session.paused and session.playback_source == "simulation"
    session.tick(FrameNeeds.none(), wall_dt=31)
    viewer.sync()
    assert viewer.panels.get("Keyframes")._playhead > original[-1]
    assert not session.paused and not session.state_take_playing
    assert tuple(session.state_take_times) == original


@pytest.mark.parametrize("stop_control", ("record", "toggle"))
def test_viewport_take_recording_rewinds_and_selects_the_same_range_as_keyframes(
    viewer, stop_control
):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.SeekStateTake(60))
    viewer.sync()
    take_id = session.active_state_take_id
    viewer.app._viewport_recording_mode = "take"
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-record"))
    assert session.state_take_recording
    assert session.state_take_recording_start_frame == 60
    session.tick(FrameNeeds.none(), wall_dt=0.3)
    _click(viewer, _item_center(viewer, "invisible_button", "##viewport-playback-" + stop_control))
    assert not session.state_take_recording and session.paused
    assert session.active_state_take_id == take_id
    assert session.state_take_cursor == 60
    assert panel._playhead == session.state_take_times[60]
    assert panel._editor.take_selection == (60, len(session.state_take_times) - 1)


def test_end_checkbox_persists_and_continues_the_playhead_without_physics(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    _click(viewer, _item_center(viewer, "invisible_button", "##timeline-options"))
    for _ in range(3):
        viewer.sync()
    _click(viewer, _item_center(viewer, "checkbox", "Pause at last frame"))
    assert not session.state_take_pause_at_end
    assert viewer.app.localizer.preference("take_pause_at_end") is False
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-play-pause"))
    session.tick(FrameNeeds.none(), wall_dt=31)
    viewer.sync()
    assert session.state_take_playing and session.paused
    assert panel._playhead > session.state_take_times[-1]
    assert session.frame.time == session.state_take_times[-1]


def test_playhead_pages_at_edge_and_stays_stationary_when_locked(viewer):
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    assert session.submit(cmd.SeekStateTake(360))
    viewer.sync()
    panel._view_start, panel._view_end = 1, 11
    choose_follow(viewer, "page")
    assert panel._view_start < panel._playhead < panel._view_start + 1
    assert panel._view_end - panel._view_start == pytest.approx(10)
    choose_follow(viewer, "locked")
    fraction = (panel._playhead - panel._view_start) / 10
    for index in (400, 450, 200):
        session.submit(cmd.SeekStateTake(index))
        viewer.sync()
        assert (panel._playhead - panel._view_start) / 10 == pytest.approx(fraction)
    point = timeline_point(viewer, panel._playhead)
    drag(viewer, point, (point[0] + 50, point[1]), button=1)
    assert panel._follow_mode == "off"


def test_escape_clears_range_without_clearing_scene_selection(viewer):
    session = viewer.session
    node = next(n for n in session.nodes if n.posable)
    session.submit(cmd.SelectNode(node.node_id))
    session.submit(cmd.SetStateTakeLoop(10, 20))
    viewer.sync()
    imgui.internal.focus_window(imgui.internal.find_window_by_name("Keyframes"))
    _click(viewer, timeline_point(viewer, 5))
    hints = viewer.app._status_tool_hints(loading=False)
    assert [hint.hint_id for hint in hints if hint.control == "Esc"] == ["keyframes.clear_range"]
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    assert session.state_take_loop is None
    assert session.selected_node is not None
    assert session.selected_node.node_id == node.node_id


def test_transport_status_geometry_stays_stable_across_time_and_frame_digits(viewer, monkeypatch):
    from mojive.ui.panels import keyframes

    original = keyframes._toolbar_status
    rows = []

    def capture(*args, **kwargs):
        original(*args, **kwargs)
        if not args[0].endswith(" s"):
            return
        lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        rows.append((lo.x, lo.y, hi.x, hi.y))

    monkeypatch.setattr(keyframes, "_toolbar_status", capture)
    baseline = None
    for index in (9, 10, 99, 100, 299, 300, 899):
        assert viewer.session.submit(cmd.SeekStateTake(index))
        rows.clear()
        viewer.sync()
        assert len(rows) == 1
        if baseline is None:
            baseline = rows.copy()
        assert rows == baseline
