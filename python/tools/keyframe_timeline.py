"""Exercise the Keyframes ruler, loop selection, and playhead scrolling in a real window."""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import pairwise
from pathlib import Path

from imgui_bundle import imgui

from mojive.capture import CaptureSurface
from mojive.capture.recording import VideoRecorder
from mojive.scene.assets import resolve

from .. import ViewerConfig, build
from .. import commands as cmd
from ..adapters.base import FrameNeeds
from ..ui.panels.keyframes import timeline_channel_width, timeline_time_to_x
from .ui_runtime import (
    _activate_panel,
    _click,
    _item_center,
    _item_rect,
    _right_click,
    _save_active_popup_crop,
    _save_window_crop,
)


def populate_take(viewer, count: int = 3945) -> None:
    """Record a long 30-sample-per-second take through the normal Session interface."""
    session = viewer.session
    assert session.submit(cmd.StartStateTakeRecording())
    for _ in range(count):
        session.tick(FrameNeeds.none(), wall_dt=1 / 30)
    assert session.submit(cmd.StopStateTakeRecording())
    assert session.submit(cmd.SeekStateTake(0))


def show_timeline(viewer) -> None:
    """Give the actual Keyframes panel enough space for transport and both tracks."""
    viewer.panels.open_panel("Keyframes")
    for _ in range(4):
        viewer.sync()
    _activate_panel(viewer, "Keyframes")
    window = imgui.internal.find_window_by_name("Keyframes")
    if window.dock_node is not None:
        imgui.internal.dock_context_process_undock_window(imgui.get_current_context(), window, True)
    begin = viewer.panels._begin_panel_window

    def place(panel, *args, **kwargs):
        if panel.name == "Keyframes":
            size = imgui.get_main_viewport().work_size
            imgui.set_next_window_pos((20, 160))
            imgui.set_next_window_size((size.x - 40, min(650, size.y - 185)))
        return begin(panel, *args, **kwargs)

    viewer.panels._begin_panel_window = place
    imgui.internal.focus_window(window)
    for _ in range(4):
        viewer.sync()


def timeline_point(viewer, time: float, row: str = "ruler") -> tuple[float, float]:
    """Map time to a point in the actual ruler or recorded-take track."""
    lo, hi = _item_rect(viewer, "invisible_button", "##keyframe-dope-sheet")
    scale = viewer.window.style_scale
    left = lo[0] + timeline_channel_width(hi[0] - lo[0], scale)
    panel = viewer.panels.get("Keyframes")
    x = timeline_time_to_x(
        time,
        panel._view_start,
        panel._view_end,
        left,
        hi[0],
    )
    row_height = (hi[1] - lo[1] - 32 * scale) / 3
    y = (
        lo[1] + 12 * scale
        if row == "ruler"
        else lo[1] + 32 * scale + row_height * ({"model": 0.5, "take": 1.5, "snapshots": 2.5}[row])
    )
    # Native ImGui floors mouse coordinates; inject the nearest pixel to the target.
    return round(x), round(y)


def drag(viewer, start, end, *, button: int = 0, shift: bool = False, cancel: bool = False) -> None:
    """Drive a complete native timeline drag, including modifier and release events."""
    io = imgui.get_io()
    io.add_mouse_pos_event(*start)
    io.add_key_event(imgui.Key.mod_shift, shift)
    viewer.sync()
    io.add_mouse_button_event(button, True)
    viewer.sync()
    for part in (0.25, 0.5, 0.75, 1):
        io.add_mouse_pos_event(
            start[0] + (end[0] - start[0]) * part, start[1] + (end[1] - start[1]) * part
        )
        viewer.sync()
    if cancel:
        io.add_key_event(imgui.Key.escape, True)
        viewer.sync()
        io.add_key_event(imgui.Key.escape, False)
    io.add_mouse_button_event(button, False)
    io.add_key_event(imgui.Key.mod_shift, False)
    viewer.sync()
    viewer.sync()


def choose_follow(viewer, mode: str) -> None:
    """Choose the follow policy directly from the production toolbar."""
    index = ("off", "page", "locked").index(mode)
    _click(viewer, _item_center(viewer, "button", f"##timeline-follow-{index}"))
    assert viewer.panels.get("Keyframes")._follow_mode == mode


def capture_take_editing(viewer, output: Path) -> None:
    """Exercise the production track selection, Take menu, and overwrite button."""
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    first_id = session.active_state_take_id
    model_id = session.scene_models[0].model_id
    for index in (90, 180, 270):
        assert session.submit(cmd.SeekStateTake(index))
        assert session.submit(cmd.AddModelKeyframe(model_id, f"pose_{index}"))
    viewer.sync()
    panel._view_start, panel._view_end = 0, 32
    panel._view_needs_fit = False
    choose_follow(viewer, "off")
    drag(viewer, timeline_point(viewer, 10, "take"), timeline_point(viewer, 15, "take"))
    bounds = panel._take_selection
    assert bounds is not None
    _save_window_crop(
        viewer,
        "Keyframes",
        output / "take-selection.png",
        padding=0,
        max_height=300 * viewer.window.style_scale,
    )
    _save_window_crop(
        viewer, "Status###application_status", output / "selection-status.png", padding=0
    )
    _right_click(viewer, timeline_point(viewer, 12, "take"))
    for _ in range(3):
        viewer.sync()
    _save_active_popup_crop(viewer, output / "selection-menu.png", padding=3)
    _click(viewer, _item_center(viewer, "menu_item", viewer.app.localizer.text("Delete selection")))
    assert not any(10.1 < time < 14.9 for time in session.state_take_times)
    imgui.get_io().add_mouse_pos_event(*timeline_point(viewer, 5, "take"))
    for _ in range(3):
        viewer.sync()
    assert any(hint.hint_id == "keyframes.select_all" for hint in viewer.app._panel_status_hints)
    _save_window_crop(viewer, "Status###application_status", output / "track-status.png", padding=0)
    imgui.get_io().add_mouse_pos_event(*timeline_point(viewer, 5, "ruler"))
    for _ in range(3):
        viewer.sync()
    assert {"keyframes.playhead", "keyframes.select_all"} <= {
        hint.hint_id for hint in viewer.app._panel_status_hints
    }
    _save_window_crop(viewer, "Status###application_status", output / "ruler-status.png", padding=0)
    _save_window_crop(
        viewer,
        "Keyframes",
        output / "take-deleted-range.png",
        padding=0,
        max_height=300 * viewer.window.style_scale,
    )
    last_before_gap = next(i for i, value in enumerate(session.state_take_times) if value > 15) - 1
    assert session.submit(cmd.SeekStateTake(last_before_gap))
    assert session.submit(cmd.PlayStateTake())
    session.tick(FrameNeeds.none(), wall_dt=2.0)
    for _ in range(3):
        viewer.sync()
    assert session.state_take_playing and 10 < panel._playhead < 15
    _save_window_crop(viewer, "Keyframes", output / "gap-playback.png", padding=0)
    assert session.submit(cmd.PauseStateTake())
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(
        viewer,
        _item_center(viewer, "selectable", viewer.app.localizer.text("New Take") + "##new-take"),
    )
    second_id = session.active_state_take_id
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    for _ in range(8):
        session.tick(FrameNeeds.none(), wall_dt=0.1)
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert second_id != first_id and len(session.state_takes) == 2
    for _ in range(3):
        viewer.sync()
    assert panel._take_selection == (0, len(session.state_take_times) - 1)
    assert panel._playhead == session.state_take_times[0]
    _save_window_crop(viewer, "Keyframes", output / "recorded-range.png", padding=0)
    _save_window_crop(
        viewer, "Status###application_status", output / "recorded-status.png", padding=0
    )
    drag(
        viewer,
        timeline_point(viewer, session.state_take_times[0]),
        timeline_point(viewer, session.state_take_times[-1]),
        shift=True,
    )
    assert session.state_take_loop is not None
    _save_window_crop(viewer, "Status###application_status", output / "loop-status.png", padding=0)
    imgui.get_io().add_key_event(imgui.Key.mod_shift, True)
    _right_click(viewer, timeline_point(viewer, panel._playhead))
    imgui.get_io().add_key_event(imgui.Key.mod_shift, False)
    viewer.sync()
    assert session.state_take_loop is None
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    viewer.sync()
    _save_active_popup_crop(viewer, output / "take-menu.png", padding=3)
    _click(
        viewer,
        _item_center(
            viewer, "selectable", f"{viewer.app.localizer.text('Take')} {first_id}##take-{first_id}"
        ),
    )
    panel._view_start, panel._view_end = 0, 32
    _click(viewer, timeline_point(viewer, 20))
    old_count = len(session.state_take_times)
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    for _ in range(8):
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert len(session.state_take_times) < old_count and len(session.state_takes) == 2
    panel._view_start, panel._view_end = 0, 32
    viewer.sync()
    _save_window_crop(
        viewer,
        "Keyframes",
        output / "take-overwrite.png",
        padding=0,
        max_height=300 * viewer.window.style_scale,
    )
    drag(viewer, timeline_point(viewer, 2, "model"), timeline_point(viewer, 8, "model"))
    assert len(panel._selected_keyframes) == 2
    _save_window_crop(
        viewer,
        "Keyframes",
        output / "model-selection.png",
        padding=0,
        max_height=300 * viewer.window.style_scale,
    )
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    for take in session.state_takes:
        _click(viewer, _item_center(viewer, "invisible_button", f"##remove-take-{take.take_id}"))
    # The empty-take toolbar and popup resize after their final row disappears.
    for _ in range(3):
        viewer.sync()
    _click(
        viewer,
        _item_center(viewer, "selectable", viewer.app.localizer.text("New Take") + "##new-take"),
    )
    assert session.state_takes[0].name == "Take 1"
    assert session.active_state_take_id not in (first_id, second_id)
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    for _ in range(3):
        viewer.sync()
    _save_active_popup_crop(viewer, output / "take-restarted-menu.png", padding=3)


def capture_transport(viewer, output: Path) -> None:
    """Verify explicit transport sources and capture the two compact preferences."""
    session, app = viewer.session, viewer.app
    app.set_take_pause_at_end(True, persist=False)
    panel = viewer.panels.get("Keyframes")
    panel._set_follow_mode("off")
    panel._view_start, panel._view_end, panel._view_needs_fit = 0.0, 2.0, False
    viewer.sync()
    _save_window_crop(viewer, "Keyframes", output / "ruler.png", padding=0)
    _click(viewer, _item_center(viewer, "invisible_button", "##timeline-options"))
    for _ in range(3):
        viewer.sync()
    _click(viewer, _item_center(viewer, "checkbox", app.localizer.text("Pause at last frame")))
    assert not session.state_take_pause_at_end
    _save_active_popup_crop(viewer, output / "timeline-options.png", padding=0)
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    assert session.submit(cmd.PlayStateTake())
    session.tick(FrameNeeds.none(), wall_dt=1.4)
    viewer.sync()
    assert session.state_take_playing and session.paused
    assert panel._playhead > session.state_take_times[-1]
    assert session.frame.time == session.state_take_times[-1]
    _save_window_crop(viewer, "Keyframes", output / "continuous-playhead.png", padding=0)
    _save_window_crop(
        viewer, "Status###application_status", output / "replay-status.png", padding=0
    )
    app._toggle_playback()
    app._toggle_playback(source="simulation")
    session.tick(FrameNeeds.none(), wall_dt=0.2)
    viewer.sync()
    assert not session.paused and not session.state_take_playing
    assert session.frame.time > session.state_take_times[-1]
    _save_window_crop(
        viewer, "Status###application_status", output / "simulation-status.png", padding=0
    )
    assert session.submit(cmd.Pause())

    panel.open = False
    viewer.panels.open_panel("Settings")
    viewer.panels.get("Settings").show_category("Recording")
    for _ in range(4):
        viewer.sync()
    _activate_panel(viewer, "Settings")
    window = imgui.internal.find_window_by_name("Settings")
    if window.dock_node is not None:
        imgui.internal.dock_context_process_undock_window(imgui.get_current_context(), window, True)
    begin = viewer.panels._begin_panel_window

    def place(settings, *args, **kwargs):
        if settings.name == "Settings":
            size = imgui.get_main_viewport().work_size
            imgui.set_next_window_pos((20, 160))
            imgui.set_next_window_size(
                (min(1100 * viewer.window.style_scale, size.x - 40), size.y - 185)
            )
        return begin(settings, *args, **kwargs)

    viewer.panels._begin_panel_window = place
    for _ in range(3):
        viewer.sync()
    assert not app.recording_config.run_simulation
    _click(viewer, _item_center(viewer, "checkbox", "##recording_run_simulation"))
    assert app.recording_config.run_simulation
    _save_window_crop(viewer, "Settings", output / "recording-settings.png", padding=0)


def capture_replay(viewer, output: Path) -> list[dict[str, float]]:
    """Capture two seconds of real replay at a reproducible 60 FPS display clock."""
    session, panel = viewer.session, viewer.panels.get("Keyframes")
    advance = session._advance_state_take
    session._advance_state_take = lambda _dt: advance(1 / 60)
    trace = []
    recorder = None
    try:
        for _ in range(120):
            image = viewer.capture_array(surface=CaptureSurface.WINDOW)
            if recorder is None:
                recorder = VideoRecorder(output, (image.shape[1], image.shape[0]), fps=60)
            recorder.append(image)
            trace.append(
                {"playhead": panel._playhead, "start": panel._view_start, "end": panel._view_end}
            )
    finally:
        session._advance_state_take = advance
        if recorder is not None:
            recorder.close()
    output.with_suffix(".json").write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def capture_instability_recovery(viewer, output: Path) -> None:
    """Reproduce native QACC failure and exercise the recovered editor controls."""
    from PIL import Image

    session, panel = viewer.session, viewer.panels.get("Keyframes")
    first_id = session.active_state_take_id
    model_id = session.scene_models[0].model_id
    choose_follow(viewer, "off")
    assert session.submit(cmd.SeekStateTake(6))
    assert session.submit(cmd.AddModelKeyframe(model_id, "before_failure"))
    keys = tuple(session.keyframes)
    assert session.submit(cmd.SeekStateTake(len(session.state_take_times) - 1))
    viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert session.state_take_recording
    before = tuple(session.state_take_times)
    primary = getattr(session.adapter, "primary", session.adapter)
    primary.data.qfrc_applied[0] = float("nan")
    session.tick(FrameNeeds(), wall_dt=0.02)
    assert session.paused and not session.state_take_recording
    assert "QACC" in session.last_message
    assert tuple(session.state_take_times) == before
    for _ in range(3):
        viewer.sync()
    assert panel._playhead == 0
    assert all(math.isfinite(value) for value in (panel._view_start, panel._view_end))
    _save_window_crop(viewer, "Keyframes", output / "recovered-timeline.png", padding=0)
    Image.fromarray(viewer.capture_array(surface=CaptureSurface.WINDOW)).save(
        output / "recovered-window.png"
    )
    _click(viewer, _item_center(viewer, "invisible_button", "##take-first"))
    _click(viewer, timeline_point(viewer, before[6]))
    assert 0 <= session.state_take_cursor < len(before)
    _click(viewer, _item_center(viewer, "invisible_button", "##take-play-pause"))
    assert session.state_take_playing
    _click(viewer, _item_center(viewer, "invisible_button", "##take-play-pause"))
    assert not session.state_take_playing
    _click(viewer, timeline_point(viewer, before[6]))
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    for _ in range(4):
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert not session.state_take_recording
    assert all(a < b for a, b in pairwise(session.state_take_times))
    assert session.active_state_take_id == first_id
    _click(viewer, _item_center(viewer, "invisible_button", "##key-view"))
    viewer.sync()
    _save_window_crop(viewer, "Keyframes", output / "recovered-overwrite.png", padding=0)
    drag(
        viewer, timeline_point(viewer, before[2], "take"), timeline_point(viewer, before[5], "take")
    )
    assert panel._take_selection is not None
    for key, ctrl in ((imgui.Key.a, True), (imgui.Key.delete, False)):
        io = imgui.get_io()
        io.add_key_event(imgui.Key.mod_ctrl, ctrl)
        io.add_key_event(key, True)
        viewer.sync()
        io.add_key_event(key, False)
        io.add_key_event(imgui.Key.mod_ctrl, False)
        viewer.sync()
    assert not session.state_take_times
    assert tuple(session.keyframes) == keys
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(
        viewer,
        _item_center(viewer, "selectable", viewer.app.localizer.text("New Take") + "##new-take"),
    )
    assert session.active_state_take_id != first_id
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    for _ in range(4):
        viewer.sync()
    _click(viewer, _item_center(viewer, "invisible_button", "##take-record"))
    assert session.state_take_times
    _click(viewer, _item_center(viewer, "begin_combo", "##timeline-take"))
    _click(viewer, _item_center(viewer, "invisible_button", f"##remove-take-{first_id}"))
    assert len(session.state_takes) == 1
    for _ in range(3):
        viewer.sync()
    _save_active_popup_crop(viewer, output / "recovered-new-take.png", padding=3)


def main(argv: list[str] | None = None) -> int:
    """Capture ruler seeking, orange loops, and both playhead-follow policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/keyframe-timeline"))
    parser.add_argument(
        "--editing", action="store_true", help="Capture Take editing and model-track selection"
    )
    parser.add_argument(
        "--recovery", action="store_true", help="Capture timeline recovery after QACC instability"
    )
    parser.add_argument(
        "--transport",
        action="store_true",
        help="Capture independent simulation, replay, and recording controls",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--language", choices=("en", "zh_CN"), default="en")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    os.environ["MOJIVE_UI_SCALE"] = str(args.scale)
    from mojive.app.composition import build_workspace

    builder = build_workspace if args.recovery else build
    with builder(
        resolve("joint_gizmo" if args.recovery else "joint_types"),
        config=ViewerConfig(threaded_physics=False),
        paused=True,
        vsync=False,
        width=min(3200, round(1800 * args.scale)),
        height=min(1800, round(1100 * args.scale)),
        show_window=False,
    ) as viewer:
        viewer.app.set_language(args.language)
        populate_take(
            viewer, 30 if args.recovery or args.transport else 900 if args.editing else 3945
        )
        if args.transport:
            show_timeline(viewer)
            capture_transport(viewer, args.output)
            return 0
        if args.recovery:
            show_timeline(viewer)
            capture_instability_recovery(viewer, args.output)
            return 0
        if args.editing:
            show_timeline(viewer)
            capture_take_editing(viewer, args.output)
            return 0
        session = viewer.session
        model_id = session.scene_models[0].model_id
        for number, index in enumerate(
            (0, 780, 1005, 1300, 1340, 1370, 2040, 2420, 2700, 3140, 3440, 3600, 3940)
        ):
            assert session.submit(cmd.SeekStateTake(index))
            assert session.submit(cmd.AddModelKeyframe(model_id, f"snapshot_{number}"))
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        choose_follow(viewer, "off")
        panel._view_start, panel._view_end = -5, 140
        _click(viewer, timeline_point(viewer, 55))
        expected = panel._playhead
        for _ in range(3):
            viewer.sync()
        assert panel._playhead == expected
        _save_window_crop(viewer, "Keyframes", args.output / "ruler-seek.png", padding=0)
        _save_window_crop(
            viewer, "Status###application_status", args.output / "status-hints.png", padding=0
        )
        drag(viewer, timeline_point(viewer, 45), timeline_point(viewer, 75), shift=True)
        assert session.state_take_loop is not None
        _save_window_crop(viewer, "Keyframes", args.output / "loop-range.png", padding=0)
        bounds = session.state_take_loop
        assert session.submit(cmd.SeekStateTake(bounds[1] - 1))
        assert session.submit(cmd.PlayStateTake())
        session.tick(FrameNeeds(), wall_dt=0.15)
        assert bounds[0] <= session.state_take_cursor < bounds[1]
        assert session.state_take_playing
        assert session.submit(cmd.SeekStateTake(bounds[1] - 10))
        assert session.submit(cmd.PlayStateTake())
        loop_trace = capture_replay(viewer, args.output / "loop-range.mp4")
        assert any(after["playhead"] < before["playhead"] for before, after in pairwise(loop_trace))
        assert session.submit(cmd.PauseStateTake())
        _click(
            viewer,
            _item_center(viewer, "invisible_button", "##timeline-loop"),
        )
        assert session.state_take_loop is None
        assert session.submit(cmd.SeekStateTake(1720))
        viewer.sync()
        panel._view_start, panel._view_end = 45.5, 57.25
        _save_window_crop(viewer, "Keyframes", args.output / "before-follow.png", padding=0)
        choose_follow(viewer, "page")
        _save_window_crop(viewer, "Keyframes", args.output / "page-follow.png", padding=0)
        choose_follow(viewer, "locked")
        assert session.submit(cmd.PlayStateTake())
        trace = capture_replay(viewer, args.output / "locked-playhead.mp4")
        fractions = [(p["playhead"] - p["start"]) / (p["end"] - p["start"]) for p in trace]
        assert max(fractions) - min(fractions) < 1e-9
        assert session.submit(cmd.PauseStateTake())
        _save_window_crop(viewer, "Keyframes", args.output / "locked-playhead.png", padding=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
