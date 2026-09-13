"""Exercise the Keyframes ruler, loop selection, and playhead scrolling in a real window."""

from __future__ import annotations

import argparse
import json
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
from .ui_runtime import _activate_panel, _click, _item_center, _item_rect, _save_window_crop


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
    y = lo[1] + (12 if row == "ruler" else 122) * scale
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


def main(argv: list[str] | None = None) -> int:
    """Capture ruler seeking, orange loops, and both playhead-follow policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/keyframe-timeline"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    with build(
        resolve("joint_types"),
        config=ViewerConfig(threaded_physics=False),
        paused=True,
        vsync=False,
        width=1800,
        height=1100,
        show_window=False,
    ) as viewer:
        populate_take(viewer)
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
        drag(viewer, timeline_point(viewer, 45), timeline_point(viewer, 75), button=1, shift=True)
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
