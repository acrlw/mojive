"""Capture native take-video settings, countdown, automatic save, and encoded endpoint poses."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imageio_ffmpeg import read_frames
from imgui_bundle import imgui
from PIL import Image

from mojive.capture.recording import VideoRecorder
from mojive.scene.assets import resolve

from .. import CaptureSurface, RecordingConfig, ViewerConfig, build
from .. import commands as cmd
from .keyframe_timeline import populate_take
from .ui_runtime import _activate_panel, _click, _item_center, _right_click, _save_window_crop


def capture(viewer, path):
    Image.fromarray(viewer.capture_array(surface=CaptureSurface.WINDOW)).save(path)


def run(viewer, output):
    session, app = viewer.session, viewer.app
    populate_take(viewer, 60)
    viewer.panels.open_panel("Keyframes")
    for _ in range(4):
        viewer.sync()
    _activate_panel(viewer, "Keyframes")
    session.submit(cmd.SetStateTakeLoop(12, 24))
    session.submit(cmd.SeekStateTake(40))
    t = app.localizer.text
    _click(viewer, _item_center(viewer, "button", f"{t('Video Settings')}##take-video-settings"))
    assert imgui.get_current_context().open_popup_stack
    viewer.sync()
    capture(viewer, output / "settings.png")
    io = imgui.get_io()
    io.add_key_event(imgui.Key.escape, True)
    viewer.sync()
    io.add_key_event(imgui.Key.escape, False)
    viewer.sync()
    assert not imgui.get_current_context().open_popup_stack
    assert session.state_take_loop == (12, 24)
    capture(viewer, output / "ready.png")

    video = output / "take.mp4"
    video.unlink(missing_ok=True)
    app._capture_output = lambda *_args: video
    frames = []
    endpoints = {}
    append = VideoRecorder.append

    def record(recorder, image):
        frames.append(session.state_take_cursor)
        if "first" not in endpoints:
            endpoints["first"] = image.copy()
        endpoints["last"] = image.copy()
        append(recorder, image)

    with patch.object(VideoRecorder, "append", record):
        _click(viewer, _item_center(viewer, "button", f"{t('Record Take Video')}##take-video"))
        assert session.state_take_cursor == 0 and not session.state_take_playing, (
            session.state_take_cursor,
            session.state_take_loop,
            viewer.recording,
            app._popup_owned_frame,
        )
        capture(viewer, output / "countdown.png")
        assert not video.exists()
        deadline = time.monotonic() + 30
        paused_once = False
        while viewer.recording.active:
            assert time.monotonic() < deadline
            viewer.sync()
            if len(frames) >= 24 and not paused_once:
                paused_once = True
                viewer.pause_recording()
                capture(viewer, output / "paused.png")
                viewer.resume_recording()
            time.sleep(0.002)
    assert session.state_take_loop == (12, 24)
    assert session.state_take_cursor == len(session.state_take_times) - 1
    assert not session.state_take_playing
    expected_count = math.ceil(session.state_take_times[-1] * 60 - 1e-9) + 1 + 30
    assert len(frames) == expected_count, (len(frames), expected_count)
    assert frames[0] == 0 and frames[-1] == len(session.state_take_times) - 1
    viewer.sync()
    viewer.sync()
    capture(viewer, output / "saved.png")
    _save_window_crop(viewer, "Status###application_status", output / "saved-status.png", padding=0)
    point = _item_center(viewer, "invisible_button", "##status_message")
    _right_click(viewer, point)
    assert imgui.get_clipboard_text() == str(video.resolve())
    capture(viewer, output / "copy-path.png")
    reader = read_frames(str(video))
    try:
        metadata = next(reader)
        decoded = list(reader)
    finally:
        reader.close()
    assert metadata["fps"] == 60 and len(decoded) == len(frames)
    width, height = metadata["size"]
    errors = {}
    for key, raw in (("first", decoded[0]), ("last", decoded[-1])):
        image = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        Image.fromarray(image).save(output / f"video-{key}.png")
        reference = endpoints[key]
        h, w = reference.shape[:2]
        errors[key] = float(np.mean(np.abs(image[:h, :w].astype(float) - reference)))
        assert errors[key] < 5
    return {
        "backend": viewer.backend.caps.name,
        "frames": len(frames),
        "fps": metadata["fps"],
        "size": metadata["size"],
        "endpoint_mean_error": errors,
        "recorded_take_indices": frames,
        "copied_path": imgui.get_clipboard_text(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/take-video"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "preferences.json")
    scale = max(1.0, float(os.environ.get("MOJIVE_UI_SCALE", "1")))
    with build(
        resolve("joint_types"),
        paused=True,
        vsync=False,
        show_window=False,
        width=round(1600 * scale),
        height=round(1000 * scale),
        config=ViewerConfig(recording=RecordingConfig(countdown=2, end_hold=0.5)),
    ) as viewer:
        report = run(viewer, args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
