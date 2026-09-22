"""Measure interactive video readback and encoder backpressure during camera motion."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

from mojive.adapters.static import StaticSceneAdapter
from mojive.app.composition import build_from_adapter
from mojive.capture.recording import VideoRecorder
from mojive.capture.video_queue import BufferedVideoRecorder
from mojive.scene import Scene


def _summary(samples):
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered) if ordered else 0,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0,
        "max_ms": max(ordered, default=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/recording-benchmark"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument(
        "--preset", choices=("ultrafast", "fast", "medium", "slow"), default="medium"
    )
    args = parser.parse_args(argv)
    if args.frames < 30 or not 1 <= args.fps <= 120:
        parser.error("frames must be at least 30 and fps must be between 1 and 120")
    if min(args.width, args.height) < 320:
        parser.error("Window dimensions must be at least 320")
    args.output.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MOJIVE_SETTINGS")
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    try:
        report = _run(args.output, args.frames, args.fps, args.width, args.height, args.preset)
    finally:
        if previous is None:
            os.environ.pop("MOJIVE_SETTINGS", None)
        else:
            os.environ["MOJIVE_SETTINGS"] = previous
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _run(output: Path, frames: int, fps: float, width: int, height: int, preset: str) -> dict:
    scene = Scene()
    scene.plane(size=(8, 8, 0.02))
    for row in range(12):
        for col in range(12):
            scene.box(
                position=((col - 6) * 0.6, (row - 6) * 0.6, 0.3),
                size=(0.2, 0.2, 0.3),
                color=(row / 12, col / 12, (row + col) / 24, 1),
            )
    report = {"platform": platform.platform(), "target_ui_fps": fps, "cases": {}}
    append = VideoRecorder.append
    encode_ms, submit_ms = [], []
    accepted_samples = []
    submit = BufferedVideoRecorder.append

    def timed_submit(recorder, image, **kwargs):
        start = time.perf_counter_ns()
        try:
            accepted = submit(recorder, image, **kwargs)
            accepted_samples.append(accepted)
            return accepted
        finally:
            submit_ms.append((time.perf_counter_ns() - start) / 1e6)

    def timed_append(recorder, image):
        start = time.perf_counter_ns()
        try:
            return append(recorder, image)
        finally:
            encode_ms.append((time.perf_counter_ns() - start) / 1e6)

    with build_from_adapter(
        StaticSceneAdapter(scene),
        paused=True,
        vsync=False,
        show_window=False,
        width=width,
        height=height,
    ) as viewer:
        app = viewer.app
        app.recording_config = replace(
            app.recording_config, run_simulation=False, encoder_preset=preset
        )
        for _ in range(20):
            viewer.sync()
        VideoRecorder.append = timed_append
        BufferedVideoRecorder.append = timed_submit
        try:
            for recording in (False, True):
                name = "recording" if recording else "baseline"
                sync_ms, submitted_sync_ms, idle_sync_ms = [], [], []
                encode_ms.clear()
                submit_ms.clear()
                accepted_samples.clear()
                if recording:
                    app.start_recording(
                        output / "camera-motion.mp4", surface="window", fps=fps, countdown=0
                    )
                first_frame_ms = None
                for index in range(frames):
                    start = time.perf_counter()
                    app.camera.orbit(0.3, 0.1 if index < frames // 2 else -0.1)
                    previous_count = app._viewport_recording_frames
                    viewer.sync()
                    elapsed = (time.perf_counter() - start) * 1000
                    sync_ms.append(elapsed)
                    submitted = app._viewport_recording_frames > previous_count
                    (submitted_sync_ms if submitted else idle_sync_ms).append(elapsed)
                    if submitted and first_frame_ms is None:
                        first_frame_ms = elapsed
                    time.sleep(max(0, 1 / fps - (time.perf_counter() - start)))
                result = {
                    "sync": _summary(sync_ms),
                    "sync_with_submission": _summary(submitted_sync_ms),
                    "sync_without_submission": _summary(idle_sync_ms),
                    "append": _summary(encode_ms),
                    "buffer_submission": _summary(submit_ms),
                    "sampled_frames": sum(accepted_samples),
                    "skipped_samples": len(accepted_samples) - sum(accepted_samples),
                    "first_submitted_frame_ms": first_frame_ms,
                    "frames_over_16_67_ms": sum(value > 1000 / 60 for value in sync_ms),
                    "frames_over_33_33_ms": sum(value > 1000 / 30 for value in sync_ms),
                }
                if recording:
                    recorder = app._viewport_recorder
                    if recorder is None:
                        raise RuntimeError(app._recording_error or "recording produced no frames")
                    encoder = getattr(recorder, "recorder", recorder)
                    result["encoded_size"] = encoder.encoded_size
                    result["encoder"] = {
                        "codec": encoder.codec,
                        "preset": encoder.preset,
                        "crf": encoder.crf,
                    }
                    start = time.perf_counter()
                    app._request_recording_stop()
                    result["stop_request_ms"] = (time.perf_counter() - start) * 1000
                    finishing_sync_ms = []
                    while app.recording.active:
                        tick = time.perf_counter()
                        viewer.sync()
                        finishing_sync_ms.append((time.perf_counter() - tick) * 1000)
                        if time.perf_counter() - start > 35:
                            raise TimeoutError("Recording did not finalize")
                    if app.recording.error:
                        raise RuntimeError(app.recording.error)
                    result["finalization_ms"] = (time.perf_counter() - start) * 1000
                    result["finalization_sync"] = _summary(finishing_sync_ms)
                    result["append"] = _summary(encode_ms)
                    result["video_frames"] = recorder.written_frames
                report["cases"][name] = result
            report["backend"] = viewer.backend.caps.name
            report["window_points"] = viewer.window.size_points
            report["pixel_scale"] = viewer.window.pixel_scale
        finally:
            VideoRecorder.append = append
            BufferedVideoRecorder.append = submit
    return report


if __name__ == "__main__":
    raise SystemExit(main())
