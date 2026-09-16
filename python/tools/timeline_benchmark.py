"""Measure production timeline browsing and optional MuJoCo keyframe editing."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from mojive.adapters.base import KeyframeInfo, SceneModelInfo
from mojive.adapters.static import StaticSceneAdapter
from mojive.app.composition import build_from_adapter
from mojive.scene import Scene

from .keyframe_timeline import show_timeline, timeline_point
from .ui_runtime import _save_window_crop


class _TimelineAdapter(StaticSceneAdapter):
    """Provide marker metadata without model compilation or physics timing noise."""

    def __init__(self, count: int):
        scene = Scene()
        scene.box()
        super().__init__(scene)
        self.caps = replace(self.caps, keyframes=True)
        self._keys = [KeyframeInfo(i, f"pose-{i}", i * 0.1, 0) for i in range(count)]

    def scene_models(self):
        return (SceneModelInfo(0, "Timeline benchmark", Path("synthetic"), False),)

    def keyframes(self):
        return self._keys

    def load_keyframe(self, keyframe_id):
        return 0 <= keyframe_id < len(self._keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/timeline-benchmark"))
    parser.add_argument("--markers", type=int, default=20_000)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument(
        "--editable",
        action="store_true",
        help="Use real MuJoCo presets and measure held drags, selection and edit release",
    )
    args = parser.parse_args(argv)
    if args.markers < 40 or args.frames < 1:
        parser.error("markers must be at least 40 and frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("MOJIVE_SETTINGS", "MOJIVE_UI_SCALE")}
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    os.environ["MOJIVE_UI_SCALE"] = "1"
    try:
        report = _run(args.markers, args.frames, args.output, editable=args.editable)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def _run(markers: int, frames: int, output: Path, *, editable: bool = False) -> dict:
    report = {"platform": platform.platform(), "markers": markers, "cases": {}}
    if editable:
        from mojive.adapters.mujoco import MuJoCoAdapter

        path = output / "timeline.xml"
        keys = "".join(
            f'<key name="pose-{i}" time="{i * 0.1:g}" qpos="0"/>' for i in range(markers)
        )
        path.write_text(
            '<mujoco><worldbody><body><joint/><geom size="0.1"/></body></worldbody><keyframe>'
            + keys
            + "</keyframe></mujoco>"
        )
        adapter = MuJoCoAdapter(path)
    else:
        adapter = _TimelineAdapter(markers)
    report["adapter"] = adapter.caps.name
    with build_from_adapter(
        adapter, paused=True, vsync=False, show_window=False, width=1600, height=1000
    ) as viewer:
        show_timeline(viewer)
        panel = viewer.panels.get("Keyframes")
        panel._set_follow_mode("off")
        draw, paint = panel.draw, panel._paint_dope_sheet
        samples, counts = [], []

        def measured(ctx):
            start = time.perf_counter_ns()
            draw(ctx)
            samples.append((time.perf_counter_ns() - start) / 1e6)

        def counted(*args, **kwargs):
            counts.append(len(args[9]))
            return paint(*args, **kwargs)

        panel.draw, panel._paint_dope_sheet = measured, counted
        middle = markers * 0.05
        cases = (
            ("overview", (0, markers * 0.1)),
            ("zoomed", (middle, middle + 2)),
            ("overview_pan", (0, markers * 0.1)),
            ("zoomed_pan", (middle, middle + 2)),
            ("overview_hover", (0, markers * 0.1)),
        )
        if editable:
            cases += (
                ("overview_drag", (0, markers * 0.1)),
                ("overview_selection", (0, markers * 0.1)),
            )
        for name, bounds in cases:
            imgui.get_io().add_mouse_pos_event(-100, -100)
            panel._view_start, panel._view_end = bounds
            panel._view_needs_fit = False
            for _ in range(8):
                viewer.sync()
            if name.endswith("_drag") or name.endswith("_selection"):
                point = timeline_point(viewer, (bounds[0] + bounds[1]) / 2, "model")
                if name.endswith("_selection"):
                    point = (point[0], point[1] + 12)
                imgui.get_io().add_mouse_pos_event(*point)
                viewer.sync()
                imgui.get_io().add_mouse_button_event(0, True)
                viewer.sync()
                if name.endswith("_drag") and panel._editor.drag_id < 0:
                    raise RuntimeError("Benchmark did not acquire a keyframe drag")
                if name.endswith("_selection") and panel._editor.pointer_mode != "select":
                    raise RuntimeError("Benchmark did not acquire a selection gesture")
            samples.clear()
            counts.clear()
            sync_ms = []
            for index in range(frames):
                span = bounds[1] - bounds[0]
                if name.endswith("_pan"):
                    shift = span * index * 0.0005
                    panel._view_start, panel._view_end = bounds[0] + shift, bounds[1] + shift
                if name.endswith("_hover"):
                    point = timeline_point(
                        viewer, bounds[0] + span * (index + 1) / (frames + 1), "model"
                    )
                    imgui.get_io().add_mouse_pos_event(*point)
                if name.endswith("_drag") or name.endswith("_selection"):
                    imgui.get_io().add_mouse_pos_event(
                        point[0] + 60 * (index + 1) / frames, point[1]
                    )
                start = time.perf_counter_ns()
                viewer.sync()
                sync_ms.append((time.perf_counter_ns() - start) / 1e6)
            report["cases"][name] = {
                "panel_median_ms": statistics.median(samples),
                "panel_p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
                "sync_median_ms": statistics.median(sync_ms),
                "sync_p95_ms": sorted(sync_ms)[int(0.95 * (len(sync_ms) - 1))],
                "pixel_scale": viewer.window.pixel_scale,
                "projected_markers_median": statistics.median(counts),
            }
            _save_window_crop(viewer, "Keyframes", output / f"{name}.png", padding=0)
            if name.endswith("_drag") or name.endswith("_selection"):
                start = time.perf_counter_ns()
                imgui.get_io().add_mouse_button_event(0, False)
                viewer.sync()
                report["cases"][name]["release_sync_ms"] = (time.perf_counter_ns() - start) / 1e6
                if panel._error:
                    raise RuntimeError(panel._error)
        report["backend"] = viewer.backend.caps.name
        report["window_points"] = list(viewer.window.size_points)
        report["pixel_scale"] = viewer.window.pixel_scale
    return report


if __name__ == "__main__":
    raise SystemExit(main())
