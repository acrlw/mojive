"""Verify continuous viewport resizing and report application submission latency.

Native size requests and held dock-splitter gestures must update the image and
projection in the frame that resolves the new layout. Timings exclude readback
and do not measure desktop scanout or an operating system's modal resize loop.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.app.composition import build_workspace
from mojive.assets import resolve
from mojive.config import LayoutConfig, ViewerConfig
from mojive.ui import window as window_module
from mojive.ui.viewport_surface import fit_image_rect


def _observe(viewer, phase, elapsed_ms, requested_size):
    app, window = viewer.app, viewer.window
    image = app._viewport_image
    if image is None:
        raise AssertionError("A completed frame must publish its viewport image")
    panel = app.viewport_surface
    expected = app.fixed_render_size or tuple(
        max(1, int(value)) for value in window.points_to_pixels(panel.size)
    )
    actual = (image.width, image.height)
    if actual != expected or app._current_viewport_render_size() != expected:
        raise AssertionError(
            f"Viewport image is stale: {actual}, current layout requires {expected}"
        )
    aspect = app.camera.view().aspect
    if not np.isclose(aspect, expected[0] / expected[1]):
        raise AssertionError("Camera projection did not follow the current viewport")
    rect = tuple(app._viewport_rect)
    np.testing.assert_allclose(rect, fit_image_rect(panel.position, panel.size, expected))
    coverage = rect[2] * rect[3] / (panel.size[0] * panel.size[1])
    if app.fixed_render_size is None:
        # Integer render pixels can differ from fractional logical panel bounds.
        tolerance = 1.0 / window.pixel_scale + 1e-6
        if (
            max(abs(a - b) for a, b in zip(rect, (*panel.position, *panel.size), strict=True))
            > tolerance
        ):
            raise AssertionError("An adaptive viewport must cover its current panel")
    return {
        "phase": phase,
        "submission_ms": elapsed_ms,
        "requested_window_points": requested_size,
        "window_points": list(window.size_points),
        "panel_points": [*panel.position, *panel.size],
        "image_pixels": list(actual),
        "image_rect_points": list(rect),
        "camera_aspect": aspect,
        "coverage": coverage,
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    previous_settings = os.environ.get("MOJIVE_SETTINGS")
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    rows = []
    try:
        with build_workspace(
            resolve("joint_gizmo"),
            renderer=args.backend,
            width=args.width,
            height=args.height,
            vsync=args.vsync,
            show_window=args.visible,
            config=ViewerConfig(
                layout=LayoutConfig(persistence=False),
                debug_server=False,
                threaded_physics=False,
            ),
        ) as viewer:
            viewer.app.set_language("zh_CN")
            glfw = window_module.glfw
            events = []
            process_inputs = viewer.window._input.process_inputs

            def inputs():
                process_inputs()
                io = imgui.get_io()
                for method, arguments in events:
                    getattr(io, method)(*arguments)
                events.clear()

            viewer.window._input.process_inputs = inputs

            def frame(phase, size=None):
                started = time.perf_counter()
                if size is not None:
                    glfw.set_window_size(viewer.window._window, *size)
                viewer.sync()
                elapsed_ms = (time.perf_counter() - started) * 1000
                row = _observe(viewer, phase, elapsed_ms, size)
                rows.append(row)
                return row

            def event(phase, method, *arguments):
                events.append((method, arguments))
                return frame(phase)

            def capture(name):
                image = viewer.capture_array(surface="window")
                if not np.any(image):
                    raise AssertionError(
                        "A presented acceptance capture must contain the workspace"
                    )
                Image.fromarray(image).save(args.output / f"{name}.png")

            # Exercise the steady-state path beyond the former 30-frame warmup.
            for _ in range(40):
                viewer.sync()
            capture("initial")
            for fraction in np.concatenate(
                (np.linspace(0, 1, args.frames), np.linspace(1, 0, args.frames))
            ):
                size = (
                    args.width - round(220 * fraction),
                    args.height - round(120 * fraction),
                )
                frame("native-window", size)
            native_sizes = {tuple(row["image_pixels"]) for row in rows}
            if len(native_sizes) < args.frames // 2:
                raise AssertionError("Native size requests did not exercise a changing viewport")
            capture("native-window-restored")

            viewer.app.workspace.activate("Timeline")
            for _ in range(6):
                viewer.sync()
            timeline = imgui.internal.find_window_by_name("Timeline")
            if timeline is None or not timeline.dock_id:
                raise AssertionError(
                    "The production Timeline must be docked for splitter acceptance"
                )
            start = (
                timeline.pos.x + timeline.size.x * 0.5,
                timeline.pos.y - imgui.get_style().docking_separator_size * 0.5,
            )
            event("splitter-setup", "add_mouse_pos_event", *start)
            for _ in range(4):
                frame("splitter-setup")
            if imgui.get_mouse_cursor() != imgui.MouseCursor_.resize_ns:
                raise AssertionError("The pointer must hit the production dock splitter")
            event("splitter-setup", "add_mouse_button_event", 0, True)
            if not imgui.get_current_context().active_id:
                raise AssertionError("The splitter drag must retain active input ownership")
            initial_height = timeline.size.y
            distance = min(120 * viewer.window.style_scale, viewer.app.viewport_surface.size[1] / 3)
            for fraction in np.linspace(0, 1, args.frames):
                event(
                    "splitter-held",
                    "add_mouse_pos_event",
                    start[0],
                    start[1] - distance * fraction,
                )
            if timeline.size.y < initial_height + distance * 0.8:
                raise AssertionError("The held splitter did not resize the Timeline")
            capture("splitter-held")
            for fraction in np.linspace(1, 0, args.frames):
                event(
                    "splitter-held",
                    "add_mouse_pos_event",
                    start[0],
                    start[1] - distance * fraction,
                )
            event("splitter-release", "add_mouse_button_event", 0, False)
            capture("splitter-released")

            panel_width, panel_height = viewer.app.viewport_surface.size
            fixed_size = (540, 960) if panel_width >= panel_height else (960, 540)
            viewer.app.set_fixed_render_size(*fixed_size)
            fixed = frame("fixed-output")
            if fixed["coverage"] > 0.98:
                raise AssertionError(
                    "The fixed-output control must exercise intentional letterboxing"
                )
            capture("fixed-output")
            viewer.app.clear_fixed_render_size()
            frame("adaptive-restored")
            capture("adaptive-restored")

            report = {
                "backend": args.backend,
                "visible": args.visible,
                "vsync": args.vsync,
                "pixel_scale": viewer.window.pixel_scale,
                "style_scale": viewer.window.style_scale,
                "frames": rows,
                "phases": {},
                "timing_scope": "Size request or queued input through completed viewer.sync; excludes readback and desktop scanout",
            }
            for phase in dict.fromkeys(row["phase"] for row in rows):
                selected = [row for row in rows if row["phase"] == phase]
                times = [row["submission_ms"] for row in selected]
                report["phases"][phase] = {
                    "frames": len(selected),
                    "submission_ms": {
                        "median": float(np.median(times)),
                        "p95": float(np.percentile(times, 95)),
                        "maximum": max(times),
                    },
                    "minimum_coverage": min(row["coverage"] for row in selected),
                    "distinct_image_sizes": len({tuple(row["image_pixels"]) for row in selected}),
                }
    finally:
        if previous_settings is None:
            os.environ.pop("MOJIVE_SETTINGS", None)
        else:
            os.environ["MOJIVE_SETTINGS"] = previous_settings
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["phases"], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--output", type=Path, default=Path("output/viewport-resize/acceptance"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--frames", type=int, default=40, help="Frames in each direction")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    args = parser.parse_args(argv)
    if args.width < 1000 or args.height < 700 or args.frames < 8:
        parser.error("Use width >= 1000, height >= 700, and frames >= 8")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
