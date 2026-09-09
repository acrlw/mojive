"""Measure real-window submission cadence and same-frame camera publication.

These CPU timings are not physical input-to-photon latency or displayed FPS.
Run without other GPU workloads; OS composition and refresh policy remain active.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mojive.composition import build
from mojive.config import LayoutConfig, ViewerConfig


def window_state_reader(window, glfw):
    """Distinguish actual visible presentation from an occluded offscreen fallback."""

    def occluded():
        return False

    def display_sync():
        return None

    maximum_refresh_hz = None
    if sys.platform == "darwin":
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.object_getClassName.argtypes = [ctypes.c_void_p]
        objc.object_getClassName.restype = ctypes.c_char_p
        send = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        ns_window = glfw.get_cocoa_window(window._window)
        screen = send(ns_window, objc.sel_registerName(b"screen"))
        maximum_refresh_hz = int(send(screen, objc.sel_registerName(b"maximumFramesPerSecond")))
        selector = objc.sel_registerName(b"occlusionState")

        def occluded():
            return not bool(send(ns_window, selector) & 2)

        view = send(ns_window, objc.sel_registerName(b"contentView"))
        layer = send(view, objc.sel_registerName(b"layer"))
        if layer and b"MetalLayer" in objc.object_getClassName(layer):
            sync_selector = objc.sel_registerName(b"displaySyncEnabled")

            def display_sync():
                return bool(send(layer, sync_selector))

    def read():
        return {
            "focused": bool(glfw.get_window_attrib(window._window, glfw.FOCUSED)),
            "occluded": occluded(),
            "metal_display_sync": display_sync(),
        }

    read.maximum_refresh_hz = maximum_refresh_hz
    return read


def run(asset, output, backends, seconds, repeats, width, height, operations, render_size=None):
    from mojive.ui import window as window_module

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in backends:
        with build(
            asset,
            renderer=backend,
            paused=True,
            width=width,
            height=height,
            vsync=True,
            show_window=True,
            config=ViewerConfig(layout=LayoutConfig(persistence=False)),
        ) as viewer:
            if render_size is not None:
                viewer.app.set_fixed_render_size(*render_size)
            glfw = window_module.glfw
            glfw.focus_window(viewer.window._window)
            publish = viewer.backend.set_camera
            published = None

            def record_camera(camera, sink=publish):
                nonlocal published
                published = camera
                return sink(camera)

            viewer.backend.set_camera = record_camera
            for enabled in (True, False):
                viewer.window.set_vsync(enabled)
                for _ in range(45):
                    viewer.sync()
                read_window_state = window_state_reader(viewer.window, glfw)
                for operation in operations:
                    for repeat in range(repeats):
                        glfw.focus_window(viewer.window._window)
                        durations = []
                        states = Counter()
                        deadline = time.perf_counter() + seconds
                        while time.perf_counter() < deadline:
                            published = None
                            start = time.perf_counter()
                            direction = 1 if len(durations) % 2 == 0 else -1
                            if operation == "pan":
                                viewer.app.camera.pan(
                                    direction * 0.25, 0, viewer.backend.target.height
                                )
                            elif operation == "orbit":
                                viewer.app.camera.orbit(direction * 0.25, direction * 0.2)
                            else:
                                viewer.app.camera.dolly(direction * 0.005)
                            viewer.sync()
                            durations.append(time.perf_counter() - start)
                            state = read_window_state()
                            states.update({key: int(bool(value)) for key, value in state.items()})
                            if published is None or not np.allclose(
                                published.view_matrix(),
                                viewer.app.camera.view().view_matrix(),
                            ):
                                raise AssertionError(
                                    "Camera changes must reach rendering in the same frame"
                                )
                        row = {
                            "backend": backend,
                            "vsync": enabled,
                            "repeat": repeat,
                            "operation": operation,
                            "window_points": list(viewer.window.size_points),
                            "window_pixels": list(viewer.window.size_pixels),
                            "pixel_scale": viewer.window.pixel_scale,
                            "viewport_pixels": [
                                viewer.backend.target.width,
                                viewer.backend.target.height,
                            ],
                            "frames": len(durations),
                            "submission_fps": len(durations) / sum(durations),
                            "frame_ms": {
                                "median": float(np.median(durations) * 1000),
                                "p95": float(np.percentile(durations, 95) * 1000),
                                "maximum": float(max(durations) * 1000),
                            },
                            "same_frame_camera": True,
                            "display_max_refresh_hz": read_window_state.maximum_refresh_hz,
                            "window_state_frames": dict(states),
                            "metal_sync_observable": state["metal_display_sync"] is not None,
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
            Image.fromarray(viewer.capture_array(surface="window")).save(output / f"{backend}.png")
    result = {
        "scope": "CPU submission cadence, not physical input-to-photon latency or displayed FPS",
        "rows": rows,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=Path("assets/test_scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("output/native-window-benchmark"))
    parser.add_argument(
        "--backends", nargs="+", choices=("bgfx", "opengl"), default=["bgfx", "opengl"]
    )
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Fixed scene resolution; window framebuffer stays at the display scale",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=("pan", "orbit", "dolly"),
        default=["pan", "orbit", "dolly"],
    )
    args = parser.parse_args()
    if args.render_size is not None and min(args.render_size) <= 0:
        parser.error("Render dimensions must be positive")
    if args.seconds <= 0 or args.repeats <= 0:
        parser.error("Duration and repeat count must be positive")
    run(
        args.asset,
        args.output,
        args.backends,
        args.seconds,
        args.repeats,
        args.width,
        args.height,
        args.operations,
        args.render_size,
    )


if __name__ == "__main__":
    main()
