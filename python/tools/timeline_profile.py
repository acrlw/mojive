"""Measure dense Keyframes panel CPU work and optionally capture it with offscreen EGL."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path

from imgui_bundle import imgui

from mojive.adapters.base import KeyframeInfo, SceneModelInfo
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.ui import fonts, theme
from mojive.ui.panels import PanelContext
from mojive.ui.panels.keyframes import KeyframesPanel


class _TimelineAdapter(StaticSceneAdapter):
    caps = replace(StaticSceneAdapter.caps, keyframes=True)

    def __init__(self, count: int) -> None:
        super().__init__(Scene())
        self._keys = [KeyframeInfo(i, f"key{i}", i / 30, 0) for i in range(count)]

    def keyframes(self):
        return self._keys

    def scene_models(self):
        return (SceneModelInfo(0, "Timeline profile", Path("profile"), False),)


def _measure(count, frames, output, capture):
    context = imgui.create_context()
    session = Session(_TimelineAdapter(count))
    renderer = gl = target = None
    try:
        width, height = 1600, 500
        io = imgui.get_io()
        io.set_ini_filename(None)
        io.display_size = (width, height)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        fonts.load(imgui, io, allow_download=False)
        theme.apply(imgui)
        device = None
        if capture:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import moderngl
            from imgui_bundle.python_backends.opengl_backend_programmable import (
                ProgrammablePipelineRenderer,
            )

            gl = moderngl.create_standalone_context(backend="egl")
            target = gl.simple_framebuffer((width, height))
            target.use()
            renderer = ProgrammablePipelineRenderer()
            device = gl.info["GL_RENDERER"]
        panel = KeyframesPanel()
        ctx = PanelContext(session, NullBackend())

        def frame():
            start = time.perf_counter_ns()
            imgui.new_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size((width, height))
            imgui.begin("Keyframes")
            panel.draw(ctx)
            imgui.end()
            imgui.render()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if renderer is not None:
                target.use()
                target.clear()
                renderer.render(imgui.get_draw_data())
                gl.finish()
            return elapsed

        for _ in range(5):
            frame()
        assert len(panel.editor.keyframe_cache) == count
        results = []
        for view in ("all", "zoomed", "panning", "selected"):
            panel.editor.view_start, panel.editor.view_end = (
                (0, max(1, count / 30)) if view in ("all", "selected") else (0, 1)
            )
            panel.editor.view_needs_fit = False
            panel.editor.set_follow_mode("off")
            if view == "selected":
                panel.editor.selected_keyframes = {
                    key.keyframe_id for key in panel.editor.keyframe_cache
                }
            samples = []
            for index in range(frames + 5):
                if view == "panning":
                    panel.editor.view_start, panel.editor.view_end = index / 30, index / 30 + 1
                elapsed = frame()
                if index >= 5:
                    samples.append(elapsed)
            data = imgui.get_draw_data()
            results.append(
                {
                    "keys": count,
                    "view": view,
                    "median_cpu_ms": statistics.median(samples),
                    "samples_cpu_ms": samples,
                    "vertices": data.total_vtx_count,
                    "indices": data.total_idx_count,
                    "capture_device": device,
                }
            )
            if renderer is not None:
                from PIL import Image

                image = Image.frombytes("RGB", (width, height), target.read(components=3))
                image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                    output / f"{count}-{view}.png"
                )
        return results
    finally:
        if renderer is not None:
            renderer.shutdown()
        if target is not None:
            target.release()
        if gl is not None:
            gl.release()
        session.release()
        imgui.destroy_context(context)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/timeline-profile"))
    parser.add_argument("--counts", type=int, nargs="+", default=[100, 10000, 100000])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--capture", action="store_true", help="Capture production ImGui output with EGL (Linux)"
    )
    args = parser.parse_args(argv)
    if args.frames < 1 or any(count < 1 for count in args.counts):
        parser.error("frames and key counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        row
        for count in args.counts
        for row in _measure(count, args.frames, args.output, args.capture)
    ]
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "samples_cpu_ms"}
                for row in results
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
