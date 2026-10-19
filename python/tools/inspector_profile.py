"""Measure production Inspector recovery after native tab and section interaction."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import pstats
import time
from pathlib import Path
from unittest.mock import patch

from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.tools.editor_profile import _draw_counts, _summary

CHURN_ENTITIES = ("geom9", "04_free", "01_revolute_y")
STATE_FIELDS = (
    "selected_node",
    "hierarchy",
    "generic_sections",
    "inspector_tab",
    "inspector_rect",
    "inspector_child_clip",
    "inspector_scroll",
    "field_hits",
    "sections",
)


class InspectorInput:
    """Inject native pointer events after backend polling, without paint instrumentation."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.workspace = viewer.app.workspace
        self.pointer = (0, 0)
        self.events = []
        self.actions = 0

    def __enter__(self):
        self.original = self.viewer.window._input.process_inputs
        self.viewer.window._input.process_inputs = self.process
        return self

    def __exit__(self, *_):
        self.viewer.window._input.process_inputs = self.original

    def process(self):
        self.original()
        io = imgui.get_io()
        io.add_mouse_pos_event(*self.pointer)
        for pressed in self.events:
            io.add_mouse_button_event(0, pressed)
        self.events.clear()

    def sync(self, count=4):
        for _ in range(count):
            self.viewer.sync()

    def child(self):
        return next(
            window
            for window in imgui.get_current_context().windows
            if window.active and "Inspector content" in window.name
        )

    def click(self, key):
        if key.startswith("section-"):
            window = self.child()
            rect = self.workspace.ui.hits[key]
            clip = window.inner_clip_rect
            if rect[1] < clip.min.y or rect[1] + rect[3] > clip.max.y:
                scroll = window.scroll.y + rect[1] - clip.min.y - 12
                imgui.internal.set_scroll_y(window, max(0, min(window.scroll_max.y, scroll)))
                self.sync()
        x, y, width, height = self.workspace.ui.hits[key]
        self.pointer = (x + width / 2, y + height / 2)
        self.sync(2)
        self.events.append(True)
        self.sync(1)
        self.events.append(False)
        self.sync(2)
        self.actions += 1
        if key.startswith("tab-") and self.workspace.inspector_tab != key[4:]:
            raise RuntimeError(f"Inspector tab did not activate: {key}")

    def select(self, name):
        node = next(node for node in self.viewer.session.nodes if node.name == name)
        result = self.viewer.session.submit(cmd.SelectNode(node.node_id))
        if not result.ok:
            raise RuntimeError(result.message)
        self.sync()

    def top(self):
        imgui.internal.set_scroll_y(self.child(), 0)
        self.pointer = (0, 0)
        self.sync()

    def deselect(self):
        result = self.viewer.session.submit(cmd.Select(0))
        if not result.ok:
            raise RuntimeError(result.message)
        self.sync()


def _cache_counts(ui):
    primitives = ui.design.primitives
    styles = list(primitives._styles.values())
    styles += [variant for style in tuple(styles) for variant in style._variants.values()]
    layout_styles = getattr(ui, "_numeric_layout_styles", (None, None, {}))[2]
    styles.extend(layout_styles.values())
    return {
        name: {"entries": count, "limit": limit}
        for name, count, limit in (
            ("style_requests", len(primitives._style_requests), 4096),
            ("vector_layouts", len(getattr(ui, "_vector_layouts", {})), 128),
            ("numeric_layout_styles", len(layout_styles), 64),
            (
                "geometry_per_style_max",
                max((len(s._geometry_cache) for s in styles), default=0),
                256,
            ),
            ("variants_per_style_max", max((len(s._variants) for s in styles), default=0), 128),
        )
    }


def _snapshot(inputs):
    viewer, ui = inputs.viewer, inputs.workspace.ui
    runtime = viewer.app.design
    inspector = imgui.internal.find_window_by_name("Inspector")
    inspector_rect = (*inspector.pos, *inspector.size)
    hierarchy = inputs.workspace.navigation.hierarchy
    scene = next(
        window
        for window in imgui.get_current_context().windows
        if window.active and "Scene entities" in window.name
    )
    selected = viewer.session.selected_node
    row = {
        "selected_node": selected.node_id if selected else None,
        "hierarchy": {
            "open": sorted(hierarchy._open_state.items()),
            "rows": [(node.node_id, depth, leaf) for node, depth, leaf in hierarchy._row_cache],
            "scroll": tuple(scene.scroll),
        },
        "generic_sections": dict(inputs.workspace.expanded),
        "ui_draw_counts": _draw_counts(),
        "inspector_tab": inputs.workspace.inspector_tab,
        "inspector_rect": list(inspector_rect),
        "inspector_child_clip": [
            *inputs.child().inner_clip_rect.min,
            *inputs.child().inner_clip_rect.max,
        ]
        if viewer.app.entity
        else None,
        "inspector_scroll": tuple(inputs.child().scroll) if viewer.app.entity else None,
        "field_hits": {
            key: list(rect) for key, rect in ui.hits.items() if rect[0] >= inspector_rect[0]
        },
        "sections": [key for key in ui.hits if key.startswith("section-")],
        "fonts": len(imgui.get_io().fonts.fonts),
        "font_cache": len(ui._font_cache),
        "hits": len(ui.hits),
        "styles": len(ui.design.primitives._styles),
        "observations": len(runtime.observed),
        "observe_components": runtime.observe_components,
        "windows": len(imgui.get_current_context().windows),
        "native_clicks": inputs.actions,
        "cache_counts": _cache_counts(ui),
    }
    row["state_signature"] = hashlib.sha256(
        json.dumps({key: row[key] for key in STATE_FIELDS}, sort_keys=True).encode()
    ).hexdigest()
    row["cache_bounds_ok"] = all(
        cache["entries"] <= cache["limit"] for cache in row["cache_counts"].values()
    )
    return row


def _timed_window(viewer, frames):
    wall, cpu = [], []
    for _ in range(frames):
        start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
        viewer.sync()
        cpu.append((time.thread_time_ns() - cpu_start) / 1e6)
        wall.append((time.perf_counter_ns() - start) / 1e6)
    return wall, cpu


def _phase(inputs, args, label, *, windows=1):
    viewer = inputs.viewer
    inputs.pointer = (0, 0)
    inputs.sync(12)
    idle, wall, cpu = [], [], []
    for _ in range(windows):
        sample_wall, sample_cpu = _timed_window(viewer, args.frames)
        row = _snapshot(inputs)
        idle.append(
            {
                "sync_wall_ms": _summary(sample_wall),
                "main_thread_cpu_ms": _summary(sample_cpu),
                **{key: row[key] for key in ("state_signature", "cache_counts", "cache_bounds_ok")},
            }
        )
        wall.extend(sample_wall)
        cpu.extend(sample_cpu)
    row.update(
        sync_wall_ms=_summary(wall),
        main_thread_cpu_ms=_summary(cpu),
        idle_windows=idle,
        idle_state_unchanged=len({item["state_signature"] for item in idle}) == 1,
        cache_bounds_ok=all(item["cache_bounds_ok"] for item in idle),
    )
    # Keep capture/readback and profiling outside consecutive idle timing windows.
    Image.fromarray(viewer.capture_array(surface="window")).save(args.output / f"{label}.png")
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        inputs.sync(30)
        profiler.disable()
        profiler.dump_stats(str(args.output / f"{label}.prof"))
        with (args.output / f"{label}.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(80)
    print(
        label,
        json.dumps(
            {key: row[key] for key in ("sync_wall_ms", "main_thread_cpu_ms", "native_clicks")}
        ),
        flush=True,
    )
    return row


def _comparison(before, after):
    changed = [key for key in STATE_FIELDS if before[key] != after[key]]
    same_state = not changed and before["idle_state_unchanged"] and after["idle_state_unchanged"]
    return {
        "same_state": same_state,
        "changed_state_fields": changed,
        "same_draw_counts": before["ui_draw_counts"] == after["ui_draw_counts"],
        "cpu_median_ratio": after["main_thread_cpu_ms"]["median"]
        / before["main_thread_cpu_ms"]["median"]
        if same_state
        else None,
        "idle_cpu_median_ratios": [
            item["main_thread_cpu_ms"]["median"] / before["main_thread_cpu_ms"]["median"]
            for item in after["idle_windows"]
        ]
        if same_state
        else None,
        "count_changes": {
            key: after[key] - before[key] for key in ("fonts", "font_cache", "styles", "windows")
        },
        "cache_bounds_ok": before["cache_bounds_ok"] and after["cache_bounds_ok"],
    }


def _run(args):
    report = {
        "schema_version": 2,
        "frames": args.frames,
        "cycles": args.cycles,
        "language": args.language,
        "primed_entities": CHURN_ENTITIES,
        "scope": "Uninstrumented frame samples; optional cProfile runs separately. Not scanout FPS.",
        "phases": {},
    }
    with (
        build_workspace(
            resolve("joint_gizmo"),
            renderer=args.renderer,
            paused=True,
            vsync=args.vsync,
            show_window=args.visible,
            width=1600,
            height=1000,
            config=ViewerConfig(layout=LayoutConfig(persistence=False)),
        ) as viewer,
        InspectorInput(viewer) as inputs,
    ):
        viewer.app.set_language(args.language)
        inputs.sync(24)
        report.update(
            renderer=viewer.backend.caps.name,
            style_scale=viewer.window.style_scale,
            pixel_scale=viewer.window.pixel_scale,
            window_points=list(viewer.window.size_points),
            window_pixels=list(viewer.window.size_pixels),
            visible=args.visible,
            vsync=args.vsync,
            design_sha256=hashlib.sha256(
                json.dumps(viewer.app.design.spec, sort_keys=True).encode()
            ).hexdigest(),
        )

        def phase(label, *, windows=1):
            report["phases"][label] = _phase(inputs, args, label, windows=windows)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

        # Selection reveals scene ancestors. Prime the same paths before the
        # reference so hierarchy expansion does not masquerade as a UI slowdown.
        for name in CHURN_ENTITIES:
            inputs.select(name)
        inputs.click("tab-Properties")
        inputs.deselect()
        phase("unselected-before")
        inputs.select("01_revolute_y")
        inputs.click("tab-Physics")
        inputs.top()
        phase("joint-initial")
        heading = inputs.workspace.ui.translate("advanced joint properties").upper()
        advanced = "section-" + heading
        # The ordinary joint advanced section starts closed in a fresh layout.
        inputs.click(advanced)
        inputs.top()
        phase("joint-expanded")
        for cycle in range(args.cycles):
            for name in CHURN_ENTITIES:
                inputs.select(name)
                for tab in ("Properties", "Physics", "Joint", "Material", "Physics"):
                    inputs.click("tab-" + tab)
                    for key in tuple(inputs.workspace.ui.hits):
                        if key.startswith("section-") and key in inputs.workspace.ui.hits:
                            inputs.click(key)
                            inputs.click(key)
            if cycle in (0, args.cycles - 1):
                inputs.top()
                phase(f"after-churn-{cycle + 1}")
        inputs.select("01_revolute_y")
        inputs.click("tab-Physics")
        inputs.top()
        phase("joint-expanded-after", windows=3)
        inputs.click(advanced)
        inputs.top()
        phase("joint-collapsed-after")
        inputs.click("tab-Properties")
        inputs.deselect()
        phase("unselected-after", windows=3)
        report["recovery"] = {
            label: _comparison(report["phases"][before], report["phases"][after])
            for label, before, after in (
                ("expanded", "joint-expanded", "joint-expanded-after"),
                ("collapsed", "joint-initial", "joint-collapsed-after"),
                ("unselected", "unselected-before", "unselected-after"),
            )
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--language", choices=("en", "zh_CN"), default="en")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("output/inspector-profile"))
    args = parser.parse_args(argv)
    if args.frames < 12 or args.cycles < 1:
        parser.error("frames must be at least 12 and cycles must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")}):
        _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
