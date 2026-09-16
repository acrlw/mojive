"""Measure hierarchy frame cost and verify that large scenes remain fully browsable."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

from imgui_bundle import imgui

from mojive.app.composition import build_scene
from mojive.scene import Scene

from .ui_runtime import _click, _item_rect, _save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/hierarchy-benchmark"))
    parser.add_argument("--objects", type=int, default=1500)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args(argv)
    if args.objects < 1 or args.frames < 1:
        parser.error("objects and frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in ("MOJIVE_SETTINGS", "MOJIVE_UI_SCALE")}
    os.environ["MOJIVE_SETTINGS"] = str(args.output / "settings.json")
    os.environ["MOJIVE_UI_SCALE"] = "1"
    try:
        report = _run(args.objects, args.frames, args.output)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(case["last_node_selectable"] for case in report["cases"].values()) else 1


def _run(objects: int, frames: int, output: Path) -> dict:
    scene = Scene()
    for index in range(objects):
        scene.box(
            name=f"item-{index:05d}", position=(index % 50, index // 50, 0), size=(0.2, 0.2, 0.2)
        )
    report = {"platform": platform.platform(), "objects": objects, "cases": {}}
    with build_scene(scene, vsync=False, show_window=False, width=1280, height=900) as viewer:
        for _ in range(8):
            viewer.sync()
        panel = viewer.panels.get("Hierarchy")
        panel._open_state.update(
            {node.node_id: True for node in viewer.session.nodes if node.children}
        )
        target = next(
            node
            for node in reversed(viewer.session.nodes)
            if node.name == f"item-{objects - 1:05d}"
        )
        draw, row = panel.draw, panel._row
        panel_ms, frame_ms, row_counts = [], [], []
        visible = []
        calls = 0

        def draw_timed(ctx):
            nonlocal calls
            calls = 0
            visible.clear()
            start = time.perf_counter_ns()
            draw(ctx)
            panel_ms.append((time.perf_counter_ns() - start) / 1e6)
            row_counts.append(calls)

        def row_counted(ctx, node, *args, **kwargs):
            nonlocal calls
            calls += 1
            visible.append((node, imgui.get_current_context().current_window))
            return row(ctx, node, *args, **kwargs)

        panel.draw, panel._row = draw_timed, row_counted
        for name, query in (("expanded", ""), ("filtered", "item-")):
            panel._filter = query
            for _ in range(8):
                viewer.sync()
            window = visible[0][1]
            imgui.internal.set_scroll_y(window, 0)
            for _ in range(3):
                viewer.sync()
            panel_ms.clear()
            frame_ms.clear()
            row_counts.clear()
            for _ in range(frames):
                start = time.perf_counter_ns()
                viewer.sync()
                frame_ms.append((time.perf_counter_ns() - start) / 1e6)
            case = {
                "panel_median_ms": statistics.median(panel_ms),
                "panel_p95_ms": sorted(panel_ms)[int(0.95 * (len(panel_ms) - 1))],
                "sync_median_ms": statistics.median(frame_ms),
                "submitted_rows_median": statistics.median(row_counts),
            }
            imgui.internal.set_scroll_y(window, window.scroll_max.y)
            for _ in range(4):
                viewer.sync()
            selectable = any(node is target for node, _ in visible)
            if selectable:
                lo, hi = _item_rect(
                    viewer, "invisible_button", f"##hierarchy-node-{target.node_id}"
                )
                _click(viewer, ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2))
                selectable = viewer.session.selected_node is target
            case["last_node_selectable"] = selectable
            report["cases"][name] = case
            _save(viewer, output / f"{name}-last-node.png")
        report["backend"] = viewer.backend.caps.name
        report["nodes"] = len(viewer.session.nodes)
        report["window_points"] = list(viewer.window.size_points)
        report["viewport_points"] = list(viewer.app._viewport_rect)
        report["pixel_scale"] = viewer.window.pixel_scale
    return report


if __name__ == "__main__":
    raise SystemExit(main())
