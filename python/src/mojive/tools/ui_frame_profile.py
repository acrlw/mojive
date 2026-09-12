"""Profile production viewport chrome and quantify its incremental frame cost."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from .. import commands as cmd
from ..assets import resolve
from ..composition import build
from ..gizmo import GizmoMode, project

_CAPSULE_METHODS = (
    "_draw_playback_widget",
    "_draw_tool_column_widget",
    "_draw_context_hint_widget",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--profile-frames", type=int, default=180)
    parser.add_argument("--max-capsule-ms", type=float, default=0.75)
    parser.add_argument("--asset", default="gizmo", help="Asset name or model path to profile")
    parser.add_argument(
        "--diagnostics", action="store_true", help="Profile populated Output and value controls"
    )
    parser.add_argument(
        "--keyframes", action="store_true", help="Profile a populated snapshot timeline"
    )
    parser.add_argument(
        "--pending-edits",
        action="store_true",
        help="Profile coalesced dimension previews and the Apply hint",
    )
    parser.add_argument("--running", action="store_true", help="Include live simulation ticks")
    parser.add_argument(
        "--expand-hierarchy", action="store_true", help="Profile expanded hierarchy rows"
    )
    parser.add_argument(
        "--hover-gizmo", action="store_true", help="Sweep the pointer over the selected gizmo"
    )
    parser.add_argument(
        "--gizmo-mode", choices=[mode.value for mode in GizmoMode], default="translate"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    viewer = build(
        resolve(args.asset),
        paused=not args.running,
        vsync=False,
        width=1600,
        height=1000,
        show_window=False,
    )
    try:
        selected = next(
            node
            for node in viewer.session.nodes
            if node.posable and (args.gizmo_mode != "dimensions" or node.geom_index >= 0)
        )
        viewer.session.submit(cmd.Select(selected.object_id))
        viewer.app.gizmo.set_mode(args.gizmo_mode)
        if args.diagnostics:
            from .ui_runtime import _activate_panel

            for index in range(120):
                viewer.app.output.write(
                    f"Profile sample {index}", level=("info", "warning", "error")[index % 3]
                )
            _activate_panel(viewer, "Control" if viewer.session.actuators else "Joints")
            _activate_panel(viewer, "Output")
        if args.keyframes:
            from .ui_runtime import _activate_panel

            for _ in range(48):
                assert viewer.session.submit(cmd.Step(1))
                viewer.sync()
                assert viewer.session.submit(cmd.CaptureSceneSnapshot())
            viewer.panels.open_panel("Keyframes")
            for _ in range(4):
                viewer.sync()
            _activate_panel(viewer, "Keyframes")
        pending_profile = None
        if args.pending_edits:
            if args.running:
                raise ValueError("Pending model edits require a paused profile")
            session = viewer.session
            node = next(n for n in session.nodes if n.source_editable and n.geom_index >= 0)
            source = session._source
            size = np.maximum(source.geom_size[source.geom_node == node.node_id][0], 0.01)
            stage_ms = []
            for index in range(60):
                start = time.perf_counter_ns()
                result = viewer.app.model_edits.stage(
                    cmd.SetGeometrySize(node.node_id, size * (1 + index / 100))
                )
                stage_ms.append((time.perf_counter_ns() - start) / 1e6)
                if not result.ok:
                    raise RuntimeError(result.message)
            assert session._source is source
            pending_profile = {
                "stage": _summary(stage_ms),
                "commands": len(viewer.app.model_edits.commands),
                "compiled_source_unchanged": True,
            }
        for _ in range(max(1, args.warmup)):
            viewer.sync()
        if args.expand_hierarchy:
            hierarchy = viewer.panels.get("Hierarchy")
            hierarchy._open_state.update(
                {node.node_id: True for node in viewer.session.nodes if node.children}
            )

        cursor_step = 0
        origin, _rotation = viewer.app._node_pose(selected)
        center = project(viewer.app._camera_view(), (origin,), viewer.app._viewport_rect)[0, :2]

        # Match the native gallery's queued pointer events without moving the OS cursor.
        def sync():
            nonlocal cursor_step
            if args.hover_gizmo:
                angle = cursor_step * 0.31
                radius = 35.0 + 45.0 * (cursor_step % 17) / 16.0
                io = imgui.get_io()
                io.add_mouse_pos_event(
                    *(center + radius * np.array((np.cos(angle), np.sin(angle))))
                )
                cursor_step += 1
            viewer.sync()

        variants = {
            "full": (),
            "without_playback": (_CAPSULE_METHODS[0],),
            "without_tools": (_CAPSULE_METHODS[1],),
            "without_hint": (_CAPSULE_METHODS[2],),
            "without_capsules": _CAPSULE_METHODS,
        }
        samples: dict[str, list[float]] = {name: [] for name in variants}
        # Interleave variants so temperature and driver scheduling do not
        # systematically favor the first or last measurement.
        for round_index in range(3):
            order = tuple(variants) if round_index % 2 == 0 else tuple(reversed(variants))
            for name in order:
                with _disabled(viewer.app, variants[name]):
                    cursor_step = 0
                    for _ in range(4):
                        sync()
                    samples[name].extend(_sample(sync, max(1, args.frames // 3)))

        report = {name: _summary(values) for name, values in samples.items()}
        full = report["full"]["median_ms"]
        for name, method in (
            ("playback_cost_ms", "without_playback"),
            ("tools_cost_ms", "without_tools"),
            ("hint_cost_ms", "without_hint"),
            ("capsules_cost_ms", "without_capsules"),
        ):
            report[name] = full - report[method]["median_ms"]

        with _timed_widgets(viewer) as timings:
            for _ in range(max(1, args.profile_frames)):
                sync()
        report["measured_draw_ms_per_frame"] = {
            name: round(sum(values) / 1_000_000 / max(1, args.profile_frames), 5)
            for name, values in timings.items()
        }
        report["capsules_draw_ms_per_frame"] = sum(
            report["measured_draw_ms_per_frame"][name] for name in _CAPSULE_METHODS
        )
        if args.keyframes and not timings["keyframes"]:
            raise RuntimeError("The requested timeline was not drawn during profiling")
        report["draw_calls_per_frame"] = {
            name: len(values) / max(1, args.profile_frames) for name, values in timings.items()
        }

        profile_path = args.output / "ui-frame-profile.prof"
        from ..ui.draw2d import _cached_fringe_points, _concave_indices
        from ..ui.icons import (
            _icon_draw_commands,
            _production_icon_layout,
            production_helper_strokes,
        )
        from ..ui.panels.filters import severity_meshes
        from ..ui.viewport_widgets import _scaled_reset_glyph

        caches = {
            "diagnostic_meshes": severity_meshes,
            "fringes": _cached_fringe_points,
            "icon_layout": _production_icon_layout,
            "icon_commands": _icon_draw_commands,
            "helper_icon_paths": production_helper_strokes,
            "triangulation": _concave_indices,
            "reset_glyph": _scaled_reset_glyph,
        }
        before = {name: fn.cache_info() for name, fn in caches.items()}
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(max(1, args.profile_frames)):
            sync()
        profiler.disable()
        profiler.dump_stats(profile_path)
        text_path = args.output / "ui-frame-profile.txt"
        with text_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumtime").print_stats(50)

        report["geometry_cache"] = {
            name: {
                "hits": fn.cache_info().hits - before[name].hits,
                "misses": fn.cache_info().misses - before[name].misses,
                "entries": fn.cache_info().currsize,
            }
            for name, fn in caches.items()
        }
        report["profiled_cpu_ms_per_frame"] = {
            f"{path}:{name}": round(entry[3] * 1000 / max(1, args.profile_frames), 5)
            for (path, _line, name), entry in stats.stats.items()
            if name
            in {"severity_icon", "value_rail", "draw_playback", "draw_tool_column", "indexed_fill"}
        }
        report["profile"] = str(profile_path.resolve())
        report["scene"] = {
            "asset": args.asset,
            "running": args.running,
            "expanded_hierarchy": args.expand_hierarchy,
            "diagnostics": args.diagnostics,
            "keyframes": args.keyframes,
        }
        report["interaction"] = {"hover_gizmo": args.hover_gizmo, "gizmo_mode": args.gizmo_mode}
        report["interaction"]["hit_test_calls"] = sum(
            entry[1]
            for (path, _line, name), entry in stats.stats.items()
            if path == "gizmo.py" and name == "hit_test"
        )
        if args.hover_gizmo and not report["interaction"]["hit_test_calls"]:
            raise RuntimeError("The pointer sweep did not exercise gizmo hit testing")
        if pending_profile is not None:
            draft = viewer.app.model_edits
            draft.applying = True
            start = time.perf_counter_ns()
            result = viewer.session.apply_model_edits(draft.commands)
            pending_profile["apply_model_cpu_ms"] = (time.perf_counter_ns() - start) / 1e6
            if not result.ok:
                raise RuntimeError(result.message)
            draft.clear()
            report["pending_edits"] = pending_profile
        report_path = args.output / "ui-frame-profile.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(report_path.resolve())
        print(text_path.resolve())
        if report["capsules_draw_ms_per_frame"] > args.max_capsule_ms:
            raise SystemExit(
                "viewport capsule drawing "
                f"{report['capsules_draw_ms_per_frame']:.3f} ms exceeds "
                f"{args.max_capsule_ms:.3f} ms"
            )
    finally:
        viewer.release()
    return 0


@contextmanager
def _timed_widgets(viewer):
    """Measure actual widget calls independently of frame pacing and cProfile overhead."""
    from ..ui.panels import camera, control, filters, inspector, joints, output

    targets = [(viewer.app, method, method) for method in _CAPSULE_METHODS]
    targets += [(module, "severity_icon", "diagnostics") for module in (filters, output)]
    targets += [
        (module, "value_rail", "value_controls") for module in (camera, control, inspector, joints)
    ]
    targets.append((viewer.panels.get("Keyframes"), "draw", "keyframes"))
    saved, timings = [], {}

    def wrap(function, samples):
        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                samples.append(time.perf_counter_ns() - start)

        return measured

    try:
        for owner, name, group in targets:
            function = getattr(owner, name)
            saved.append((owner, name, function))
            setattr(owner, name, wrap(function, timings.setdefault(group, [])))
        yield timings
    finally:
        for owner, name, function in saved:
            setattr(owner, name, function)


def _sample(sync, frames: int) -> list[float]:
    values = []
    for _ in range(frames):
        start = time.perf_counter_ns()
        sync()
        values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return values


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, np.float64)
    return {
        "frames": len(array),
        "median_ms": round(float(np.median(array)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "mean_ms": round(float(np.mean(array)), 4),
    }


@contextmanager
def _disabled(app, names: tuple[str, ...]):
    originals = {name: getattr(app, name) for name in names}
    try:
        for name in names:
            setattr(app, name, lambda: None)
        yield
    finally:
        for name, method in originals.items():
            setattr(app, name, method)


if __name__ == "__main__":
    raise SystemExit(main())
