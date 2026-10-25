"""Attribute slow frames in a sustained hidden editor workload, retaining every event."""

from __future__ import annotations

import argparse
import gc
import json
import marshal
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from mojive.tools import editor_soak
from mojive.tools.editor_profile import _stages, _summary


class Collections:
    """Observe normal collections without changing GC thresholds or collecting explicitly."""

    def __init__(self):
        self.starts = [None, None, None]
        self.events = []

    def callback(self, phase, info):
        generation = info["generation"]
        if phase == "start":
            self.starts[generation] = (time.perf_counter_ns(), time.thread_time_ns())
        elif self.starts[generation] is not None:
            wall, cpu = self.starts[generation]
            end_cpu, end_wall = time.thread_time_ns(), time.perf_counter_ns()
            self.events.append(
                (
                    wall,
                    end_wall,
                    end_cpu - cpu,
                    generation,
                    info["collected"],
                    info["uncollectable"],
                )
            )
            self.starts[generation] = None


def stage_targets(viewer):
    """These timers overlap: submit/commit/history and panels/gizmo are inclusive."""
    from imgui_bundle import imgui

    from mojive.ui import window

    return (
        *_stages(viewer),
        (viewer.app, "_poll_gizmo", "gizmo.poll"),
        (viewer.session, "submit", "command.submit"),
        (viewer.session, "_capture_document_state", "history.capture"),
        (viewer.session, "_commit_edit", "history.commit"),
        (viewer.session._history, "append", "history.append"),
        (viewer.app.workspace, "panel", lambda name, *_: "panel." + name),
        (viewer.app, "_draw_main_menu", "chrome.main"),
        (viewer.app, "_draw_viewport_contents", "chrome.viewport"),
        (viewer.app, "_draw_application_status_bar", "chrome.status"),
        (viewer.app, "_draw_playback_widget", "chrome.playback"),
        (viewer.app, "_draw_tool_column_widget", "chrome.tools"),
        (imgui, "render", "present.imgui_finalize"),
        (window.glfw, "poll_events", "window.poll_events"),
        (viewer.window, "make_current", "window.make_current"),
        (viewer.window, "_refresh_scales", "window.scales"),
        (viewer.window, "_sync_style_scale", "window.style_scale"),
        (imgui, "new_frame", "window.imgui_new_frame"),
        *(
            (owner, method, label)
            for owner, method, label in (
                (getattr(viewer.window, "_gl_surface", None), "clear", "present.clear"),
                (getattr(viewer.window, "_impl", None), "render", "present.imgui_submit"),
                (getattr(viewer.window, "_input", None), "process_inputs", "window.input"),
                (viewer.window, "_renderer_new_frame", "window.renderer_new_frame"),
            )
            if owner is not None and callable(getattr(owner, method, None))
        ),
        (window.glfw, "swap_buffers", "present.swap"),
        *(
            (render_pass, method, "render." + name + "." + method)
            for name, render_pass in getattr(viewer.backend, "_passes", {}).items()
            for method in ("prepare_instances", "prepare", "execute")
        ),
    )


class TailSamples(editor_soak.FrameSamples):
    """Retain chronological frame, command and command-plus-frame event timings.

    Event construction and storage occur outside child command/frame intervals.
    The combined edit interval includes that observer gap without subtracting it.
    Optional stage wrappers are diagnostic instrumentation and must be run separately
    from the baseline. They are inclusive, not an additive frame-time decomposition.
    """

    def __init__(self, viewer, *, stack, stages=False, warmup_cycles=0):
        super().__init__(viewer)
        self.events = []
        self.origin_ns = time.perf_counter_ns()
        self.cycle = -warmup_cycles
        self.warmup_cycles = warmup_cycles
        self.command = None
        self._trace_stages = stages
        self._active_stages = None
        self._last_frame_wall_ms = 0.0
        self._last_frame_cpu_ms = 0.0
        if stages:
            for owner, method, label in stage_targets(viewer):
                stack.enter_context(
                    patch.object(owner, method, self.stage(getattr(owner, method), label))
                )

    def stage(self, original, label):
        def timed(*args, **kwargs):
            active = self._active_stages
            if active is None:
                return original(*args, **kwargs)
            key = label(*args) if callable(label) else label
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                end_cpu, end_wall = time.thread_time_ns(), time.perf_counter_ns()
                value = active.setdefault(key, [0, 0, 0, 0])
                value[0] += 1
                value[1] += end_wall - start
                value[2] += end_cpu - cpu
                value[3] = max(value[3], end_wall - start)

        return timed

    def event(self, kind):
        node = self.viewer.session.selected_node
        return {
            "kind": kind,
            "phase": self.phase,
            "cycle": self.cycle,
            "frame": self.frames + 1,
            "command": self.command,
            "selected_node": node.node_id if node is not None else None,
            # A dict born with a mutable value remains GC-tracked until a full
            # collection, even after replacing that value. Baseline records must
            # contain only atomic values from creation, not merely when finished.
            "stages": {} if self._trace_stages else (),
        }

    def finish(self, event, start, cpu, end_wall, end_cpu):
        # Stage instrumentation is a separate diagnostic. Compact its mutable
        # maps after timing; baseline records are already untracked at creation.
        if self._trace_stages:
            event["stages"] = tuple((label, *value) for label, value in event["stages"].items())
        event.update(
            start_ns=start,
            end_ns=end_wall,
            wall_ms=(end_wall - start) / 1e6,
            cpu_ms=(end_cpu - cpu) / 1e6,
        )
        # Even untracked dictionaries contribute to GC allocation thresholds.
        # Retain bytes, not a growing population of GC-capable record objects.
        # These process-local payloads are decoded only after timed work ends.
        self.events.append(marshal.dumps(event))

    def sync(self):
        event = self.event("frame")
        event["gizmo_before"] = bool(self.viewer.app.gizmo.using)
        self._active_stages = event["stages"] if self._trace_stages else None
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            self.original()
        finally:
            end_cpu, end_wall = time.thread_time_ns(), time.perf_counter_ns()
            self._active_stages = None
        event["gizmo_after"] = bool(self.viewer.app.gizmo.using)
        self.frames += 1
        self.finish(event, start, cpu, end_wall, end_cpu)
        self._last_frame_wall_ms = event["wall_ms"]
        self._last_frame_cpu_ms = event["cpu_ms"]
        row = self.samples[self.phase]
        row["wall"].append(event["wall_ms"])
        row["cpu"].append(event["cpu_ms"])

    def edit(self, command):
        name = getattr(command, "name", "")
        if name.startswith("soak_"):
            self.cycle = int(name.split("_")[1]) - self.warmup_cycles
        previous_command, self.command = self.command, type(command).__name__
        event, combined = self.event("command"), self.event("edit")
        self._active_stages = event["stages"] if self._trace_stages else None
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        try:
            result = editor_soak.checked(self.viewer, command)
        finally:
            command_cpu, command_wall = time.thread_time_ns(), time.perf_counter_ns()
            self._active_stages = None
        self.finish(event, start, cpu, command_wall, command_cpu)
        try:
            self.viewer.sync()
        finally:
            end_cpu, end_wall = time.thread_time_ns(), time.perf_counter_ns()
            self.command = previous_command
        combined["command_ms"] = event["wall_ms"]
        combined["frame_ms"] = self._last_frame_wall_ms
        combined["command_cpu_ms"] = event["cpu_ms"]
        combined["frame_cpu_ms"] = self._last_frame_cpu_ms
        combined["observer_gap_ms"] = (
            (end_wall - start) / 1e6 - event["wall_ms"] - self._last_frame_wall_ms
        )
        self.finish(combined, start, cpu, end_wall, end_cpu)
        row = self.edits[type(command).__name__]
        row["wall"].append(combined["wall_ms"])
        row["cpu"].append(combined["cpu_ms"])
        return result


def serialize_events(timer, collections):
    """Join GC timestamps after the run, keeping report work outside frame timing."""
    result = []
    for index, packed in enumerate(timer.events):
        source = marshal.loads(packed)
        row = {key: value for key, value in source.items() if key not in ("start_ns", "end_ns")}
        row.update(
            event=index,
            start_ms=(source["start_ns"] - timer.origin_ns) / 1e6,
            off_cpu_ms=max(0, source["wall_ms"] - source["cpu_ms"]),
        )
        stages = source["stages"]
        if isinstance(stages, dict):
            stages = tuple((label, *value) for label, value in stages.items())
        row["stages"] = {
            label: {
                "calls": calls,
                "wall_ms": wall / 1e6,
                "cpu_ms": cpu / 1e6,
                "max_call_ms": longest / 1e6,
            }
            for label, calls, wall, cpu, longest in stages
        }
        relevant = [
            item
            for item in collections.events
            if item[0] < source["end_ns"] and item[1] > source["start_ns"]
        ]
        row["gc"] = {
            str(generation): {
                "count": len(items),
                "wall_ms": sum(item[1] - item[0] for item in items) / 1e6,
                "cpu_ms": sum(item[2] for item in items) / 1e6,
                "collected": sum(item[4] for item in items),
            }
            for generation in range(3)
            if (items := [item for item in relevant if item[3] == generation])
        }
        result.append(row)
    return sorted(result, key=lambda event: (event["start_ms"], event["event"]))


def summarize_events(events, top):
    groups = defaultdict(list)
    for row in events:
        if row["cycle"] <= 0:
            continue
        key = "/".join((row["kind"], row["phase"], row["command"] or "none"))
        groups[key].append(row)
    return {
        key: {
            "count": len(rows),
            "wall_ms": _summary([row["wall_ms"] for row in rows]),
            "cpu_ms": _summary([row["cpu_ms"] for row in rows]),
            "off_cpu_ms": _summary([row["off_cpu_ms"] for row in rows]),
            "over_16_67_ms": sum(row["wall_ms"] > 1000 / 60 for row in rows),
            "over_33_33_ms": sum(row["wall_ms"] > 1000 / 30 for row in rows),
            "slowest": sorted(rows, key=lambda row: row["wall_ms"], reverse=True)[:top],
        }
        for key, rows in groups.items()
    }


def run(args):
    collections, timers = Collections(), []
    gc_stats_start = gc.get_stats()
    with ExitStack() as stack:

        def factory(viewer):
            timer = TailSamples(
                viewer,
                stack=stack,
                stages=args.stages,
                warmup_cycles=args.warmup_cycles,
            )
            timers.append(timer)
            return timer

        gc.callbacks.append(collections.callback)
        stack.callback(gc.callbacks.remove, collections.callback)
        stack.enter_context(patch.object(editor_soak, "FrameSamples", factory))
        stack.enter_context(
            patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")})
        )
        report = editor_soak.run(args)
    gc_stats_end = gc.get_stats()
    observer_records = {
        "count": len(timers[0].events),
        "gc_tracked": sum(gc.is_tracked(event) for event in timers[0].events),
        "packed_bytes": sum(len(event) for event in timers[0].events),
        "storage": "process-local marshal bytes, decoded after timing",
    }
    events = serialize_events(timers[0], collections)
    result = {
        "schema_version": 2,
        "mode": "inclusive-stage-attribution" if args.stages else "event-timing-baseline",
        "observer_records": observer_records,
        "scope": (
            "Same hidden production workload as editor_soak. No OS input. Event recording and "
            "GC correlation are outside frame intervals; normal GC callbacks record collection "
            "duration. Stage timings, when enabled, are inclusive and overlap; do not add them. "
            "Completed records are packed into bytes outside timed intervals and decoded only "
            "after the workload. Event allocation and callbacks can still affect GC "
            "cadence: compare separate baseline runs, not stage-run throughput. "
            "Edit events include the command and following frame; those child events must not "
            "be summed with the parent. Combined edit latency includes intervening event metadata "
            "and tracing work: observer_gap_ms reports this difference without subtracting it. "
            "Frame gizmo_before/gizmo_after are inspected outside child frame intervals to identify "
            "press, held and release transitions. Warmup cycles remain in raw events but not summaries. "
            "Off-CPU is wall minus calling-thread CPU, not proof of a particular scheduler/GPU cause."
        ),
        "cycles": args.cycles,
        "asset": args.asset,
        "gc_thresholds": gc.get_threshold(),
        "gc_stats_start": gc_stats_start,
        "gc_stats_end": gc_stats_end,
        "collections": [
            {
                "start_ms": (start - timers[0].origin_ns) / 1e6,
                "wall_ms": (end - start) / 1e6,
                "cpu_ms": cpu / 1e6,
                "generation": generation,
                "collected": collected,
                "uncollectable": uncollectable,
            }
            for start, end, cpu, generation, collected, uncollectable in collections.events
        ],
        "observer": {
            "stage_wrappers": args.stages,
            "retained_stage_storage": "process-local byte records packed after each timed interval",
            "gc_callbacks": "start/stop timestamps and generation counts; default GC retained",
            "gc_observer_bias": "callbacks and event allocations may shift collection cadence",
        },
        "summaries": summarize_events(events, args.top),
        "events": events,
        "soak_report": "report.json",
        "elapsed_s": report["elapsed_s"],
    }
    (args.output / "tail-events.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "events": len(events),
                "output": str(args.output / "tail-events.json"),
            }
        )
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", help="Keep this model alongside authored objects")
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--shading", choices=("shaded", "wireframe"), default="shaded")
    parser.add_argument(
        "--stages", action="store_true", help="Separate diagnostic run with inclusive stage timing"
    )
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--warmup-cycles", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--undo-every", type=int, default=10)
    parser.add_argument("--drag-frames", type=int, default=32)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--profile", action="store_true", help="Profile initial/final idle outside timed samples"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("output/editor-tail-profile"))
    args = parser.parse_args(argv)
    if (
        args.undo_every < 0
        or min(args.cycles, args.warmup_cycles, args.checkpoint_every, args.objects, args.top) < 1
        or min(args.frames, args.drag_frames) < 12
    ):
        parser.error("counts must be positive and frame counts at least 12")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
