"""Measure sustained production editing with hidden native input and matched checkpoints."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.scene.queries import node_world_pose
from mojive.tools.editor_profile import _drag, _summary
from mojive.tools.inspector_profile import InspectorInput, _snapshot
from mojive.types import MeshShape
from mojive.ui.workspace.icon_mesh import native_icon


class FrameSamples:
    """Time normal sync calls; inspection, captures and commands stay outside the timer."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.original = viewer.sync
        self.phase = "warmup"
        self.samples = defaultdict(lambda: {"wall": [], "cpu": []})
        self.frames = 0
        self.edits = defaultdict(lambda: {"wall": [], "cpu": []})

    def sync(self):
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        self.original()
        row = self.samples[self.phase]
        row["cpu"].append((time.thread_time_ns() - cpu) / 1e6)
        row["wall"].append((time.perf_counter_ns() - start) / 1e6)
        self.frames += 1

    def edit(self, command):
        """Include synchronous command/history work and its following production frame."""
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        result = checked(self.viewer, command)
        self.viewer.sync()
        row = self.edits[type(command).__name__]
        row["cpu"].append((time.thread_time_ns() - cpu) / 1e6)
        row["wall"].append((time.perf_counter_ns() - start) / 1e6)
        return result

    def drain_edits(self):
        rows = {}
        for key, value in self.edits.items():
            row = summarize(value)
            row["operations"] = row.pop("frames")
            row["command_and_frame_wall_ms"] = row.pop("sync_wall_ms")
            rows[key] = row
        self.edits.clear()
        return rows

    def drain(self):
        result = {key: summarize(value) for key, value in self.samples.items()}
        self.samples.clear()
        return result


def summarize(samples):
    wall, cpu = samples["wall"], samples["cpu"]
    return {
        "frames": len(wall),
        "sync_wall_ms": _summary(wall),
        "main_thread_cpu_ms": _summary(cpu),
        "over_16_67_ms": sum(value > 1000 / 60 for value in wall),
        "over_33_33_ms": sum(value > 1000 / 30 for value in wall),
        "wall_samples_ms": wall,
        "cpu_samples_ms": cpu,
    }


def checked(viewer, command):
    result = viewer.session.submit(command)
    if not result.ok:
        raise RuntimeError(f"{type(command).__name__}: {result.message}")
    return result


def cpu_reference():
    """Record an independent Python workload outside frame timing, without normalization."""
    samples = {"wall": [], "cpu": []}
    for _ in range(5):
        start, cpu = time.perf_counter_ns(), time.thread_time_ns()
        sum(value * value for value in range(50000))
        samples["cpu"].append((time.thread_time_ns() - cpu) / 1e6)
        samples["wall"].append((time.perf_counter_ns() - start) / 1e6)
    return samples


def resources(viewer):
    app, session, backend = viewer.app, viewer.session, viewer.backend
    hierarchy = app.workspace.navigation.hierarchy
    cache = native_icon.cache_info()
    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024
    row = {
        "rss_bytes": rss,
        "window_readback_bytes": (
            viewer.window._readback.nbytes if viewer.window._readback is not None else 0
        ),
        "live_nodes": len(session.nodes),
        "instances": session.source.instance_count,
        "restore_snapshots": len(app.initial_entities),
        "entity_cache": len(app.projection._entity_cache),
        "hierarchy_rows": len(hierarchy._row_cache),
        "hierarchy_open": len(hierarchy._open_state),
        "undo_records": len(session._undo_stack),
        "redo_records": len(session._redo_stack),
        "history_bytes": session.history_bytes,
        "icon_meshes": cache.currsize,
        "icon_mesh_limit": cache.maxsize,
        "editing": session.editing,
        "gizmo_using": app.gizmo.using,
        "numeric_edit_active": app.workspace._numeric_edit is not None,
    }
    if backend.caps.name == "opengl":
        row.update(
            meshes=len(backend.meshes._meshes),
            textures=len(backend.textures._textures),
            vaos=len(backend.instances._vaos),
            instance_capacity=backend.instances.capacity,
        )
    return row


def assert_released(viewer):
    if viewer.session.editing or viewer.app.gizmo.using:
        raise RuntimeError("A completed gesture retained editing or gizmo ownership")
    if viewer.app.workspace._numeric_edit is not None:
        raise RuntimeError("A completed gesture retained a numeric edit")


def compare(before, after):
    same = before["state"]["state_signature"] == after["state"]["state_signature"]
    draws = before["state"]["ui_draw_counts"] == after["state"]["ui_draw_counts"]
    return {
        "same_state": same,
        "same_draw_counts": draws,
        "idle_cpu_median_ratio": (
            after["idle"]["main_thread_cpu_ms"]["median"]
            / before["idle"]["main_thread_cpu_ms"]["median"]
        )
        if same and draws
        else None,
        "idle_wall_median_ratio": (
            after["idle"]["sync_wall_ms"]["median"] / before["idle"]["sync_wall_ms"]["median"]
        )
        if same and draws
        else None,
    }


def run(args):
    output = args.output
    started = time.perf_counter()
    report = {
        "platform": platform.platform(),
        "asset": args.asset,
        "cycles": args.cycles,
        "warmup_cycles": args.warmup_cycles,
        "objects_per_cycle": args.objects,
        "drag_frames_per_cycle": args.drag_frames,
        "undo_every": args.undo_every,
        "scope": (
            "One continuously running hidden production workspace; no OS input. "
            "Frame timers exclude commands, snapshots and captures; edit timers include commands "
            "plus their next frame. Uncapped sync throughput "
            "does not measure display scanout. Undo history is never cleared after setup."
        ),
        "checkpoints": [],
        "workload": [],
    }
    with build_workspace(
        resolve(args.asset or "joint_gizmo"),
        renderer=args.renderer,
        paused=True,
        vsync=False,
        show_window=False,
        width=1600,
        height=1000,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        if not args.asset:
            checked(viewer, cmd.NewScene())
        anchors = [
            checked(
                viewer,
                cmd.AddSceneObject(
                    MeshShape.BOX,
                    name=f"soak_anchor_{index}",
                    position=(index - 1, 0, 0.5),
                ),
            ).entity_id
            for index in range(3)
        ]
        app, workspace = viewer.app, viewer.app.workspace
        app.set_language("en")
        app.tool = "move"
        reference = anchors[1]
        checked(viewer, cmd.Select(reference))
        for _ in range(24):
            viewer.sync()
        report.update(
            renderer=viewer.backend.caps.name,
            window_points=viewer.window.size_points,
            window_pixels=viewer.window.size_pixels,
            style_scale=viewer.window.style_scale,
            pixel_scale=viewer.window.pixel_scale,
            visible=False,
            vsync=False,
            design_sha256=hashlib.sha256(
                json.dumps(app.design.spec, sort_keys=True).encode()
            ).hexdigest(),
        )
        timer = FrameSamples(viewer)
        operations = {
            "created": 0,
            "deleted": 0,
            "drag_gestures": 0,
            "panel_clicks": 0,
            "undo_redo_pairs": 0,
        }
        with patch.object(viewer, "sync", timer.sync), InspectorInput(viewer) as inputs:

            def select(object_id):
                checked(viewer, cmd.Select(object_id))
                inputs.sync(2)

            def panels():
                for name in ("Pose", "Sensors", "Assets", "Scene"):
                    inputs.click("rail-" + name)
                    if workspace.left_active != name or not workspace.open[name]:
                        raise RuntimeError(f"Panel did not activate: {name}")
                for name in ("Physics", "Material", "Properties"):
                    inputs.click("tab-" + name)
                operations["panel_clicks"] = inputs.actions

            def drag(frames):
                node = viewer.session.selected_node
                position, rotation = node_world_pose(viewer.session, node)
                position, rotation = position.copy(), rotation.copy()
                held_phase = timer.phase
                timer.phase = "drag-setup"
                with _drag(viewer) as advance:
                    timer.phase = held_phase
                    for frame in range(frames):
                        advance(frame)
                        viewer.sync()
                    moved, _ = node_world_pose(viewer.session, viewer.session.selected_node)
                    if np.allclose(position, moved):
                        raise RuntimeError("Held native gizmo input did not change the pose")
                    timer.phase = "drag-release"
                operations["drag_gestures"] += 1
                assert_released(viewer)
                checked(
                    viewer, cmd.SetPose(viewer.session.selected_node.node_id, position, rotation)
                )
                inputs.pointer = (0, 0)
                inputs.sync(2)

            def cycle(index):
                timer.phase = "create-delete"
                ids = []
                for number in range(args.objects):
                    created = timer.edit(
                        cmd.AddSceneObject(
                            (MeshShape.BOX, MeshShape.SPHERE, MeshShape.CYLINDER)[number % 3],
                            name=f"soak_{index}_{number}",
                            position=(number - 1, 1, 0.5),
                        ),
                    )
                    ids.append(created.entity_id)
                    operations["created"] += 1
                # Exercise Inspector lookup for the newest identity, not only old anchors.
                select(ids[-1])
                for object_id in reversed(ids):
                    timer.edit(cmd.RemoveSceneObject(object_id))
                    operations["deleted"] += 1
                if args.undo_every and index % args.undo_every == 0:
                    timer.edit(cmd.Undo())
                    if viewer.session.node_by_object_id(ids[0]) is None:
                        raise RuntimeError("Undo did not restore the last deleted object")
                    timer.edit(cmd.Redo())
                    if viewer.session.node_by_object_id(ids[0]) is not None:
                        raise RuntimeError("Redo did not remove the restored object")
                    operations["undo_redo_pairs"] += 1
                select(anchors[index % len(anchors)])
                timer.phase = "drag"
                drag(args.drag_frames)
                timer.phase = "panel-switch"
                panels()
                assert_released(viewer)

            def state_snapshot():
                row = _snapshot(inputs)
                camera = app._camera_view()
                row["draw_counts_by_window"] = {
                    window.name: {
                        "vertices": len(window.draw_list.vtx_buffer),
                        "indices": len(window.draw_list.idx_buffer),
                    }
                    for window in imgui.get_current_context().windows
                    if window.active
                }
                row["scene_poses"] = {
                    str(node.node_id): [
                        value.round(10).tolist() for value in node_world_pose(viewer.session, node)
                    ]
                    for node in viewer.session.nodes
                }
                row["camera"] = {
                    key: getattr(camera, key).tolist() for key in ("eye", "target", "up")
                }
                row["state_signature"] = hashlib.sha256(
                    json.dumps(
                        {key: row[key] for key in ("state_signature", "scene_poses", "camera")},
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                return row

            def checkpoint(cycle_number):
                timer.phase = "settle"
                select(reference)
                inputs.top()
                scene_window = next(
                    window
                    for window in imgui.get_current_context().windows
                    if window.active and "Scene entities" in window.name
                )
                imgui.internal.set_scroll_y(scene_window, 0)
                inputs.sync(12)
                timer.drain()
                before = state_snapshot()
                timer.phase = "idle"
                inputs.sync(args.frames * 3)
                state = state_snapshot()
                if before["state_signature"] != state["state_signature"]:
                    raise RuntimeError("Checkpoint state changed during contiguous idle samples")
                idle = timer.drain()["idle"]
                timer.phase = "held-drag"
                drag(args.frames)
                active = timer.drain()["held-drag"]
                timer.phase = "settle"
                inputs.sync(12)
                # Include the reusable window readback in both endpoint baselines.
                # Sampling before the first capture mislabels that buffer as growth.
                if cycle_number in (0, args.cycles):
                    Image.fromarray(viewer.capture_array(surface="window")).save(
                        output / f"cycle-{cycle_number}.png"
                    )
                row = {
                    "cycle": cycle_number,
                    "elapsed_s": time.perf_counter() - started,
                    "total_frames": timer.frames,
                    "operations": dict(operations),
                    "state": state,
                    "resources": resources(viewer),
                    "cpu_reference": cpu_reference(),
                    "idle": idle,
                    "held_drag": active,
                    "idle_windows": [
                        summarize(
                            {
                                "wall": idle["wall_samples_ms"][offset : offset + args.frames],
                                "cpu": idle["cpu_samples_ms"][offset : offset + args.frames],
                            }
                        )
                        for offset in range(0, len(idle["wall_samples_ms"]), args.frames)
                    ],
                }
                if args.profile and cycle_number in (0, args.cycles):
                    timer.phase = "profile"
                    profiler = cProfile.Profile()
                    profiler.runcall(inputs.sync, 60)
                    profiler.dump_stats(str(output / f"cycle-{cycle_number}.prof"))
                    with (output / f"cycle-{cycle_number}.txt").open("w") as stream:
                        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(80)
                report["checkpoints"].append(row)
                report["comparisons"] = [
                    compare(report["checkpoints"][0], item) for item in report["checkpoints"]
                ]
                report["elapsed_s"] = row["elapsed_s"]
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "cycle": cycle_number,
                            "elapsed_s": round(row["elapsed_s"], 1),
                            "idle_ms": idle["sync_wall_ms"],
                            "drag_ms": active["sync_wall_ms"],
                            "resources": row["resources"],
                            "comparison": report["comparisons"][-1],
                        }
                    ),
                    flush=True,
                )
                timer.drain()

            for index in range(args.warmup_cycles):
                cycle(index)
            checkpoint(0)
            timer.drain_edits()
            for index in range(1, args.cycles + 1):
                cycle(index + args.warmup_cycles)
                if index % args.checkpoint_every == 0 or index == args.cycles:
                    report["workload"].append(
                        {
                            "through_cycle": index,
                            "phases": timer.drain(),
                            "edits": timer.drain_edits(),
                        }
                    )
                    checkpoint(index)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset", help="Keep this model alongside authored objects; default is a new scene"
    )
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument(
        "--profile", action="store_true", help="Profile initial/final idle separately after timing"
    )
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--warmup-cycles", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument(
        "--undo-every",
        type=int,
        default=0,
        help="Undo/redo the last deletion every N cycles; 0 disables",
    )
    parser.add_argument("--drag-frames", type=int, default=32)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/editor-soak"))
    args = parser.parse_args(argv)
    if (
        args.undo_every < 0
        or min(args.cycles, args.checkpoint_every, args.objects, args.warmup_cycles) < 1
        or min(args.frames, args.drag_frames) < 12
    ):
        parser.error("counts must be positive and frame counts at least 12")
    args.output.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")}):
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
