"""Measure CPU interaction geometry and scene queries without graphics or physics."""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import platform
import pstats
import statistics
import time
from pathlib import Path

import numpy as np

from mojive import commands as cmd
from mojive.adapters.static import StaticSceneAdapter
from mojive.drawing.drag_link import smooth_drag_link_mesh
from mojive.interaction.gizmo import (
    GizmoMode,
    hit_test,
    prepare_projection,
    project,
    visibility,
    world_scale,
)
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import CameraView
from mojive.ui.gizmo import _hinge_range_path_hit, _HingeRangeProjection


def measure(function, count: int, batches: int):
    for i in range(count):
        function(i)
    samples = []
    for _ in range(batches):
        start = time.perf_counter_ns()
        for i in range(count):
            function(i)
        samples.append((time.perf_counter_ns() - start) / count / 1000.0)
    return {"median_us": statistics.median(samples), "batch_us": samples, "calls": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output/interaction-performance.json")
    )
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    camera = CameraView(eye=np.array((4.0, 3.0, 2.0)), aspect=1.6)
    origin, rotation, rect = np.zeros(3), np.eye(3), (0.0, 0.0, 1600.0, 1000.0)
    prepared = prepare_projection(camera)
    scale = world_scale(camera, origin, rect[3], prepared=prepared)
    cursors = np.random.default_rng(428).uniform((680, 380), (920, 620), size=(256, 2))
    points = np.random.default_rng(428).uniform(-1, 1, size=(4096, 3))
    angles = np.linspace(-2.1, 2.1, 128)
    joint = _HingeRangeProjection(
        1.0,
        np.column_stack((800 + 80 * np.cos(angles), 500 + 80 * np.sin(angles))),
        False,
        None,
        None,
        None,
    )
    cases = {
        "joint_range_hit": (lambda i: _hinge_range_path_hit(cursors[i % 256], joint, 1.0), 256),
        "drag_overlap_cold": (lambda i: smooth_drag_link_mesh(5.0 + i / 64.0, 5.0, 2.0), 256),
        "gizmo_visibility": (
            lambda i: visibility(camera, origin, rotation, rect, scale, prepared=prepared),
            256,
        ),
    }
    for count in (1, 64, 4096):
        cases[f"project_{count}"] = (
            lambda i, count=count: project(camera, points[:count], rect, prepared=prepared),
            512,
        )
    for mode in GizmoMode:
        cases[f"hit_{mode.value}"] = (
            lambda i, mode=mode: hit_test(camera, origin, rotation, rect, cursors[i % 256], mode),
            256,
        )
    sessions = []
    try:
        for count in (100, 1000, 10000):
            scene = Scene()
            for i in range(count):
                scene.box(position=(i % 100, i // 100, 0), size=(0.2, 0.2, 0.2))
            session = Session(StaticSceneAdapter(scene))
            sessions.append(session)
            query = cmd.Bounds()
            node = next(node for node in reversed(session.nodes) if node.geom_index >= 0)
            cases[f"scene_bounds_{count}"] = (
                lambda i, session=session, query=query: session.query(query),
                64,
            )
            cases[f"node_bounds_{count}"] = (
                lambda i, session=session, node=node: session.node_world_bounds(node.node_id),
                64,
            )
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            report = {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cases": {},
            }
            for name, (function, count) in cases.items():
                report["cases"][name] = measure(function, count, args.batches)
                print(f"{name}: {report['cases'][name]['median_us']:.3f} us", flush=True)
            if args.profile:
                profiler = cProfile.Profile()
                profiler.enable()
                for function, count in cases.values():
                    for i in range(min(count, 64)):
                        function(i)
                profiler.disable()
                profiler.dump_stats(args.output.with_suffix(".prof"))
                with args.output.with_suffix(".txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                        "cumtime"
                    ).print_stats(60)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            if gc_enabled:
                gc.enable()
    finally:
        for session in sessions:
            session.release()


if __name__ == "__main__":
    main()
