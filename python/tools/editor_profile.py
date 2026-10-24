"""Profile editor frames and document edits without mixing timers with CPU attribution."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.app.composition import build, build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.scene.assets import resolve
from mojive.types import MeshShape


def _summary(values):
    return {
        "mean": float(np.mean(values)),
        **{
            key: float(np.percentile(values, percentile))
            for key, percentile in (("median", 50), ("p95", 95), ("p99", 99), ("max", 100))
        },
    }


def _stages(viewer):
    return (
        (viewer.session, "tick", "session.tick"),
        (viewer.app, "_sync_structure", "source_publication"),
        (viewer.backend, "set_scene", "backend.set_scene"),
        (viewer.backend, "update", "backend.update"),
        (viewer.backend, "render", "backend.render"),
        (viewer.app.scene_entities, "publish", "helpers"),
        (viewer.panels, "draw", "panels"),
        (viewer.window, "begin_frame", "window.begin"),
        (viewer.app, "_present_frame", "presentation"),
    )


def _draw_counts():
    from imgui_bundle import imgui

    data = imgui.get_draw_data()
    return {
        "vertices": data.total_vtx_count,
        "indices": data.total_idx_count,
        "commands": sum(len(commands.cmd_buffer) for commands in data.cmd_lists),
    }


@contextmanager
def _drag(viewer):
    """Hold a real move handle through native input, without touching the OS cursor."""
    from imgui_bundle import imgui

    from mojive.interaction.gizmo import AXIS_END, SIZE_PT, GizmoHandle, project, world_scale

    app = viewer.app
    app.tool = "move"
    pointer, pressed = (0.0, 0.0), False
    original = viewer.window._input.process_inputs

    def process():
        original()
        io = imgui.get_io()
        io.add_mouse_pos_event(*pointer)
        io.add_mouse_button_event(0, pressed)

    with patch.object(viewer.window._input, "process_inputs", process):
        for _ in range(3):
            viewer.sync()
        gizmo, camera, rect = app.gizmo, app._camera_view(), app._viewport_rect
        frame = gizmo._frame
        length = world_scale(camera, frame.position, rect[3], SIZE_PT * viewer.window.style_scale)
        center = project(camera, (frame.position,), rect)[0, :2]
        for axis, handle in enumerate((GizmoHandle.X, GizmoHandle.Y, GizmoHandle.Z)):
            point = frame.position + frame.rotation[:, axis] * length * AXIS_END * 0.8
            pointer = tuple(project(camera, (point,), rect)[0, :2])
            viewer.sync()
            if gizmo.hovered_handle is handle:
                break
        else:
            raise RuntimeError("No visible move handle for the selected object")
        start = np.array(pointer)
        direction = (start - center) / np.linalg.norm(start - center)
        pressed = True
        viewer.sync()

        def advance(index):
            nonlocal pointer
            pointer = tuple(
                start + direction * np.sin(index * 0.12) * 30 * viewer.window.style_scale
            )

        try:
            yield advance
            if not gizmo.using:
                raise RuntimeError("Gizmo lost input during the held-drag workload")
        finally:
            pressed = False
            viewer.sync()
            viewer.sync()
            if gizmo.using:
                raise RuntimeError("Gizmo did not release the drag")


def _run(args):
    report = {
        "platform": platform.platform(),
        "asset": args.asset,
        "workspace": args.workspace,
        "settings_category": args.settings,
        "frames": args.frames,
        "empty_objects": args.empty_objects,
        "tool": args.tool,
        "scope": (
            "Uninstrumented wall/CPU samples, inclusive stage timers and cProfile run separately. "
            "Nested stage times must not be added. sync return is not display scanout latency. "
            "OpenGL drain waits for GPU completion; other backends include color readback."
        ),
        "cases": {},
    }
    compose = build_workspace if args.workspace else build
    with compose(
        resolve(args.asset),
        renderer=args.renderer,
        paused=True,
        vsync=args.vsync,
        show_window=args.visible,
        width=args.width,
        height=args.height,
        config=ViewerConfig(layout=LayoutConfig(persistence=False)),
    ) as viewer:
        if args.empty_objects is not None:
            result = viewer.session.submit(cmd.NewScene())
            if not result.ok:
                raise RuntimeError(result.message)
            for index in range(args.empty_objects):
                result = viewer.session.submit(
                    cmd.AddSceneObject(
                        MeshShape.BOX,
                        name=f"box_{index}",
                        position=(float(index % 3) - 1, float(index // 3), 0.5),
                    )
                )
                if not result.ok:
                    raise RuntimeError(result.message)
            if args.empty_objects:
                result = viewer.session.submit(cmd.Select(result.entity_id))
                if not result.ok:
                    raise RuntimeError(result.message)
        if args.settings:
            viewer.panels.get("Settings").show_category(args.settings)
            viewer.panels.open_panel("Settings")
        if args.joint:
            joint = next(
                (
                    node
                    for node in viewer.session.nodes
                    if node.type is NodeType.JOINT and node.name == args.joint
                ),
                None,
            )
            if joint is None:
                raise ValueError(f"No joint node named {args.joint!r}")
            result = viewer.session.submit(cmd.SelectNode(joint.node_id))
            if not result.ok:
                raise RuntimeError(result.message)
            viewer.set_gizmo_mode("rotate")
            report["selected_joint"] = args.joint
        if args.tool:
            viewer.app.tool = args.tool
        for _ in range(24):
            viewer.sync()
        body = next(
            (node for node in viewer.session.nodes if node.posable and node.object_id), None
        )
        geom = next((node for node in viewer.session.nodes if node.geom_index >= 0), None)
        editing_cases = set(args.cases) & {"visibility", "color", "transform", "selection", "drag"}
        if editing_cases and (body is None or geom is None):
            raise ValueError("Editing cases require at least one editable object")
        report.update(
            renderer=viewer.backend.caps.name,
            window_points=viewer.window.size_points,
            window_pixels=viewer.window.size_pixels,
            pixel_scale=viewer.window.pixel_scale,
            style_scale=viewer.window.style_scale,
            visible=args.visible,
            vsync=args.vsync,
            application=type(viewer.app).__name__,
            instances=viewer.session.source.instance_count,
            nodes=len(viewer.session.nodes),
            gpu_timing_supported=viewer.backend.caps.gpu_timing,
        )
        design = getattr(viewer.app, "design", None)
        if design is not None:
            report["design_sha256"] = hashlib.sha256(
                json.dumps(design.spec, sort_keys=True).encode()
            ).hexdigest()

        def drain():
            if viewer.backend.caps.name == "opengl":
                viewer.backend.ctx.finish()
            else:
                viewer.backend.target.read_color()

        drag_step = None

        def execute(mode, index):
            result = None
            if mode == "drag":
                drag_step(index)
            elif mode == "orbit":
                viewer.app.camera.orbit(0.2, 0.05)
            elif mode == "visibility":
                result = viewer.session.submit(cmd.SetVisible(body.node_id, bool(index % 2)))
            elif mode == "selection":
                nodes = [n for n in viewer.session.nodes if n.posable and n.object_id]
                result = viewer.session.submit(cmd.SelectNode(nodes[index % len(nodes)].node_id))
            elif mode == "transform":
                result = viewer.session.submit(
                    cmd.SetPose(body.node_id, np.array(((index % 10) * 0.1, 0.0, 0.5)), np.eye(3))
                )
            elif mode == "create-delete":
                created = viewer.session.submit(
                    cmd.AddSceneObject(MeshShape.SPHERE, name="temporary")
                )
                if not created.ok:
                    raise RuntimeError(created.message)
                viewer.sync()
                result = viewer.session.submit(cmd.RemoveSceneObject(created.entity_id))
            elif mode == "color":
                result = viewer.session.submit(
                    cmd.SetGeometryColor(geom.node_id, ((index % 10) / 10, 0.4, 0.8, 1))
                )
            if result is not None and not result.ok:
                raise RuntimeError(result.message)
            viewer.sync()

        for mode in args.cases:
            with ExitStack() as case:
                drag_step = case.enter_context(_drag(viewer)) if mode == "drag" else None
                count = (
                    args.frames
                    if mode in ("idle", "orbit", "drag", "idle-after-edits")
                    else max(12, args.frames // 3)
                )
                for index in range(5):
                    execute(mode, index)
                drain()
                wall, cpu, waits = [], [], []
                for index in range(count):
                    start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
                    execute(mode, index)
                    cpu.append((time.thread_time_ns() - cpu_start) / 1e6)
                    wall.append((time.perf_counter_ns() - start) / 1e6)
                for index in range(10):
                    execute(mode, index)
                    start = time.perf_counter_ns()
                    drain()
                    waits.append((time.perf_counter_ns() - start) / 1e6)
                row = {
                    "frames_per_sample": 2 if mode == "create-delete" else 1,
                    "sync_samples_ms": wall,
                    "cpu_samples_ms": cpu,
                    "sync_wall_ms": _summary(wall),
                    "main_thread_cpu_ms": _summary(cpu),
                    "post_sync_gpu_wait_or_readback_ms": _summary(waits),
                    "gpu_pass_ms": dict(viewer.backend.stats.gpu_ms),
                    "ui_draw_counts": _draw_counts(),
                }
                report["cases"][mode] = row
                if args.timing_only:
                    continue
                stages = {}
                with ExitStack() as stack:
                    for owner, name, label in _stages(viewer):
                        function = getattr(owner, name)

                        def measured(
                            *values, _function=function, _label=label, _stages=stages, **kwargs
                        ):
                            start = time.perf_counter_ns()
                            try:
                                return _function(*values, **kwargs)
                            finally:
                                _stages.setdefault(_label, []).append(
                                    (time.perf_counter_ns() - start) / 1e6
                                )

                        stack.enter_context(patch.object(owner, name, measured))
                    for index in range(min(30, count)):
                        execute(mode, index)
                row["inclusive_stages_ms"] = {key: _summary(value) for key, value in stages.items()}
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    for index in range(min(30, count)):
                        execute(mode, index)
                finally:
                    profiler.disable()
                profiler.dump_stats(str(args.output / f"{mode}.prof"))
                with (args.output / f"{mode}.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(60)
        Image.fromarray(viewer.capture_array(surface="window")).save(args.output / "window.png")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="joint_gizmo")
    parser.add_argument(
        "--empty-objects", type=int, help="Start a new scene and add this many boxes"
    )
    parser.add_argument("--tool", choices=("select", "move", "rotate", "scale"))
    parser.add_argument("--joint", help="Select a named joint and enable its rotation gizmo")
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--visible", action="store_true", help="Present the native window")
    parser.add_argument(
        "--vsync", action="store_true", help="Synchronize presentation to the display"
    )
    parser.add_argument(
        "--workspace", action="store_true", help="Use the production editor workspace"
    )
    parser.add_argument(
        "--settings",
        choices=("General", "Camera", "Interaction", "Rendering", "Recording", "MuJoCo Visuals"),
        help="Open a settings category in the production workspace",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "idle",
            "orbit",
            "visibility",
            "color",
            "transform",
            "selection",
            "create-delete",
            "idle-after-edits",
            "drag",
        ),
        default=("idle", "orbit", "visibility", "color"),
    )
    parser.add_argument(
        "--timing-only", action="store_true", help="Skip stage instrumentation and cProfile"
    )
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/editor-profile"))
    args = parser.parse_args(argv)
    if args.settings or args.empty_objects is not None:
        args.workspace = True
    if args.empty_objects is not None and args.empty_objects < 0:
        parser.error("empty-objects cannot be negative")
    if args.frames < 12 or min(args.width, args.height) < 320:
        parser.error("frames must be at least 12 and window dimensions at least 320")
    args.output.mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"MOJIVE_SETTINGS": str(args.output / "settings.json")}):
        report = _run(args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
