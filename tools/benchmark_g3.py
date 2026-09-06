"""Measure G3 geometry and ImGui submission on the CPU without a window."""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import platform
import pstats
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from mojive.curves2d import (
    arc_ribbon_mesh,
    arc_ribbon_points,
    arrow_points,
    arrow_triangles,
    smooth_affine_corners,
    smooth_capsule_points,
    smooth_line_cap,
    smooth_rect_points,
)
from mojive.draglink2d import smooth_drag_link_mesh
from mojive.gizmo import GizmoMode
from mojive.render.debugdraw import DebugDraw, Occlusion
from mojive.types import CameraView
from mojive.ui.draw2d import ImguiDraw2D, draw_drag_link
from mojive.ui.gizmo import ObjectGizmo, _JointRangeState
from mojive.ui.panels.keyframes import _draw_command_icon, _rounded_command_icon_path
from mojive.ui.viewport_widgets import (
    _draw_axis_arrow_glyph,
    _move_glyph_path,
    _rounded_playback_triangle,
    draw_playback_glyph,
    draw_tool_glyph,
)


class NullDraw:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def measure(function, count, batches):
    for i in range(count):
        function(i)
    samples = []
    for batch in range(batches):
        start = time.perf_counter_ns()
        for i in range(count):
            function((batch + 1) * count + i)
        samples.append((time.perf_counter_ns() - start) / count / 1000)
    return {"median_us": statistics.median(samples), "batch_us": samples, "calls": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/g3-perf/cpu.json"))
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--profile", action="store_true", help="Also save untimed call profiles")
    args = parser.parse_args()
    context = imgui.create_context()
    draw_list = imgui.ImDrawList(imgui.get_draw_list_shared_data())
    draw = ImguiDraw2D(draw_list)
    null = NullDraw()
    debug = DebugDraw()
    screen = debug.layer("screen", Occlusion.ALWAYS)
    path = _move_glyph_path(100.0, 100.0, 1.0, 6.0, 9.0, 3**0.5, 0.7)
    angles = np.linspace(0.0, 2.0, 64)
    arc = np.column_stack((np.cos(angles), np.sin(angles))) * 60.0
    sector = ((0.0, 0.0), *map(tuple, arc.tolist()))
    camera = CameraView(eye=np.array((3.0, -5.0, 3.0)), target=np.zeros(3), aspect=4.0 / 3.0)
    viewport = (0.0, 0.0, 800.0, 600.0)
    dimensions = ObjectGizmo("dimensions")
    dimensions._frame.mode = GizmoMode.DIMENSIONS
    joints = {}
    for kind, mode in (("hinge", "rotate"), ("slide", "translate")):
        joint = ObjectGizmo(mode)
        joint._joint_range = _JointRangeState(kind, 0.0, -0.34, 0.34)
        joints[kind] = joint

    def reset():
        draw_list._reset_for_new_frame()
        draw_list.flags = (
            imgui.ImDrawListFlags_.anti_aliased_fill.value
            | imgui.ImDrawListFlags_.anti_aliased_lines.value
        )

    def fringe(_):
        reset()
        draw._write_anti_alias_fringe(path, 0xFFFFFFFF)

    def rectangle(i):
        reset()
        draw_list.path_rect((0, 0), (100 + i % 2048 * 0.125, 36), 8)

    def rounded_control(radius):
        reset()
        draw_list.add_rect_filled((0, 0), (100, 36), 0xFFFFFFFF, radius)

    def scale_handles(_):
        reset()
        dimensions._draw_flat(draw, camera, viewport, 1.0)

    def joint_range(kind):
        reset()
        joints[kind]._draw_joint_range(draw, camera, viewport, 1.0)

    def arrow(i):
        reset()
        x, y = i * 0.125, i * 0.0625
        draw.arrow((x, y), (x + 60.0, y + 80.0), (1.0,) * 4)

    def frame_arrow(i):
        reset()
        _draw_axis_arrow_glyph(
            draw,
            (i * 0.125, i * 0.0625),
            (0.6, 0.8),
            (1.0,) * 4,
            1.18,
            1.46,
            clear_radius=3.0,
            base=7.6,
            tip=10.0,
            wing=1.8,
            corner_radius=0.25,
        )

    def command(i):
        reset()
        _draw_command_icon(draw, (100.0, 100.0), "key-next", (1, 1, 1, 1), 1.0)

    def drag(i):
        reset()
        draw_drag_link(
            draw, (0.0, 0.0), (100.0, 0.0), (1.0,) * 4, (0.2, 0.2, 0.2, 1.0), 2.0, 5.0, 1.0
        )

    def drag_moving(i):
        reset()
        x = i % 2048 * 0.125
        draw_drag_link(
            draw, (x, 0.0), (100.0 + x, 0.0), (1.0,) * 4, (0.2, 0.2, 0.2, 1.0), 2.0, 5.0, 1.0
        )

    def tool(i):
        reset()
        draw_tool_glyph(draw, (100, 100), (1, 1, 1, 1), 1, "move", "world")

    def fan(i):
        reset()
        draw.triangle_fan_fill(sector, (1.0,) * 4)

    def arc_draw(i):
        reset()
        vertices, indices = arc_ribbon_mesh(arc, None, None, 4, round_caps=True)
        draw.indexed_fill(vertices, indices, (1.0,) * 4, outline=vertices)

    cases = {
        "rectangle_static": (lambda i: smooth_rect_points(0, 0, 100, 36, 8), 4096),
        "rectangle_translated": (
            lambda i: smooth_rect_points(i * 0.125, 0, 100 + i * 0.125, 36, 8),
            2048,
        ),
        "rectangle_resized": (
            lambda i: smooth_rect_points(0, 0, 100 + i % 2048 * 0.125, 36, 8),
            2048,
        ),
        "capsule_translated": (lambda i: smooth_capsule_points(i * 0.125, 0, 40, 36), 1024),
        "capsule_short_resized": (
            lambda i: smooth_capsule_points(0, 0, 36 + i % 1024 / 1024 * 10, 36),
            1024,
        ),
        "cap_static": (lambda i: smooth_line_cap((0, 0), (1, 0), 8), 1024),
        "cap_short_dynamic": (
            lambda i: smooth_line_cap((0, 0), (1, 0), 8, max_inset=i % 512 / 512),
            512,
        ),
        "move_translated": (lambda i: _move_glyph_path(i * 0.125, 100, 1, 6, 9, 3**0.5, 0.7), 512),
        "snap_static": (
            lambda i: draw_tool_glyph(null, (100, 100), (1, 1, 1, 1), 1, "snap", "world"),
            1024,
        ),
        "fringe_static": (fringe, 1024),
        "playback_translated": (
            lambda i: draw_playback_glyph(null, (i * 0.125, 100), (1, 1, 1, 1), 1, "previous"),
            1024,
        ),
        "keyframe_translated": (
            lambda i: _draw_command_icon(null, (i * 0.125, 100), "key-next", (1, 1, 1, 1), 1),
            1024,
        ),
        "keyframe_native_static": (command, 1024),
        "plane_translated": (
            lambda i: smooth_affine_corners(
                ((i * 0.125, 0), (i * 0.125 + 12, 4), (i * 0.125 + 12, 18), (i * 0.125, 14)), 2
            ),
            1024,
        ),
        "native_rectangle_resized": (rectangle, 2048),
        "native_control_radius_4_5": (lambda i: rounded_control(4.5), 2048),
        "native_control_radius_16": (lambda i: rounded_control(16), 2048),
        "native_control_radius_8_circular": (lambda i: rounded_control(8), 2048),
        "scale_gizmo_native": (scale_handles, 512),
        "joint_hinge_native": (lambda i: joint_range("hinge"), 512),
        "joint_slide_native": (lambda i: joint_range("slide"), 512),
        "axis_arrow_translated": (
            lambda i: arrow_points((i * 0.125, 0), (100 + i * 0.125, 0)),
            1024,
        ),
        "arrow_native_static": (lambda i: arrow(0), 1024),
        "arrow_native_translated": (arrow, 1024),
        "frame_arrow_native_static": (lambda i: frame_arrow(0), 1024),
        "frame_arrow_native_translated": (frame_arrow, 1024),
        "screen_arrow_update": (
            lambda i: screen.arrow_2d("a", (i * 0.125, 0), (100 + i * 0.125, 0), (1, 1, 1, 1)),
            1024,
        ),
        "drag_static": (lambda i: smooth_drag_link_mesh(100.0, 5.0, 2.0), 2048),
        "drag_stretched": (lambda i: smooth_drag_link_mesh(100 + i * 0.125, 5.0, 2.0), 512),
        "drag_overlap_dynamic": (lambda i: smooth_drag_link_mesh(5 + i % 512 / 128, 5.0, 2.0), 64),
        "drag_native_static": (drag, 512),
        "drag_native_translated": (drag_moving, 512),
        "move_native_static": (tool, 1024),
        "arc_ribbon_static": (
            lambda i: arc_ribbon_points(arc, None, None, 4, round_caps=True),
            1024,
        ),
        "arc_ribbon_translated": (
            lambda i: arc_ribbon_points(
                np.add(arc, (i * 0.125, 0)), None, None, 4, round_caps=True
            ),
            1024,
        ),
        "triangle_fan_native_static": (fan, 1024),
        "arc_ribbon_native_static": (arc_draw, 1024),
    }
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "imgui_bundle": version("imgui-bundle"),
        "native_indexed_fill": hasattr(draw_list, "add_indexed_fill"),
        "native_poly_fringe": hasattr(draw_list, "add_poly_fringe"),
        "fringe_points": len(path),
        "cases": {},
    }
    try:
        gc.disable()
        for name, (function, count) in cases.items():
            result = measure(function, count, args.batches)
            report["cases"][name] = result
            if name in {
                "native_control_radius_4_5",
                "native_control_radius_16",
                "native_control_radius_8_circular",
                "scale_gizmo_native",
                "joint_hinge_native",
                "joint_slide_native",
            }:
                result["vertices"] = len(draw_list.vtx_buffer)
                result["indices"] = len(draw_list.idx_buffer)
            print(f"{name}: {result['median_us']:.3f} us", flush=True)
        report["cache_stats"] = {
            "playback_profiles": _rounded_playback_triangle.cache_info()._asdict(),
            "keyframe_profiles": _rounded_command_icon_path.cache_info()._asdict(),
            "drag_meshes": smooth_drag_link_mesh.cache_info()._asdict(),
            "screen_arrows": arrow_triangles.cache_info()._asdict(),
        }
        report["drag_topology"] = {
            str(distance): {"vertices": len(mesh[0]), "triangles": len(mesh[1]) // 3}
            for distance in (20.0, 100.0, 10000.0)
            for mesh in (smooth_drag_link_mesh(distance, 5.0, 2.0),)
        }
        report["screen_arrow_packing"] = {}
        for count in (100, 1000):
            scene = DebugDraw(limit=1_000_000)
            layer = scene.layer("arrows", Occlusion.ALWAYS)
            for index in range(count):
                layer.arrow_2d(str(index), (0, index), (100, index), (1, 1, 1, 1))
            report["screen_arrow_packing"][str(count)] = measure(
                lambda i, scene=scene: scene.build(), 128, args.batches
            )
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            for function, count in cases.values():
                for i in range(min(count, 256)):
                    function(i)
            profiler.disable()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(args.output.with_suffix(".prof"))
            with args.output.with_suffix(".txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumtime"
                ).print_stats(60)
    finally:
        gc.enable()
        draw._dl = None
        draw_list = None
        imgui.destroy_context(context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
