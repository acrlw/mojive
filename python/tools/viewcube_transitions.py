"""Capture view-cube depth swaps and an orbit through the production UI painter."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from imgui_bundle import imgui
from PIL import Image

from mojive.app.ui.window import create_window
from mojive.types import CameraView
from mojive.ui.imgui_draw import ImguiDraw2D
from mojive.ui.viewcube import BALL_PT, MARGIN_PT, RADIUS_PT, SPOKE_INSET_PT, ViewCube
from mojive.ui.window import WindowConfig


def depth_swaps():
    for first, second in ((0, 1), (0, 2), (1, 2)):
        for sign in (-1.0, 1.0):
            for third in (-0.6, 0.6):
                eyes = []
                for delta in (-0.00001, 0.00001):
                    eye = [third] * 3
                    eye[first], eye[second] = sign + delta, sign
                    eyes.append(eye)
                yield f"{'XYZ'[first]}{'XYZ'[second]} {sign:+g} {third:+g}", eyes


def circle_limits():
    for angle in (0.0, 0.5, 1.1):
        eyes = []
        for delta in (-1e-6, 1e-6):
            reach = (BALL_PT + SPOKE_INSET_PT) / RADIUS_PT + delta
            eyes.append(
                (math.sqrt(1 - reach * reach), reach * math.cos(angle), reach * math.sin(angle))
            )
        yield f"circle {angle:g}", eyes


def capture_orientation(
    window, cube, eye, scale, *, background=(0.10, 0.12, 0.14, 1.0), hover_origin=False
):
    window.begin_frame()
    draw = ImguiDraw2D(imgui.get_foreground_draw_list())
    width, height = window.size_points
    draw.rect_filled((0, 0), (width, height), background)
    reach = (RADIUS_PT + BALL_PT + MARGIN_PT) * scale
    rect = (0, height * 0.5 - reach, width * 0.5 + reach, height)
    view = CameraView(eye=np.asarray(eye), target=np.zeros(3), up=np.array((0.0, 0.0, 1.0)))
    start = time.perf_counter_ns()
    cursor = (width * 0.5, height * 0.5) if hover_origin else (-1000, -1000)
    cube.update(view, rect, cursor, scale)
    cube.draw(draw, scale)
    elapsed_us = (time.perf_counter_ns() - start) / 1000
    vertices = len(draw._dl.vtx_buffer)
    pixels = window.end_frame(readback=True)[::-1].copy()
    return pixels, elapsed_us, vertices


def run(renderer, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for scale in (0.65, 1.0, 1.25, 1.5):
        window = create_window(
            WindowConfig(
                width=180,
                height=180,
                vsync=False,
                docking=False,
                ini_path="",
                show_on_start=False,
                ui_scale=scale,
            ),
            renderer,
        )
        cube = ViewCube()
        try:
            for _ in range(3):
                capture_orientation(window, cube, (1, 1, 0.6), scale)
            rows, swaps = [], {}
            for name, eyes in (*depth_swaps(), *circle_limits()):
                pair = [capture_orientation(window, cube, eye, scale)[0] for eye in eyes]
                delta = np.abs(pair[0].astype(int) - pair[1].astype(int))
                swaps[name] = int(delta.max())
                rows.append(np.concatenate(pair, axis=1))
            Image.fromarray(np.concatenate(rows, axis=0)).save(output / f"transitions-{scale}.png")
            frames, elapsed, counts = [], [], []
            for index in range(180):
                yaw = math.tau * index / 180
                pitch = math.radians(35) * math.sin(yaw * 2)
                eye = (
                    math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                )
                pixels, cost, vertices = capture_orientation(window, cube, eye, scale)
                frames.append(Image.fromarray(pixels))
                elapsed.append(cost)
                counts.append(vertices)
            frames[0].save(
                output / f"orbit-{scale}.webp",
                save_all=True,
                append_images=frames[1:],
                duration=33,
                loop=0,
                lossless=True,
            )
            frames[25].save(output / f"viewcube-{scale}.png")
            hover_pair = [
                capture_orientation(window, cube, (1, 1, 0.6), scale, hover_origin=hover)[0]
                for hover in (False, True)
            ]
            Image.fromarray(np.concatenate(hover_pair, axis=1)).save(
                output / f"origin-hover-{scale}.png"
            )
            for name, background in (
                ("blue", (0.2, 0.4, 0.7, 1.0)),
                ("warm", (0.7, 0.3, 0.15, 1.0)),
            ):
                pixels = capture_orientation(
                    window, cube, (1, 1, 0.6), scale, background=background
                )[0]
                Image.fromarray(pixels).save(output / f"origin-{name}-{scale}.png")
            report[str(scale)] = {
                "transition_max_channel_delta": swaps,
                "update_draw_median_us": float(np.median(elapsed)),
                "update_draw_p95_us": float(np.percentile(elapsed, 95)),
                "max_vertices": max(counts),
                "framebuffer_scale": window.pixel_scale,
            }
        finally:
            window.close()
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", default="opengl")
    parser.add_argument("--output", type=Path, default=Path("output/viewcube-transitions"))
    args = parser.parse_args()
    run(args.renderer, args.output)


if __name__ == "__main__":
    main()
