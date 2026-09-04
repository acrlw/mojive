"""Capture the three shadow quality presets for visual acceptance."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..assets import resolve
from ..render.backend import ShadowQuality
from ..types import CameraView
from ._harness import OffscreenHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/shadow-quality"))
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    captures: list[tuple[ShadowQuality, Image.Image]] = []
    report: dict[str, object] = {}
    with OffscreenHarness(resolve("sunlight_shadow"), args.width, args.height) as harness:
        harness.backend.set_camera(
            CameraView(
                eye=np.asarray((3.0, -5.0, 2.6), np.float32),
                target=np.asarray((0.0, 0.0, 0.8), np.float32),
                up=np.asarray((0.0, 0.0, 1.0), np.float32),
                fov_y=float(np.radians(34.0)),
                near=0.03,
                far=80.0,
                aspect=args.width / args.height,
            )
        )
        harness.warmup(3)
        backend_name = harness.backend.caps.name
        for quality in ShadowQuality:
            if not harness.backend.set_shadow_quality(quality):
                raise RuntimeError(f"{backend_name} does not support shadow quality presets")
            harness.step_and_render(0)
            for _ in range(6):
                harness.step_and_render(0)
            wall_ms: list[float] = []
            frame_cpu_ms: list[float] = []
            gpu_ms: list[float] = []
            for _ in range(max(args.frames, 1)):
                started = time.perf_counter()
                harness.step_and_render(0)
                wall_ms.append((time.perf_counter() - started) * 1000.0)
                frame_cpu_ms.append(float(harness.backend.stats.frame_cpu_ms))
                if harness.backend.stats.gpu_ms:
                    gpu_ms.append(
                        sum(float(value) for value in harness.backend.stats.gpu_ms.values())
                    )
            pixels = harness.backend.target.read_color(flip=True)
            image = Image.fromarray(pixels, "RGBA").convert("RGB")
            output = args.output / f"{backend_name}-{quality.value}.png"
            image.save(output)
            captures.append((quality, image))
            report[quality.value] = {
                "cascade_divisors": quality.cascade_divisors,
                "shader_quality": quality.level,
                "render_stats": harness.backend.stats.notes,
                "median_wall_ms": statistics.median(wall_ms),
                "median_frame_cpu_ms": statistics.median(frame_cpu_ms),
                "median_gpu_ms": statistics.median(gpu_ms) if gpu_ms else None,
            }

    comparison = _comparison(captures)
    comparison_path = args.output / f"{backend_name}-comparison.png"
    comparison.save(comparison_path)
    report_path = args.output / f"{backend_name}-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for path in (*sorted(args.output.glob(f"{backend_name}-*.png")), report_path):
        print(path.resolve())
    return 0


def _comparison(captures: list[tuple[ShadowQuality, Image.Image]]) -> Image.Image:
    title_height = 30
    width = sum(image.width for _, image in captures)
    height = max(image.height for _, image in captures) + title_height
    canvas = Image.new("RGB", (width, height), (25, 27, 31))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for quality, image in captures:
        draw.text((x + 10, 8), quality.value, fill=(235, 238, 242))
        canvas.paste(image, (x, title_height))
        x += image.width
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
