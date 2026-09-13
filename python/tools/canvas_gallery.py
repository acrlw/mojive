"""Capture the retained Canvas2D primitives on a selected production backend."""

import argparse
import math
from pathlib import Path

from PIL import Image

from mojive import Scene, SceneRenderer
from mojive.canvas2d import Affine2D, PathBuilder2D


def populate(canvas):
    layer = canvas.layer("gallery")
    cyan, orange, green = (0.25, 0.75, 1, 1), (1, 0.55, 0.2, 1), (0.5, 0.85, 0.4, 1)
    layer.rounded_rectangle("round", (0.5, 4), (3.4, 6), 0.35, cyan, filled=True)
    layer.text("round-label", (0.5, 6.4), "Rounded fill / ellipse")
    layer.ellipse("ellipse", (1.95, 5), (1, 0.55), (0.1, 0.25, 0.4, 1), rotation=0.3, filled=True)
    layer.text("path-label", (4.2, 6.4), "Even-odd hole")
    path = PathBuilder2D(fill_rule="evenodd")
    for x0, y0, x1, y1 in ((4.2, 4, 7.1, 6), (4.8, 4.5, 6.5, 5.5)):
        path.move_to(x0, y0).line_to(x1, y0).line_to(x1, y1).line_to(x0, y1).close()
    layer.fill_path("hole", path.finish(), (*green[:3], 0.6))
    layer.text("curve-label", (7.8, 6.4), "Bezier / arc")
    layer.bezier("curve", ((7.8, 4), (8.2, 6.8), (9.6, 3), (10.8, 6)), cyan, 4)
    layer.arc("arc", (9.3, 4.6), (1, 0.6), 0, math.pi, orange, 3)
    layer.text("stroke-label", (0.5, 3.2), "Round / square / butt caps")
    for index, cap in enumerate(("round", "square", "butt")):
        path = (
            PathBuilder2D().move_to(0.8, 2.5 - index * 0.65).line_to(3, 2.5 - index * 0.65).finish()
        )
        layer.stroke_path(f"cap-{cap}", path, cyan, width=0.25, cap=cap)
    layer.text("transform-label", (4.2, 3.2), "Nested transform")
    with layer.transformed(Affine2D.translation(5.6, 1.8) @ Affine2D.rotation(0.5)):
        layer.rectangle("transformed", (-0.9, -0.55), (0.9, 0.55), orange, filled=True)
        layer.arrow("arrow", (-0.9, 0), (0.9, 0), (1, 1, 1, 1), 2)
    layer.text("batch-label", (7.8, 3.2), "Triangle batch / point / text")
    layer.triangles(
        "batch", (((7.8, 1), (8.5, 2.5), (9.2, 1)), ((9.3, 1), (10, 2.5), (10.7, 1))), (cyan, green)
    )
    layer.points("points", ((8.5, 1.5), (10, 1.5)), orange, 5)


def capture(backend, output):
    scene = Scene()
    with SceneRenderer(scene.source, width=1000, height=600, renderer=backend) as renderer:
        canvas = renderer.canvas2d
        populate(canvas)
        renderer.update(
            scene.frame, camera=canvas.camera((0, 0, 11.5, 7), aspect=1000 / 600, padding=0.03)
        )
        pixels = renderer.render()
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        capture(args.backend, args.output or Path(f"output/canvas2d/{args.backend}.png")).resolve()
    )


if __name__ == "__main__":
    main()
