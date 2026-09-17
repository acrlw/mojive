"""Capture material replacement, skybox and editable floor grids through public APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import RenderProduct, Scene, SceneRenderer
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.debugdraw import Occlusion
from mojive.types import CameraView, Material, TextureData, TextureType


def capture(output: Path, renderer: str = "opengl") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    box = scene.box(name="box", position=(0, 0, 0.5), size=(0.5, 0.5, 0.5), color=None)
    floor = scene.plane(name="floor", size=(4, 4, 0.1), color=None)
    box.set_material(Material(name="red", rgba=np.array((0.8, 0.08, 0.04, 1), np.float32)))
    floor.set_material(Material(name="floor", rgba=np.array((0.2, 0.23, 0.26, 1), np.float32)))
    camera = CameraView(
        eye=np.array((3.5, -5.5, 3.5), np.float32),
        target=np.array((0, 0, 0.3), np.float32),
        up=np.array((0, 0, 1), np.float32),
    )
    adapter = StaticSceneAdapter(scene)
    with SceneRenderer(width=800, height=600, camera=camera, renderer=renderer) as view:

        def snapshot(name):
            view.update_from(adapter)
            pixels = view.render()
            Image.fromarray(pixels).save(output / f"{name}.png")
            return pixels.copy()

        before = snapshot("01-original")
        box.set_material(Material(name="blue", rgba=np.array((0.04, 0.35, 0.9, 1), np.float32)))
        tiles = np.full((32, 32, 3), (175, 180, 185), np.uint8)
        tiles[:16, :16] = tiles[16:, 16:] = (70, 78, 86)
        scene.add_texture(TextureData("floor-tiles", TextureType.TWO_D, tiles))
        floor.set_material(
            Material(
                name="tiled-floor",
                rgba=np.ones(4, np.float32),
                texture="floor-tiles",
                tex_repeat=np.array((3, 3), np.float32),
            )
        )
        materials = snapshot("02-materials")
        box_mask = view.render(product=RenderProduct.OBJECT_ID) == box.object_id
        red = before[box_mask].mean(axis=0)
        blue = materials[box_mask].mean(axis=0)
        assert red[0] > red[2] + 30, red
        assert blue[2] > blue[0] + 30, blue
        sky = np.full((6, 32, 32, 3), (55, 95, 145), np.uint8)
        sky[:, :16] = (95, 135, 180)
        scene.add_texture(TextureData("studio-sky", TextureType.SKYBOX, sky))
        assert scene.set_skybox("studio-sky")
        skybox = snapshot("03-skybox")
        grid = view.canvas2d.layer("floor-grid", depth=0.003, occlusion=Occlusion.DEPTH)
        grid.grid("lines", (-4, -4, 4, 4), 0.5, (0.2, 0.9, 0.9, 0.8), 1.5)
        grid_image = snapshot("04-grid")
        # Updating the same retained ID changes spacing, color and width without appending grids.
        grid.grid("lines", (-4, -4, 4, 4), 1, (1, 0.7, 0.1, 1), 2.5)
        view.debug.layer("measurement", Occlusion.DEPTH).arrow(
            "height",
            (1.2, 0, 0),
            (1.2, 0, 1),
            (1, 0.8, 0.1, 1),
            3,
        )
        edited = snapshot("05-grid-edited")
        pairs = (
            (before, materials),
            (materials, skybox),
            (skybox, grid_image),
            (grid_image, edited),
        )
        changes = [int(np.any(a != b, axis=2).sum()) for a, b in pairs]
        assert all(count > 100 for count in changes), changes
        scene.save(output / "materials.scene.json")
    report = {
        "renderer": renderer,
        "changed_pixels": dict(
            zip(
                ("replace_materials", "skybox", "add_grid", "edit_grid_and_add_debug_arrow"),
                changes,
                strict=True,
            )
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/material-workflow"))
    parser.add_argument("--renderer", choices=("opengl", "bgfx"), default="opengl")
    args = parser.parse_args()
    print(json.dumps(capture(args.output, args.renderer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
