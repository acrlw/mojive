"""Matched native/OpenGL/WebGPU product captures and numerical acceptance."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from mojive import (
    CameraView,
    Light,
    LightSet,
    Material,
    RenderFlag,
    RenderProduct,
    Scene,
    SceneRenderer,
    ShadingModel,
)
from mojive.interaction.gizmo import GizmoFrame, GizmoMode
from mojive.render.debugdraw import Occlusion
from mojive.types import LightType, TextureData, TextureType


def cases():
    for shading in ShadingModel:
        for variant in (
            "tendon",
            "tendon-transparent",
            "tendon-texture",
            "tendon-island",
            "ambient",
            "headlight",
            "directional",
            "point",
            "spot",
            "texture",
            "texture-r",
            "texture-rg",
            "texture-rgba",
            "transparent",
            "materials",
            "view-albedo",
            "view-normal",
            "view-depth",
            "view-overdraw",
            "view-wireframe",
            "debug-line",
            "debug-arrow",
            "debug-point",
            "debug-stroke",
            "debug-solid",
            "debug-sector",
            "debug-drag",
            "debug-screen",
            "debug-text",
            "gizmo-translate",
            "gizmo-rotate",
            "selection",
            "selection-xray",
            "reflection",
            "reflection-multiple",
            "reflection-transparent",
            "cube",
            "image",
            "image-missing",
            "sky",
            "horizon",
            "shadow-directional-low",
            "shadow-directional-high",
            "shadow-directional",
            "shadow-directional-oblique",
            "shadow-point",
            "shadow-spot",
            "shadow-area",
        ):
            lights = LightSet(ambient=np.array([0.2, 0.25, 0.3]))
            light = Light(
                direction=np.array([-0.4, 0.5, -1]),
                diffuse=np.array([0.85, 0.65, 0.5]),
                specular=np.array([0.6, 0.8, 0.9]),
            )
            if variant == "headlight":
                lights = replace(lights, headlight=light)
            elif variant in {"directional", "point", "spot"}:
                light = replace(light, cast_shadow=False)
                if variant != "directional":
                    light = replace(
                        light,
                        type=LightType.POINT if variant == "point" else LightType.SPOT,
                        position=np.array([0, -1, 3]),
                        direction=np.array([0, 0.4, -1]),
                        attenuation=np.array([1, 0.1, 0.05]),
                        cutoff=55,
                    )
                lights = replace(lights, lights=(light,))
            elif variant not in {"ambient", "texture"}:
                lights = replace(lights, headlight=light)
            if variant.startswith("shadow-"):
                kind = variant.removeprefix("shadow-").split("-")[0].upper()
                lights = replace(
                    lights,
                    headlight=None,
                    lights=(
                        replace(
                            light,
                            type=LightType[kind],
                            position=np.array([0, -1, 3]),
                            direction=np.array(
                                [-0.35, 0.45, -1] if variant.endswith("oblique") else [0, 0.4, -1]
                            ),
                            cutoff=55,
                            area_radius=0.25 if kind == "AREA" else 0,
                            cast_shadow=True,
                        ),
                    ),
                )
            cube = None
            if variant in {"cube", "image", "image-missing", "sky", "horizon"}:
                colors = np.array(
                    [
                        [180, 70, 30],
                        [40, 160, 70],
                        [40, 80, 180],
                        [180, 160, 40],
                        [140, 50, 160],
                        [30, 140, 160],
                    ],
                    np.uint8,
                )
                pixels = np.broadcast_to(colors[:, None, None], (6, 32, 32, 3)).copy()
                pixels[:, :, :16] = (pixels[:, :, :16].astype(float) * 0.65).astype(np.uint8)
                cube = TextureData("environment", TextureType.CUBE, pixels)
                if variant.startswith("image"):
                    lights = replace(
                        lights,
                        lights=(
                            Light(
                                type=LightType.IMAGE,
                                texture="missing" if variant == "image-missing" else "environment",
                                intensity=4000,
                            ),
                        ),
                    )
                if variant == "horizon":
                    lights = replace(
                        lights,
                        horizon_haze=True,
                        haze_density=0.3,
                        haze_color=np.array([0.5, 0.6, 0.7]),
                    )
            scene = Scene(lights=lights)
            if cube is not None:
                scene.add_texture(cube)
            scene.plane(
                size=(0, 0, 0.01) if variant == "horizon" else (3, 3, 0.01),
                color=(0.4, 0.45, 0.5, 1),
                material=Material(reflectance=0.6 if "reflection" in variant else 0),
            )
            if variant == "reflection-multiple":
                scene.box(
                    position=(1.2, 1.2, 0.25),
                    size=(0.6, 0.6, 0.25),
                    material=Material(reflectance=0.5),
                )
            mat = Material(specular=0.3, shininess=0.4)
            if variant.startswith("texture"):
                rgba = np.array(
                    [[[240, 60, 30], [40, 200, 80]], [[20, 60, 230], [200, 190, 30]]], np.uint8
                )
                if variant == "texture-r":
                    rgba = rgba[..., :1]
                elif variant == "texture-rg":
                    rgba = rgba[..., :2]
                elif variant == "texture-rgba":
                    rgba = np.concatenate((rgba, np.full((2, 2, 1), 180, np.uint8)), axis=2)
                scene.add_texture(
                    TextureData(
                        "quadrants", TextureType.TWO_D, np.repeat(np.repeat(rgba, 16, 0), 16, 1)
                    )
                )
                mat = Material(texture="quadrants", specular=0, tex_uniform=True)
            scene.box(
                position=(-0.7, 0, 0.5),
                size=(0.45, 0.5, 0.5),
                color=(0.8, 0.35, 0.2, 1),
                material=mat,
            )
            scene.sphere(
                position=(0.65, 0, 0.55),
                size=(0.5, 0.5, 0.5),
                color=(0.25, 0.65, 0.9, 0.4 if "transparent" in variant else 1),
                material=Material(
                    emission=0.25 if variant == "materials" else 0, specular=0.8, shininess=0.8
                ),
            )
            if "transparent" in variant:
                scene.box(position=(0.7, 0.6, 0.7), color=(0.9, 0.2, 0.7, 0.5))
            if variant == "cube":
                scene.box(position=(0, -0.8, 0.4), material=Material(texture="environment"))
            source = scene.source
            if variant in {"sky", "horizon"}:
                source.skybox = "environment"
            if variant == "horizon":
                source.geom_infinite_plane[0] = True
            frame = scene.frame
            if variant.startswith("tendon"):
                source.tendon_visible = np.array([True, True])
                source.tendon_rgba = np.array(
                    [
                        [0.95, 0.85, 0.15, 0.5 if "transparent" in variant else 1],
                        [0.2, 0.85, 0.9, 1],
                    ],
                    np.float32,
                )
                source.tendon_material = np.array([0, 0], np.int32)
                frame.tendon_segments = np.array(
                    [[[-1, -0.5, 0.2], [0, -0.5, 1.3]], [[0, -0.5, 1.3], [1, -0.5, 0.2]]],
                    np.float32,
                )
                frame.tendon_ids = np.array([0, 1], np.int32)
                frame.tendon_widths = np.array([0.035, 0.045], np.float32)
                frame.tendon_island_rgba = np.array(
                    [[0.1, 0.8, 0.3, 1], [0.8, 0.1, 0.6, 1]], np.float32
                )
                if "texture" in variant:
                    pixels = np.zeros((16, 16, 3), np.uint8)
                    pixels[:8] = [230, 120, 10]
                    pixels[8:] = [10, 160, 230]
                    source.textures["tendon"] = TextureData("tendon", TextureType.TWO_D, pixels)
                    source.materials = (
                        *source.materials,
                        Material(texture="tendon", tex_repeat=(3, 3)),
                    )
                    source.tendon_material[:] = len(source.materials) - 1
            source.shading_model = shading
            source.geom_segmentation = np.column_stack(
                (source.geom_object_id, np.full(source.instance_count, 7))
            ).astype(np.int32)
            camera = CameraView(eye=np.array([4, -4, 3]), target=np.array([0, 0, 0.4]))
            if variant == "horizon":
                camera = replace(camera, eye=np.array([4, -4, 1.4]), target=np.array([0, 0, 1.4]))
            yield f"{shading.value}-{variant}", source, frame, camera


def debug_scene(backend, name):
    kind = name.rsplit("debug-", 1)[1]
    for index, occlusion in enumerate(Occlusion):
        layer = backend.debug.layer(f"parity-{index}", occlusion)
        a = np.array([-0.8, -0.4 + 0.4 * index, 0.2])
        b = np.add(a, [1.5, 0.2, 0.6])
        color = (0.9, 0.3 + index * 0.2, 0.2, 0.8)
        if kind == "line":
            layer.line("item", a, b, color, 4)
        elif kind == "arrow":
            layer.arrow("item", a, b, color, 4)
        elif kind == "point":
            layer.point("item", np.add(a, [0.8, 0, 0.4]), color, 14)
        elif kind == "stroke":
            layer.polyline("item", [a, np.add(a, [0.6, -0.3, 0.4]), b], color, 5)
        elif kind == "solid":
            model = np.eye(4, dtype=np.float32)
            model[:3, :3] *= 0.3
            model[:3, 3] = np.add(a, [0.6, 0, 0.4])
            layer.sphere("item", model, color)
        elif kind == "sector":
            layer.sector(
                "item", a, np.add(a, [0, 0, 1.8]), np.add(a, [1, 0, 0]), color, radius_px=42
            )
        elif kind == "drag":
            layer.drag_link("item", a, b, color, (0.1, 0.1, 0.1, 1), width_px=3, radius_px=10)
        elif kind == "screen" and occlusion is Occlusion.ALWAYS:
            layer.arrow_2d(
                "item", (50, 40), (240, 60), color, width_px=8, head_width_px=25, head_length_px=28
            )
        elif kind == "text":
            layer.text("item", a, f"Native {index}", color, offset_px=(0, -10))


def compare(a, b, *, infinite_ground=False):
    diff = np.abs(a[RenderProduct.COLOR].astype(float) - b[RenderProduct.COLOR])
    ids_a, ids_b = a[RenderProduct.OBJECT_ID], b[RenderProduct.OBJECT_ID]
    shared = (ids_a == ids_b) & (ids_a != 0)
    if infinite_ground:
        shared &= ids_a != 1
    depth = np.abs(a[RenderProduct.METRIC_DEPTH][shared] - b[RenderProduct.METRIC_DEPTH][shared])
    return {
        "color_mean": float(diff.mean()),
        "color_p99": float(np.percentile(diff, 99)),
        "id_disagreement": float(np.mean(ids_a != ids_b)),
        "depth_p99": float(np.percentile(depth, 99)) if depth.size else None,
        "segmentation_disagreement": float(
            np.mean(np.any(a[RenderProduct.SEGMENTATION] != b[RenderProduct.SEGMENTATION], axis=2))
        ),
    }


def ground_depth_error(capture, camera):
    """Bound distant plane interpolation against analytic rays in pixel space.

    Rasterizers snap triangle vertices to a subpixel grid. Near the horizon,
    a subpixel displacement becomes a large absolute distance, so compare the
    implied pixel coordinate as well as reporting the metric depth error.
    """
    depth = capture[RenderProduct.METRIC_DEPTH].astype(float)
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    camera = camera.with_aspect(w / h)
    view_inverse = np.linalg.inv(camera.view_matrix().astype(float))
    projection = camera.proj_matrix().astype(float)
    a = view_inverse[2, 0] * (2 * (x + 0.5) / w - 1) / projection[0, 0] - view_inverse[2, 2]
    b = view_inverse[2, 1] / projection[1, 1]
    expected = -view_inverse[2, 3] / (a + b * (1 - 2 * (y + 0.5) / h))
    mask = (capture[RenderProduct.OBJECT_ID] == 1) & (expected > 0) & (expected < camera.far)
    implied_y = (1 - (-view_inverse[2, 3] / np.maximum(depth, 1e-9) - a) / b) * h / 2
    return {
        "ground_depth_absolute_p99": float(np.percentile(np.abs(depth[mask] - expected[mask]), 99)),
        "ground_depth_subpixel_max": float(np.abs(implied_y[mask] - (y[mask] + 0.5)).max()),
    }


def run(output, selected="", check=False):
    report = {}
    for name, source, frame, camera in cases():
        if selected and selected not in name:
            continue
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        captures = {}
        baselines = {}
        for backend in ("opengl", "wgpu", "bgfx"):
            with SceneRenderer(
                source, width=320, height=240, samples=0, renderer=backend, camera=camera
            ) as renderer:
                renderer.set_flag(RenderFlag.SHADOW, "shadow-" in name)
                renderer.set_flag(RenderFlag.REFLECTION, "reflection" in name)
                if name.endswith("-low"):
                    renderer._backend.set_shadow_quality("performance")
                if name.endswith("-high"):
                    renderer._backend.set_shadow_quality("high")
                if "island" in name:
                    renderer.set_flag(RenderFlag.ISLAND, True)
                renderer.update(frame)
                feature_flag = next(
                    (
                        flag
                        for token, flag in (
                            ("shadow-", RenderFlag.SHADOW),
                            ("reflection", RenderFlag.REFLECTION),
                            ("tendon", RenderFlag.TENDON),
                            ("horizon", RenderFlag.HAZE),
                            ("-sky", RenderFlag.SKYBOX),
                        )
                        if token in name
                    ),
                    None,
                )
                feature = feature_flag is not None or any(
                    token in name for token in ("debug-", "gizmo", "selection", "view-")
                )
                if feature:
                    if feature_flag is not None:
                        renderer.set_flag(feature_flag, False)
                        renderer.update(frame)
                    baselines[backend] = renderer.render(product=RenderProduct.COLOR)
                    if feature_flag is not None:
                        renderer.set_flag(feature_flag, True)
                        renderer.update(frame)
                if "view-" in name:
                    renderer.set_debug_view(name.rsplit("view-", 1)[1])
                if "debug-" in name:
                    debug_scene(renderer._backend, name)
                if "gizmo" in name:
                    renderer._backend.set_gizmo(
                        GizmoFrame(
                            mode=GizmoMode.ROTATE if "rotate" in name else GizmoMode.TRANSLATE,
                            position=np.array([0, 0, 0.7]),
                            rotation=np.eye(3, dtype=np.float32),
                        )
                    )
                if "selection" in name:
                    renderer._backend.highlight(2, xray="xray" in name)
                captures[backend] = {
                    product: renderer.render(product=product) for product in RenderProduct
                }
            Image.fromarray(captures[backend][RenderProduct.COLOR]).save(
                directory / f"{backend}.png"
            )
        metrics = {
            backend: compare(
                captures["opengl"], captures[backend], infinite_ground="horizon" in name
            )
            for backend in ("wgpu", "bgfx")
        }
        if "horizon" in name:
            for backend, result in metrics.items():
                result.update(ground_depth_error(captures[backend], camera))
        if baselines:
            reference = captures["opengl"][RenderProduct.COLOR].astype(float) - baselines["opengl"]
            mask = np.max(np.abs(reference), axis=2) > 5
            for backend, result in metrics.items():
                effect = captures[backend][RenderProduct.COLOR].astype(float) - baselines[backend]
                result["effect_pixels"] = int(mask.sum())
                result["effect_relative_error"] = (
                    float(
                        np.abs(effect[mask] - reference[mask]).mean()
                        / max(1, np.abs(reference[mask]).mean())
                    )
                    if mask.any()
                    else None
                )
        report[name] = metrics
        comparison = np.concatenate(
            [captures[backend][RenderProduct.COLOR] for backend in ("opengl", "wgpu", "bgfx")],
            axis=1,
        )
        Image.fromarray(comparison).save(directory / "comparison.png")
        print(name, json.dumps(metrics), flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if check:
        failures = {
            name: result
            for name, rows in report.items()
            for result in (rows["bgfx"],)
            if result["color_mean"] >= 1
            or result["color_p99"] > 5
            or result["id_disagreement"] >= 0.001
            or result["depth_p99"] is None
            or result["depth_p99"] >= 1e-4
            or result.get("ground_depth_subpixel_max", 0) > 1 / 256
            or result["segmentation_disagreement"] >= 0.001
            or (
                "effect_pixels" in result
                and (
                    result["effect_pixels"] < 8
                    or result["effect_relative_error"] is None
                    or result["effect_relative_error"] > 0.15
                )
            )
        }
        if failures:
            raise SystemExit(f"Native parity failed: {json.dumps(failures)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/native-parity"))
    parser.add_argument("--case", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.output, args.case, args.check)


if __name__ == "__main__":
    main()
