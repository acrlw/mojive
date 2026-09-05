"""Compare render products from the same shared contracts on both backends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import CameraView, Material, RenderProduct, Scene, SceneRenderer, ShadingModel
from mojive.types import TextureData, TextureType

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("shading", list(ShadingModel))
def test_backends_agree_on_textured_transparent_color_depth_and_identity(shading):
    scene = Scene()
    pixels = np.full((8, 8, 3), 220, np.uint8)
    pixels[::2, ::2] = [45, 90, 180]
    scene.add_texture(TextureData("checker", TextureType.TWO_D, pixels))
    scene.plane(size=(3, 3, 0.01))
    scene.box(position=(0, -0.6, 0.5), material=Material(texture="checker"))
    scene.sphere(position=(0, 0.6, 0.5), color=(0.9, 0.2, 0.1, 0.45))
    source = scene.source
    source.shading_model = shading
    source.geom_segmentation = np.column_stack(
        (
            np.arange(source.instance_count, dtype=np.int32) + 100,
            np.full(source.instance_count, 7, np.int32),
        )
    )
    camera = CameraView(eye=np.array([4, -4, 3]), target=np.array([0, 0, 0.4]))
    captures = {}
    output = Path("output/quality-improvements/backend-parity") / shading.value
    output.mkdir(parents=True, exist_ok=True)
    for backend in ("opengl", "wgpu"):
        with SceneRenderer(
            source, width=320, height=240, samples=0, renderer=backend, camera=camera
        ) as renderer:
            renderer.update(scene.frame)
            captures[backend] = {
                product: renderer.render(product=product) for product in RenderProduct
            }
        Image.fromarray(captures[backend][RenderProduct.COLOR]).save(output / f"{backend}.png")
    a, b = captures["opengl"], captures["wgpu"]
    color_difference = np.abs(a[RenderProduct.COLOR].astype(float) - b[RenderProduct.COLOR])
    ids_a, ids_b = a[RenderProduct.OBJECT_ID], b[RenderProduct.OBJECT_ID]
    shared = (ids_a == ids_b) & (ids_a != 0)
    depth_difference = np.abs(
        a[RenderProduct.METRIC_DEPTH][shared] - b[RenderProduct.METRIC_DEPTH][shared]
    )
    metrics = {
        "color_mean": float(color_difference.mean()),
        "color_p99": float(np.percentile(color_difference, 99)),
        "id_disagreement": float(np.mean(ids_a != ids_b)),
        "depth_p99": float(np.percentile(depth_difference, 99)),
        "segmentation_disagreement": float(
            np.mean(np.any(a[RenderProduct.SEGMENTATION] != b[RenderProduct.SEGMENTATION], axis=2))
        ),
    }
    (output / "report.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Identical geometry should differ only at rasterization/texture-filter edges.
    # Tolerances allow one display level on average and 0.1% identity edge pixels.
    assert shared.sum() > 5000
    assert metrics["color_mean"] < 1.0, metrics
    assert metrics["color_p99"] <= 5.0, metrics
    assert metrics["id_disagreement"] < 0.001, metrics
    assert metrics["depth_p99"] < 1e-4, metrics
    assert metrics["segmentation_disagreement"] < 0.001, metrics


@pytest.mark.parametrize("backend", ["opengl", "wgpu"])
def test_classic_lighting_preserves_gray_texture_under_saturated_lights(backend):
    from mojive import Light, LightSet

    scene = Scene(
        lights=LightSet(
            ambient=np.ones(3), headlight=Light(diffuse=np.ones(3), specular=np.zeros(3))
        )
    )
    scene.add_texture(TextureData("gray", TextureType.TWO_D, np.full((4, 4, 3), 128, np.uint8)))
    scene.box(color=(1, 1, 1, 1), material=Material(rgba=np.ones(4), texture="gray", specular=0))
    scene.source.shading_model = ShadingModel.MUJOCO_CLASSIC
    with SceneRenderer(scene.source, width=64, height=48, samples=0, renderer=backend) as renderer:
        renderer.update(scene.frame)
        rgb = renderer.render()
    np.testing.assert_allclose(rgb[24, 32], [128, 128, 128], atol=1)


@pytest.mark.parametrize("backend", ["opengl", "wgpu"])
def test_classic_generated_texture_coordinates_keep_object_xy_orientation(backend):
    from mojive import LightSet

    scene = Scene(lights=LightSet(ambient=np.ones(3)))
    colors = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], np.uint8)
    scene.add_texture(
        TextureData(
            "quadrants", TextureType.TWO_D, np.repeat(np.repeat(colors, 8, axis=0), 8, axis=1)
        )
    )
    scene.box(
        size=(1, 1, 0.1),
        color=(1, 1, 1, 1),
        material=Material(rgba=np.ones(4), texture="quadrants", specular=0),
    )
    scene.source.shading_model = ShadingModel.MUJOCO_CLASSIC
    camera = CameraView(
        eye=np.array([0, 0, 4]), up=np.array([0, 1, 0]), orthographic=True, ortho_height=4
    )
    with SceneRenderer(
        scene.source, width=96, height=96, samples=0, renderer=backend, camera=camera
    ) as renderer:
        renderer.update(scene.frame)
        rgb = renderer.render()
    # Object X maps left-to-right, object Y maps to the reverse texture row.
    actual = rgb[np.array([36, 36, 60, 60]), np.array([36, 60, 36, 60])]
    np.testing.assert_allclose(actual, colors.reshape(4, 3), atol=1)
