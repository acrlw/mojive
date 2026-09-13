"""Compare render products from the same shared contracts on both backends."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mojive import CameraView, Material, RenderProduct, Scene, SceneRenderer, ShadingModel
from mojive.types import TextureData, TextureType

pytestmark = pytest.mark.gpu
BACKENDS = ["opengl", "wgpu"] + (["bgfx"] if os.environ.get("MOJIVE_NATIVE_BUILD") else [])


@pytest.mark.skipif("bgfx" not in BACKENDS, reason="Native build required")
@pytest.mark.parametrize("samples", [0, 2, 4, 8])
def test_bgfx_matches_gl_coverage_at_exact_triangle_sample_boundaries(samples):
    from mojive.types import MeshData

    # Pixel-aligned orthographic coordinates isolate edge ownership from lighting
    # and projection precision. Thin faces reproduce the corpus's MSAA failures.
    triangles = []
    for index, fraction in enumerate((0.125, 0.375, 0.625, 0.875)):
        y = index * 4 + fraction
        triangles.extend(
            [
                [(-24, y, 0), (-24, y - 0.0625, 0), (24, y, 0)],
                [(-24, y + 1, 0), (24, y + 1, 0), (-24, y + 1.0625, 0)],
            ]
        )
    triangles.append([(-24, -20, 0), (24, -20, 0), (-24, -8, 0)])
    vertices = np.array(triangles, np.float32).reshape(-1, 3)
    scene = Scene()
    scene.mesh(
        MeshData(
            vertices,
            np.tile(np.array([0, 0, 1], np.float32), (len(vertices), 1)),
            np.zeros((len(vertices), 2), np.float32),
            np.arange(len(vertices), dtype=np.uint32),
        ),
        size=(1, 1, 1),
        color=(1, 1, 1, 1),
    )
    camera = CameraView(
        eye=np.array([0, 0, 5], np.float32),
        up=np.array([0, 1, 0], np.float32),
        orthographic=True,
        ortho_height=64,
    )
    captures = []
    for backend in ("opengl", "bgfx"):
        with SceneRenderer(
            scene.source, renderer=backend, width=64, height=64, samples=samples, camera=camera
        ) as renderer:
            renderer.update(scene.frame)
            renderer.set_debug_view("albedo")
            captures.append(renderer.render())
    assert np.count_nonzero(captures[0][..., 0] == 255) > 200
    if samples:
        assert np.count_nonzero((captures[0][..., 0] > 0) & (captures[0][..., 0] < 255)) > 20
    np.testing.assert_array_equal(*captures)


@pytest.mark.parametrize("shading", list(ShadingModel))
@pytest.mark.parametrize("compared_backend", BACKENDS[1:])
def test_backends_agree_on_textured_transparent_color_depth_and_identity(shading, compared_backend):
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
    output = Path("output/quality-improvements/backend-parity") / shading.value / compared_backend
    output.mkdir(parents=True, exist_ok=True)
    for backend in ("opengl", compared_backend):
        with SceneRenderer(
            source, width=320, height=240, samples=0, renderer=backend, camera=camera
        ) as renderer:
            renderer.update(scene.frame)
            captures[backend] = {
                product: renderer.render(product=product) for product in RenderProduct
            }
        Image.fromarray(captures[backend][RenderProduct.COLOR]).save(output / f"{backend}.png")
    a, b = captures["opengl"], captures[compared_backend]
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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
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


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("samples", [0, 4])
def test_identity_only_requests_use_current_camera_depth_without_a_color_render(backend, samples):
    scene = Scene()
    scene.box(position=(-0.25, 0.0, 0.5))
    scene.box(position=(0.35, 0.6, 0.5), color=(1.0, 0.2, 0.1, 0.5))
    source = scene.source
    source.geom_segmentation = np.column_stack(
        (source.geom_object_id, np.ones(source.instance_count))
    ).astype(np.int32)
    with SceneRenderer(source, width=128, height=96, samples=samples, renderer=backend) as renderer:
        for eye in ((3.0, -3.0, 2.0), (-3.0, 3.0, 1.0), (4.0, 0.0, 2.0)):
            renderer.update(
                scene.frame, camera=CameraView(eye=np.array(eye), target=np.array((0, 0, 0.5)))
            )
            ids = renderer.render(product=RenderProduct.OBJECT_ID)
            semantics = renderer.render(product=RenderProduct.SEGMENTATION)[..., 0]
            expected = np.maximum(semantics, 0).astype(np.uint32)
            # The ID buffer may be multisampled; compare fully covered pixels
            # against the independent single-sample semantic export.
            interior = np.ones(expected.shape, bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    interior &= expected == np.roll(expected, (dy, dx), axis=(0, 1))
            assert np.count_nonzero(interior & (expected != 0)) > 100
            np.testing.assert_array_equal(ids[interior], expected[interior])


@pytest.mark.parametrize("compared_backend", BACKENDS[1:])
@pytest.mark.parametrize("distance", [2.0, 10.0])
def test_npot_texture_minification_preserves_cross_backend_color(compared_backend, distance):
    pixels = np.full((100, 200, 3), 210, np.uint8)
    y, x = np.indices(pixels.shape[:2])
    pixels[(x // 7 + y // 7) % 2 == 0] = [35, 80, 120]
    scene = Scene()
    scene.add_texture(TextureData("npot", TextureType.TWO_D, pixels, srgb=True))
    scene.plane(size=(20, 20, 0.01), material=Material(texture="npot"))
    camera = CameraView(eye=np.array([distance, -distance, distance * 0.4]), target=np.zeros(3))
    captures = []
    for backend in ("opengl", compared_backend):
        with SceneRenderer(
            scene.source, width=320, height=240, samples=0, renderer=backend, camera=camera
        ) as renderer:
            renderer.update(scene.frame)
            captures.append(renderer.render())
    error = np.abs(captures[0].astype(float) - captures[1])
    assert error.mean() < 0.1
    assert np.percentile(error, 99) <= 1
    assert captures[0].std() > 5, "Minification must retain visible texture detail"


def test_opengl_uploads_complete_linear_light_npot_mips(gl_ctx, require_opengl):
    from mojive.render.opengl.resources import TextureStore
    from mojive.render.texture import mip_chain

    pixels = np.zeros((3, 5, 4), np.uint8)
    pixels[-1] = [255, 180, 90, 30]
    pixels[:, -1] = [80, 255, 200, 240]
    store = TextureStore(gl_ctx)
    try:
        store.sync({"npot": TextureData("npot", TextureType.TWO_D, pixels, srgb=True)})
        texture = store.get("npot")
        for level, expected in enumerate(mip_chain(pixels[None], srgb=True)):
            actual = np.frombuffer(texture.read(level=level, alignment=1), np.uint8)
            np.testing.assert_array_equal(actual, expected.ravel())
    finally:
        store.release()


@pytest.mark.skipif("bgfx" not in BACKENDS, reason="native development build not selected")
@pytest.mark.parametrize("samples", [2, 4, 8])
def test_bgfx_msaa_preserves_color_and_depth_across_overlay_passes(samples):
    scene = Scene()
    scene.plane(size=(3, 3, 0.01))
    scene.box(position=(-0.2, 0, 0.5), color=(0.15, 0.6, 0.8, 1))
    scene.sphere(position=(0.4, -0.3, 0.5), color=(0.9, 0.2, 0.1, 0.4))
    images = {}
    camera = CameraView(eye=np.array([3.2, -3.8, 2.1]), target=np.array([0, 0, 0.4]))
    for backend in ("opengl", "bgfx"):
        with SceneRenderer(
            scene.source, width=193, height=137, samples=samples, renderer=backend, camera=camera
        ) as renderer:
            renderer.update(scene.frame)
            renderer._backend.highlight(int(scene.source.geom_object_id[1]))
            color = renderer.render()
            depth = renderer.render(product=RenderProduct.METRIC_DEPTH)
            np.testing.assert_array_equal(renderer.render(), color)
            images[backend] = color, depth
    a, b = images["opengl"], images["bgfx"]
    error = np.abs(a[0].astype(float) - b[0])
    assert error.mean() < 1
    assert np.percentile(error, 99) <= 5
    np.testing.assert_allclose(a[1], b[1], atol=0.001, rtol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("view", ["segment", "idcolor"])
def test_identity_color_views_preserve_uint32_ids_and_selection(backend, view):
    from mojive.render.backend import DebugView

    scene = Scene()
    scene.box()
    source = scene.source
    selected = 0xFEDCBA98
    source.geom_object_id[:] = selected
    with SceneRenderer(source, width=97, height=73, samples=4, renderer=backend) as renderer:
        renderer.update(scene.frame)
        assert renderer.set_debug_view(DebugView(view))
        renderer._backend.highlight(selected)
        color = renderer.render()
        ids = renderer.render(product=RenderProduct.OBJECT_ID)
        assert np.count_nonzero(ids == selected) > 100
        h = selected * 2654435761 & 0xFFFFFFFF
        expected = [255, 255, 255] if view == "segment" else [h >> 16 & 255, h >> 8 & 255, h & 255]
        np.testing.assert_array_equal(
            color[ids == selected], np.tile(expected, (np.count_nonzero(ids == selected), 1))
        )
        np.testing.assert_array_equal(color[ids == 0], 0)
        # A data read must not replace the cached debug-color result.
        renderer.render(product=RenderProduct.METRIC_DEPTH)
        np.testing.assert_array_equal(renderer.render(), color)
