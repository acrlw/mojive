"""Real GPU acceptance of the private Python/native vertical integration."""

from __future__ import annotations

import gc
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

OUTPUT = Path("output/cpp-python/bgfx")


@pytest.fixture
def runtime(native):
    assert native.has_renderer
    with native.RenderRuntime(os.environ["MOJIVE_NATIVE_SHADER_DIR"]) as renderer:
        yield renderer


def fixture_scene(native):
    source = native.SceneSource()
    points = np.array([[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]], np.float32)
    source.add_mesh(points, np.tile([0, 0, 1], (4, 1)), np.array([0, 1, 2, 0, 2, 3], np.uint32))
    source.set_instances(
        np.zeros(3, np.uint32),
        np.array([0x01000001, 0xFEDCBA98, 0xFFFFFFFF], np.uint32),
        np.array([[-1, 0x1000001], [-0x80000000, 0x7FFFFFFF], [-23456789, 42]], np.int32),
        np.array([[1, 0.2, 0.1, 1], [0.1, 1, 0.2, 1], [0.2, 0.3, 1, 1]], np.float32),
    )
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    transforms[:, :2, 3] = [[-1, 0.7], [1, 0.7], [0, -1]]
    camera = native.CameraView()
    camera.view = native.look_at([0, 0, 5], [0, 0, 0], [0, 1, 0])
    camera.projection = native.orthographic(4, 4, 0.1, 20)
    camera.far_plane = 20
    camera.revision = 9
    return source, transforms, camera


def test_render_products_own_data_and_preserve_existing_conventions(native, runtime):
    source, transforms, camera = fixture_scene(native)
    runtime.set_scene(source)
    # Noncontiguous and Fortran arrays retain their mathematical meaning.
    padded = np.zeros((3, 4, 8), np.float32)
    padded[:, :, ::2] = transforms
    runtime.update(padded[:, :, ::2], sequence=7)
    padded[:] = 0
    source.set_instances(
        np.empty(0, np.uint32),
        np.empty(0, np.uint32),
        np.empty((0, 2), np.int32),
        np.empty((0, 4), np.float32),
    )
    target = runtime.create_target(256, 256, samples=4)
    frame = runtime.render(target, camera)
    assert frame.sequence == 7 and frame.camera_revision == 9
    products = [
        native.Product.COLOR,
        native.Product.OBJECT_ID,
        native.Product.SEGMENTATION,
        native.Product.METRIC_DEPTH,
    ]
    tickets = [runtime.readback(frame, product) for product in products]
    results = [runtime.wait(ticket) for ticket in tickets]
    assert all(result.state == native.ReadbackState.READY for result in results)
    color, ids, segments, depth = [result.image for result in results]
    assert color.shape == (256, 256, 3) and color.dtype == np.uint8
    assert ids.dtype == np.uint32 and segments.dtype == np.int32 and depth.dtype == np.float32
    assert all(array.flags.c_contiguous for array in (color, ids, segments, depth))
    for (x, y), object_id, pair in zip(
        [(64, 83), (192, 83), (128, 192)],
        [0x01000001, 0xFEDCBA98, 0xFFFFFFFF],
        [[-1, 0x1000001], [-0x80000000, 0x7FFFFFFF], [-23456789, 42]],
        strict=True,
    ):
        assert ids[y, x] == object_id
        np.testing.assert_array_equal(segments[y, x], pair)
        assert depth[y, x] == pytest.approx(5, abs=1e-4)
    assert ids[0, 0] == 0 and depth[0, 0] == pytest.approx(20)
    np.testing.assert_array_equal(segments[0, 0], [-1, -1])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(OUTPUT / "products.png")
    pending = runtime.readback(frame, native.Product.COLOR)
    runtime.resize(target, 193, 137)
    canceled = runtime.wait(pending)
    assert canceled.state == native.ReadbackState.CANCELED and canceled.image is None
    with pytest.raises(ValueError):
        runtime.readback(frame, native.Product.COLOR)
    runtime.update(np.asfortranarray(transforms), sequence=8)
    fresh = runtime.render(target, camera)
    assert runtime.wait(runtime.readback(fresh, native.Product.COLOR)).image.shape == (137, 193, 3)
    runtime.close()
    del results, source, padded
    gc.collect()
    # Output ownership does not depend on the runtime, frame, or result wrappers.
    assert ids[83, 192] == 0xFEDCBA98


def test_owner_thread_peer_targets_errors_and_concurrent_close(native, runtime):
    source, transforms, camera = fixture_scene(native)
    runtime.set_scene(source)
    runtime.update(transforms)
    with pytest.raises(ValueError):
        runtime.update(transforms, source_revision=999)
    with pytest.raises(ValueError):
        runtime.create_target(0, 100)
    camera.far_plane = float("nan")
    target = runtime.create_target(16, 16)
    with pytest.raises(ValueError, match="far plane"):
        runtime.render(target, camera)
    runtime.destroy(target)
    camera.far_plane = 20
    with ThreadPoolExecutor(4) as pool:
        targets = list(pool.map(lambda _: runtime.create_target(128, 128), range(4)))
        frames = list(pool.map(lambda target: runtime.render(target, camera), targets))
        tickets = [runtime.readback(frame, native.Product.OBJECT_ID) for frame in frames]
        outputs = list(pool.map(runtime.wait, tickets))
        assert all(output.state == native.ReadbackState.READY for output in outputs)
        runtime.destroy(targets.pop())
        for target in targets:
            frame = runtime.render(target, camera)
            assert runtime.wait(runtime.readback(frame, native.Product.COLOR)).image.shape == (
                128,
                128,
                3,
            )
        list(pool.map(lambda _: runtime.close(), range(4)))
    assert runtime.closed
    records = runtime.log.read().records
    assert [record.message for record in records] == [
        "Render owner started",
        "Render owner stopped",
    ]
    assert records[0].thread_id == records[1].thread_id
    assert all(record.origin == native.LogOrigin.NATIVE for record in records)
    with pytest.raises(RuntimeError, match="closed"):
        runtime.advance()


def test_failed_initialization_and_automatic_teardown(native, tmp_path):
    with pytest.raises(RuntimeError, match="shader"):
        native.RenderRuntime(str(tmp_path))
    # Failed construction releases the process-wide device reservation.
    with native.RenderRuntime(os.environ["MOJIVE_NATIVE_SHADER_DIR"]) as runtime:
        assert runtime.capabilities.readback
        with pytest.raises(RuntimeError, match="Only one"):
            native.RenderRuntime(os.environ["MOJIVE_NATIVE_SHADER_DIR"])
        assert not runtime.closed
    script = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('mojive._native', {os.environ["MOJIVE_NATIVE_TEST_MODULE"]!r})
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
runtime = native.RenderRuntime({os.environ["MOJIVE_NATIVE_SHADER_DIR"]!r})
target = runtime.create_target(64, 64)
# Interpreter shutdown must destroy GPU resources on their owner thread.
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)


def test_official_hundred_humanoids_through_python(native, runtime):
    path = Path(os.environ["MOJIVE_NATIVE_SCENE"])
    with path.open("rb") as stream:
        assert stream.read(8) == b"MJVPROB1"
        mesh_count, instance_count, frame_count = struct.unpack("<III", stream.read(12))
        camera = native.CameraView()
        camera.view = np.frombuffer(stream.read(64), "<f4").reshape(4, 4)
        camera.projection = np.frombuffer(stream.read(64), "<f4").reshape(4, 4)
        camera.far_plane = struct.unpack("<f", stream.read(4))[0]
        source = native.SceneSource()
        for _ in range(mesh_count):
            vertices, indices = struct.unpack("<II", stream.read(8))
            data = np.frombuffer(stream.read(vertices * 24), "<f4").reshape(-1, 6)
            source.add_mesh(
                data[:, :3], data[:, 3:], np.frombuffer(stream.read(indices * 4), "<u4")
            )
        instances = np.frombuffer(
            stream.read(instance_count * 32),
            np.dtype(
                [
                    ("mesh", "<u4"),
                    ("id", "<u4"),
                    ("seg", "<i4", 2),
                    ("color", "<f4", 4),
                ]
            ),
        )
        source.set_instances(
            *[np.ascontiguousarray(instances[key]) for key in ["mesh", "id", "seg", "color"]]
        )
        assert instance_count >= 5000 and frame_count >= 12
        runtime.set_scene(source)
        target = runtime.create_target(640, 360)
        for sequence in range(12):
            transforms = np.frombuffer(stream.read(instance_count * 64), "<f4").reshape(-1, 4, 4)
            runtime.update(transforms, sequence=sequence)
            token = runtime.render(target, camera)
            runtime.advance()
        ids = runtime.wait(runtime.readback(token, native.Product.OBJECT_ID)).image
        color = runtime.wait(runtime.readback(token, native.Product.COLOR)).image
        assert len(np.unique(ids)) > 100
        assert np.isin(ids, np.append(instances["id"], np.uint32(0))).all()
        assert np.std(color) > 10
        OUTPUT.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(OUTPUT / "humanoids100.png")
        (OUTPUT / "acceptance.json").write_text(
            json.dumps(
                {
                    "backend": runtime.capabilities.backend,
                    "instances": instance_count,
                    "frames": 12,
                    "visible_object_ids": len(np.unique(ids)),
                    "scope": "Private Python/native integration; production material and shadow parity pending",
                },
                indent=2,
            )
            + "\n"
        )


def test_texture_orientation_and_incremental_colors_preserve_submitted_readback(native, runtime):
    source = native.SceneSource()
    positions = np.array(
        [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0.8, 0.8, 0], [-0.8, 0.8, 0]], np.float32
    )
    source.add_mesh(positions, np.tile([0, 0, 1], (4, 1)), np.array([0, 1, 2, 0, 2, 3], np.uint32))
    source.set_mesh_texcoords(0, np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32))
    pixels = np.array(
        [[[255, 0, 0, 255], [0, 255, 0, 255]], [[0, 0, 255, 255], [255, 255, 255, 255]]], np.uint8
    )
    material = native.Material()
    material.texture = source.add_texture(pixels)
    material.specular = 0
    source.materials = [material]
    source.set_instances(
        np.array([0], np.uint32),
        np.array([7], np.uint32),
        np.array([[3, 5]], np.int32),
        np.ones((1, 4), np.float32),
    )
    source.set_material_indices(np.array([0], np.uint32))
    scene = runtime.create_scene(source)
    # GPU upload must own immutable pixels after the Python inputs are gone.
    pixels[:] = 0
    del pixels, source
    transform = np.eye(4, dtype=np.float32)[None]
    uv = np.array([[1, 1, 0, 0]], np.float32)
    runtime.update_textured(scene, transform, uv, np.ones((1, 4), np.float32), 1, 0)
    camera = native.CameraView()
    camera.view = native.look_at([0, 0, 5], [0, 0, 0], [0, 1, 0])
    camera.projection = native.orthographic(2, 2, 0.1, 20)
    camera.far_plane = 20
    target = runtime.create_target(64, 64, scene=scene)
    token = runtime.render(target, camera)
    ticket = runtime.readback(token, native.Product.COLOR)
    runtime.update_textured(scene, transform, uv, np.array([[0, 0, 0, 1]], np.float32), 1, 1)
    black = runtime.render(target, camera)
    first = runtime.wait(ticket)
    assert first.state == native.ReadbackState.READY
    color = first.image
    assert color[20, 20, 0] > max(color[20, 20, 1:]) + 100
    assert color[20, 44, 1] > max(color[20, 44, [0, 2]]) + 100
    assert color[44, 20, 2] > max(color[44, 20, :2]) + 100
    assert runtime.wait(runtime.readback(black, native.Product.COLOR)).image[32, 32].max() == 0
    assert runtime.wait(runtime.readback(black, native.Product.OBJECT_ID)).image[32, 32] == 7


def test_render_request_prunes_color_and_rejects_unavailable_products(native, runtime):
    source, transforms, camera = fixture_scene(native)
    runtime.set_scene(source)
    runtime.update(transforms)
    target = runtime.create_target(64, 64)
    runtime.advance()
    frame = runtime.render(target, camera, color=False, scene_data=True)
    stats = runtime.advance()
    assert stats.draw_calls == 1
    with pytest.raises(ValueError, match="unavailable"):
        runtime.readback(frame, native.Product.COLOR)
    products = (native.Product.OBJECT_ID, native.Product.SEGMENTATION, native.Product.METRIC_DEPTH)
    references = {
        product: runtime.wait(runtime.readback(frame, product)).image.copy() for product in products
    }
    for product in products:
        target = runtime.create_target(64, 64)
        frame = runtime.render(target, camera, data_product=product)
        np.testing.assert_array_equal(
            runtime.wait(runtime.readback(frame, product)).image, references[product]
        )
        for other in products:
            if other != product:
                with pytest.raises(ValueError, match="unavailable"):
                    runtime.readback(frame, other)
    with pytest.raises(ValueError, match="Invalid scene data product"):
        runtime.render(target, camera, data_product=native.Product.COLOR)
    # Returning to the full product set must populate the previously omitted attachments.
    frame = runtime.render(target, camera)
    assert runtime.wait(runtime.readback(frame, native.Product.METRIC_DEPTH)).image.min() < 20
    frame = runtime.render(target, camera, color=True, scene_data=False)
    with pytest.raises(ValueError, match="unavailable"):
        runtime.readback(frame, native.Product.METRIC_DEPTH)
    assert runtime.wait(runtime.readback(frame, native.Product.COLOR)).image.std() > 10
    with pytest.raises(ValueError, match="Empty"):
        runtime.render(target, camera, color=False, scene_data=False)


def test_invalid_native_visual_updates_preserve_the_scene(native, runtime):
    source, transforms, camera = fixture_scene(native)
    runtime.set_scene(source)
    runtime.update(transforms)
    target = runtime.create_target(64, 64)
    expected = runtime.wait(
        runtime.readback(runtime.render(target, camera), native.Product.COLOR)
    ).image
    lighting = native.Lighting()
    lighting.image_texture = 0
    with pytest.raises(ValueError, match="cube texture"):
        runtime.set_lighting(native.Scene(), lighting)
    style = native.SceneStyle()
    style.shadow_quality = 5
    with pytest.raises(ValueError, match="render mode"):
        runtime.configure(native.Scene(), style)
    packet = native.OverlayFrame()
    surface = native.SurfaceBatch()
    surface.count = 1
    packet.surface_batches = [surface]
    with pytest.raises(ValueError, match="surface batch"):
        runtime.set_overlays(native.Scene(), packet)
    actual = runtime.wait(
        runtime.readback(runtime.render(target, camera), native.Product.COLOR)
    ).image
    np.testing.assert_array_equal(expected, actual)
