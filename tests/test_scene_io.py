from __future__ import annotations

import json

import numpy as np
import pytest

from forge_viewer.adapters.base import LIGHT_OBJECT_BASE
from forge_viewer.render.builder import SceneSourceBuilder
from forge_viewer.scene import Scene
from forge_viewer.types import (
    CameraView,
    Environment,
    Light,
    LightType,
    Material,
    MeshData,
    TextureData,
    TextureType,
)

pytestmark = pytest.mark.integration


def test_forge_scene_round_trip_preserves_authored_content(tmp_path):
    shared = Material(
        name="paint",
        rgba=np.array([0.2, 0.4, 0.8, 0.9], np.float32),
        emission=0.1,
        metallic=0.25,
        roughness=0.65,
        texture="checker",
        tex_repeat=np.array([2.0, 3.0], np.float32),
    )
    mesh = MeshData(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        normals=np.array([[0, 0, 1]] * 3, np.float32),
        uvs=np.array([[0, 0], [1, 0], [0, 1]], np.float32),
        indices=np.array([0, 1, 2], np.uint32),
    )
    scene = Scene()
    first = scene.box(name="box", material=shared, position=(1.0, 2.0, 3.0))
    scene.mesh(mesh, name="triangle", material=shared)
    scene.add_texture(
        TextureData(
            "checker",
            TextureType.TWO_D,
            np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8),
        )
    )
    scene.add_texture(
        TextureData(
            "sky",
            TextureType.CUBE,
            np.full((6, 2, 2, 3), 128, np.uint8),
        )
    )
    assert scene.set_skybox("sky")
    removed_light = scene.add_light("removed", Light())
    fill = scene.add_light(
        "fill",
        Light(
            type=LightType.POINT,
            position=np.array([2.0, -1.0, 4.0], np.float32),
            diffuse=np.array([0.3, 0.5, 0.9], np.float32),
            texture="checker",
            intensity=4.5,
        ),
    )
    removed_light.remove()
    scene.set_environment(
        Environment(
            ambient=np.array([0.1, 0.15, 0.2], np.float32),
            fog_color=np.array([0.4, 0.5, 0.6], np.float32),
            fog_start=3.0,
            fog_end=18.0,
            horizon_haze=True,
            horizon_haze_slices=23,
        )
    )
    removed_camera = scene.add_camera("removed", CameraView())
    camera = scene.add_camera("shot", CameraView(eye=np.array([5.0, -4.0, 3.0], np.float32)))
    scene.remove_camera(removed_camera)

    path = scene.save(tmp_path / "authored.forge.json")
    restored = Scene.load(path)
    source = restored.source
    rendered = SceneSourceBuilder().set_source(source)

    assert json.loads(path.read_text())["format"] == "forge-viewer.scene"
    assert source.geom_object_id.tolist() == [first.object_id, 2]
    assert source.geom_material == [0, 0]
    assert source.materials[0].metallic == pytest.approx(0.25)
    assert source.materials[0].roughness == pytest.approx(0.65)
    assert np.allclose(restored.frame.geom_xpos[0], [1.0, 2.0, 3.0])
    assert np.array_equal(restored.textures["checker"].pixels, scene.textures["checker"].pixels)
    assert source.skybox == "sky"
    assert restored.textures["sky"].type is TextureType.CUBE
    assert np.array_equal(next(iter(restored.source.meshes.values())).indices, mesh.indices)
    assert restored.light("fill").light_id == fill.light_id
    assert restored.light("fill").value.texture == "checker"
    assert restored.light("fill").value.intensity == 4.5
    light_node = next(node for node in source.nodes if node.name == "fill")
    assert light_node.object_id == LIGHT_OBJECT_BASE + fill.light_id
    assert np.allclose(source.lights.fog_color, [0.4, 0.5, 0.6])
    assert source.lights.horizon_haze
    assert source.lights.horizon_haze_slices == 23
    assert [info.camera_id for info in restored.camera_infos()] == [camera]
    assert restored.camera_infos()[0].name == "shot"
    assert rendered.count == 2
