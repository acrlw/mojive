from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .adapters.base import CAMERA_OBJECT_BASE
from .scene import Scene, _CameraItem, _Item, _LightItem
from .types import (
    CameraView,
    Environment,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    TextureData,
    TextureKind,
)

FORMAT = "forge-viewer.scene"
VERSION = 1


def save_scene(scene: Scene, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_scene_document(scene), indent=2), encoding="utf-8")
    return target


def load_scene(path: str | Path) -> Scene:
    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError(f"Unsupported Forge scene format in {source}")
    return _scene_from_document(document)


def _scene_document(scene: Scene) -> dict:
    scene._sync_light_items()
    materials: list[Material] = []
    material_ids: dict[int, int] = {}
    objects = []
    for item in scene._items:
        token = id(item.material)
        if token not in material_ids:
            material_ids[token] = len(materials)
            materials.append(item.material)
        objects.append(
            {
                "id": item.object_id,
                "name": item.name,
                "mesh": {"shape": item.mesh.shape.value, "index": item.mesh.index},
                "size": item.size.tolist(),
                "color": item.color.tolist(),
                "material": material_ids[token],
                "position": item.position.tolist(),
                "rotation": item.rotation.tolist(),
                "initial_position": item.initial_position.tolist(),
                "initial_rotation": item.initial_rotation.tolist(),
            }
        )

    return {
        "format": FORMAT,
        "version": VERSION,
        "next_ids": {
            "object": scene._next_object_id,
            "mesh": scene._next_mesh_id,
            "light": scene._next_light_id,
            "camera": scene._next_camera_id,
        },
        "materials": [_material_document(material) for material in materials],
        "meshes": [
            {"index": key.index, **_mesh_document(mesh)}
            for key, mesh in scene._meshes.items()
            if key.shape is MeshShape.ASSET
        ],
        "textures": [_texture_document(texture) for texture in scene.textures.values()],
        "objects": objects,
        "environment": _environment_document(scene.lights.environment()),
        "lights": [
            {
                "id": item.light_id,
                "name": item.name,
                "light": _light_document(light),
            }
            for item, light in zip(scene._lights, scene.lights.lights, strict=True)
        ],
        "cameras": [
            {
                "id": item.camera_id,
                "name": item.name,
                "view": _camera_document(item.view),
            }
            for item in scene._cameras
        ],
    }


def _scene_from_document(document: dict) -> Scene:
    scene = Scene()
    materials = [_material_from_document(item) for item in document["materials"]]
    scene._meshes = {
        MeshKey(MeshShape.ASSET, int(item["index"])): _mesh_from_document(item)
        for item in document["meshes"]
    }
    scene.textures = {
        texture.name: texture
        for texture in (_texture_from_document(item) for item in document["textures"])
    }
    scene._items = [
        _Item(
            object_id=int(item["id"]),
            name=str(item["name"]),
            mesh=MeshKey(MeshShape(item["mesh"]["shape"]), int(item["mesh"]["index"])),
            size=_f32(item["size"], (3,)),
            color=_f32(item["color"], (4,)),
            material=materials[int(item["material"])],
            position=_f32(item["position"], (3,)),
            rotation=_f32(item["rotation"], (3, 3)),
            initial_position=_f32(item["initial_position"], (3,)),
            initial_rotation=_f32(item["initial_rotation"], (3, 3)),
        )
        for item in document["objects"]
    ]
    light_entries = document["lights"]
    lights = tuple(_light_from_document(item["light"]) for item in light_entries)
    scene.lights = LightSet(lights=lights).with_environment(
        _environment_from_document(document["environment"])
    )
    scene._lights = [_LightItem(int(item["id"]), str(item["name"])) for item in light_entries]
    scene._cameras = [
        _CameraItem(
            camera_id=int(item["id"]),
            object_id=CAMERA_OBJECT_BASE + int(item["id"]),
            name=str(item["name"]),
            view=_camera_from_document(item["view"]),
        )
        for item in document["cameras"]
    ]
    scene.camera = scene._cameras[0].view if scene._cameras else None
    ids = document["next_ids"]
    scene._next_object_id = int(ids["object"])
    scene._next_mesh_id = int(ids["mesh"])
    scene._next_light_id = int(ids["light"])
    scene._next_camera_id = int(ids["camera"])
    scene._revision = 1
    scene._built_revision = -1
    return scene


def _light_document(light: Light | None) -> dict | None:
    if light is None:
        return None
    return {
        "kind": light.kind.name.lower(),
        "position": light.position.tolist(),
        "direction": light.direction.tolist(),
        "diffuse": light.diffuse.tolist(),
        "specular": light.specular.tolist(),
        "ambient": light.ambient.tolist(),
        "attenuation": light.attenuation.tolist(),
        "range": light.range,
        "area_radius": light.area_radius,
        "cutoff": light.cutoff,
        "exponent": light.exponent,
        "texture": light.texture,
        "intensity": light.intensity,
        "cast_shadow": light.cast_shadow,
        "active": light.active,
    }


def _light_from_document(item: dict | None) -> Light | None:
    if item is None:
        return None
    return Light(
        kind=LightKind[item["kind"].upper()],
        position=_f32(item["position"], (3,)),
        direction=_f32(item["direction"], (3,)),
        diffuse=_f32(item["diffuse"], (3,)),
        specular=_f32(item["specular"], (3,)),
        ambient=_f32(item["ambient"], (3,)),
        attenuation=_f32(item["attenuation"], (3,)),
        range=float(item["range"]),
        area_radius=float(item["area_radius"]),
        cutoff=float(item["cutoff"]),
        exponent=float(item["exponent"]),
        texture=item.get("texture"),
        intensity=float(item.get("intensity", 1.0)),
        cast_shadow=bool(item["cast_shadow"]),
        active=bool(item["active"]),
    )


def _environment_document(environment: Environment) -> dict:
    return {
        "headlight": _light_document(environment.headlight),
        "ambient": environment.ambient.tolist(),
        "fog_color": environment.fog_color.tolist(),
        "fog_start": environment.fog_start,
        "fog_end": environment.fog_end,
        "haze_color": environment.haze_color.tolist(),
        "haze_density": environment.haze_density,
    }


def _environment_from_document(item: dict) -> Environment:
    return Environment(
        headlight=_light_from_document(item["headlight"]),
        ambient=_f32(item["ambient"], (3,)),
        fog_color=_f32(item["fog_color"], (3,)),
        fog_start=float(item["fog_start"]),
        fog_end=float(item["fog_end"]),
        haze_color=_f32(item["haze_color"], (3,)),
        haze_density=float(item["haze_density"]),
    )


def _camera_document(camera: CameraView) -> dict:
    return {
        "eye": camera.eye.tolist(),
        "target": camera.target.tolist(),
        "up": camera.up.tolist(),
        "fov_y": camera.fov_y,
        "near": camera.near,
        "far": camera.far,
        "aspect": camera.aspect,
        "orthographic": camera.orthographic,
        "ortho_height": camera.ortho_height,
        "focal_length": camera.focal_length.tolist(),
        "sensor_size": camera.sensor_size.tolist(),
        "principal_offset": camera.principal_offset.tolist(),
    }


def _camera_from_document(item: dict) -> CameraView:
    return CameraView(
        eye=_f32(item["eye"], (3,)),
        target=_f32(item["target"], (3,)),
        up=_f32(item["up"], (3,)),
        fov_y=float(item["fov_y"]),
        near=float(item["near"]),
        far=float(item["far"]),
        aspect=float(item["aspect"]),
        orthographic=bool(item["orthographic"]),
        ortho_height=float(item["ortho_height"]),
        focal_length=_f32(item["focal_length"], (2,)),
        sensor_size=_f32(item["sensor_size"], (2,)),
        principal_offset=_f32(item["principal_offset"], (2,)),
    )


def _material_document(material: Material) -> dict:
    return {
        "name": material.name,
        "rgba": material.rgba.tolist(),
        "emission": material.emission,
        "specular": material.specular,
        "shininess": material.shininess,
        "reflectance": material.reflectance,
        "texture": material.texture,
        "tex_repeat": material.tex_repeat.tolist(),
        "tex_uniform": material.tex_uniform,
    }


def _material_from_document(item: dict) -> Material:
    return Material(
        name=str(item["name"]),
        rgba=_f32(item["rgba"], (4,)),
        emission=float(item["emission"]),
        specular=float(item["specular"]),
        shininess=float(item["shininess"]),
        reflectance=float(item["reflectance"]),
        texture=item["texture"],
        tex_repeat=_f32(item["tex_repeat"], (2,)),
        tex_uniform=bool(item["tex_uniform"]),
    )


def _mesh_document(mesh: MeshData) -> dict:
    return {
        "positions": _array_document(mesh.positions),
        "normals": _array_document(mesh.normals),
        "uvs": _array_document(mesh.uvs),
        "indices": _array_document(mesh.indices),
    }


def _mesh_from_document(item: dict) -> MeshData:
    return MeshData(
        positions=_array_from_document(item["positions"]),
        normals=_array_from_document(item["normals"]),
        uvs=_array_from_document(item["uvs"]),
        indices=_array_from_document(item["indices"]),
    )


def _texture_document(texture: TextureData) -> dict:
    return {
        "name": texture.name,
        "kind": texture.kind.value,
        "srgb": texture.srgb,
        "pixels": _array_document(texture.pixels),
    }


def _texture_from_document(item: dict) -> TextureData:
    return TextureData(
        name=str(item["name"]),
        kind=TextureKind(item["kind"]),
        pixels=_array_from_document(item["pixels"]),
        srgb=bool(item["srgb"]),
    )


def _array_document(array: np.ndarray) -> dict:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "data": base64.b64encode(value.tobytes()).decode("ascii"),
    }


def _array_from_document(item: dict) -> np.ndarray:
    data = base64.b64decode(item["data"])
    return np.frombuffer(data, dtype=np.dtype(item["dtype"])).reshape(item["shape"]).copy()


def _f32(value, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(value, np.float32).reshape(shape)
