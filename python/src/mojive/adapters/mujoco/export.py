"""MJCF export, asset staging and authored scene serialization."""

from __future__ import annotations

import shutil
import warnings
from html import escape
from pathlib import Path

import numpy as np

from ... import math3d
from ...types import (
    LightType,
    MeshData,
    MeshShape,
    TextureType,
)
from ..base import (
    NodeType,
    SceneFrame,
    SceneSaveOptions,
    SceneSource,
)
from .constants import _MOJIVE_AREA_LIGHTS_TEXT
from .engine import mujoco
from .spec import (
    _apply_environment,
    _camera_quaternion,
    _editable_spec_xml,
    _mjcf_name,
    _set_text_names,
    _spec_text_names,
)


def _authored_builtin_mesh(shape: MeshShape) -> MeshData | None:
    if shape not in (MeshShape.CYLINDER, MeshShape.CONE):
        return None
    segments = 32
    positions: list[tuple[float, float, float]] = []
    indices: list[int] = []
    radius_top = 1.0 if shape is MeshShape.CYLINDER else 0.0
    for segment in range(segments):
        angle = 2.0 * np.pi * segment / segments
        x, y = float(np.cos(angle)), float(np.sin(angle))
        positions.extend(((x, y, -1.0), (radius_top * x, radius_top * y, 1.0)))
    bottom_center = len(positions)
    positions.append((0.0, 0.0, -1.0))
    top_center = len(positions)
    positions.append((0.0, 0.0, 1.0))
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        bottom = 2 * segment
        top = bottom + 1
        next_bottom = 2 * next_segment
        next_top = next_bottom + 1
        indices.extend((bottom, next_bottom, top, top, next_bottom, next_top))
        indices.extend((bottom_center, next_bottom, bottom))
        if radius_top > 0.0:
            indices.extend((top_center, top, next_top))
    vertex_count = len(positions)
    return MeshData(
        positions=np.asarray(positions, np.float32),
        normals=np.zeros((vertex_count, 3), np.float32),
        uvs=np.zeros((vertex_count, 2), np.float32),
        indices=np.asarray(indices, np.uint32),
    )


class _MjcfExport:
    """MJCF export, asset staging and authored scene serialization.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def export_mjcf(
        self,
        path: Path,
        source: SceneSource,
        frame: SceneFrame,
        options: SceneSaveOptions | None = None,
    ) -> Path:
        """Write the composed MuJoCo model and Mojive-authored entities as MJCF."""
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self._validate_authored_mjcf(source)
        spec = self._composed_spec()
        self._append_authored_scene(spec, target, source, frame)
        key_name = (options or SceneSaveOptions()).current_pose_keyframe
        if key_name:
            previous = spec.key(key_name)
            if previous is not None:
                spec.delete(previous)
            state = self.capture_state()
            spec.add_key(
                name=key_name,
                qpos=state.qpos,
                qvel=np.zeros_like(state.qvel),
                act=state.act,
                mpos=state.mocap_pos.reshape(-1),
                mquat=state.mocap_quat.reshape(-1),
                ctrl=state.ctrl,
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attach conflict.*")
            preview = spec.compile()
            _apply_environment(spec, source.lights, float(preview.stat.extent))
            spec.compile()
        assets = self._stage_mjcf_assets(spec, target)
        xml = _editable_spec_xml(spec)
        for source_file, exported_file in assets:
            xml = xml.replace(
                f'="{escape(source_file, quote=True)}"',
                f'="{escape(exported_file, quote=True)}"',
            )
        target.write_text(xml.rstrip() + "\n", encoding="utf-8")
        mujoco.MjModel.from_xml_path(str(target))
        return target

    @staticmethod
    def _validate_authored_mjcf(source: SceneSource) -> None:
        unsupported: list[str] = []
        for light in source.lights.lights:
            texture = source.textures.get(light.texture or "")
            if light.type is LightType.IMAGE and (
                texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX)
            ):
                unsupported.append("image light without a cube texture")
        if source.skybox is not None:
            texture = source.textures.get(source.skybox)
            if texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX):
                unsupported.append("skybox without a cube texture")
        if unsupported:
            details = ", ".join(dict.fromkeys(unsupported))
            raise RuntimeError(f"MJCF export cannot preserve {details}")

    @staticmethod
    def _stage_mjcf_assets(spec, target: Path) -> list[tuple[str, str]]:
        assets = target.parent / f"{target.stem}_assets"
        exported: list[tuple[str, str]] = []
        groups = (
            ("mesh", spec.meshes),
            ("texture", spec.textures),
            ("hfield", spec.hfields),
            ("skin", spec.skins),
        )
        for asset_type, items in groups:
            for index, asset in enumerate(items):
                files = [str(asset.file or "")]
                files.extend(str(file or "") for file in getattr(asset, "cubefiles", ()))
                for file_index, file in enumerate(files):
                    if not file:
                        continue
                    source = Path(file).expanduser().resolve()
                    suffix = source.suffix or ".bin"
                    name = _mjcf_name(asset_type, str(asset.name), index)
                    face = f"_{file_index - 1}" if file_index else ""
                    destination = (
                        source
                        if source.parent == assets.resolve()
                        else assets / f"{name}{face}{suffix}"
                    )
                    assets.mkdir(parents=True, exist_ok=True)
                    if source != destination.resolve():
                        shutil.copy2(source, destination)
                    exported.append(
                        (str(source), destination.relative_to(target.parent).as_posix())
                    )
        return exported

    def _append_authored_scene(
        self, spec, target: Path, source: SceneSource, frame: SceneFrame
    ) -> None:
        texture_names = self._export_authored_textures(spec, target, source)
        material_names: list[str] = []
        for index, material in enumerate(source.materials):
            name = _mjcf_name("mojive_material", material.name, index)
            textures = (
                ["", texture_names[material.texture]] if material.texture in texture_names else []
            )
            spec.add_material(
                name=name,
                textures=textures,
                texuniform=material.tex_uniform,
                texrepeat=material.tex_repeat,
                emission=material.emission,
                specular=material.specular,
                shininess=material.shininess,
                reflectance=material.reflectance,
                metallic=material.metallic,
                roughness=material.roughness,
                rgba=material.rgba,
            )
            material_names.append(name)

        positions = frame.geom_xpos
        rotations = frame.geom_xmat
        if positions is None or rotations is None:
            positions = np.zeros((source.instance_count, 3), np.float32)
            rotations = np.repeat(np.eye(3, dtype=np.float32)[None], source.instance_count, axis=0)
        nodes = {node.node_id: node for node in source.nodes}
        for index, key in enumerate(source.geom_mesh):
            node = nodes.get(int(source.geom_node[index]))
            owner = nodes.get(node.parent) if node is not None else None
            name = _mjcf_name(
                "mojive_object",
                owner.name if owner is not None else (node.name if node is not None else "object"),
                index,
            )
            body = spec.worldbody.add_body(
                name=name,
                pos=positions[index],
                quat=math3d.mat3_to_quat(rotations[index]),
            )
            material = (
                material_names[int(source.geom_material[index])]
                if material_names and 0 <= int(source.geom_material[index]) < len(material_names)
                else ""
            )
            geom = {
                "name": f"{name}_geom",
                "rgba": source.geom_rgba[index],
                "material": material,
                "contype": 0,
                "conaffinity": 0,
            }
            shape = key.shape
            size = np.asarray(source.geom_size[index], np.float64)
            if shape is MeshShape.SPHERE:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=size, **geom)
            elif shape is MeshShape.BOX:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, **geom)
            elif shape is MeshShape.PLANE:
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_PLANE,
                    size=(float(size[0]), float(size[1]), max(float(size[2]), 1e-3)),
                    **geom,
                )
            elif shape is MeshShape.CYLINDER and np.isclose(size[0], size[1]):
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=(float(size[0]), float(size[2])),
                    **geom,
                )
            else:
                mesh = source.meshes.get(key) or _authored_builtin_mesh(shape)
                if mesh is None:
                    continue
                mesh_name = f"{name}_mesh"
                spec.add_mesh(
                    name=mesh_name,
                    scale=size,
                    uservert=np.asarray(mesh.positions, np.float64).reshape(-1),
                    userface=np.asarray(mesh.indices, np.int32).reshape(-1),
                )
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=mesh_name,
                    **geom,
                )

        camera_names = {
            node.camera_index: node.name
            for node in source.nodes
            if node.type is NodeType.CAMERA and node.camera_index >= 0
        }
        for index, camera in enumerate(source.cameras):
            name = _mjcf_name(
                "mojive_camera",
                camera_names.get(index, "camera"),
                index,
            )
            spec.worldbody.add_camera(
                name=name,
                pos=camera.eye,
                quat=_camera_quaternion(camera),
                mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                proj=(
                    mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
                    if camera.orthographic
                    else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
                ),
                fovy=(camera.ortho_height if camera.orthographic else np.degrees(camera.fov_y)),
                focal_length=camera.focal_length if camera.uses_intrinsics() else None,
                sensor_size=camera.sensor_size if camera.uses_intrinsics() else None,
                principal_length=camera.principal_offset if camera.uses_intrinsics() else None,
            )

        light_names = {
            node.light_index: node.name
            for node in source.nodes
            if node.type is NodeType.LIGHT and node.light_index >= 0
        }
        light_types = {
            LightType.DIRECTIONAL: mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            LightType.POINT: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.SPOT: mujoco.mjtLightType.mjLIGHT_SPOT,
            # MuJoCo has no area-light enum.  OpenGL records the semantic type in
            # custom metadata and uses the native point-light bulb radius as the
            # portable fallback representation.
            LightType.AREA: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.IMAGE: mujoco.mjtLightType.mjLIGHT_IMAGE,
        }
        area_names = set(_spec_text_names(spec, _MOJIVE_AREA_LIGHTS_TEXT))
        for index, light in enumerate(source.lights.lights):
            name = _mjcf_name(
                "mojive_light",
                light_names.get(index, "light"),
                index,
            )
            spec.worldbody.add_light(
                name=name,
                pos=light.position,
                dir=light.direction,
                mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                active=light.active,
                type=light_types[light.type],
                texture=texture_names.get(light.texture or "", ""),
                castshadow=light.cast_shadow,
                bulbradius=light.area_radius,
                intensity=light.intensity,
                range=light.range,
                attenuation=light.attenuation,
                cutoff=light.cutoff,
                exponent=light.exponent,
                ambient=light.ambient,
                diffuse=light.diffuse,
                specular=light.specular,
            )
            if light.type is LightType.AREA:
                area_names.add(name)
        _set_text_names(spec, _MOJIVE_AREA_LIGHTS_TEXT, area_names)

    @staticmethod
    def _export_authored_textures(spec, target: Path, source: SceneSource) -> dict[str, str]:
        from PIL import Image

        texture_names: dict[str, str] = {}
        assets = target.parent / f"{target.stem}_assets"
        for index, texture in enumerate(source.textures.values()):
            assets.mkdir(parents=True, exist_ok=True)
            name = _mjcf_name("mojive_texture", texture.name, index)
            if texture.type is TextureType.TWO_D:
                file = assets / f"{name}.png"
                Image.fromarray(texture.pixels).save(file)
                spec.add_texture(
                    name=name,
                    type=mujoco.mjtTexture.mjTEXTURE_2D,
                    file=str(file.resolve()),
                )
            else:
                files = []
                for face, pixels in enumerate(np.asarray(texture.pixels)):
                    file = assets / f"{name}_{face}.png"
                    Image.fromarray(pixels).save(file)
                    files.append(str(file.resolve()))
                texture_type = (
                    mujoco.mjtTexture.mjTEXTURE_SKYBOX
                    if texture.name == source.skybox
                    else mujoco.mjtTexture.mjTEXTURE_CUBE
                )
                spec.add_texture(name=name, type=texture_type, cubefiles=files)
            texture_names[texture.name] = name
        return texture_names
