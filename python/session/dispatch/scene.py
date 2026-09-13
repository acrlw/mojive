"""Session commands: scene."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.commands import CommandResult
from mojive.types import (
    TextureType,
)

if TYPE_CHECKING:
    from .. import Session


from ..state import _LightOverride


def set_light(self: Session, c: cmd.SetLight) -> CommandResult:
    index = int(c.light_index)
    if self._source is None or not 0 <= index < len(self._source.lights.lights):
        return CommandResult.bad(f"light index {index} is unavailable")
    writeback = self._adapter.set_light(index, c.light)
    node = next((node for node in self._nodes if node.light_index == index), None)
    object_id = int(node.object_id) if node is not None else 0
    override_key = object_id if object_id > 0 else -(index + 1)
    if self._preserve_authored_override(writeback):
        self._authored.lights[override_key] = _LightOverride(object_id, index, c.light)
    else:
        self._authored.lights.pop(override_key, None)
    lights = list(self._source.lights.lights)
    lights[index] = c.light
    self._source.lights = replace(self._source.lights, lights=tuple(lights))
    for node in self._nodes:
        if node.light_index == index:
            node.visible = c.light.active
            break
    self._compose_lights()
    message = "" if writeback else "edited in the viewer; adapter write-back is unavailable"
    return CommandResult.good(message)


def set_environment(self: Session, c: cmd.SetEnvironment) -> CommandResult:
    if self._source is None:
        return CommandResult.bad("environment is unavailable")
    writeback = self._adapter.set_environment(c.environment)
    self._authored.environment = (
        c.environment if self._preserve_authored_override(writeback) else None
    )
    self._source.lights = self._source.lights.with_environment(c.environment)
    self._compose_lights()
    message = "" if writeback else "edited in the viewer; adapter write-back is unavailable"
    return CommandResult.good(message)


def set_skybox(self: Session, c: cmd.SetSkybox) -> CommandResult:
    if self._source is None:
        return CommandResult.bad("skybox is unavailable")
    texture = self._source.textures.get(c.texture or "")
    if c.texture is not None and (
        texture is None or texture.type not in (TextureType.CUBE, TextureType.SKYBOX)
    ):
        return CommandResult.bad(f"cube texture {c.texture!r} is unavailable")
    if not self._adapter.set_skybox(c.texture):
        return CommandResult.bad("skybox update failed")
    for name, item in tuple(self._source.textures.items()):
        next_type = (
            TextureType.SKYBOX
            if name == c.texture
            else TextureType.CUBE
            if item.type is TextureType.SKYBOX
            else item.type
        )
        if next_type is not item.type:
            self._source.textures[name] = replace(item, type=next_type)
    self._source.skybox = c.texture
    self._structure_generation += 1
    return CommandResult.good("")


def set_material(self: Session, c: cmd.SetMaterial) -> CommandResult:
    index = int(c.material_index)
    if self._source is None or not 0 <= index < len(self._source.materials):
        return CommandResult.bad(f"material index {index} is unavailable")
    writeback = self._adapter.set_material(index, c.material)
    if self._preserve_authored_override(writeback):
        self._authored.materials[index] = c.material
    else:
        self._authored.materials.pop(index, None)
    self._source.materials[index] = c.material
    self._structure_generation += 1
    message = "" if writeback else "edited in the viewer; adapter write-back is unavailable"
    return CommandResult.good(message)


def set_geometry_color(self: Session, c: cmd.SetGeometryColor) -> CommandResult:
    if self._source is None:
        return CommandResult.bad("geometry is unavailable")
    node_id = self._command_node_id(c, c.node_key)
    instances = np.flatnonzero(self._source.geom_node == node_id)
    if not len(instances):
        return CommandResult.bad(
            f"geometry node {c.node_id} is unavailable"
            if not c.node_key
            else f"pending geometry {c.node_key} is unavailable"
        )
    rgba = np.asarray(c.rgba, np.float32).reshape(4).copy()
    writeback = self._adapter.set_geometry_color(node_id, rgba)
    if self._preserve_authored_override(writeback):
        self._authored.geometry_colors[node_id] = rgba
        node = self.node(node_id)
        self._authored.geometry_color_targets[node_id] = (
            int(self._source.geom_object_id[instances[0]]),
            node.model_id,
            node.type,
            node.name,
        )
    else:
        self._authored.geometry_colors.pop(node_id, None)
        self._authored.geometry_color_targets.pop(node_id, None)
    self._source.geom_rgba[instances] = rgba
    self._structure_generation += 1
    message = "" if writeback else "edited in the viewer; adapter write-back is unavailable"
    return CommandResult.good(message)


def set_geometry_size(self: Session, c: cmd.SetGeometrySize) -> CommandResult:
    if self._source is None:
        return CommandResult.bad("geometry is unavailable")
    instances = np.flatnonzero(self._source.geom_node == int(c.node_id))
    if not len(instances):
        return CommandResult.bad(f"geometry node {c.node_id} is unavailable")
    node = self.node(c.node_id)
    authored_scene_node = bool(node is not None and node.model_id < 0)
    if node is None or (not node.source_editable and not authored_scene_node):
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    size = np.asarray(c.size, np.float32).reshape(3)
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        return CommandResult.bad("geometry size must contain three positive values")
    if not self._adapter.set_geometry_size(c.node_id, size):
        return CommandResult.bad("geometry size cannot be edited")
    self._refresh_structure()
    return CommandResult.good("")


def set_scene_camera(self: Session, c: cmd.SetSceneCamera) -> CommandResult:
    camera_id = int(c.camera_id)
    slot = self._camera_slot(camera_id)
    if self._source is None or slot < 0 or slot >= len(self._source.cameras):
        return CommandResult.bad(f"camera {camera_id} is unavailable")
    writeback = self._adapter.set_camera_view(camera_id, c.camera)
    cameras = list(self._source.cameras)
    cameras[slot] = c.camera
    self._source.cameras = tuple(cameras)
    if self._preserve_authored_override(writeback):
        self._authored.cameras[camera_id] = c.camera
    else:
        self._authored.cameras.pop(camera_id, None)
    self._compose_cameras()
    message = "" if writeback else "edited in the viewer; adapter write-back is unavailable"
    return CommandResult.good(message)


def add_scene_object(self: Session, c: cmd.AddSceneObject) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    object_id = self._adapter.add_scene_object(
        c.shape, c.name, c.size, c.position, c.rotation, c.color, c.material
    )
    if object_id < 0:
        return CommandResult.bad("Object creation failed")
    self._refresh_structure()
    return CommandResult.good(f"Added {c.name}", object_id)


def remove_scene_object(self: Session, c: cmd.RemoveSceneObject) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    if not self._adapter.remove_scene_object(c.object_id):
        return CommandResult.bad(f"object {c.object_id} is unavailable")
    if self._selected == c.object_id:
        self._selected = 0
        self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good("Object removed", c.object_id)


def add_scene_light(self: Session, c: cmd.AddSceneLight) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    light_id = self._adapter.add_scene_light(c.name, c.light)
    if light_id < 0:
        return CommandResult.bad("Light creation failed")
    self._refresh_structure()
    return CommandResult.good(f"Added {c.name}", light_id)


def remove_scene_light(self: Session, c: cmd.RemoveSceneLight) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    if not self._adapter.remove_scene_light(c.light_id):
        return CommandResult.bad(f"light {c.light_id} is unavailable")
    self._refresh_structure()
    return CommandResult.good("Light removed", c.light_id)


def add_scene_camera(self: Session, c: cmd.AddSceneCamera) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    camera_id = self._adapter.add_scene_camera(c.name, c.camera)
    if camera_id < 0:
        return CommandResult.bad("Camera creation failed")
    self._refresh_structure()
    return CommandResult.good(f"Added {c.name}", camera_id)


def remove_scene_camera(self: Session, c: cmd.RemoveSceneCamera) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    if not self._adapter.remove_scene_camera(c.camera_id):
        return CommandResult.bad(f"camera {c.camera_id} is unavailable")
    self._authored.cameras.pop(c.camera_id, None)
    self._refresh_structure()
    return CommandResult.good("Camera removed", c.camera_id)


def duplicate_scene_entity(self: Session, c: cmd.DuplicateSceneEntity) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    object_id = self._adapter.duplicate_scene_entity(c.object_id)
    if not object_id:
        return CommandResult.bad(f"entity {c.object_id} cannot be duplicated")
    self._selected = object_id
    self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good("Entity duplicated", object_id)


def remove_scene_entity(self: Session, c: cmd.RemoveSceneEntity) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    if not self._adapter.remove_scene_entity(c.object_id):
        return CommandResult.bad(f"entity {c.object_id} cannot be removed")
    if self._selected == c.object_id:
        self._selected = 0
        self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good("Entity removed", c.object_id)


def rename_scene_entity(self: Session, c: cmd.RenameSceneEntity) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_authoring:
        return CommandResult.bad(f"{caps.name} does not support scene authoring")
    if not self._adapter.rename_scene_entity(c.object_id, c.name):
        return CommandResult.bad(f"entity {c.object_id} cannot be renamed")
    self._refresh_structure()
    return CommandResult.good("Entity renamed", c.object_id)
