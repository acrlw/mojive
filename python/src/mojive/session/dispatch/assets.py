"""Session commands: assets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


def import_model_geometry_resource(
    self: Session, c: cmd.ImportModelGeometryResource
) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before importing geometry resources")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    resource_type = str(c.resource_type).strip().lower()
    if resource_type not in ("mesh", "hfield"):
        return CommandResult.bad("Geometry resource type must be mesh or hfield")
    path = Path(c.path).expanduser().resolve()
    name = str(c.name).strip()
    if not path.is_file():
        return CommandResult.bad(f"Geometry resource file does not exist: {path}")
    allowed_suffixes = {".stl", ".obj", ".msh", ".ply"} if resource_type == "mesh" else {".png"}
    if path.suffix.lower() not in allowed_suffixes:
        suffixes = ", ".join(sorted(allowed_suffixes))
        return CommandResult.bad(f"{resource_type} resources must use one of: {suffixes}")
    if not name:
        return CommandResult.bad("A geometry resource name cannot be empty")
    try:
        changed = self._adapter.import_model_geometry_resource(c.node_id, resource_type, path, name)
    except Exception as exc:
        return CommandResult.bad(f"Geometry resource {name!r} could not be imported: {exc}")
    if not changed:
        return CommandResult.bad(f"Geometry resource {name!r} could not be imported")
    self._refresh_structure()
    return CommandResult.good(f"Imported and assigned {resource_type} {name}")


def import_model_asset(self: Session, c: cmd.ImportModelAsset) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before importing model assets")
    asset_type = str(c.asset_type).strip().lower()
    if asset_type not in ("mesh", "hfield"):
        return CommandResult.bad("Standalone import currently supports mesh or hfield")
    path = Path(c.path).expanduser().resolve()
    allowed_suffixes = {".stl", ".obj", ".msh", ".ply"} if asset_type == "mesh" else {".png"}
    if not path.is_file():
        return CommandResult.bad(f"Model asset file does not exist: {path}")
    if path.suffix.lower() not in allowed_suffixes:
        suffixes = ", ".join(sorted(allowed_suffixes))
        return CommandResult.bad(f"{asset_type} assets must use one of: {suffixes}")
    name = str(c.name).strip()
    if not name:
        return CommandResult.bad("A model asset name cannot be empty")
    if any(
        asset.type == asset_type and asset.name == name
        for asset in self._adapter.model_assets(c.model_id)
    ):
        return CommandResult.bad(f"{asset_type} asset {name!r} already exists")
    try:
        changed = self._adapter.import_model_asset(c.model_id, asset_type, path, name, c.fields)
    except Exception as exc:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be imported: {exc}")
    if not changed:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be imported")
    self._refresh_structure()
    return CommandResult.good(f"Imported {asset_type} asset {name}")


def set_height_field_size(self: Session, c: cmd.SetHeightFieldSize) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing height fields")
    try:
        size = np.asarray(c.size, np.float64).reshape(4)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Height-field dimensions have invalid value types")
    if not np.all(np.isfinite(size)):
        return CommandResult.bad("Height-field dimensions must be finite")
    if np.any(size[:3] <= 0.0) or size[3] < 0.0:
        return CommandResult.bad(
            "Height-field half-sizes and elevation scale must be positive; "
            "base depth cannot be negative"
        )
    name = str(c.name).strip()
    if not name:
        return CommandResult.bad("A height-field name cannot be empty")
    current = next(
        (
            asset
            for asset in self._adapter.model_assets(c.model_id)
            if asset.type == "hfield" and asset.name == name
        ),
        None,
    )
    if current is None:
        return CommandResult.bad(f"Height-field asset {name!r} is unavailable")
    try:
        changed = self._adapter.set_height_field_size(
            c.model_id,
            name,
            tuple(float(value) for value in size),
        )
    except Exception as exc:
        return CommandResult.bad(
            f"Height-field asset {name!r} dimensions could not be applied: {exc}"
        )
    if not changed:
        return CommandResult.bad(f"Height-field asset {name!r} dimensions could not be applied")
    self._refresh_structure()
    return CommandResult.good(f"Updated height-field dimensions for {name}")


def rename_model_asset(self: Session, c: cmd.RenameModelAsset) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before renaming model assets")
    asset_type = str(c.asset_type).strip().lower()
    name = str(c.name).strip()
    new_name = str(c.new_name).strip()
    if not name or not new_name:
        return CommandResult.bad("Model asset names cannot be empty")
    if any(
        asset.type == asset_type and asset.name == new_name
        for asset in self._adapter.model_assets(c.model_id)
    ):
        return CommandResult.bad(f"{asset_type} asset {new_name!r} already exists")
    try:
        changed = self._adapter.rename_model_asset(c.model_id, asset_type, name, new_name)
    except Exception as exc:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be renamed: {exc}")
    if not changed:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be renamed")
    self._refresh_structure()
    return CommandResult.good(f"Renamed {asset_type} asset {name} to {new_name}")


def duplicate_model_asset(self: Session, c: cmd.DuplicateModelAsset) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before duplicating model assets")
    asset_type = str(c.asset_type).strip().lower()
    name = str(c.name).strip()
    new_name = str(c.new_name).strip()
    if not name or not new_name:
        return CommandResult.bad("Model asset names cannot be empty")
    if any(
        asset.type == asset_type and asset.name == new_name
        for asset in self._adapter.model_assets(c.model_id)
    ):
        return CommandResult.bad(f"{asset_type} asset {new_name!r} already exists")
    try:
        changed = self._adapter.duplicate_model_asset(c.model_id, asset_type, name, new_name)
    except Exception as exc:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be duplicated: {exc}")
    if not changed:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be duplicated")
    self._refresh_structure()
    return CommandResult.good(f"Duplicated {asset_type} asset as {new_name}")


def replace_model_asset_file(self: Session, c: cmd.ReplaceModelAssetFile) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before replacing model asset files")
    asset_type = str(c.asset_type).strip().lower()
    if asset_type not in ("mesh", "hfield", "texture"):
        return CommandResult.bad("File replacement currently supports mesh, hfield, or texture")
    path = Path(c.path).expanduser().resolve()
    allowed_suffixes = {".stl", ".obj", ".msh", ".ply"} if asset_type == "mesh" else {".png"}
    if not path.is_file():
        return CommandResult.bad(f"Model asset file does not exist: {path}")
    if path.suffix.lower() not in allowed_suffixes:
        suffixes = ", ".join(sorted(allowed_suffixes))
        return CommandResult.bad(f"{asset_type} assets must use one of: {suffixes}")
    name = str(c.name).strip()
    try:
        changed = self._adapter.replace_model_asset_file(c.model_id, asset_type, name, path)
    except Exception as exc:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be replaced: {exc}")
    if not changed:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be replaced")
    self._refresh_structure()
    return CommandResult.good(f"Replaced source for {asset_type} asset {name}")


def remove_model_asset(self: Session, c: cmd.RemoveModelAsset) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before removing model assets")
    asset_type = str(c.asset_type).strip().lower()
    name = str(c.name).strip()
    asset = next(
        (
            item
            for item in self._adapter.model_assets(c.model_id)
            if item.type == asset_type and item.name == name
        ),
        None,
    )
    if asset is None:
        return CommandResult.bad(f"{asset_type} asset {name!r} is unavailable")
    if asset.references:
        return CommandResult.bad(
            f"{asset_type} asset {name!r} is used by {len(asset.references)} element(s)"
        )
    try:
        changed = self._adapter.remove_model_asset(c.model_id, asset_type, name)
    except Exception as exc:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be removed: {exc}")
    if not changed:
        return CommandResult.bad(f"{asset_type} asset {name!r} could not be removed")
    self._refresh_structure()
    return CommandResult.good(f"Removed {asset_type} asset {name}")


def create_model_material(self: Session, c: cmd.CreateModelMaterial) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before creating materials")
    value = str(c.name).strip()
    if not value:
        return CommandResult.bad("A material name cannot be empty")
    if any(
        asset.type == "material" and asset.name == value
        for asset in self._adapter.model_assets(c.model_id)
    ):
        return CommandResult.bad(f"Material {value!r} already exists")
    material_index = self._adapter.create_model_material(c.model_id, value)
    if material_index < 0:
        return CommandResult.bad(f"Material {value!r} could not be created")
    self._refresh_structure()
    return CommandResult.good(f"Created material {value}", material_index)


def add_model_material(self: Session, c: cmd.AddModelMaterial) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before creating materials")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    value = str(c.name).strip()
    if not value:
        return CommandResult.bad("A material name cannot be empty")
    material_index = self._adapter.add_model_material(c.node_id, value, int(c.copy_from))
    if material_index < 0:
        return CommandResult.bad(f"Material {value!r} could not be created")
    self._refresh_structure()
    return CommandResult.good(f"Created material {value}", material_index)


def import_model_texture(self: Session, c: cmd.ImportModelTexture) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before importing textures")
    path = Path(c.path).expanduser().resolve()
    value = str(c.name).strip()
    texture_type = str(c.texture_type).strip().lower()
    if not path.is_file():
        return CommandResult.bad(f"Texture file does not exist: {path}")
    if path.suffix.lower() != ".png":
        return CommandResult.bad("MuJoCo image textures must use PNG files")
    if not value:
        return CommandResult.bad("A texture name cannot be empty")
    if texture_type not in ("2d", "cube", "skybox"):
        return CommandResult.bad("Texture type must be 2d, cube, or skybox")
    if texture_type != "2d" and int(c.material_index) >= 0:
        return CommandResult.bad("Only 2D textures can be assigned to a material")
    try:
        changed = self._adapter.import_model_texture(
            c.model_id,
            path,
            value,
            int(c.material_index),
            texture_type,
        )
    except (RuntimeError, ValueError) as exc:
        return CommandResult.bad(f"Texture {value!r} could not be imported: {exc}")
    if not changed:
        return CommandResult.bad(f"Texture {value!r} could not be imported")
    self._refresh_structure()
    return CommandResult.good(f"Imported texture {value}")


def set_geometry_material(self: Session, c: cmd.SetGeometryMaterial) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_assets:
        return CommandResult.bad(f"{caps.name} does not support model asset editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before binding materials")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    if not self._adapter.set_geometry_material(c.node_id, c.material_index):
        return CommandResult.bad("The material is unavailable for this model geometry")
    self._refresh_structure()
    return CommandResult.good("")
