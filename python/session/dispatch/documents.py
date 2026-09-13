"""Session commands: documents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mojive import commands as cmd
from mojive.adapters.base import (
    NodeType,
    SceneSaveOptions,
)
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


from ..state import PerturbState


def reload(self: Session, c: cmd.Reload) -> CommandResult:
    caps = self._adapter.caps
    if not caps.reload:
        return CommandResult.bad(f"{caps.name} does not support reload")
    if not self._pause_before_model_change():
        return CommandResult.bad("physics backend rejected pause before reload")
    try:
        self._adapter.reload()
    except Exception as exc:
        return CommandResult.bad(str(exc))
    self._pause_loaded_scene()
    self._step_counter = 0
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._authored.clear()
    self._refresh_structure()
    self._reset_edit_history()
    return CommandResult.good("Scene reloaded")


def new_scene(self: Session, c: cmd.NewScene) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_files:
        return CommandResult.bad(f"{caps.name} does not support scene files")
    self._adapter.new_scene()
    self._pause_loaded_scene()
    self._asset_path = None
    self._selected = 0
    self._selected_node_id = -1
    self._authored.clear()
    self._refresh_structure()
    self._reset_edit_history()
    return CommandResult.good("New scene")


def open_scene(self: Session, c: cmd.OpenScene) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_files:
        return CommandResult.bad(f"{caps.name} does not support scene files")
    if not self._pause_before_model_change():
        return CommandResult.bad("physics backend rejected pause before opening scene")
    path = Path(c.path).expanduser().resolve()
    try:
        self._adapter.open_scene(path)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    self._pause_loaded_scene()
    self._asset_path = path
    self._selected = 0
    self._selected_node_id = -1
    self._authored.clear()
    self._refresh_structure()
    self._reset_edit_history()
    return CommandResult.good(f"Opened {path.name}")


def save_scene(self: Session, c: cmd.SaveScene) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_files:
        return CommandResult.bad(f"{caps.name} does not support scene files")
    path = Path(c.path).expanduser().resolve()
    try:
        self._adapter.save_scene(
            path,
            SceneSaveOptions(current_pose_keyframe=c.current_pose_keyframe),
        )
    except Exception as exc:
        return CommandResult.bad(str(exc))
    self._asset_path = path
    self._saved_revision = self._document_revision
    return CommandResult.good(f"Saved {path.name}")


def load_asset(self: Session, c: cmd.LoadAsset) -> CommandResult:
    caps = self._adapter.caps
    if not caps.asset_loading:
        return CommandResult.bad(f"{caps.name} does not support model loading")
    if not self._pause_before_model_change():
        return CommandResult.bad("physics backend rejected pause before loading model")
    path = Path(c.path).expanduser().resolve()
    try:
        self._adapter.load(path)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    self._pause_loaded_scene()
    self._asset_path = path
    self._step_counter = 0
    self._selected = 0
    self._selected_node_id = -1
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._authored.clear()
    self._refresh_structure()
    self._reset_edit_history()
    return CommandResult.good(f"Loaded {c.path.name}")


def add_scene_model(self: Session, c: cmd.AddSceneModel) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_composition:
        return CommandResult.bad(f"{caps.name} does not support model composition")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    path = Path(c.path).expanduser().resolve()
    try:
        model_id = self._adapter.add_scene_model(path, c.position, c.rotation)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if model_id < 0:
        return CommandResult.bad(f"Failed to add {path.name}")
    self._selected = 0
    self._selected_node_id = -1
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._refresh_structure()
    return CommandResult.good(f"Added {path.name}", model_id)


def remove_scene_model(self: Session, c: cmd.RemoveSceneModel) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_composition:
        return CommandResult.bad(f"{caps.name} does not support model composition")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    info = next((item for item in self.scene_models if item.model_id == int(c.model_id)), None)
    if info is None or not info.removable:
        return CommandResult.bad(f"Model {c.model_id} cannot be removed")
    try:
        removed = self._adapter.remove_scene_model(c.model_id)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not removed:
        return CommandResult.bad(f"Failed to remove {info.name}")
    self._selected = 0
    self._selected_node_id = -1
    self._perturb = PerturbState()
    self._active_keyframe = -1
    self._refresh_structure()
    return CommandResult.good(f"Removed {info.name}")


def set_scene_model_transform(self: Session, c: cmd.SetSceneModelTransform) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_composition:
        return CommandResult.bad(f"{caps.name} does not support model composition")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before moving a model root")
    try:
        changed = self._adapter.set_scene_model_transform(c.model_id, c.position, c.rotation)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"Model {c.model_id} cannot be transformed")
    self._refresh_structure()
    return CommandResult.good("Updated model transform")


def preview_scene_model_transform(
    self: Session, c: cmd.PreviewSceneModelTransform
) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_composition:
        return CommandResult.bad(f"{caps.name} does not support model composition")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before moving a model root")
    try:
        changed = self._adapter.preview_scene_model_transform(c.model_id, c.position, c.rotation)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"Model {c.model_id} cannot be transformed")
    return CommandResult.good()


def clear_scene_model_transform_preview(
    self: Session, c: cmd.ClearSceneModelTransformPreview
) -> CommandResult:
    try:
        cleared = self._adapter.clear_scene_model_transform_preview(c.model_id)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not cleared:
        return CommandResult.bad(f"Model {c.model_id} has no transform preview")
    return CommandResult.good()


def add_model_element(self: Session, c: cmd.AddModelElement) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    parent = self.node(c.parent_node_id)
    if parent is None:
        return CommandResult.bad(f"Unknown parent node_id={c.parent_node_id}")
    if parent.type not in (NodeType.WORLD, NodeType.MODEL) and not parent.source_editable:
        return CommandResult.bad(f"{parent.name} has no editable source element")
    try:
        node_id = self._adapter.add_model_element(c.parent_node_id, c.element_type, c.name)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if node_id < 0:
        return CommandResult.bad(f"Failed to add {c.element_type}")
    self._refresh_structure()
    return CommandResult.good(f"Added {c.name}", node_id)


def duplicate_model_element(self: Session, c: cmd.DuplicateModelElement) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    node = self.node(c.node_id)
    if node is None:
        return CommandResult.bad(f"Unknown node_id={c.node_id}")
    if not node.source_editable:
        return CommandResult.bad(f"{node.name} has no editable source element")
    try:
        node_id = self._adapter.duplicate_model_element(c.node_id)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if node_id < 0:
        return CommandResult.bad(f"{node.name} cannot be duplicated")
    self._selected = 0
    self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good(f"Duplicated {node.name}", node_id)


def remove_model_element(self: Session, c: cmd.RemoveModelElement) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    node = self.node(c.node_id)
    if node is None:
        return CommandResult.bad(f"Unknown node_id={c.node_id}")
    if not node.source_editable:
        return CommandResult.bad(f"{node.name} has no editable source element")
    try:
        changed = self._adapter.remove_model_element(c.node_id)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"{node.name} cannot be removed from the model")
    self._selected = 0
    self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good(f"Removed {node.name}")


def rename_model_element(self: Session, c: cmd.RenameModelElement) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    node = self.node(c.node_id)
    if node is None:
        return CommandResult.bad(f"Unknown node_id={c.node_id}")
    if not node.source_editable:
        return CommandResult.bad(f"{node.name} has no editable source element")
    try:
        changed = self._adapter.rename_model_element(c.node_id, c.name)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"{node.name} cannot be renamed")
    self._refresh_structure()
    return CommandResult.good(f"Renamed {node.name}")


def model_edit_batch(self: Session, c: cmd.ModelEditBatch) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    if not c.edits:
        return CommandResult.bad("A model edit batch cannot be empty")
    if not all(isinstance(edit, cmd.ModelEdit) for edit in c.edits):
        return CommandResult.bad("A model edit batch contains an unsupported operation")
    selected = self.selected_node
    selected_identity = (
        (selected.model_id, selected.type, selected.name) if selected is not None else None
    )
    if selected is not None:
        for edit in c.edits:
            if (
                not isinstance(edit, (cmd.RemoveModelElementEdit, cmd.RenameModelElementEdit))
                or int(edit.target.node_id) != selected.node_id
            ):
                continue
            if isinstance(edit, cmd.RemoveModelElementEdit):
                selected_identity = None
            elif isinstance(edit, cmd.RenameModelElementEdit):
                selected_identity = (selected.model_id, selected.type, edit.name)
    try:
        node_ids = self._adapter.apply_model_edit_batch(c.edits)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if len(node_ids) != len(c.edits):
        return CommandResult.bad("The model edit batch was not applied")
    self._selected = 0
    self._selected_node_id = -1
    self._refresh_structure()
    if selected_identity is not None:
        model_id, node_type, name = selected_identity
        body_types = {NodeType.LINK, NodeType.ROBOT}
        restored = next(
            (
                node
                for node in self._nodes
                if node.model_id == model_id
                and node.name == name
                and (
                    node.type is node_type or (node.type in body_types and node_type in body_types)
                )
            ),
            None,
        )
        if restored is not None:
            self._selected = restored.object_id
            self._selected_node_id = restored.node_id
    entity_id = next((node_id for node_id in reversed(node_ids) if node_id >= 0), -1)
    return CommandResult.good(f"Applied {len(c.edits)} model edits", entity_id)


def set_model_source(self: Session, c: cmd.SetModelSource) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    try:
        changed = self._adapter.set_scene_model_xml(c.model_id, c.mjcf)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"Model {c.model_id} source cannot be updated")
    self._selected = 0
    self._selected_node_id = -1
    self._refresh_structure()
    return CommandResult.good("Updated MJCF source")


def add_model_component(self: Session, c: cmd.AddModelComponent) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    try:
        component_id = self._adapter.add_model_component(c.model_id, c.category, c.subtype, c.name)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if component_id < 0:
        return CommandResult.bad(f"Failed to add {c.category} {c.name}")
    self._refresh_structure()
    return CommandResult.good(f"Added {c.category} {c.name}", component_id)


def update_model_component(self: Session, c: cmd.UpdateModelComponent) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    try:
        changed = self._adapter.update_model_component(
            c.model_id,
            c.category,
            c.component_id,
            c.name,
            c.fields,
            c.path,
        )
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"{c.category} {c.component_id} cannot be updated")
    self._refresh_structure()
    return CommandResult.good(f"Updated {c.category} {c.name}")


def remove_model_component(self: Session, c: cmd.RemoveModelComponent) -> CommandResult:
    caps = self._adapter.caps
    if not caps.topology_editing:
        return CommandResult.bad(f"{caps.name} does not support topology editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before changing model topology")
    try:
        changed = self._adapter.remove_model_component(c.model_id, c.category, c.component_id)
    except Exception as exc:
        return CommandResult.bad(str(exc))
    if not changed:
        return CommandResult.bad(f"{c.category} {c.component_id} cannot be removed")
    self._refresh_structure()
    return CommandResult.good(f"Removed {c.category}")


def add_resource_root(self: Session, c: cmd.AddResourceRoot) -> CommandResult:
    caps = self._adapter.caps
    if not caps.scene_files or not c.path.is_dir():
        return CommandResult.bad(f"Resource directory is unavailable: {c.path}")
    return (
        CommandResult.good(f"Added resource directory {c.path}")
        if self._adapter.add_resource_root(c.path)
        else CommandResult.bad(f"Failed to add resource directory {c.path}")
    )


def remove_resource_root(self: Session, c: cmd.RemoveResourceRoot) -> CommandResult:
    return (
        CommandResult.good(f"Removed resource directory {c.path}")
        if self._adapter.remove_resource_root(c.path)
        else CommandResult.bad(f"Resource directory is unavailable: {c.path}")
    )
