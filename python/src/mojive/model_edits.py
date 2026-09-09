"""Pending model commands and render-only editor previews."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import fields, replace

import numpy as np

from . import commands as cmd

_COMMAND_SCOPE = ContextVar("model_edit_command_scope", default=None)

# These commands change model declarations or derived constants. Runtime camera,
# material, pose, joint position and actuator control writes keep their direct path.
MODEL_REBUILD_COMMANDS = (
    cmd.SetGeometrySize,
    cmd.SetSceneModelTransform,
    cmd.AddModelElement,
    cmd.DuplicateModelElement,
    cmd.RemoveModelElement,
    cmd.RemoveSceneModel,
    cmd.CreateModelMaterial,
    cmd.RenameModelElement,
    cmd.ModelEditBatch,
    cmd.SetModelSource,
    cmd.AddModelKeyframe,
    cmd.SetModelKeyframe,
    cmd.RemoveModelKeyframe,
    cmd.AddModelComponent,
    cmd.UpdateModelComponent,
    cmd.RemoveModelComponent,
    cmd.SetJointAdvancedProperties,
    cmd.SetSiteProperties,
    cmd.SetGeometryAdvancedProperties,
    cmd.SetGeometryShape,
    cmd.SetBodyProperties,
    cmd.ImportModelGeometryResource,
    cmd.ImportModelAsset,
    cmd.SetHeightFieldSize,
    cmd.RenameModelAsset,
    cmd.DuplicateModelAsset,
    cmd.ReplaceModelAssetFile,
    cmd.RemoveModelAsset,
    cmd.AddModelMaterial,
    cmd.ImportModelTexture,
)

_PROPERTY_COMMANDS = {
    "joint_advanced_properties": cmd.SetJointAdvancedProperties,
    "site_properties": cmd.SetSiteProperties,
    "geometry_advanced_properties": cmd.SetGeometryAdvancedProperties,
    "geometry_shape_properties": cmd.SetGeometryShape,
    "body_properties": cmd.SetBodyProperties,
    "keyframe_properties": cmd.SetModelKeyframe,
}


@contextmanager
def model_edit_scope(session, handler):
    """Intercept UI commands only on the calling thread and for this Session."""
    token = _COMMAND_SCOPE.set((session, handler) if handler is not None else None)
    try:
        yield
    finally:
        _COMMAND_SCOPE.reset(token)


def intercept_model_edit(session, command):
    scope = _COMMAND_SCOPE.get()
    if scope is None or scope[0] is not session:
        return None
    with model_edit_scope(session, None):
        return scope[1](command)


class ModelEditDraft:
    """Coalesce pending edits without changing compiled physics or stored source."""

    def __init__(self, session):
        self.session = session
        self.commands = []
        self.applying = False
        self.source = None
        self.nodes = []
        self.by_node_id = {}
        self.error = ""
        self._document_id = None
        self._adapter_revision = None
        self._checkpoint = None
        self._transaction_label = None
        self._property_overrides = {}
        session._model_edit_preview = self

    @property
    def active(self):
        return bool(self.commands)

    @property
    def visible(self):
        return self.active and not self.applying and self.compatible()

    @staticmethod
    def _key(command):
        if not type(command).__name__.startswith(("Set", "Update", "Rename")):
            return None
        return (
            type(command),
            *(
                getattr(command, name, None)
                for name in (
                    "node_id",
                    "joint_id",
                    "keyframe_id",
                    "model_id",
                    "component_id",
                    "category",
                    "asset_type",
                    *(
                        ("name",)
                        if (
                            hasattr(command, "asset_type")
                            or isinstance(command, cmd.SetHeightFieldSize)
                        )
                        else ()
                    ),
                )
            ),
        )

    def stage(self, command):
        if not self.session.paused:
            return cmd.CommandResult.bad("Pause simulation before editing model declarations")
        if self.active and not self.compatible():
            return cmd.CommandResult.bad("The scene changed; discard the pending model edits")
        if isinstance(command, cmd.SetGeometrySize):
            size = np.asarray(command.size)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                return cmd.CommandResult.bad("Geometry size must contain three positive values")
        if isinstance(command, cmd.RenameModelElement):
            name = command.name.strip()
            node = self.session.node(command.node_id)
            if not name or node is None:
                return cmd.CommandResult.bad("An existing entity and a nonempty name are required")
            if any(
                n.node_id != node.node_id
                and n.model_id == node.model_id
                and n.type == node.type
                and n.name == name
                for n in self.session.nodes
            ):
                return cmd.CommandResult.bad(f"An entity named {name!r} already exists")
        if isinstance(command, cmd.SetSceneModelTransform):
            result = self.session.submit(
                cmd.PreviewSceneModelTransform(command.model_id, command.position, command.rotation)
            )
            if not result.ok:
                return result
        if not self.active:
            self._document_id = self.session.document_id
            self._adapter_revision = self.session.adapter.structure_revision
        key = self._key(command)
        index = next(
            (
                i
                for i, item in enumerate(self.commands)
                if key is not None and self._key(item) == key
            ),
            None,
        )
        if index is None:
            if len(self.commands) >= 512:
                return cmd.CommandResult.bad("Apply pending edits before adding more operations")
            self.commands.append(deepcopy(command))
        else:
            self.commands[index] = deepcopy(command)
        self.error = ""
        self.refresh_preview()
        return cmd.CommandResult.good()

    def compatible(self):
        return (
            self._document_id == self.session.document_id
            and self._adapter_revision == self.session.adapter.structure_revision
        )

    def refresh_preview(self):
        session = self.session
        base = session._source
        self.source = replace(base, geom_size=base.geom_size.copy()) if base is not None else None
        self.nodes = list(session._nodes)
        for command in self.commands:
            if isinstance(command, cmd.RenameModelElement):
                self.nodes = [
                    replace(node, name=command.name.strip())
                    if node.node_id == command.node_id
                    else node
                    for node in self.nodes
                ]
            elif isinstance(command, cmd.SetGeometrySize) and self.source is not None:
                self.source.geom_size[self.source.geom_node == command.node_id] = command.size
        self.by_node_id = {node.node_id: node for node in self.nodes}
        self._property_overrides.clear()
        for command in self.commands:
            for name, kind in _PROPERTY_COMMANDS.items():
                if isinstance(command, kind):
                    identity = getattr(
                        command,
                        "node_id",
                        getattr(command, "joint_id", getattr(command, "keyframe_id", None)),
                    )
                    self._property_overrides[name, identity] = {
                        field.name: getattr(command, field.name)
                        for field in fields(command)
                        if getattr(command, field.name) is not None
                    }
        session._mesh_bounds_cache.clear()
        session._scene_bounds = None
        session._preview_generation += 1

    def rebase_after_failure(self):
        self.applying = False
        self._adapter_revision = self.session.adapter.structure_revision
        with model_edit_scope(self.session, None):
            for command in self.commands:
                if isinstance(command, cmd.SetSceneModelTransform):
                    self.session.submit(
                        cmd.PreviewSceneModelTransform(
                            command.model_id, command.position, command.rotation
                        )
                    )
        self.refresh_preview()

    def properties(self, name, identity, original):
        if original is None or not self.visible:
            return original
        values = self._property_overrides.get((name, identity))
        if not values:
            return original
        names = {item.name for item in fields(original)}
        return replace(original, **{key: value for key, value in values.items() if key in names})

    def begin_transaction(self, label):
        self._checkpoint = list(self.commands)
        self._transaction_label = label

    def end_transaction(self, cancel=False):
        if cancel and self._checkpoint is not None:
            self.commands = self._checkpoint
            self.refresh_preview()
        self._checkpoint = self._transaction_label = None

    def clear(self):
        with model_edit_scope(self.session, None):
            for command in self.commands:
                if isinstance(command, cmd.SetSceneModelTransform):
                    self.session.adapter.clear_scene_model_transform_preview(command.model_id)
        self.commands.clear()
        self.applying = False
        self.source = None
        self.nodes = []
        self.by_node_id.clear()
        self.error = ""
        self.end_transaction()
        self.session._mesh_bounds_cache.clear()
        self.session._scene_bounds = None
        self.session._preview_generation += 1
