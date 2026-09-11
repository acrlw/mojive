"""Pending model commands and render-only editor previews."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import numpy as np

from . import commands as cmd
from .adapters.base import NodeType
from .model_preview import GeometryPreview

if TYPE_CHECKING:
    from .session import Session

_NodeIdentity = tuple[int, NodeType, str]
_Commands = tuple[cmd.Command, ...]

_COMMAND_SCOPE = ContextVar("model_edit_command_scope", default=None)


@dataclass(frozen=True)
class _StagedCreation:
    """One staged command that adds a hierarchy element under a known parent.

    A duplicated element is named when the batch runs, so its recorded identity carries
    the source node instead of a final name.
    """

    key: str
    parent_node_id: int
    node_type: NodeType
    name: str
    source_node_id: int = -1
    node_id: int = -1

    @property
    def copied(self) -> bool:
        return self.source_node_id >= 0


@dataclass(frozen=True)
class _ModelEditPlan:
    """One pending batch split into element creation and the edits that address it.

    ``direct`` holds the commands that apply against existing nodes, and ``creations``
    holds the commands that add elements. ``bind`` maps creation keys to real node IDs
    after the creation phase applied, so styling and selection follow-ups can join the
    same undoable transaction.
    """

    direct: _Commands
    creations: _Commands = ()
    bind: Callable[[Session, tuple[_NodeIdentity, ...]], _Commands] | None = None
    rebind: Callable[[Session, _Commands], _Commands] | None = None


# These commands change model declarations or derived constants. Runtime camera,
# material, pose, joint position and actuator control writes keep their direct path.
# A geometry color keeps its direct path too, unless it styles an element that this
# batch creates, which only becomes addressable after Apply.
MODEL_REBUILD_COMMANDS = (
    cmd.SetGeometrySize,
    cmd.SetGeometryColor,
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

# Element tags accepted by AddModelElement, mapped to the node each one creates.
_ELEMENT_NODE_TYPES = {
    "body": NodeType.LINK,
    "geom": NodeType.GEOM,
    "joint": NodeType.JOINT,
    "site": NodeType.SITE,
    "camera": NodeType.CAMERA,
    "light": NodeType.LIGHT,
}


def _element_node_type(element_type) -> NodeType | None:
    """Return the hierarchy node type an authored element tag produces."""

    return _ELEMENT_NODE_TYPES.get(str(element_type).partition(":")[0])


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
        self.by_object_id = {}
        self.geometry = None
        self.error = ""
        self._document_id = None
        self._adapter_revision = None
        self._checkpoint = None
        self._transaction_label = None
        self._property_overrides = {}
        self._creations = {}
        self._next_creation_key = 0
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
                    "node_key",
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

    def applies_immediately(self, command) -> bool:
        """Return whether a command must keep its immediate path while edits are deferred.

        A geometry color that addresses an existing node applies at once, so a color
        picker still writes through. Only a color bound to an element this batch creates
        waits for Apply, because that element has no node ID yet.
        """

        if not isinstance(command, cmd.SetGeometryColor):
            return False
        if self.creation_for_node(command.node_id) is not None:
            return False
        key = str(getattr(command, "node_key", ""))
        if not key:
            return True
        return key not in self._creations

    def stage(self, command):
        if self.applies_immediately(command):
            # The command owns scene state this batch does not create, so it applies now.
            return None
        if not self.session.paused:
            return cmd.CommandResult.bad("Pause simulation before editing model declarations")
        if self.active and not self.compatible():
            return cmd.CommandResult.bad("The scene changed; discard the pending model edits")
        if isinstance(command, cmd.SetGeometrySize):
            size = np.asarray(command.size)
            if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                return cmd.CommandResult.bad("Geometry size must contain three positive values")
        if isinstance(command, cmd.AddModelKeyframe):
            name = str(command.name).strip()
            if not name:
                return cmd.CommandResult.bad("Keyframe name cannot be empty")
            if name in self.model_keyframe_names(command.model_id):
                return cmd.CommandResult.bad(f"Keyframe {name} already exists")
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
        # A command that creates an element, or addresses one this draft created, reports
        # its batch-local key so the caller can stage the styling and selection with it.
        created = self._creation_key(command) or self._follow_up_key(command)
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
        if isinstance(command, cmd.SelectNode):
            node = self.by_node_id.get(self.command_node_id(command))
            if node is not None:
                self.session._selected_node_id = node.node_id
                self.session._selected = node.object_id
                self.session._selection_revision += 1
        creation = self._creations.get(created)
        return cmd.CommandResult.good(
            "", entity_id=creation.node_id if creation else -1, entity_key=created
        )

    def model_keyframe_names(self, model_id: int) -> set[str]:
        """Return compiled and pending names reserved by one model edit batch."""

        target = int(model_id)
        names = {
            keyframe.name for keyframe in self.session.keyframes if int(keyframe.model_id) == target
        }
        for command in self.commands:
            if (
                isinstance(command, (cmd.AddModelKeyframe, cmd.SetModelKeyframe))
                and int(command.model_id) == target
            ):
                name = str(command.name).strip()
                if name:
                    names.add(name)
        return names

    def creation_for_node(self, node_id):
        return next((item for item in self._creations.values() if item.node_id == node_id), None)

    def command_node_id(self, command):
        key = str(getattr(command, "node_key", ""))
        creation = self._creations.get(key)
        return creation.node_id if creation else getattr(command, "node_id", -1)

    def compatible(self):
        return (
            self._document_id == self.session.document_id
            and self._adapter_revision == self.session.adapter.structure_revision
        )

    def resolve_commands(self, session):
        """Split pending edits into the creation phase and its bound follow-up phase.

        An element created by a staged command has no node ID until that command applies,
        so its styling and selection commands address it by a batch-local key. The
        returned plan exposes a binder that maps those keys to the real nodes once the
        creation phase ran, and keeps every edit inside the same undoable transaction.
        """

        if not self._creations:
            return _ModelEditPlan(tuple(self.commands))
        selected_creation = self.creation_for_node(session._selected_node_id)
        selected_object = session.selected
        original_nodes = {
            node.node_id: (node.model_id, node.type, node.name) for node in session._nodes
        }
        creations = []
        creation_keys = []
        follow_ups = {}
        for command in self.commands:
            key = self._follow_up_key(command)
            if key:
                follow_ups.setdefault(key, []).append(command)
                continue
            creation = self._creation(command)
            if creation is not None:
                creations.append(command)
                creation_keys.append(creation.key)
                follow_ups.setdefault(creation.key, [])
            else:
                follow_ups.setdefault("", []).append(command)
        return _ModelEditPlan(
            tuple(follow_ups.get("", ())),
            tuple(creations),
            lambda session, identities: self._bind_creations(
                session,
                follow_ups,
                dict(zip(creation_keys, identities, strict=True)),
                selected_creation,
                selected_object,
            ),
            lambda session, commands: self._bind_existing(session, commands, original_nodes),
        )

    @staticmethod
    def _bind_existing(session, commands, original_nodes):
        nodes = {(node.model_id, node.type, node.name): node.node_id for node in session.nodes}
        bound = []
        for command in commands:
            changes = {}
            for field in ("node_id", "parent_node_id"):
                identity = original_nodes.get(getattr(command, field, -1))
                if identity is not None:
                    if identity not in nodes:
                        raise ValueError(f"Edited entity {identity[2]!r} no longer exists")
                    changes[field] = nodes[identity]
            bound.append(replace(command, **changes) if changes else command)
        return tuple(bound)

    def _bind_creations(
        self, session, follow_ups, identities, selected_creation, selected_object
    ) -> _Commands:
        """Bind to the identities returned by creation, after all nodes are installed."""
        nodes = {(node.model_id, node.type, node.name): node.node_id for node in session.nodes}
        bound = []
        for key, identity in identities.items():
            node_id = nodes.get(identity)
            if node_id is None:
                raise ValueError(f"Created entity {identity[2]!r} no longer exists")
            for command in follow_ups.get(key, ()):
                if isinstance(command, cmd.SelectNode):
                    continue
                changes = {"node_id": node_id}
                if hasattr(command, "node_key"):
                    changes["node_key"] = ""
                bound.append(replace(command, **changes))
        if selected_creation is not None:
            bound.append(cmd.SelectNode(nodes[identities[selected_creation.key]]))
        elif selected_object and session.node_by_object_id(selected_object) is not None:
            bound.append(cmd.Select(selected_object))
        return tuple(bound)

    def _creation(self, command):
        """Return the staged creation that a creation command produces."""

        for creation in self._creations.values():
            if isinstance(command, cmd.AddModelElement):
                if creation.copied:
                    continue
                matches = (
                    command.parent_node_id == creation.parent_node_id
                    and _element_node_type(command.element_type) is creation.node_type
                    and command.name == creation.name
                )
            elif isinstance(command, cmd.DuplicateModelElement):
                matches = command.node_id == creation.source_node_id
            else:
                continue
            if matches:
                return creation
        return None

    def _follow_up_key(self, command) -> str:
        """Return the creation key of an element this draft already queued.

        Applying such a command now would fail on a missing node ID, so a non-empty key
        keeps it queued until the batch that creates the element applies.
        """

        key = str(getattr(command, "node_key", ""))
        creation = self.creation_for_node(getattr(command, "node_id", -1))
        return key if key in self._creations else creation.key if creation is not None else ""

    def _creation_key(self, command) -> str:
        """Assign a batch-local key to a command that creates one hierarchy element."""

        if isinstance(command, cmd.AddModelElement):
            node_type = _element_node_type(command.element_type)
            if node_type is None:
                return ""
            identity = (command.parent_node_id, node_type, command.name)
            source_node_id = -1
        elif isinstance(command, cmd.DuplicateModelElement):
            node = self.session.node(command.node_id)
            if node is None:
                return ""
            identity = (node.parent, node.type, node.name)
            source_node_id = node.node_id
        else:
            return ""
        for existing in self._creations.values():
            if (
                existing.parent_node_id,
                existing.node_type,
                existing.name,
                existing.source_node_id,
            ) == (*identity, source_node_id):
                return existing.key
        key = f"create#{self._next_creation_key}"
        self._next_creation_key += 1
        self._creations[key] = _StagedCreation(
            key, *identity, source_node_id, 0x7E000000 + self._next_creation_key
        )
        return key

    def refresh_preview(self):
        session = self.session
        base = session._source
        self.geometry = GeometryPreview(session) if base is not None else None
        for command in self.commands:
            if self.geometry is not None:
                if isinstance(command, cmd.AddModelElement):
                    self.geometry.add(session, command, self._creation(command))
                else:
                    self.geometry.edit(command, self.command_node_id(command))
        self.source = self.geometry.source if self.geometry is not None else None
        self.nodes = self.geometry.nodes if self.geometry is not None else list(session._nodes)
        if self.source is not None:
            self.source.nodes = self.nodes
        self.by_node_id = {node.node_id: node for node in self.nodes}
        self.by_object_id = {node.object_id: node for node in self.nodes if node.object_id}
        if session._selected_node_id not in self.by_node_id:
            session._selected_node_id = -1
            session._selected = 0
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
        if not self.visible:
            return original
        if name == "geometry_shape_properties" and self.geometry is not None:
            original = self.geometry.shapes.get(identity, original)
        if original is None:
            return None
        values = self._property_overrides.get((name, identity))
        if not values:
            return original
        names = {item.name for item in fields(original)}
        return replace(original, **{key: value for key, value in values.items() if key in names})

    def begin_transaction(self, label):
        self._checkpoint = (list(self.commands), dict(self._creations), self._next_creation_key)
        self._transaction_label = label

    def end_transaction(self, cancel=False):
        if cancel and self._checkpoint is not None:
            # A cancelled gesture drops the elements it staged along with their keys, so
            # keystrokes bound to them never reach an Apply batch.
            self.commands, self._creations, self._next_creation_key = (
                self._checkpoint[0],
                self._checkpoint[1],
                self._checkpoint[2],
            )
            self.refresh_preview()
        self._checkpoint = self._transaction_label = None

    def clear(self):
        with model_edit_scope(self.session, None):
            for command in self.commands:
                if isinstance(command, cmd.SetSceneModelTransform):
                    self.session.adapter.clear_scene_model_transform_preview(command.model_id)
        self.commands.clear()
        # Creation keys only mean something inside one pending command list.
        self._creations.clear()
        self._next_creation_key = 0
        self.applying = False
        self.source = None
        self.geometry = None
        self.nodes = []
        self.by_node_id.clear()
        self.by_object_id.clear()
        if self.session._selected_node_id not in self.session._by_node_id:
            self.session._selected_node_id = -1
            self.session._selected = 0
        self.error = ""
        self.end_transaction()
        self.session._mesh_bounds_cache.clear()
        self.session._scene_bounds = None
        self.session._preview_generation += 1
