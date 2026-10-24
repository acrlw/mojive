"""Entity view models projected from Session, with edits routed through commands."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import numpy as np

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import NodeType
from mojive.scene.queries import node_geometry_indices, node_world_pose
from mojive.session import Session


def rgba(value):
    """Convert a native palette color into a geometry color command value."""
    return (*[int(value[i : i + 2], 16) / 255 for i in (1, 3, 5)], 1.0)


class EntityProjection:
    def __init__(self, app):
        self.app = app
        self.entities = []
        self.initial_entities = []
        self._initial_by_identity = {}
        self._retained_identities = None
        self._entity_cache = {}
        self._document_id = None
        self._structure_generation = -1
        self._node_ids_by_identity = {}
        self._projected_entities = []
        self.refresh_document()

    @property
    def selected(self):
        node = self.app.session.selected_node
        return node.name if node is not None else None

    @selected.setter
    def selected(self, name):
        if name is None:
            self.app.session.submit(cmd.Select(0))
            return
        matches = [n for n in self.app.session.nodes if n.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one scene node named {name!r}, found {len(matches)}")
        self.app.session.submit(cmd.SelectNode(matches[0].node_id))

    @property
    def entity(self):
        node = self.app.session.selected_node
        if node is None:
            return None
        entry = next((e for e in self.entities if e["node_id"] == node.node_id), None)
        if entry is None:
            self.refresh_document()
            entry = next((e for e in self.entities if e["node_id"] == node.node_id), None)
        if entry is not None:
            entry["hidden"] = not node.visible
        return entry

    def refresh_document(self):
        """Project selected and pinned entities; retain load-time values for Restore."""
        session = self.app.session
        new_document = self._document_id != self.app.session.document_id
        if new_document:
            self.initial_entities.clear()
            self._initial_by_identity.clear()
            if getattr(self.app, "workspace", None) is not None:
                self.app.workspace.pinned = None
        retained = session.retained_node_identities
        originals_changed = False
        if retained is not self._retained_identities:
            self._retained_identities = retained
            removed = self._initial_by_identity.keys() - retained
            for identity in removed:
                del self._initial_by_identity[identity]
            originals_changed = bool(removed)
        if new_document or self._structure_generation != session.structure_generation:
            self._entity_cache.clear()
            self._node_ids_by_identity = {
                self._identity(node): node.node_id for node in session.nodes
            }
            for node in session.nodes:
                identity = self._identity(node)
                if identity not in self._initial_by_identity:
                    self._initial_by_identity[identity] = {
                        **self._transform_snapshot(node),
                        "rgba": self._node_rgba(node),
                    }
                    originals_changed = True
            self._structure_generation = session.structure_generation
            self._document_id = session.document_id
        if originals_changed:
            self.initial_entities[:] = self._initial_by_identity.values()
        active = []
        node = session.selected_node
        if node is not None:
            active.append(node.node_id)
        pinned = getattr(getattr(self.app, "workspace", None), "pinned", None)
        pinned_id = self._node_ids_by_identity.get(pinned)
        if pinned_id is not None and pinned_id not in active:
            active.append(pinned_id)
        self.entities.clear()
        for node_id in active:
            node = session.node(node_id)
            if node is not None:
                entry = self._entity_cache.setdefault(node_id, {})
                entry.update(self._project_node(node))
                self.entities.append(entry)
        # The edit baseline is limited to the at most two presented entities;
        # physics frames never deep-copy the entire scene into UI state.
        self._projected_entities = copy.deepcopy(self.entities)

    @staticmethod
    def _identity(node):
        return Session.node_identity(node)

    def initial_entity(self, identity):
        """Look up a retained load-time Restore baseline without scanning history."""
        return self._initial_by_identity.get(identity)

    def _project_node(self, node):
        from .navigation import node_accent, node_icon

        rgba = self._node_rgba(node)
        return {
            "id": str(node.node_id),
            "node_id": node.node_id,
            "object_id": node.object_id,
            "parent": node.parent,
            "name": node.name,
            "type": node.type.value,
            **self._transform_snapshot(node),
            "icon": node_icon(node, self.app.session),
            "icon_color": node_accent(self.app.theme, node.type.value),
            "color": "#" + "".join(f"{round(np.clip(v, 0, 1) * 255):02x}" for v in rgba[:3]),
            "rgba": rgba,
            "hidden": not node.visible,
            "posable": node.posable,
            "scalable": self.app.session.scale_target(node.node_id) is not None,
            "colorable": node.type is NodeType.GEOM
            or any(self.app.session.node(child).type is NodeType.GEOM for child in node.children),
        }

    def _transform_snapshot(self, node):
        """Capture numeric transform defaults without building display fields."""
        position, rotation = node_world_pose(self.app.session, node)
        return {
            "identity": self._identity(node),
            "position": position.tolist(),
            "rotation": np.degrees(math3d.mat3_to_euler_xyz(rotation)).tolist(),
            "scale": list(self.app.session.scale_factors(node.node_id)),
        }

    def _node_rgba(self, node):
        indices = node_geometry_indices(self.app.session, node)
        value = (
            self.app.session.source.geom_rgba[indices[0]] if len(indices) else (0.6, 0.7, 0.8, 1)
        )
        return [float(channel) for channel in value]

    def apply_document_edits(self):
        """Route reference field changes through the same command boundary as gizmos."""
        previous = {e["node_id"]: e for e in self._projected_entities}
        changed = False
        for entry in self.entities:
            old = previous.get(entry["node_id"])
            if old is None:
                continue
            nid = entry["node_id"]
            edits = []
            if (entry["position"], entry["rotation"]) != (old["position"], old["rotation"]):
                edits.append(
                    cmd.SetPose(
                        nid,
                        np.array(entry["position"]),
                        math3d.euler_xyz_to_mat3(np.radians(entry["rotation"])),
                    )
                )
            if entry["scale"] != old["scale"]:
                edits.append(cmd.SetScale(nid, np.array(entry["scale"])))
            if entry["hidden"] != old["hidden"]:
                edits.append(cmd.SetVisible(nid, not entry["hidden"]))
            if entry["color"] != old["color"] or entry["rgba"] != old["rgba"]:
                node = self.app.session.node(nid)
                geometry = (
                    [node]
                    if node.type is NodeType.GEOM
                    else [
                        self.app.session.node(c)
                        for c in node.children
                        if self.app.session.node(c).type is NodeType.GEOM
                    ]
                )
                if entry["rgba"] == old["rgba"]:
                    entry["rgba"][:3] = [
                        int(entry["color"][i : i + 2], 16) / 255 for i in (1, 3, 5)
                    ]
                rgba = np.array(entry["rgba"])
                edits.extend(cmd.SetGeometryColor(g.node_id, rgba) for g in geometry)
            if edits:
                changed = True
                # Single fields retain the UI draft interceptor; color gestures
                # retain their existing transaction rather than nesting a batch.
                active = self.app.session.editing or self.app.model_edits._checkpoint is not None
                group = (
                    nullcontext()
                    if len(edits) == 1 or active
                    else self.app.session.edit("Edit " + entry["name"])
                )
                try:
                    with group:
                        for command in edits:
                            result = self.app.session.submit(command)
                            if not result.ok:
                                raise RuntimeError(result.message)
                except RuntimeError as error:
                    self.app.session.report_message(str(error), level="warning")
        # Re-project accepted state, including rejected or normalized edits.
        if changed:
            self.refresh_document()
