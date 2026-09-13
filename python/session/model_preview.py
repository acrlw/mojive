"""Backend-neutral geometry and poses for pending model edits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import GeometryShapeProperties, NodeType, SceneNode
from mojive.types import InstancePoseSource, Material, MeshKey, MeshShape


def primitive_parts(kind, size):
    """Expand authored dimensions into render meshes, scales and local transforms."""
    size = np.asarray(size, np.float32)
    local = np.eye(4, dtype=np.float32)
    if kind == "capsule":
        top, bottom = local.copy(), local.copy()
        top[2, 3], bottom[2, 3] = size[2], -size[2]
        bottom[1, 1] = bottom[2, 2] = -1
        return (
            (MeshKey(MeshShape.CAPSULE_SHAFT), (size[0], size[0], size[2]), local),
            (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, size[0]), top),
            (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, size[0]), bottom),
        )
    shape = MeshShape.SPHERE if kind == "ellipsoid" else MeshShape(kind)
    scale = size.copy()
    if kind == "sphere":
        scale[:] = size[0]
    elif kind == "cylinder":
        scale[1] = size[0]
    elif kind == "plane":
        scale[2] = 1
    return ((MeshKey(shape), scale, local),)


class GeometryPreview:
    """Compose draft primitives without compiling or changing a physics adapter."""

    def __init__(self, session):
        base = session._source
        self.base_source = base
        self._instance_indices = {}
        self.source = replace(
            base,
            geom_mesh=list(base.geom_mesh),
            geom_size=base.geom_size.copy(),
            geom_local=base.geom_local.copy(),
            geom_rgba=base.geom_rgba.copy(),
        )
        self.nodes = [replace(node, children=list(node.children)) for node in session._nodes]
        self.shapes = {}
        self.poses = {"geom": {}, "site": {}}
        self._base_frame = None
        self._frame = None
        self._buffers = {}

    def reset_sizes(self):
        """Restore dimension buffers before replaying a coalesced scale/size gesture."""
        np.copyto(self.source.geom_size, self.base_source.geom_size)
        np.copyto(self.source.geom_local, self.base_source.geom_local)

    def _instances(self, node_id):
        if node_id not in self._instance_indices:
            self._instance_indices[node_id] = np.flatnonzero(self.source.geom_node == node_id)
        return self._instance_indices[node_id]

    def add(self, session, command, creation):
        kind, _, subtype = command.element_type.partition(":")
        if kind not in ("geom", "site"):
            return
        parent = next((node for node in self.nodes if node.node_id == command.parent_node_id), None)
        if parent is None:
            return
        subtype = subtype or "sphere"
        size = (4, 4, 0.02) if subtype == "plane" else (0.03,) * 3 if kind == "site" else (0.1,) * 3
        parts = primitive_parts(subtype, size)
        base_positions = getattr(session._frame, f"{kind}_xpos")
        index = len(base_positions) if base_positions is not None else 0
        index += len(self.poses[kind])
        body = max(0, parent.body_index)
        position, rotation = np.zeros(3, np.float32), np.eye(3, dtype=np.float32)
        if session._frame.body_xpos is not None and body < len(session._frame.body_xpos):
            position = session._frame.body_xpos[body].copy()
            rotation = session._frame.body_xmat[body].copy()
        if parent.type is NodeType.MODEL:
            model = next(item for item in session.scene_models if item.model_id == parent.model_id)
            position, rotation = np.asarray(model.position), np.asarray(model.rotation)
        self.poses[kind][index] = (position, rotation)
        node = SceneNode(
            creation.node_id,
            creation.name,
            NodeType(kind),
            parent=parent.node_id,
            object_id=creation.node_id,
            posable=True,
            body_index=body,
            geom_index=index if kind == "geom" else -1,
            site_index=index if kind == "site" else -1,
            model_id=max(0, parent.model_id),
            source_editable=True,
        )
        parent.children.append(node.node_id)
        self.nodes.append(node)
        self.shapes[node.node_id] = GeometryShapeProperties(node.node_id, subtype, "")
        source = self.source
        material = next(
            (i for i, item in enumerate(source.materials) if item.name == "__geom__"), -1
        )
        if material < 0:
            material = len(source.materials)
            source.materials = [*source.materials, Material(name="__geom__")]
        self._append(parts, node, index, material, kind)

    def _append(self, parts, node, index, material, kind):
        self._instance_indices.clear()
        source = self.source
        count, added = source.instance_count, len(parts)
        for name, values in (
            ("geom_mesh", [part[0] for part in parts]),
            ("geom_convex_mesh", [part[0] for part in parts]),
            ("geom_material", [material] * added),
        ):
            old = getattr(source, name)
            if name == "geom_convex_mesh" and len(old) != count:
                old = source.geom_mesh[:count]
            setattr(source, name, [*old, *values])
        for name, values, default in (
            ("geom_size", [part[1] for part in parts], 1),
            ("geom_local", [part[2] for part in parts], np.eye(4)),
            ("geom_rgba", [(0.5, 0.5, 0.5, 1)] * added, 1),
            ("geom_object_id", [node.object_id] * added, 0),
            ("geom_body", [node.body_index] * added, 0),
            ("geom_source", [index] * added, 0),
            (
                "geom_pose_source",
                [int(InstancePoseSource.GEOM if kind == "geom" else InstancePoseSource.SITE)]
                * added,
                0,
            ),
            ("geom_visual", [0] * added, 0),
            ("geom_static", [True] * added, False),
            ("instance_island_body", [-1] * added, -1),
            ("geom_node", [node.node_id] * added, -1),
            ("geom_infinite_plane", [False] * added, False),
            ("geom_segmentation", [(-1, -1)] * added, -1),
        ):
            old = getattr(source, name)
            extra = np.asarray(values, dtype=old.dtype)
            if len(old) != count:
                old = np.broadcast_to(default, (count, *extra.shape[1:])).astype(old.dtype)
            setattr(source, name, np.concatenate((old, extra)))

    def edit(self, command, node_id):
        source = self.source
        if isinstance(command, cmd.SetGeometrySize):
            for index in self._instances(node_id):
                shape = source.geom_mesh[index].shape
                size = np.asarray(command.size, np.float32)
                if shape is MeshShape.CAPSULE_CAP:
                    source.geom_size[index] = size[0]
                    local = source.geom_local[index]
                    local[2, 3] = np.copysign(size[2], local[2, 2])
                else:
                    source.geom_size[index] = size
        elif isinstance(command, cmd.SetScale):
            source.geom_size[self._instances(node_id)] *= command.scale
        elif isinstance(command, cmd.SetGeometryColor):
            source.geom_rgba[source.geom_node == node_id] = command.rgba
        elif isinstance(command, cmd.SetPose):
            node = next((node for node in self.nodes if node.node_id == node_id), None)
            if node is not None and node.type in (NodeType.GEOM, NodeType.SITE):
                kind = "geom" if node.type is NodeType.GEOM else "site"
                index = node.geom_index if kind == "geom" else node.site_index
                self.poses[kind][index] = (
                    np.asarray(command.position),
                    np.asarray(command.rotation),
                )
        elif isinstance(command, cmd.RenameModelElement):
            for node in self.nodes:
                if node.node_id == node_id:
                    node.name = command.name.strip()
        elif isinstance(command, cmd.RemoveModelElement):
            self._instance_indices.clear()
            removed = {node_id}
            for node in self.nodes:
                if node.parent in removed:
                    removed.add(node.node_id)
            keep = np.flatnonzero(~np.isin(source.geom_node, tuple(removed)))
            count = source.instance_count
            for name in vars(source):
                if name.startswith("geom_") or name == "instance_island_body":
                    value = getattr(source, name)
                    if isinstance(value, (list, np.ndarray)) and len(value) == count:
                        setattr(
                            source,
                            name,
                            value[keep]
                            if isinstance(value, np.ndarray)
                            else [value[i] for i in keep],
                        )
            self.nodes = [node for node in self.nodes if node.node_id not in removed]
            for node in self.nodes:
                node.children[:] = [child for child in node.children if child not in removed]

    def frame(self, base):
        if self._base_frame is base:
            return self._frame
        values = {}
        for kind, poses in self.poses.items():
            if not poses:
                continue
            for suffix, component, shape in (("xpos", 0, (3,)), ("xmat", 1, (3, 3))):
                name = f"{kind}_{suffix}"
                original = getattr(base, name)
                count = max(len(original) if original is not None else 0, max(poses) + 1)
                buffer = self._buffers.get(name)
                if buffer is None or buffer.shape != (count, *shape):
                    buffer = self._buffers[name] = np.zeros((count, *shape), np.float32)
                if original is not None:
                    buffer[: len(original)] = original
                for index, pose in poses.items():
                    buffer[index] = pose[component]
                values[name] = buffer
        self._base_frame = base
        self._frame = replace(base, **values) if values else base
        return self._frame
