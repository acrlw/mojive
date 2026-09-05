"""Shared finite bounds for rendered instances, scene framing, and authoring tools."""

from __future__ import annotations

import numpy as np

from .adapters.base import NodeType, SceneFrame, SceneNode, SceneSource
from .types import Bounds, CenteredBounds, InstancePoseSource, MeshData, MeshKey, MeshShape

_BOUND_CORNER_SIGNS = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    np.float64,
)

_MeshBoundsCache = dict[MeshKey, tuple[MeshData, Bounds]]


def _instance_mesh_bounds(
    source: SceneSource,
    instance: int,
    frame: SceneFrame | None = None,
    cache: _MeshBoundsCache | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    key = source.geom_mesh[instance]
    mesh = source.meshes.get(key)
    if mesh is not None:
        dynamic = key in source.dynamic_meshes
        cached = None if dynamic or cache is None else cache.get(key)
        if cached is not None and cached[0] is mesh:
            return cached[1]
        update = (frame.mesh_updates or {}).get(key) if frame is not None else None
        positions = update.positions if update is not None else mesh.positions
        points = np.asarray(positions, np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return None
        result = Bounds(points.min(axis=0), points.max(axis=0))
        if not dynamic and cache is not None:
            cache[key] = mesh, result
        return result

    if key.shape in {
        MeshShape.ASSET,
        MeshShape.HEIGHTFIELD,
        MeshShape.FLEX,
        MeshShape.FLEX_FACE,
        MeshShape.SKIN,
        MeshShape.CONVEX_HULL,
    }:
        return None
    if key.shape in {MeshShape.PLANE, MeshShape.DISK}:
        return np.array((-1.0, -1.0, 0.0)), np.array((1.0, 1.0, 0.0))
    if key.shape in {
        MeshShape.CAPSULE_CAP,
        MeshShape.ARROW_SHAFT,
        MeshShape.ARROW_HEAD,
        MeshShape.ARROW,
    }:
        return np.array((-1.0, -1.0, 0.0)), np.ones(3, np.float64)
    return -np.ones(3, np.float64), np.ones(3, np.float64)


def _instance_world_corners(
    source: SceneSource,
    frame: SceneFrame,
    instance: int,
    cache: _MeshBoundsCache | None = None,
) -> np.ndarray | None:
    """Return one finite rendered instance's eight world-space bound corners."""

    count = source.instance_count
    if len(source.geom_size) != count:
        return None
    pose_index = int(source.geom_source[instance]) if len(source.geom_source) == count else instance
    pose_source = (
        InstancePoseSource(int(source.geom_pose_source[instance]))
        if len(source.geom_pose_source) == count
        else InstancePoseSource.GEOM
    )
    if pose_source is InstancePoseSource.WORLD:
        geom_position, geom_rotation = np.zeros(3), np.eye(3)
    else:
        positions, rotations = (
            (frame.site_xpos, frame.site_xmat)
            if pose_source is InstancePoseSource.SITE
            else (frame.geom_xpos, frame.geom_xmat)
        )
        if (
            positions is None
            or rotations is None
            or not 0 <= pose_index < min(len(positions), len(rotations))
        ):
            return None
        geom_position = np.asarray(positions[pose_index], np.float64).reshape(3)
        geom_rotation = np.asarray(rotations[pose_index], np.float64).reshape(3, 3)
    mesh_bounds = _instance_mesh_bounds(source, instance, frame, cache)
    if mesh_bounds is None:
        return None
    mesh_lo, mesh_hi = mesh_bounds
    mesh_center = (mesh_lo + mesh_hi) * 0.5
    mesh_half = (mesh_hi - mesh_lo) * 0.5
    size = np.asarray(source.geom_size[instance], np.float64).reshape(3)
    local = (
        np.asarray(source.geom_local[instance], np.float64).reshape(4, 4)
        if len(source.geom_local) == count
        else np.eye(4, dtype=np.float64)
    )
    if not all(
        np.isfinite(value).all()
        for value in (mesh_center, mesh_half, size, local, geom_position, geom_rotation)
    ):
        return None
    points = _BOUND_CORNER_SIGNS * mesh_half + mesh_center
    points = (points * size) @ local[:3, :3].T + local[:3, 3]
    return points @ geom_rotation.T + geom_position


def _node_local_bounds(
    source: SceneSource | None,
    frame: SceneFrame,
    node: SceneNode,
    cache: _MeshBoundsCache | None = None,
) -> CenteredBounds | None:
    """Return the selected geom or body's finite geometry bound in its body frame."""

    if source is None or source.instance_count == 0 or node.body_index < 0:
        return None
    if frame.body_xpos is None or frame.body_xmat is None:
        return None
    if frame.geom_xpos is None or frame.geom_xmat is None:
        return None

    body = int(node.body_index)
    if body >= len(frame.body_xpos) or body >= len(frame.body_xmat):
        return None
    body_position = np.asarray(frame.body_xpos[body], np.float64).reshape(3)
    body_rotation = np.asarray(frame.body_xmat[body], np.float64).reshape(3, 3)
    if not np.isfinite(body_position).all() or not np.isfinite(body_rotation).all():
        return None

    count = source.instance_count
    if len(source.geom_body) != count or len(source.geom_size) != count:
        return None
    geom_only = len(source.geom_pose_source) == count
    target_geom = int(node.geom_index) if node.type is NodeType.GEOM else -1
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    for instance in range(count):
        if int(source.geom_body[instance]) != body:
            continue
        if len(source.geom_infinite_plane) == count and source.geom_infinite_plane[instance]:
            continue
        if geom_only and int(source.geom_pose_source[instance]) != int(InstancePoseSource.GEOM):
            continue

        pose_index = (
            int(source.geom_source[instance]) if len(source.geom_source) == count else instance
        )
        if target_geom >= 0 and pose_index != target_geom:
            continue
        if not 0 <= pose_index < len(frame.geom_xpos) or pose_index >= len(frame.geom_xmat):
            continue

        points = _instance_world_corners(source, frame, instance, cache)
        if points is None:
            continue
        points = (points - body_position) @ body_rotation
        part_lo, part_hi = points.min(axis=0), points.max(axis=0)
        lo = part_lo if lo is None else np.minimum(lo, part_lo)
        hi = part_hi if hi is None else np.maximum(hi, part_hi)

    if lo is None or hi is None:
        return None
    return CenteredBounds(
        ((lo + hi) * 0.5).astype(np.float32), ((hi - lo) * 0.5).astype(np.float32)
    )


def _node_world_bounds(
    source: SceneSource | None,
    frame: SceneFrame,
    node: SceneNode,
    nodes: list[SceneNode],
    cache: _MeshBoundsCache | None = None,
) -> CenteredBounds | None:
    """Return selected finite geometry bounds without nesting body/geom scans."""

    if source is None or source.instance_count == 0:
        return None
    if frame.geom_xpos is None or frame.geom_xmat is None:
        return None
    count = source.instance_count
    if len(source.geom_body) != count:
        return None

    target_geoms: set[int] | None = None
    target_bodies: set[int] | None = None
    if node.type is NodeType.MODEL:
        nodes_by_id = {candidate.node_id: candidate for candidate in nodes}
        pending = list(node.children)
        target_geoms = set()
        while pending:
            candidate = nodes_by_id.get(pending.pop())
            if candidate is None:
                continue
            pending.extend(candidate.children)
            if candidate.type is NodeType.GEOM and candidate.geom_index >= 0:
                target_geoms.add(int(candidate.geom_index))
        if not target_geoms:
            return None
    elif node.body_index >= 0:
        target_bodies = {int(node.body_index)}
        if node.type is NodeType.GEOM and node.geom_index >= 0:
            target_geoms = {int(node.geom_index)}
    else:
        return None

    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    for instance in range(count):
        if len(source.geom_infinite_plane) == count and source.geom_infinite_plane[instance]:
            continue
        if len(source.geom_pose_source) == count and int(source.geom_pose_source[instance]) != int(
            InstancePoseSource.GEOM
        ):
            continue
        pose_index = (
            int(source.geom_source[instance]) if len(source.geom_source) == count else instance
        )
        if target_geoms is not None and pose_index not in target_geoms:
            continue
        if target_bodies is not None and int(source.geom_body[instance]) not in target_bodies:
            continue
        points = _instance_world_corners(source, frame, instance, cache)
        if points is None:
            continue
        part_lo, part_hi = points.min(axis=0), points.max(axis=0)
        lo = part_lo if lo is None else np.minimum(lo, part_lo)
        hi = part_hi if hi is None else np.maximum(hi, part_hi)

    if lo is None or hi is None:
        return None
    return CenteredBounds(
        ((lo + hi) * 0.5).astype(np.float32), ((hi - lo) * 0.5).astype(np.float32)
    )


class SceneBounds:
    """Reuse local mesh bounds and batched transform buffers for scene framing."""

    def __init__(self, source: SceneSource, cache: _MeshBoundsCache | None = None) -> None:
        self.source = source
        self.cache = cache if cache is not None else {}
        count = source.instance_count
        self._centers = np.zeros((count, 3), np.float64)
        self._halves = np.zeros((count, 3), np.float64)
        self._mesh_valid = np.zeros(count, bool)
        self._valid = np.zeros(count, bool)
        self._linear = np.empty((count, 3, 3), np.float64)
        self._absolute = np.empty_like(self._linear)
        self._translation = np.empty((count, 3), np.float64)
        self._world_center = np.empty_like(self._translation)
        self._world_half = np.empty_like(self._translation)
        self._minimum = np.empty_like(self._translation)
        self._maximum = np.empty_like(self._translation)
        groups: dict[MeshKey, list[int]] = {}
        for instance, key in enumerate(source.geom_mesh):
            groups.setdefault(key, []).append(instance)
        self._dynamic = []
        for key, members in groups.items():
            indices = np.asarray(members, np.intp)
            self._set_mesh_bounds(
                indices, _instance_mesh_bounds(source, members[0], cache=self.cache)
            )
            if key in source.dynamic_meshes:
                self._dynamic.append(indices)
        kinds = (
            source.geom_pose_source
            if len(source.geom_pose_source) == count
            else np.zeros(count, np.uint8)
        )
        self._world = kinds == int(InstancePoseSource.WORLD)
        self._poses = []
        for kind in (InstancePoseSource.GEOM, InstancePoseSource.SITE):
            indices = np.flatnonzero(kinds == int(kind))
            pose_indices = (
                source.geom_source[indices] if len(source.geom_source) == count else indices
            )
            self._poses.append((kind, indices, pose_indices))

    def _set_mesh_bounds(self, indices: np.ndarray, bounds: Bounds | None) -> None:
        self._mesh_valid[indices] = bounds is not None
        if bounds is not None:
            low, high = bounds
            self._centers[indices] = (low + high) * 0.5
            self._halves[indices] = (high - low) * 0.5

    def world(self, frame: SceneFrame) -> Bounds | None:
        """Return the finite world AABB, including current dynamic mesh updates."""
        source = self.source
        count = source.instance_count
        if not count or len(source.geom_size) != count:
            return None
        for indices in self._dynamic:
            self._set_mesh_bounds(indices, _instance_mesh_bounds(source, int(indices[0]), frame))
        if len(source.geom_local) == count:
            self._linear[:] = source.geom_local[:, :3, :3]
            self._translation[:] = source.geom_local[:, :3, 3]
        else:
            self._linear[:] = np.eye(3)
            self._translation.fill(0)
        self._linear *= source.geom_size[:, None, :]
        self._valid[:] = self._world
        for kind, indices, pose_indices in self._poses:
            positions, rotations = (
                (frame.geom_xpos, frame.geom_xmat)
                if kind is InstancePoseSource.GEOM
                else (frame.site_xpos, frame.site_xmat)
            )
            if not len(indices) or positions is None or rotations is None:
                continue
            valid = (pose_indices >= 0) & (pose_indices < min(len(positions), len(rotations)))
            indices, pose_indices = indices[valid], pose_indices[valid]
            rotation = rotations[pose_indices].reshape(-1, 3, 3)
            self._linear[indices] = rotation @ self._linear[indices]
            self._translation[indices] = (
                np.einsum("nij,nj->ni", rotation, self._translation[indices])
                + positions[pose_indices]
            )
            self._valid[indices] = True
        # Transform the local center and half extent with the complete linear
        # transform. Applying abs to each rotation separately would overestimate
        # bounds when the instance and body rotations cancel one another.
        np.einsum("nij,nj->ni", self._linear, self._centers, out=self._world_center)
        self._world_center += self._translation
        np.abs(self._linear, out=self._absolute)
        np.einsum("nij,nj->ni", self._absolute, self._halves, out=self._world_half)
        np.subtract(self._world_center, self._world_half, out=self._minimum)
        np.add(self._world_center, self._world_half, out=self._maximum)
        self._valid &= self._mesh_valid
        if len(source.geom_infinite_plane) == count:
            self._valid &= ~source.geom_infinite_plane
        self._valid &= np.isfinite(self._minimum).all(axis=1) & np.isfinite(self._maximum).all(
            axis=1
        )
        if not self._valid.any():
            return None
        low = np.min(self._minimum, axis=0, where=self._valid[:, None], initial=np.inf)
        high = np.max(self._maximum, axis=0, where=self._valid[:, None], initial=-np.inf)
        return Bounds(low.astype(np.float32), high.astype(np.float32))
