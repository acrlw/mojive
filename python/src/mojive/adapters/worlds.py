"""Render independent rigid worlds with shared geometry and batched pose updates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..types import CameraView, InstancePoseSource
from .base import AdapterCaps, FrameNeeds, NodeType, SceneAdapterBase, SceneFrame, SceneNode


class WorldInstances(SceneAdapterBase):
    """Repeat a rigid source without copying meshes, textures, or physics models.

    Static instances (for example an infinite ground) appear once. Every other
    instance appears once per world. ``set_poses`` accepts local geometry poses
    in ``moving_instances`` order; grid offsets affect display coordinates only.
    There is no physics or cross-world collision in this adapter. A publisher
    owns simulation, replay timing, and any synchronization of calls.

    Selection identifies a whole world. Segmentation is ``(world, template
    instance)``; world -1 denotes shared static geometry. Inputs use Mojive's
    Z-up, row-major rotation contract. Dynamic meshes, sites, and tendons require
    a different provider and are rejected rather than silently discarded.
    """

    caps = AdapterCaps(name="worlds", external_clock=True, clock_control=False)

    def __init__(self, source, frame: SceneFrame, offsets: np.ndarray) -> None:
        offsets = np.array(offsets, dtype=np.float32, copy=True)
        if offsets.ndim != 2 or offsets.shape[1] != 3 or not len(offsets):
            raise ValueError("offsets must have shape (worlds, 3), with at least one world")
        if not np.isfinite(offsets).all():
            raise ValueError("world offsets must be finite")
        count = source.instance_count
        pose = source.geom_pose_source
        if (
            (len(pose) and np.any(pose != int(InstancePoseSource.GEOM)))
            or source.dynamic_meshes
            or len(source.tendon_rgba)
            or len(source.flex_vertex_indices)
        ):
            raise ValueError("WorldInstances requires rigid geometry poses without deformables")
        static = source.geom_static if len(source.geom_static) else np.zeros(count, bool)
        self.moving_instances = np.flatnonzero(~static)
        self.moving_instances.flags.writeable = False
        fixed = np.flatnonzero(static)
        if not len(self.moving_instances):
            raise ValueError("The template has no moving geometry")
        self.world_count = len(offsets)
        self.offsets = offsets
        self.offsets.flags.writeable = False
        n = len(self.moving_instances)
        self._moving_count = n * self.world_count
        rows = np.concatenate((np.tile(self.moving_instances, self.world_count), fixed))
        size = len(rows)
        world = np.repeat(np.arange(self.world_count, dtype=np.int32), n)
        ids = np.concatenate((world + 1, np.zeros(len(fixed), np.int32)))
        template_slots = source.geom_source if len(source.geom_source) else np.arange(count)
        self.pose_indices = np.array(template_slots[self.moving_instances], copy=True)
        self.pose_indices.flags.writeable = False
        nodes = [SceneNode(0, "world", NodeType.WORLD, children=list(range(1, len(offsets) + 1)))]
        nodes.extend(
            SceneNode(i + 1, f"World {i}", NodeType.MODEL, parent=0, object_id=i + 1)
            for i in range(len(offsets))
        )

        def expand_optional(values):
            if len(values) not in (0, count):
                raise ValueError("Optional instance arrays must be empty or match instance count")
            return values[rows] if len(values) else values

        # Stable arrays are expanded once. Resources retain their original identity.
        self._source = replace(
            source,
            geom_mesh=[source.geom_mesh[i] for i in rows],
            geom_convex_mesh=[source.geom_convex_mesh[i] for i in rows]
            if len(source.geom_convex_mesh)
            else [],
            geom_material=[source.geom_material[i] for i in rows],
            geom_size=source.geom_size[rows],
            geom_rgba=source.geom_rgba[rows],
            geom_local=expand_optional(source.geom_local),
            geom_visual=expand_optional(source.geom_visual),
            geom_infinite_plane=expand_optional(source.geom_infinite_plane),
            geom_static=static[rows],
            geom_object_id=ids.astype(np.uint32),
            geom_node=ids,
            geom_body=np.zeros(size, np.int32),
            geom_source=np.arange(size, dtype=np.int32),
            geom_pose_source=np.zeros(size, np.uint8),
            geom_segmentation=np.column_stack(
                (np.concatenate((world, np.full(len(fixed), -1))), rows)
            ).astype(np.int32),
            instance_island_body=np.full(size, -1, np.int32),
            nodes=nodes,
            body_names=(),
            geom_names=(),
            joint_names=(),
            site_names=(),
            cameras=(),
            scene_center=offsets.mean(axis=0),
            scene_extent=float(np.linalg.norm(np.ptp(offsets, axis=0)) + source.scene_extent),
        )
        self._frame = SceneFrame(
            geom_xpos=np.empty((size, 3), np.float32),
            geom_xmat=np.empty((size, 3, 3), np.float32),
        )
        self._positions = self._frame.geom_xpos[: self._moving_count].reshape(
            self.world_count, n, 3
        )
        self._rotations = self._frame.geom_xmat[: self._moving_count].reshape(
            self.world_count, n, 3, 3
        )
        self._frame.geom_xpos[self._moving_count :] = frame.geom_xpos[template_slots[fixed]]
        self._frame.geom_xmat[self._moving_count :] = frame.geom_xmat[template_slots[fixed]]
        self.set_poses(
            np.broadcast_to(frame.geom_xpos[self.pose_indices], self._positions.shape),
            np.broadcast_to(frame.geom_xmat[self.pose_indices], self._rotations.shape),
        )

    def set_poses(self, positions, rotations, *, time: float = 0.0) -> None:
        """Copy one complete world batch into reusable frame buffers, adding grid offsets."""
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        if positions.shape != self._positions.shape or rotations.shape != self._rotations.shape:
            raise ValueError("poses must have shape (worlds, moving_instances, 3) and (..., 3, 3)")
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(rotations).all()
            or not np.isfinite(time)
        ):
            raise ValueError("poses and time must be finite")
        np.add(positions, self.offsets[:, None, :], out=self._positions)
        np.copyto(self._rotations, rotations)
        self._frame.time = float(time)
        self._frame.step += 1

    def scene_source(self):
        return self._source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        return self._frame

    def nodes(self):
        return self._source.nodes

    def camera_hint(self):
        center = self._source.scene_center + np.array([0, 0, 0.7])
        extent = max(2.0, self._source.scene_extent * 1.1)
        return CameraView(
            eye=center + np.array([0, -extent, extent * 0.8]), target=center, far=extent * 5
        )
