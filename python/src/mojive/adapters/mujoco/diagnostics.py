"""Reusable diagnostic buffers, debug geometry and preview transforms."""

from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import dataclass

import numpy as np

from ..base import (
    ActuatorVisualType,
    BvhType,
    DiagnosticFrame,
    DiagnosticSource,
    FrameNeeds,
    JointVisualType,
    SceneSource,
)
from .constants import (
    _ACTUATOR_POSE_GEOM,
    _ACTUATOR_POSE_JOINT_AXIS,
    _ACTUATOR_POSE_JOINT_BODY,
    _ACTUATOR_POSE_SITE,
    _BVH_POSE_BODY,
    _BVH_POSE_DYNAMIC,
    _BVH_POSE_GEOM,
    _GEOM_RGBA_DEFAULT,
)
from .engine import mujoco
from .spec import _RAY_NORMAL


def _grid_node(start: int, ny: int, nz: int, i: int, j: int, k: int) -> int:
    return start + i * ny * nz + j * nz + k


def _grid_boundary(nx: int, ny: int, nz: int, i: int, j: int, k: int) -> bool:
    return i in (0, nx - 1) or j in (0, ny - 1) or k in (0, nz - 1)


_RAY_FIELD_WIDTHS = (1, 3, 3, 3, 3, 1)


_RAY_DIST = 1 << 0


_RAY_DIR = 1 << 1


_RAY_ORIGIN = 1 << 2


_RAY_POINT = 1 << 3


@dataclass(frozen=True)
class _RangefinderSpec:
    sensor: int
    fields: int
    ray_count: int
    stride: int
    frame_offset: int
    object_type: int
    object_id: int


class _Diagnostics:
    """Reusable diagnostic buffers, debug geometry and preview transforms.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def _apply_model_transform_preview_poses(self) -> None:
        preview = self._model_transform_preview
        if preview is None:
            return
        self._transform_preview_group(
            self._body_xpos_buf, self._body_xmat_buf, preview.body_indices
        )
        self._transform_preview_group(
            self._geom_xpos_buf, self._geom_xmat_buf, preview.geom_indices
        )
        self._transform_preview_group(
            self._site_xpos_buf, self._site_xmat_buf, preview.site_indices
        )

    def _apply_model_transform_preview_diagnostics(
        self,
        diagnostics: DiagnosticFrame,
        *,
        include_full_frame: bool,
    ) -> None:
        preview = self._model_transform_preview
        if preview is None:
            return
        self._transform_preview_points(diagnostics.joint_xpos, preview.joint_indices)
        self._transform_preview_directions(diagnostics.joint_xaxis, preview.joint_indices)
        if not include_full_frame:
            return
        self._transform_preview_points(diagnostics.subtree_com, preview.body_indices)
        self._transform_preview_group(
            diagnostics.body_xipos, diagnostics.body_ximat, preview.body_indices
        )

    def _transform_preview_group(
        self, positions: np.ndarray, matrices: np.ndarray, indices: np.ndarray
    ) -> None:
        self._transform_preview_points(positions, indices)
        self._transform_preview_matrices(matrices, indices)

    def _transform_preview_points(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.point_input[:count]
        target = preview.point_output[:count]
        np.take(values, indices, axis=0, out=source)
        source -= preview.previous_position
        np.einsum(
            "ij,nj->ni",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        target += preview.position
        values[indices] = target

    def _transform_preview_directions(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.point_input[:count]
        target = preview.point_output[:count]
        np.take(values, indices, axis=0, out=source)
        np.einsum(
            "ij,nj->ni",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        values[indices] = target

    def _transform_preview_matrices(self, values: np.ndarray, indices: np.ndarray) -> None:
        preview = self._model_transform_preview
        count = len(indices)
        if preview is None or count == 0:
            return
        source = preview.matrix_input[:count]
        target = preview.matrix_output[:count]
        np.take(values, indices, axis=0, out=source)
        np.einsum(
            "ij,njk->nik",
            preview.delta_rotation,
            source,
            out=target,
            casting="unsafe",
        )
        values[indices] = target

    def prepare_frame(self, needs: FrameNeeds) -> bool:
        """Build potentially huge BVH diagnostics only after they are requested."""
        if not needs.bvh or self._bvh_source_ready:
            return False
        self._bvh_source_ready = True
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def _fill_tendons(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = 0
        self._tendon_ids.fill(-1)
        for ti in range(self._m.ntendon):
            start = int(self._d.ten_wrapadr[ti])
            points = int(self._d.ten_wrapnum[ti])
            segments = max(points - 1, 0)
            stop = count + segments
            self._tendon_segments[count:stop, 0] = self._mj_wrap_points[start : start + segments]
            self._tendon_segments[count:stop, 1] = self._mj_wrap_points[
                start + 1 : start + 1 + segments
            ]
            self._tendon_ids[count:stop] = ti
            widths = self._tendon_widths[count:stop]
            widths[:] = float(self._m.tendon_width[ti])
            inside = (self._mj_wrap_objects[start : start + segments] >= 0) & (
                self._mj_wrap_objects[start + 1 : start + 1 + segments] >= 0
            )
            widths[inside] *= 0.5
            count = stop
        return (
            self._tendon_segments[:count],
            self._tendon_ids[:count],
            self._tendon_widths[:count],
        )

    def _fill_contacts(self, islands: bool = False) -> np.ndarray:
        d, m = self._d, self._m
        n = int(d.ncon)
        if n > len(self._contact_buf):
            capacity = max(n, 2 * len(self._contact_buf))
            self._contact_buf = np.zeros((capacity, 7), np.float32)
            self._contact_force_buf = np.zeros((capacity, 2, 3), np.float32)
            self._contact_island_rgba_buf = np.zeros((capacity, 4), np.float32)
            self._contact_view = self._contact_buf[:0]
            self._contact_force_view = self._contact_force_buf[:0]
            self._contact_island_rgba_view = self._contact_island_rgba_buf[:0]
        for i in range(n):
            c = d.contact[i]
            self._contact_buf[i, 0:3] = c.pos
            self._contact_buf[i, 3:6] = c.frame[0:3]
            mujoco.mj_contactForce(m, d, i, self._contact_force)
            local = self._contact_force[:3].copy()
            if int(c.dim) < 3:
                local[int(c.dim) :] = 0.0
            rotation = np.asarray(c.frame, np.float64).reshape(3, 3).T
            self._contact_force_buf[i, 0] = rotation[:, 0] * local[0]
            self._contact_force_buf[i, 1] = rotation[:, 1:] @ local[1:]
            first = int(m.geom_bodyid[c.geom[0]]) if c.geom[0] >= 0 else m.nbody + int(c.flex[0])
            second = int(m.geom_bodyid[c.geom[1]]) if c.geom[1] >= 0 else m.nbody + int(c.flex[1])
            if first > second:
                self._contact_force_buf[i] *= -1.0
            self._contact_buf[i, 6] = np.linalg.norm(local)
            if islands:
                address = int(c.efc_address)
                if address >= 0:
                    island = int(d.efc_island[address]) if int(d.nisland) else -1
                    key = int(d.island_dofadr[island]) if island >= 0 else -1
                    self._write_island_color(self._contact_island_rgba_buf[i], key, True)
                else:
                    self._contact_island_rgba_buf[i] = m.vis.rgba.contactgap
        if len(self._contact_view) != n:
            self._contact_view = self._contact_buf[:n]
            self._contact_force_view = self._contact_force_buf[:n]
            self._contact_island_rgba_view = self._contact_island_rgba_buf[:n]
        return self._contact_view

    def _fill_island_colors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        source = self.scene_source()
        bodies = source.instance_island_body
        moving = bodies >= 0
        np.copyto(self._island_rgba_buf, source.geom_rgba)
        for body in np.unique(bodies[moving]):
            key, awake = self._body_island_key(int(body))
            self._write_island_color(self._body_island_rgba_buf[int(body)], key, awake)
        if np.any(moving):
            self._island_rgba_buf[moving] = self._body_island_rgba_buf[bodies[moving]]

        model, data = self._m, self._d
        for flex in range(model.nflex):
            body = self._flex_island_body(flex)
            if body >= 0:
                self._flex_island_rgba_buf[flex] = self._body_island_rgba_buf[body]
            else:
                color = np.asarray(model.flex_rgba[flex], np.float32)
                material = int(model.flex_matid[flex])
                if material >= 0 and np.array_equal(color, _GEOM_RGBA_DEFAULT):
                    color = model.mat_rgba[material]
                self._flex_island_rgba_buf[flex] = color

        for tendon in range(model.ntendon):
            key = -1
            address = int(data.tendon_efcadr[tendon])
            if int(data.nisland) and address >= 0:
                island = int(data.efc_island[address])
                key = int(data.island_dofadr[island])
            self._write_island_color(self._tendon_island_rgba_buf[tendon], key, True)
        return (
            self._island_rgba_buf,
            self._tendon_island_rgba_buf,
            self._flex_island_rgba_buf,
        )

    def _body_island_key(self, body: int) -> tuple[int, bool]:
        model, data = self._m, self._d
        weld = int(model.body_weldid[body])
        dof = int(model.body_dofadr[weld])
        island = int(data.dof_island[dof]) if int(data.nisland) else -1
        key = int(data.island_dofadr[island]) if island >= 0 else -1
        awake = bool(data.body_awake[body])
        sleep = int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        if key < 0 and sleep:
            tree = int(model.dof_treeid[dof])
            if not awake:
                tree = self._sleep_cycle(data.tree_asleep, tree)
            if tree >= 0:
                key = int(model.tree_dofadr[tree])
        return key, awake

    def _flex_island_body(self, flex: int) -> int:
        model = self._m
        if bool(model.flex_interp[flex]):
            address = int(model.flex_nodeadr[flex])
            count = int(model.flex_nodenum[flex])
            bodies = model.flex_nodebodyid[address : address + count]
        else:
            address = int(model.flex_vertadr[flex])
            count = int(model.flex_vertnum[flex])
            bodies = model.flex_vertbodyid[address : address + count]
        return next((int(body) for body in bodies if int(model.body_treeid[body]) >= 0), -1)

    def _sleep_cycle(self, tree_asleep: np.ndarray, start: int) -> int:
        if start < 0 or start >= self._m.ntree:
            return -1
        smallest = current = start
        for _ in range(self._m.ntree + 1):
            current = int(tree_asleep[current])
            if current < 0 or current >= self._m.ntree:
                return -1
            smallest = min(smallest, current)
            if current == start:
                return smallest
        return -1

    @staticmethod
    def _write_island_color(out: np.ndarray, key: int, awake: bool) -> None:
        hue, saturation, value = 1.0, 0.0, 0.7
        if key >= 0:
            hue = float(mujoco.mju_Halton(key + 1, 7))
            saturation = 0.5 + 0.5 * float(mujoco.mju_Halton(key + 1, 3))
            value = 0.6 + 0.4 * float(mujoco.mju_Halton(key + 1, 5))
        if not awake:
            saturation *= 0.7
            value *= 0.6
        out[:3] = hsv_to_rgb(hue, saturation, value)
        out[3] = 1.0

    def _build_flex_debug_source(self, source: SceneSource) -> None:
        model = self._m
        vertex_indices: list[np.ndarray] = []
        edges: list[np.ndarray] = []
        vertex_colors: list[np.ndarray] = []
        edge_colors: list[np.ndarray] = []
        vertex_owners: list[np.ndarray] = []
        edge_owners: list[np.ndarray] = []
        ranges = np.zeros((model.nflex, 2), np.int32)
        for flex in range(model.nflex):
            if not self._visual_groups["flex"][int(model.flex_group[flex])]:
                continue
            vertex_address = int(model.flex_vertadr[flex])
            vertex_count = int(model.flex_vertnum[flex])
            edge_address = int(model.flex_edgeadr[flex])
            edge_count = int(model.flex_edgenum[flex])
            ranges[flex] = (vertex_address, vertex_count)
            color = np.asarray(model.flex_rgba[flex], np.float32).copy()
            material = int(model.flex_matid[flex])
            if material >= 0 and np.array_equal(color, _GEOM_RGBA_DEFAULT):
                color = np.asarray(model.mat_rgba[material], np.float32).copy()
            vertex_indices.append(
                np.arange(vertex_address, vertex_address + vertex_count, dtype=np.int32)
            )
            edges.append(
                np.asarray(model.flex_edge[edge_address : edge_address + edge_count], np.int32)
            )
            vertex_colors.append(np.repeat(color[None], vertex_count, axis=0))
            edge_colors.append(np.repeat(color[None], edge_count, axis=0))
            vertex_owners.append(np.full(vertex_count, flex, np.int32))
            edge_owners.append(np.full(edge_count, flex, np.int32))
        source.flex_vertex_indices = (
            np.concatenate(vertex_indices) if vertex_indices else np.zeros(0, np.int32)
        )
        source.flex_edges = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), np.int32)
        source.flex_vertex_rgba = (
            np.concatenate(vertex_colors, axis=0) if vertex_colors else np.zeros((0, 4), np.float32)
        )
        source.flex_edge_rgba = (
            np.concatenate(edge_colors, axis=0) if edge_colors else np.zeros((0, 4), np.float32)
        )
        source.flex_vertex_owner = (
            np.concatenate(vertex_owners) if vertex_owners else np.zeros(0, np.int32)
        )
        source.flex_edge_owner = (
            np.concatenate(edge_owners) if edge_owners else np.zeros(0, np.int32)
        )
        source.flex_vertex_ranges = ranges

    def _build_bvh_records(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self._m
        record_types: list[int] = []
        depths: list[int] = []
        leaves: list[bool] = []
        pose_types: list[int] = []
        pose_sources: list[int] = []
        global_indices: list[int] = []
        centers: list[np.ndarray] = []
        sizes: list[np.ndarray] = []

        def append(
            record_type: int,
            depth: int,
            leaf: bool,
            pose_type: int,
            pose_source: int,
            global_index: int,
            center,
            size,
        ) -> None:
            record_types.append(record_type)
            depths.append(depth)
            leaves.append(leaf)
            pose_types.append(pose_type)
            pose_sources.append(pose_source)
            global_indices.append(global_index)
            centers.append(np.asarray(center, np.float32))
            sizes.append(np.asarray(size, np.float32))

        for body in range(m.nbody):
            start = int(m.body_bvhadr[body])
            for index in range(start, start + int(m.body_bvhnum[body])):
                leaf = bool(np.all(np.asarray(m.bvh_child[index]) == -1))
                geom = int(m.bvh_nodeid[index])
                if leaf:
                    center, size = m.geom_aabb[geom, :3], m.geom_aabb[geom, 3:]
                    pose_type, pose_source = _BVH_POSE_GEOM, geom
                else:
                    center, size = m.bvh_aabb[index, :3], m.bvh_aabb[index, 3:]
                    pose_type, pose_source = _BVH_POSE_BODY, body
                append(
                    int(BvhType.BODY),
                    int(m.bvh_depth[index]),
                    leaf,
                    pose_type,
                    pose_source,
                    index,
                    center,
                    size,
                )

        for flex in range(m.nflex):
            if not self._visual_groups["flex"][int(m.flex_group[flex])]:
                continue
            start = int(m.flex_bvhadr[flex])
            for index in range(start, start + int(m.flex_bvhnum[flex])):
                append(
                    int(BvhType.FLEX),
                    int(m.bvh_depth[index]),
                    bool(np.all(np.asarray(m.bvh_child[index]) == -1)),
                    _BVH_POSE_DYNAMIC,
                    index,
                    index,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                )

        mesh_geom = int(mujoco.mjtGeom.mjGEOM_MESH)
        hfield_geom = int(mujoco.mjtGeom.mjGEOM_HFIELD)
        for geom in range(m.ngeom):
            mesh = int(m.geom_dataid[geom])
            if mesh < 0:
                continue
            if int(m.geom_type[geom]) == mesh_geom and int(m.mesh_octadr[mesh]) < 0:
                start = int(m.mesh_bvhadr[mesh])
                for index in range(start, start + int(m.mesh_bvhnum[mesh])):
                    append(
                        int(BvhType.MESH),
                        int(m.bvh_depth[index]),
                        bool(np.all(np.asarray(m.bvh_child[index]) == -1)),
                        _BVH_POSE_GEOM,
                        geom,
                        index,
                        m.bvh_aabb[index, :3],
                        m.bvh_aabb[index, 3:],
                    )
            if int(m.geom_type[geom]) not in (hfield_geom,) and int(m.mesh_octadr[mesh]) >= 0:
                start = int(m.mesh_octadr[mesh])
                for index in range(start, start + int(m.mesh_octnum[mesh])):
                    append(
                        int(BvhType.OCTREE),
                        int(m.oct_depth[index]),
                        False,
                        _BVH_POSE_GEOM,
                        geom,
                        -1,
                        m.oct_aabb[index, :3],
                        m.oct_aabb[index, 3:],
                    )

        self._bvh_pose_type = np.asarray(pose_types, np.uint8)
        self._bvh_pose_source = np.asarray(pose_sources, np.int32)
        self._bvh_global_index = np.asarray(global_indices, np.int32)
        self._bvh_local_center = np.stack(centers) if centers else np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.stack(sizes) if sizes else np.zeros((0, 3), np.float32)
        return (
            np.asarray(record_types, np.uint8),
            np.asarray(depths, np.int32),
            np.asarray(leaves, bool),
        )

    def _build_bvh_control_cages(self) -> int:
        m = self._m
        edges: list[tuple[int, int]] = []
        centered = np.zeros(len(m.flex_nodebodyid), bool)
        for flex in range(m.nflex):
            order = abs(int(m.flex_interp[flex]))
            if not order:
                continue
            start = int(m.flex_nodeadr[flex])
            centered[start : start + int(m.flex_nodenum[flex])] = bool(m.flex_centered[flex])
            nx, ny, nz = np.asarray(m.flex_cellnum[flex], np.int32) * order + 1
            shell = int(m.flex_interp[flex]) < 0

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        current = _grid_node(start, ny, nz, i, j, k)
                        if not m.body_jntnum[m.flex_nodebodyid[current]] or (
                            shell and not _grid_boundary(nx, ny, nz, i, j, k)
                        ):
                            continue
                        for neighbor in (
                            _grid_node(start, ny, nz, i + 1, j, k) if i + 1 < nx else -1,
                            _grid_node(start, ny, nz, i, j + 1, k) if j + 1 < ny else -1,
                            _grid_node(start, ny, nz, i, j, k + 1) if k + 1 < nz else -1,
                        ):
                            if neighbor >= 0 and m.body_jntnum[m.flex_nodebodyid[neighbor]]:
                                local = neighbor - start
                                ni, rem = divmod(local, ny * nz)
                                nj, nk = divmod(rem, nz)
                                if not shell or _grid_boundary(nx, ny, nz, ni, nj, nk):
                                    edges.append((current, neighbor))

        nodes = np.asarray(edges, np.int32).reshape(-1, 2)
        self._bvh_control_body = np.asarray(m.flex_nodebodyid[nodes], np.int32)
        self._bvh_control_local = np.asarray(m.flex_node[nodes], np.float32).copy()
        self._bvh_control_local[centered[nodes]] = 0.0
        return len(nodes)

    def _build_diagnostic_source(self) -> DiagnosticSource:
        m = self._m
        if self._bvh_source_ready:
            bvh_type, bvh_depth, bvh_leaf = self._build_bvh_records()
            bvh_control_count = self._build_bvh_control_cages()
        else:
            bvh_type = np.zeros(0, np.uint8)
            bvh_depth = np.zeros(0, np.int32)
            bvh_leaf = np.zeros(0, bool)
            bvh_control_count = 0
        bvh_rgba = np.asarray(m.vis.rgba.bv, np.float32)
        if bvh_rgba.shape != (4,):
            bvh_rgba = np.array([0.0, 1.0, 0.0, 0.5], np.float32)
        bvh_active_rgba = np.asarray(m.vis.rgba.bvactive, np.float32)
        if bvh_active_rgba.shape != (4,):
            bvh_active_rgba = np.array([1.0, 0.0, 0.0, 0.5], np.float32)
        joint_types = np.empty(m.njnt, np.uint8)
        joint_type_map = {
            int(mujoco.mjtJoint.mjJNT_FREE): JointVisualType.FREE,
            int(mujoco.mjtJoint.mjJNT_BALL): JointVisualType.BALL,
            int(mujoco.mjtJoint.mjJNT_SLIDE): JointVisualType.SLIDE,
            int(mujoco.mjtJoint.mjJNT_HINGE): JointVisualType.HINGE,
        }
        for source_type, visual_type in joint_type_map.items():
            joint_types[np.asarray(m.jnt_type) == source_type] = int(visual_type)

        meansize = float(m.stat.meansize)
        meanmass = float(m.stat.meanmass)
        com_bodies = np.flatnonzero(np.asarray(m.body_parentid[1:]) == 0).astype(np.int32) + 1
        inertia_bodies = np.flatnonzero(
            (np.asarray(m.body_dofnum) > 0) & (np.asarray(m.body_mass) > 0.0)
        ).astype(np.int32)
        inertia = np.asarray(m.body_inertia[inertia_bodies], np.float64)
        mass = np.asarray(m.body_mass[inertia_bodies], np.float64)
        inertia_sizes = np.sqrt(
            np.maximum(
                1.5
                * np.column_stack(
                    (
                        inertia[:, 1] + inertia[:, 2] - inertia[:, 0],
                        inertia[:, 0] + inertia[:, 2] - inertia[:, 1],
                        inertia[:, 0] + inertia[:, 1] - inertia[:, 2],
                    )
                )
                / mass[:, None],
                0.0,
            )
        )
        volume_scale = np.cbrt(mass / (8000.0 * np.prod(inertia_sizes, axis=1)))
        scaled_inertia_sizes = inertia_sizes * volume_scale[:, None]

        visual_types: list[int] = []
        visual_actuators: list[int] = []
        visual_sizes: list[np.ndarray] = []
        pose_types: list[int] = []
        pose_indices: list[int] = []
        slider_crank_actuators: list[int] = []
        primitive_types = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE): ActuatorVisualType.SPHERE,
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ActuatorVisualType.ELLIPSOID,
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): ActuatorVisualType.CAPSULE,
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): ActuatorVisualType.CYLINDER,
            int(mujoco.mjtGeom.mjGEOM_BOX): ActuatorVisualType.BOX,
        }

        def append_primitive(actuator: int, geom_type: int, size, pose_type: int, source: int):
            visual_type = primitive_types.get(int(geom_type))
            if visual_type is None:
                return
            raw = np.asarray(size, np.float32)
            if visual_type is ActuatorVisualType.SPHERE:
                normalized = np.full(3, raw[0], np.float32)
            elif visual_type in (ActuatorVisualType.CAPSULE, ActuatorVisualType.CYLINDER):
                normalized = np.array((raw[0], raw[0], raw[1]), np.float32)
            else:
                normalized = raw[:3].copy()
            visual_types.append(int(visual_type))
            visual_actuators.append(actuator)
            visual_sizes.append(1.05 * normalized)
            pose_types.append(pose_type)
            pose_indices.append(source)

        joint_transmissions = {
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
        }
        for actuator, transmission in enumerate(np.asarray(m.actuator_trntype)):
            source = int(m.actuator_trnid[actuator, 0])
            transmission = int(transmission)
            if transmission in joint_transmissions:
                joint_type = int(m.jnt_type[source])
                if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    visual_type = ActuatorVisualType.SLIDE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    visual_type = ActuatorVisualType.HINGE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                    visual_type = ActuatorVisualType.BALL
                else:
                    visual_type = ActuatorVisualType.FREE
                if visual_type in (ActuatorVisualType.SLIDE, ActuatorVisualType.HINGE):
                    size = (
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorlength),
                    )
                    pose_type = _ACTUATOR_POSE_JOINT_AXIS
                else:
                    radius = meansize * float(m.vis.scale.jointlength) * 0.33
                    size = (radius, radius, radius)
                    pose_type = _ACTUATOR_POSE_JOINT_BODY
                visual_types.append(int(visual_type))
                visual_actuators.append(actuator)
                visual_sizes.append(np.asarray(size, np.float32))
                pose_types.append(pose_type)
                pose_indices.append(source)
            elif transmission == int(mujoco.mjtTrn.mjTRN_SITE):
                append_primitive(
                    actuator,
                    int(m.site_type[source]),
                    m.site_size[source],
                    _ACTUATOR_POSE_SITE,
                    source,
                )
            elif transmission == int(mujoco.mjtTrn.mjTRN_BODY):
                start = int(m.body_geomadr[source])
                stop = start + int(m.body_geomnum[source])
                for geom in range(start, stop):
                    append_primitive(
                        actuator,
                        int(m.geom_type[geom]),
                        m.geom_size[geom],
                        _ACTUATOR_POSE_GEOM,
                        geom,
                    )
            elif transmission == int(mujoco.mjtTrn.mjTRN_SLIDERCRANK):
                slider_crank_actuators.append(actuator)

        count = len(visual_types)
        self._actuator_visual_pose_types = np.asarray(pose_types, np.uint8)
        self._actuator_visual_pose_indices = np.asarray(pose_indices, np.int32)
        self._slider_crank_actuators = np.asarray(slider_crank_actuators, np.int32)
        self._diagnostic_frame.actuator_xpos = np.zeros((count, 3), np.float32)
        self._diagnostic_frame.actuator_xmat = np.zeros((count, 3, 3), np.float32)
        slider_crank_count = len(slider_crank_actuators)
        self._diagnostic_frame.slider_crank_points = np.zeros(
            (slider_crank_count, 3, 3), np.float32
        )
        self._diagnostic_frame.slider_crank_broken = np.zeros(slider_crank_count, bool)
        autoconnect_count = sum(
            1 + int(m.body_jntnum[body])
            for body in range(1, m.nbody)
            if int(m.body_parentid[body]) != 0
        )
        self._diagnostic_frame.autoconnect_segments = np.zeros(
            (autoconnect_count, 2, 3), np.float32
        )
        bvh_count = len(bvh_type)
        self._diagnostic_frame.bvh_centers = np.zeros((bvh_count, 3), np.float32)
        self._diagnostic_frame.bvh_matrices = np.zeros((bvh_count, 3, 3), np.float32)
        self._diagnostic_frame.bvh_sizes = self._bvh_local_size.copy()
        self._diagnostic_frame.bvh_active = np.zeros(bvh_count, bool)
        self._diagnostic_frame.bvh_control_segments = np.zeros(
            (bvh_control_count, 2, 3), np.float32
        )

        return DiagnosticSource(
            joint_types=joint_types,
            joint_visible=self._group_visibility(m.jnt_group, "joint"),
            joint_length=meansize * float(m.vis.scale.jointlength),
            joint_width=meansize * float(m.vis.scale.jointwidth),
            joint_rgba=np.asarray(m.vis.rgba.joint, np.float32).copy(),
            com_bodies=com_bodies,
            com_radius=meansize * float(m.vis.scale.com),
            com_rgba=np.asarray(m.vis.rgba.com, np.float32).copy(),
            inertia_bodies=inertia_bodies,
            inertia_sizes=np.asarray(inertia_sizes, np.float32),
            scaled_inertia_sizes=np.asarray(scaled_inertia_sizes, np.float32),
            inertia_rgba=np.asarray(m.vis.rgba.inertia, np.float32).copy(),
            actuator_visual_types=np.asarray(visual_types, np.uint8),
            actuator_visual_actuators=np.asarray(visual_actuators, np.int32),
            actuator_visual_sizes=(
                np.stack(visual_sizes) if count else np.zeros((0, 3), np.float32)
            ),
            slider_crank_actuators=self._slider_crank_actuators.copy(),
            slider_crank_width=meansize * float(m.vis.scale.slidercrank),
            slider_crank_rgba=np.asarray(m.vis.rgba.slidercrank, np.float32).copy(),
            slider_crank_broken_rgba=np.asarray(m.vis.rgba.crankbroken, np.float32).copy(),
            camera_rgba=np.asarray(m.vis.rgba.camera, np.float32).copy(),
            light_rgba=np.asarray(m.vis.rgba.light, np.float32).copy(),
            rangefinder_rgba=np.asarray(m.vis.rgba.rangefinder, np.float32).copy(),
            rangefinder_normal_length=meansize * 0.25,
            constraint_radius=meansize * float(m.vis.scale.constraint),
            constraint_connect_rgba=np.asarray(m.vis.rgba.connect, np.float32).copy(),
            constraint_rgba=np.asarray(m.vis.rgba.constraint, np.float32).copy(),
            contact_point_rgba=np.asarray(m.vis.rgba.contactpoint, np.float32).copy(),
            contact_force_rgba=np.asarray(m.vis.rgba.contactforce, np.float32).copy(),
            contact_friction_rgba=np.asarray(m.vis.rgba.contactfriction, np.float32).copy(),
            contact_force_scale=float(m.vis.map.force) / meanmass if meanmass > 0.0 else 0.0,
            autoconnect_width=meansize * float(m.vis.scale.connect),
            autoconnect_rgba=np.asarray(m.vis.rgba.connect, np.float32).copy(),
            bvh_type=bvh_type,
            bvh_depth=bvh_depth,
            bvh_leaf=bvh_leaf,
            bvh_active_highlight=bool(m.vis.global_.bvactive),
            bvh_rgba=bvh_rgba.copy(),
            bvh_active_rgba=bvh_active_rgba.copy(),
            bvh_control_count=bvh_control_count,
        )

    def _fill_constraint_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        visible = diagnostics.constraint_visible
        visible.fill(False)
        connect = int(mujoco.mjtEq.mjEQ_CONNECT)
        weld = int(mujoco.mjtEq.mjEQ_WELD)
        site = int(mujoco.mjtObj.mjOBJ_SITE)
        for equality in np.flatnonzero(np.asarray(d.eq_active)):
            constraint_type = int(m.eq_type[equality])
            if constraint_type not in (connect, weld):
                continue
            first = int(m.eq_obj1id[equality])
            second = int(m.eq_obj2id[equality])
            if int(m.eq_objtype[equality]) == site:
                diagnostics.constraint_starts[equality] = d.site_xpos[first]
                diagnostics.constraint_ends[equality] = d.site_xpos[second]
            else:
                data = m.eq_data[equality]
                start_offset = 3 if constraint_type == weld else 0
                end_offset = 0 if constraint_type == weld else 3
                diagnostics.constraint_starts[equality] = (
                    d.xpos[first]
                    + d.xmat[first].reshape(3, 3) @ data[start_offset : start_offset + 3]
                )
                diagnostics.constraint_ends[equality] = (
                    d.xpos[second]
                    + d.xmat[second].reshape(3, 3) @ data[end_offset : end_offset + 3]
                )
            visible[equality] = True

    def _fill_bvh_visuals(self, diagnostics: DiagnosticFrame) -> None:
        if not len(self._bvh_pose_type) and not len(self._bvh_control_body):
            return
        m, d = self._m, self._d
        centers = diagnostics.bvh_centers
        matrices = diagnostics.bvh_matrices
        np.copyto(diagnostics.bvh_sizes, self._bvh_local_size)

        body = self._bvh_pose_type == _BVH_POSE_BODY
        if np.any(body):
            source = self._bvh_pose_source[body]
            rotation = d.ximat[source].reshape(-1, 3, 3)
            matrices[body] = rotation
            centers[body] = d.xipos[source] + np.einsum(
                "nij,nj->ni", rotation, self._bvh_local_center[body]
            )

        geom = self._bvh_pose_type == _BVH_POSE_GEOM
        if np.any(geom):
            source = self._bvh_pose_source[geom]
            rotation = d.geom_xmat[source].reshape(-1, 3, 3)
            matrices[geom] = rotation
            centers[geom] = d.geom_xpos[source] + np.einsum(
                "nij,nj->ni", rotation, self._bvh_local_center[geom]
            )

        dynamic = self._bvh_pose_type == _BVH_POSE_DYNAMIC
        if np.any(dynamic):
            source = self._bvh_global_index[dynamic] - int(m.nbvhstatic)
            aabb = d.bvh_aabb_dyn[source]
            centers[dynamic] = aabb[:, :3]
            matrices[dynamic] = np.eye(3, dtype=np.float32)
            diagnostics.bvh_sizes[dynamic] = aabb[:, 3:]

        diagnostics.bvh_active.fill(False)
        indexed = self._bvh_global_index >= 0
        diagnostics.bvh_active[indexed] = d.bvh_active[self._bvh_global_index[indexed]]

        body = self._bvh_control_body
        if len(body):
            rotation = d.xmat[body].reshape(-1, 2, 3, 3)
            diagnostics.bvh_control_segments[:] = d.xpos[body] + np.einsum(
                "...ij,...j->...i", rotation, self._bvh_control_local
            )

    @staticmethod
    def _build_rangefinder_specs(model) -> tuple[_RangefinderSpec, ...]:
        sensor_type = int(mujoco.mjtSensor.mjSENS_RANGEFINDER)
        offset = 0
        specs = []
        for sensor in np.flatnonzero(np.asarray(model.sensor_type) == sensor_type):
            sensor = int(sensor)
            fields = int(model.sensor_intprm[sensor, 0])
            stride = sum(
                width for field, width in enumerate(_RAY_FIELD_WIDTHS) if fields & (1 << field)
            )
            ray_count = int(model.sensor_dim[sensor]) // stride
            specs.append(
                _RangefinderSpec(
                    sensor,
                    fields,
                    ray_count,
                    stride,
                    offset,
                    int(model.sensor_objtype[sensor]),
                    int(model.sensor_objid[sensor]),
                )
            )
            offset += ray_count
        return tuple(specs)

    def _fill_rangefinder_visuals(self, diagnostics: DiagnosticFrame) -> None:
        diagnostics.rangefinder_lines.fill(False)
        diagnostics.rangefinder_points.fill(False)
        diagnostics.rangefinder_normal_arrows.fill(False)
        for spec in self._rangefinder_specs:
            output = slice(spec.frame_offset, spec.frame_offset + spec.ray_count)
            data_start = int(self._m.sensor_adr[spec.sensor])
            values = self._d.sensordata[
                data_start : data_start + spec.ray_count * spec.stride
            ].reshape(spec.ray_count, spec.stride)
            fields = {}
            cursor = 0
            for field, width in enumerate(_RAY_FIELD_WIDTHS):
                bit = 1 << field
                if spec.fields & bit:
                    fields[bit] = values[:, cursor : cursor + width]
                    cursor += width

            origins, directions = self._rangefinder_rays(spec)
            if _RAY_ORIGIN in fields:
                origins = fields[_RAY_ORIGIN]
            if _RAY_DIR in fields:
                directions = fields[_RAY_DIR]
            diagnostics.rangefinder_starts[output] = origins

            distance = fields.get(_RAY_DIST)
            point = fields.get(_RAY_POINT)
            normal = fields.get(_RAY_NORMAL)
            if distance is not None:
                hit = distance[:, 0] >= 0.0
            elif point is not None:
                hit = np.linalg.norm(point, axis=1) > 1e-12
            else:
                hit = np.zeros(spec.ray_count, bool)

            ends = origins.copy()
            if point is not None:
                ends[hit] = point[hit]
            elif distance is not None:
                ends[hit] += directions[hit] * distance[hit]
            diagnostics.rangefinder_ends[output] = ends
            diagnostics.rangefinder_lines[output] = hit & bool(spec.fields & _RAY_DIST)
            diagnostics.rangefinder_points[output] = hit & bool(spec.fields & _RAY_POINT)

            if normal is not None:
                diagnostics.rangefinder_normals[output] = normal
                diagnostics.rangefinder_normal_arrows[output] = hit & (
                    np.linalg.norm(normal, axis=1) > 1e-12
                )

    def _rangefinder_rays(self, spec: _RangefinderSpec) -> tuple[np.ndarray, np.ndarray]:
        if spec.object_type == int(mujoco.mjtObj.mjOBJ_SITE):
            origin = np.asarray(self._d.site_xpos[spec.object_id], np.float32)
            rotation = np.asarray(self._d.site_xmat[spec.object_id], np.float32).reshape(3, 3)
            return (
                np.repeat(origin[None], spec.ray_count, axis=0),
                np.repeat(rotation[:, 2][None], spec.ray_count, axis=0),
            )

        width, height = (int(value) for value in self._m.cam_resolution[spec.object_id])
        view = self.camera_view(spec.object_id).with_aspect(width / height)
        columns = np.tile(np.arange(width), height)
        rows = np.repeat(np.arange(height), width)
        clip = np.column_stack(
            (
                2.0 * (columns + 0.5) / width - 1.0,
                1.0 - 2.0 * (rows + 0.5) / height,
                np.full(spec.ray_count, -1.0),
                np.ones(spec.ray_count),
            )
        )
        local = clip @ np.linalg.inv(view.proj_matrix()).T
        local = local[:, :3] / local[:, 3:4]
        rotation = np.asarray(self._d.cam_xmat[spec.object_id], np.float32).reshape(3, 3)
        if view.orthographic:
            origins = view.eye + local[:, :2] @ rotation[:, :2].T
            directions = np.repeat((-rotation[:, 2])[None], spec.ray_count, axis=0)
        else:
            origins = np.repeat(np.asarray(view.eye)[None], spec.ray_count, axis=0)
            directions = local @ rotation.T
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return origins.astype(np.float32), directions.astype(np.float32)

    def _fill_actuator_visual_poses(self, diagnostics: DiagnosticFrame) -> None:
        d, m = self._d, self._m
        for record, (pose_type, source) in enumerate(
            zip(self._actuator_visual_pose_types, self._actuator_visual_pose_indices, strict=True)
        ):
            source = int(source)
            if pose_type == _ACTUATOR_POSE_JOINT_AXIS:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = self._axis_rotation(d.xaxis[source])
            elif pose_type == _ACTUATOR_POSE_JOINT_BODY:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = d.xmat[int(m.jnt_bodyid[source])].reshape(3, 3)
            elif pose_type == _ACTUATOR_POSE_SITE:
                diagnostics.actuator_xpos[record] = d.site_xpos[source]
                diagnostics.actuator_xmat[record] = d.site_xmat[source].reshape(3, 3)
            else:
                diagnostics.actuator_xpos[record] = d.geom_xpos[source]
                diagnostics.actuator_xmat[record] = d.geom_xmat[source].reshape(3, 3)

    def _fill_slider_crank_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        for record, actuator in enumerate(self._slider_crank_actuators):
            actuator = int(actuator)
            crank_site, slider_site = map(int, m.actuator_trnid[actuator])
            axis = d.site_xmat[slider_site].reshape(3, 3)[:, 2]
            offset = d.site_xpos[crank_site] - d.site_xpos[slider_site]
            axial = float(np.dot(offset, axis))
            rod = float(m.actuator_cranklength[actuator])
            determinant = axial * axial + rod * rod - float(np.dot(offset, offset))
            diagnostics.slider_crank_broken[record] = determinant < 0.0
            slider_length = axial - np.sqrt(max(determinant, 0.0))
            points = diagnostics.slider_crank_points[record]
            points[0] = d.site_xpos[slider_site]
            points[1] = points[0] + axis * slider_length
            points[2] = d.site_xpos[crank_site]

    def _fill_autoconnect_visuals(self, diagnostics: DiagnosticFrame) -> None:
        m, d = self._m, self._d
        record = 0
        for body in range(1, m.nbody):
            parent = int(m.body_parentid[body])
            if parent == 0:
                continue
            current = d.xipos[body]
            start = int(m.body_jntadr[body])
            for joint in range(start + int(m.body_jntnum[body]) - 1, start - 1, -1):
                diagnostics.autoconnect_segments[record] = current, d.xanchor[joint]
                current = d.xanchor[joint]
                record += 1
            diagnostics.autoconnect_segments[record] = current, d.xipos[parent]
            record += 1
