"""Convert compiled MuJoCo structure into neutral scene sources."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np

from ...types import (
    DEFAULT_HEADLIGHT,
    InstancePoseSource,
    InstanceVisual,
    Light,
    LightSet,
    LightType,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    ShadingModel,
    TextureData,
    TextureType,
)
from ..base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    MODEL_OBJECT_BASE,
    NodeType,
    SceneNode,
    SceneSource,
)
from .constants import (
    _GEOM_RGBA_DEFAULT,
    _MOJIVE_AMBIENT_NUMERIC,
    _MOJIVE_HAZE_NUMERIC,
    _TEXROLE_RGB,
    _TEXROLE_RGBA,
)
from .deformables import build_deformables
from .engine import mujoco
from .spec import _numeric_values


def _object_names(model, object_type, count: int, prefix: str) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, object_type, index) or f"{prefix}{index}" for index in range(count)
    )


class _SceneConversion:
    """Convert compiled MuJoCo structure into neutral scene sources.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def scene_source(self) -> SceneSource:
        if self._source is None:
            self._source = self._build_source()
        return self._source

    def nodes(self) -> list[SceneNode]:
        if not self._nodes:
            self._nodes = self._build_nodes()
        return self._nodes

    def _build_source(self) -> SceneSource:
        m = self._m
        src = SceneSource()
        src.shading_model = ShadingModel.MUJOCO_CLASSIC
        src.nodes = self.nodes()
        src.diagnostics = self._build_diagnostic_source()
        trn_tendon = np.asarray(m.actuator_trntype) == int(mujoco.mjtTrn.mjTRN_TENDON)
        src.actuator_tendon = np.where(trn_tendon, m.actuator_trnid[:, 0], -1).astype(np.int32)
        src.actuator_visible = self._group_visibility(m.actuator_group, "actuator")
        disabled_groups = int(m.opt.disableactuator)
        groups = np.clip(np.asarray(m.actuator_group, np.int32), 0, 30)
        src.actuator_visible &= (disabled_groups & (1 << groups)) == 0
        src.actuator_ctrl_address = self._actuator_ctrl_address.copy()
        src.actuator_ctrl_limited = np.asarray(m.actuator_ctrllimited, bool).copy()
        src.actuator_ctrl_range = np.asarray(m.actuator_ctrlrange, np.float32).copy()
        src.actuator_act_limited = np.asarray(m.actuator_actlimited, bool).copy()
        src.actuator_act_range = np.asarray(m.actuator_actrange, np.float32).copy()
        src.actuator_dynamic = (np.asarray(m.actuator_dyntype) != 0).astype(bool)
        src.actuator_rgba = np.asarray(
            [m.vis.rgba.actuatornegative, m.vis.rgba.actuator, m.vis.rgba.actuatorpositive],
            np.float32,
        )
        src.actuator_tendon_scale = float(m.vis.map.actuatortendon)

        textures = self._build_textures()
        src.textures = textures
        src.skybox = next((t.name for t in textures.values() if t.type is TextureType.SKYBOX), None)
        materials, mat_of_matid = self._build_materials(textures)
        src.materials = materials

        meshes: dict[MeshKey, MeshData] = {}
        mesh_keys: list[MeshKey] = []
        convex_mesh_keys: list[MeshKey] = []
        mats: list[int] = []
        sizes: list[np.ndarray] = []
        rgbas: list[np.ndarray] = []
        object_ids: list[int] = []
        bodies: list[int] = []
        sources: list[int] = []
        pose_sources: list[int] = []
        visuals: list[int] = []
        statics: list[bool] = []
        island_bodies: list[int] = []
        node_ids: list[int] = []
        locals_: list[np.ndarray] = []
        infinite: list[bool] = []
        geom_groups = set(np.flatnonzero(self._visual_groups["geom"]))
        site_groups = set(np.flatnonzero(self._visual_groups["site"]))
        flex_groups = set(np.flatnonzero(self._visual_groups["flex"]))
        skin_groups = set(np.flatnonzero(self._visual_groups["skin"]))
        skipped: set[int] = set()

        def append_parts(
            parts,
            *,
            mat_index: int,
            rgba: np.ndarray,
            body: int,
            source: int,
            pose_source: InstancePoseSource,
            node_id: int,
            object_id: int,
            is_infinite: bool = False,
            visual: InstanceVisual = InstanceVisual.DEFAULT,
            is_static: bool = False,
            island_body: int = -1,
            convex_mesh: MeshKey | None = None,
        ) -> None:
            for key, scale, cap_offset in parts:
                if key.shape not in (MeshShape.ASSET, MeshShape.CONVEX_HULL) and key not in meshes:
                    meshes[key] = None
                mesh_keys.append(key)
                convex_mesh_keys.append(convex_mesh or key)
                mats.append(mat_index)
                sizes.append(np.asarray(scale, np.float32))
                rgbas.append(rgba)
                object_ids.append(object_id)
                bodies.append(body)
                sources.append(source)
                pose_sources.append(int(pose_source))
                visuals.append(int(visual))
                statics.append(is_static)
                island_bodies.append(island_body)
                node_ids.append(node_id)
                local = np.eye(4, dtype=np.float32)
                if cap_offset is not None:
                    local[2, 3] = cap_offset
                    if cap_offset < 0.0:
                        local[1, 1] = -1.0
                        local[2, 2] = -1.0
                locals_.append(local)
                infinite.append(is_infinite)

        for gi in range(m.ngeom):
            if int(m.geom_group[gi]) not in geom_groups:
                continue
            gtype = int(m.geom_type[gi])
            size = np.asarray(m.geom_size[gi], np.float64)
            body = int(m.geom_bodyid[gi])
            matid = int(m.geom_matid[gi])
            rgba = self._geom_rgba(gi, matid)
            mat_index = mat_of_matid[matid] if matid >= 0 else mat_of_matid[-1]
            is_infinite = False
            hull_key = None

            if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
                key = MeshKey(MeshShape.PLANE)

                scale = np.array([size[0], size[1], 1.0], np.float64)
                is_infinite = size[0] == 0.0 or size[1] == 0.0
                parts = [(key, scale, None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_HFIELD:
                data_id = int(m.geom_dataid[gi])
                if data_id < 0:
                    skipped.add(gtype)
                    continue
                key = MeshKey(MeshShape.HEIGHTFIELD, data_id)
                if key not in meshes:
                    meshes[key] = self._build_heightfield(data_id)
                hs = np.asarray(m.hfield_size[data_id], np.float64)
                parts = [(key, hs[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                parts = [(MeshKey(MeshShape.SPHERE), np.full(3, size[0]), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                parts = [(MeshKey(MeshShape.SPHERE), size[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                parts = [(MeshKey(MeshShape.BOX), size[:3].copy(), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                parts = [(MeshKey(MeshShape.CYLINDER), np.array([size[0], size[0], size[1]]), None)]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, half = float(size[0]), float(size[1])
                parts = [
                    (MeshKey(MeshShape.CAPSULE_SHAFT), np.array([r, r, half]), None),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), +half),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), -half),
                ]
            elif gtype in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SDF):
                data_id = int(m.geom_dataid[gi])
                if data_id < 0:
                    skipped.add(gtype)
                    continue
                key = MeshKey(MeshShape.ASSET, data_id)
                if key not in meshes:
                    meshes[key] = self._build_mesh(data_id)
                parts = [(key, np.ones(3), None)]
                if int(m.mesh_graphadr[data_id]) >= 0 and (
                    int(m.geom_contype[gi]) or int(m.geom_conaffinity[gi])
                ):
                    hull_key = MeshKey(MeshShape.CONVEX_HULL, data_id)
                    if hull_key not in meshes:
                        meshes[hull_key] = self._build_convex_hull(data_id)
            else:
                skipped.add(gtype)
                continue

            geom_node = self.nodes()[self._geom_nodes[gi]]
            append_parts(
                parts,
                mat_index=mat_index,
                rgba=rgba,
                body=body,
                source=gi,
                pose_source=InstancePoseSource.GEOM,
                node_id=self._geom_nodes.get(gi, -1),
                object_id=int(geom_node.object_id or body),
                is_infinite=is_infinite,
                is_static=int(m.body_weldid[body]) == 0,
                island_body=body if int(m.body_dofnum[int(m.body_weldid[body])]) else -1,
                convex_mesh=hull_key,
            )

        for si in range(m.nsite):
            if int(m.site_group[si]) not in site_groups:
                continue
            stype = int(m.site_type[si])
            size = np.asarray(m.site_size[si], np.float64)
            if stype == mujoco.mjtGeom.mjGEOM_SPHERE:
                parts = [(MeshKey(MeshShape.SPHERE), np.full(3, size[0]), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                parts = [(MeshKey(MeshShape.SPHERE), size[:3].copy(), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_BOX:
                parts = [(MeshKey(MeshShape.BOX), size[:3].copy(), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                parts = [(MeshKey(MeshShape.CYLINDER), np.array([size[0], size[0], size[1]]), None)]
            elif stype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, half = float(size[0]), float(size[1])
                parts = [
                    (MeshKey(MeshShape.CAPSULE_SHAFT), np.array([r, r, half]), None),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), +half),
                    (MeshKey(MeshShape.CAPSULE_CAP), np.full(3, r), -half),
                ]
            else:
                skipped.add(stype)
                continue
            body = int(m.site_bodyid[si])
            matid = int(m.site_matid[si])
            mat_index = mat_of_matid[matid] if matid >= 0 else mat_of_matid[-1]
            rgba = self._site_rgba(si, matid)
            append_parts(
                parts,
                mat_index=mat_index,
                rgba=rgba,
                body=body,
                source=si,
                pose_source=InstancePoseSource.SITE,
                node_id=self._site_nodes.get(si, -1),
                object_id=0,
                is_static=int(m.body_weldid[body]) == 0,
            )

        self._deformables = build_deformables(m, self._d, flex_groups, skin_groups)
        self._mesh_updates = {spec.key: spec.update_data for spec in self._deformables}
        for spec in self._deformables:
            meshes[spec.key] = spec.mesh
            mat_index = mat_of_matid[spec.matid] if spec.matid >= 0 else mat_of_matid[-1]
            rgba = spec.rgba
            if spec.matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
                rgba = np.asarray(m.mat_rgba[spec.matid], np.float32)
            is_flex = spec.key.shape in (MeshShape.FLEX, MeshShape.FLEX_FACE)
            node_id = (
                self._flex_nodes.get(spec.key.index, -1)
                if is_flex
                else self._skin_nodes.get(spec.key.index, -1)
            )
            object_id = m.nbody + spec.key.index if is_flex else m.nbody + m.nflex + spec.key.index
            append_parts(
                [(spec.key, np.ones(3), None)],
                mat_index=mat_index,
                rgba=np.asarray(rgba, np.float32).copy(),
                body=0,
                source=0,
                pose_source=InstancePoseSource.WORLD,
                node_id=node_id,
                object_id=object_id,
                visual=spec.visual,
                island_body=self._flex_island_body(spec.key.index) if is_flex else -1,
            )

        src.meshes = {k: v for k, v in meshes.items() if v is not None}
        src.dynamic_meshes = frozenset(spec.key for spec in self._deformables)
        src.geom_mesh = mesh_keys
        src.geom_convex_mesh = convex_mesh_keys
        src.geom_material = mats
        n = len(mesh_keys)
        src.geom_size = np.stack(sizes) if n else np.zeros((0, 3), np.float32)
        src.geom_rgba = np.stack(rgbas) if n else np.zeros((0, 4), np.float32)
        src.geom_object_id = np.array(object_ids, np.uint32)
        src.geom_body = np.array(bodies, np.int32)
        src.geom_source = np.array(sources, np.int32)
        src.geom_pose_source = np.array(pose_sources, np.uint8)
        src.geom_visual = np.array(visuals, np.uint8)
        # Semantic pairs belong to the adapter; rendering selection IDs stay
        # unchanged so the viewer and offscreen captures identify the same objects.
        src.geom_segmentation = np.full((n, 2), -1, np.int32)
        for pose_source, object_type in (
            (InstancePoseSource.GEOM, mujoco.mjtObj.mjOBJ_GEOM),
            (InstancePoseSource.SITE, mujoco.mjtObj.mjOBJ_SITE),
        ):
            mask = src.geom_pose_source == int(pose_source)
            src.geom_segmentation[mask, 0] = src.geom_source[mask]
            src.geom_segmentation[mask, 1] = int(object_type)
        flex = np.isin(
            src.geom_visual,
            (
                int(InstanceVisual.FLEX_EDGE),
                int(InstanceVisual.FLEX_FACE),
                int(InstanceVisual.FLEX_SKIN),
            ),
        )
        src.geom_segmentation[flex, 0] = src.geom_object_id[flex].astype(np.int32) - m.nbody
        src.geom_segmentation[flex, 1] = int(mujoco.mjtObj.mjOBJ_FLEX)
        skin = src.geom_visual == int(InstanceVisual.SKIN)
        src.geom_segmentation[skin, 0] = (
            src.geom_object_id[skin].astype(np.int32) - m.nbody - m.nflex
        )
        src.geom_segmentation[skin, 1] = int(mujoco.mjtObj.mjOBJ_SKIN)
        src.geom_static = np.array(statics, bool)
        src.instance_island_body = np.array(island_bodies, np.int32)
        src.geom_node = np.array(node_ids, np.int32)
        src.geom_local = np.stack(locals_) if n else np.zeros((0, 4, 4), np.float32)
        src.geom_infinite_plane = np.array(infinite, bool)
        src.body_names = _object_names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody, "body")
        src.joint_names = _object_names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt, "joint")
        src.geom_names = _object_names(m, mujoco.mjtObj.mjOBJ_GEOM, m.ngeom, "geom")
        src.site_names = _object_names(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite, "site")
        src.camera_names = _object_names(m, mujoco.mjtObj.mjOBJ_CAMERA, m.ncam, "camera")
        src.light_names = _object_names(m, mujoco.mjtObj.mjOBJ_LIGHT, m.nlight, "light")
        src.tendon_names = _object_names(m, mujoco.mjtObj.mjOBJ_TENDON, m.ntendon, "tendon")
        src.actuator_names = _object_names(m, mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu, "actuator")
        src.constraint_names = _object_names(m, mujoco.mjtObj.mjOBJ_EQUALITY, m.neq, "constraint")
        src.flex_names = _object_names(m, mujoco.mjtObj.mjOBJ_FLEX, m.nflex, "flex")
        self._build_flex_debug_source(src)
        src.lights = self._build_lights()
        src.cameras = tuple(self.camera_view(i) for i in range(m.ncam))
        src.scene_extent = float(m.stat.extent)

        src.shadow_clip = float(m.vis.map.shadowclip) or 1.0
        src.scene_center = np.asarray(m.stat.center, np.float32)
        src.debug_frame_length = float(m.stat.meansize) * float(m.vis.scale.framelength)

        src.initial_qpos = np.asarray(m.qpos0, np.float32).copy()

        src.initial_ctrl = np.zeros(m.nu, np.float32)
        tendon_matid = np.asarray(m.tendon_matid, np.int32)
        src.tendon_material = np.asarray(
            [mat_of_matid[int(matid)] for matid in tendon_matid], np.int32
        )
        src.tendon_rgba = np.asarray(m.tendon_rgba, np.float32).copy()
        material_color = (tendon_matid >= 0) & np.all(src.tendon_rgba == _GEOM_RGBA_DEFAULT, axis=1)
        src.tendon_rgba[material_color] = m.mat_rgba[tendon_matid[material_color]]
        src.tendon_visible = self._group_visibility(m.tendon_group, "tendon")
        self._island_rgba_buf = src.geom_rgba.copy()
        self._tendon_island_rgba_buf = np.zeros((m.ntendon, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((m.nflex, 4), np.float32)

        if skipped:
            names = ", ".join(sorted(str(mujoco.mjtGeom(t)) for t in skipped))

            note = f"Skipped unsupported geom types: {names}"
            if note not in self._notes:
                self._notes.append(note)
            self.caps = replace(self.caps, notes=tuple(self._notes))
        return src

    @staticmethod
    def _axis_rotation(axis) -> np.ndarray:
        z = np.asarray(axis, np.float32)
        z = z / np.linalg.norm(z)
        reference = np.array((1.0, 0.0, 0.0), np.float32)
        if abs(float(z[0])) > 0.9:
            reference = np.array((0.0, 1.0, 0.0), np.float32)
        x = np.cross(reference, z)
        x /= np.linalg.norm(x)
        return np.column_stack((x, np.cross(z, x), z)).astype(np.float32)

    def _site_rgba(self, si: int, matid: int) -> np.ndarray:
        rgba = np.asarray(self._m.site_rgba[si], np.float32)
        if matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
            return np.asarray(self._m.mat_rgba[matid], np.float32).copy()
        return rgba.copy()

    def _geom_rgba(self, gi: int, matid: int) -> np.ndarray:
        rgba = np.asarray(self._m.geom_rgba[gi], np.float32)
        if matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
            return np.asarray(self._m.mat_rgba[matid], np.float32).copy()
        return rgba.copy()

    def _build_mesh(self, mesh_id: int) -> MeshData:
        m = self._m
        va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
        fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
        na = int(m.mesh_normaladr[mesh_id])
        ta, tn = int(m.mesh_texcoordadr[mesh_id]), int(m.mesh_texcoordnum[mesh_id])

        verts = np.asarray(m.mesh_vert[va : va + vn], np.float32)
        face = np.asarray(m.mesh_face[fa : fa + fn], np.int32)
        fnorm = np.asarray(m.mesh_facenormal[fa : fa + fn], np.int32)
        normals_all = np.asarray(
            m.mesh_normal[na : na + int(m.mesh_normalnum[mesh_id])], np.float32
        )

        has_uv = ta >= 0 and tn > 0
        if has_uv:
            uvs_all = np.asarray(m.mesh_texcoord[ta : ta + tn], np.float32)
            ftex = np.asarray(m.mesh_facetexcoord[fa : fa + fn], np.int32)
        else:
            uvs_all = np.zeros((0, 2), np.float32)
            ftex = face

        aligned = np.array_equal(fnorm, face) and (not has_uv or np.array_equal(ftex, face))
        aligned = aligned and len(normals_all) == vn and (not has_uv or tn == vn)
        if aligned:
            uvs = uvs_all.copy() if has_uv else np.zeros((vn, 2), np.float32)
            return MeshData(
                positions=verts.copy(),
                normals=normals_all.copy(),
                uvs=uvs,
                indices=face.reshape(-1).astype(np.uint32),
            )

        corner_v = face.reshape(-1)
        corner_n = fnorm.reshape(-1)
        positions = verts[corner_v]
        normals = normals_all[corner_n] if len(normals_all) else np.zeros_like(positions)
        if has_uv:
            uvs = uvs_all[ftex.reshape(-1)]
        else:
            uvs = np.zeros((len(positions), 2), np.float32)
        return MeshData(
            positions=np.ascontiguousarray(positions, np.float32),
            normals=np.ascontiguousarray(normals, np.float32),
            uvs=np.ascontiguousarray(uvs, np.float32),
            indices=np.arange(len(positions), dtype=np.uint32),
        )

    def _build_convex_hull(self, mesh_id: int) -> MeshData:
        m = self._m
        graph_adr = int(m.mesh_graphadr[mesh_id])
        vertex_count = int(m.mesh_graph[graph_adr])
        face_count = int(m.mesh_graph[graph_adr + 1])
        face_adr = graph_adr + 2 + 3 * vertex_count + 3 * face_count
        faces = np.asarray(m.mesh_graph[face_adr : face_adr + 3 * face_count], np.int32).reshape(
            -1, 3
        )

        vertex_adr = int(m.mesh_vertadr[mesh_id])
        vertices = np.asarray(
            m.mesh_vert[vertex_adr : vertex_adr + int(m.mesh_vertnum[mesh_id])], np.float32
        )
        triangles = vertices[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(lengths, 1e-12)

        texcoord_adr = int(m.mesh_texcoordadr[mesh_id])
        texcoord_count = int(m.mesh_texcoordnum[mesh_id])
        if texcoord_adr >= 0 and texcoord_count > int(faces.max(initial=-1)):
            texcoords = np.asarray(
                m.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_count], np.float32
            )[faces]
        else:
            texcoords = np.zeros((face_count, 3, 2), np.float32)

        positions = triangles.reshape(-1, 3)
        normals = np.repeat(face_normals[:, None, :], 3, axis=1).reshape(-1, 3)
        return MeshData(
            positions=np.ascontiguousarray(positions, np.float32),
            normals=np.ascontiguousarray(normals, np.float32),
            uvs=np.ascontiguousarray(texcoords.reshape(-1, 2), np.float32),
            indices=np.arange(len(positions), dtype=np.uint32),
        )

    def _build_heightfield(self, field_id: int) -> MeshData:
        m = self._m
        rows = int(m.hfield_nrow[field_id])
        cols = int(m.hfield_ncol[field_id])
        adr = int(m.hfield_adr[field_id])
        height = float(m.hfield_size[field_id][2])
        base = float(m.hfield_size[field_id][3])
        z0 = -base / max(height, 1e-12)
        data = np.asarray(m.hfield_data[adr : adr + rows * cols], np.float32).reshape(rows, cols)

        positions: list[tuple[float, float, float]] = []
        uvs: list[tuple[float, float]] = []
        indices: list[int] = []

        def vertex(x: float, y: float, z: float, u: float, v: float) -> int:
            positions.append((x, y, z))
            uvs.append((u, v))
            return len(positions) - 1

        top = np.zeros((rows, cols), np.int32)
        for r in range(rows):
            v = r / max(rows - 1, 1)
            y = 2.0 * v - 1.0
            for c in range(cols):
                u = c / max(cols - 1, 1)
                top[r, c] = vertex(2.0 * u - 1.0, y, float(data[r, c]), u, v)
        for r in range(rows - 1):
            for c in range(cols - 1):
                a, b = int(top[r, c]), int(top[r, c + 1])
                d, e = int(top[r + 1, c]), int(top[r + 1, c + 1])
                indices += (a, b, e, a, e, d)

        boundary = [
            [(float(top[0, c]), c / (cols - 1)) for c in range(cols)],
            [(float(top[r, cols - 1]), r / (rows - 1)) for r in range(rows)],
            [(float(top[rows - 1, c]), 1.0 - c / (cols - 1)) for c in range(cols - 1, -1, -1)],
            [(float(top[r, 0]), 1.0 - r / (rows - 1)) for r in range(rows - 1, -1, -1)],
        ]
        pos = positions
        for edge in boundary:
            for (top_a, u0), (top_b, u1) in pairwise(edge):
                pa, pb = pos[int(top_a)], pos[int(top_b)]
                a = vertex(*pa, u0, 1.0)
                b = vertex(*pb, u1, 1.0)
                c = vertex(pb[0], pb[1], z0, u1, 0.0)
                d = vertex(pa[0], pa[1], z0, u0, 0.0)
                indices += (a, b, c, a, c, d)

        bottom = [
            vertex(-1.0, -1.0, z0, 0.0, 0.0),
            vertex(1.0, -1.0, z0, 1.0, 0.0),
            vertex(1.0, 1.0, z0, 1.0, 1.0),
            vertex(-1.0, 1.0, z0, 0.0, 1.0),
        ]
        indices += (bottom[0], bottom[2], bottom[1], bottom[0], bottom[3], bottom[2])

        p = np.asarray(positions, np.float32)
        idx = np.asarray(indices, np.uint32)
        tri = idx.reshape(-1, 3)
        face_n = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        normals = np.zeros_like(p)
        for corner in range(3):
            np.add.at(normals, tri[:, corner], face_n)
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.maximum(length, 1e-12)
        return MeshData(p, normals, np.asarray(uvs, np.float32), idx)

    def _build_textures(self) -> dict[str, TextureData]:
        """Convert compiled MuJoCo textures to Mojive texture arrays."""

        m = self._m
        out: dict[str, TextureData] = {}
        for ti in range(m.ntex):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, ti) or f"tex{ti}"
            w, h = int(m.tex_width[ti]), int(m.tex_height[ti])
            c = int(m.tex_nchannel[ti])
            adr = int(m.tex_adr[ti])
            raw = np.asarray(m.tex_data[adr : adr + w * h * c], np.uint8)
            ttype = int(m.tex_type[ti])
            if ttype == mujoco.mjtTexture.mjTEXTURE_CUBE:
                texture_type = TextureType.CUBE
            elif ttype == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                texture_type = TextureType.SKYBOX
            else:
                texture_type = TextureType.TWO_D
            if texture_type is TextureType.TWO_D:
                pixels = raw.reshape(h, w, c)
            elif h == w:
                # MuJoCo repeats a square cube texture on all six faces.
                face = raw.reshape(h, w, c)
                pixels = np.repeat(face[None, ...], 6, axis=0)
            else:
                # Separate cube faces are stored as a vertical 6-by-1 strip.
                pixels = raw.reshape(6, w, w, c)
            out[name] = TextureData(name=name, type=texture_type, pixels=pixels.copy(), srgb=True)
        return out

    def _build_materials(
        self, textures: dict[str, TextureData]
    ) -> tuple[list[Material], dict[int, int]]:
        m = self._m
        tex_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i) or f"tex{i}" for i in range(m.ntex)
        ]
        out: list[Material] = []
        index: dict[int, int] = {}
        for mi in range(m.nmat):
            texid = -1
            for role in (_TEXROLE_RGB, _TEXROLE_RGBA):
                cand = int(m.mat_texid[mi][role])
                if cand >= 0:
                    texid = cand
                    break
            tex = tex_names[texid] if 0 <= texid < len(tex_names) else None
            if tex is not None and tex not in textures:
                tex = None
            index[mi] = len(out)
            out.append(
                Material(
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mi) or f"mat{mi}",
                    rgba=np.asarray(m.mat_rgba[mi], np.float32).copy(),
                    emission=float(m.mat_emission[mi]),
                    specular=float(m.mat_specular[mi]),
                    shininess=float(m.mat_shininess[mi]),
                    reflectance=float(m.mat_reflectance[mi]),
                    metallic=float(m.mat_metallic[mi]),
                    roughness=float(m.mat_roughness[mi]),
                    texture=tex,
                    tex_repeat=np.asarray(m.mat_texrepeat[mi], np.float32).copy(),
                    tex_uniform=bool(m.mat_texuniform[mi]),
                )
            )

        index[-1] = len(out)
        out.append(Material(name="__geom__"))
        return out, index

    def _light(self, i: int, pos: np.ndarray, direction: np.ndarray) -> Light:
        m = self._m
        ltype = int(m.light_type[i])

        # MuJoCo values: spot=0, directional=1, point=2, image=3.

        if i < len(self._area_lights) and self._area_lights[i]:
            light_type = LightType.AREA
        elif ltype == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
            light_type = LightType.DIRECTIONAL
        elif ltype == mujoco.mjtLightType.mjLIGHT_POINT:
            light_type = LightType.POINT
        elif ltype == mujoco.mjtLightType.mjLIGHT_SPOT:
            light_type = LightType.SPOT
        else:
            light_type = LightType.IMAGE
        texid = int(m.light_texid[i])
        texture = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, texid) if texid >= 0 else None
        return Light(
            type=light_type,
            position=np.asarray(pos, np.float32).copy(),
            direction=np.asarray(direction, np.float32).copy(),
            diffuse=np.asarray(m.light_diffuse[i], np.float32).copy(),
            specular=np.asarray(m.light_specular[i], np.float32).copy(),
            ambient=np.asarray(m.light_ambient[i], np.float32).copy(),
            attenuation=np.asarray(m.light_attenuation[i], np.float32).copy(),
            range=float(m.light_range[i]),
            area_radius=float(m.light_bulbradius[i]),
            cutoff=float(m.light_cutoff[i]),
            exponent=float(m.light_exponent[i]),
            texture=texture,
            intensity=float(m.light_intensity[i]),
            cast_shadow=bool(m.light_castshadow[i]),
            active=bool(m.light_active[i]),
        )

    def _build_lights(self) -> LightSet:
        d = self._d
        return self._light_set(
            tuple(self._light(i, d.light_xpos[i], d.light_xdir[i]) for i in range(self._m.nlight))
        )

    def _dynamic_lights(self) -> LightSet:
        d = self._d
        preview = self._model_transform_preview
        lights = []
        for i in range(self._m.nlight):
            position = d.light_xpos[i]
            direction = d.light_xdir[i]
            if preview is not None and preview.light_mask[i]:
                position = preview.position + preview.delta_rotation @ (
                    np.asarray(position, np.float64) - preview.previous_position
                )
                direction = preview.delta_rotation @ np.asarray(direction, np.float64)
            lights.append(self._light(i, position, direction))
        return self._light_set(tuple(lights))

    def _light_set(self, lights: tuple[Light, ...]) -> LightSet:
        m = self._m
        extent = float(m.stat.extent) or 1.0
        ambient = _numeric_values(m, _MOJIVE_AMBIENT_NUMERIC)
        haze = _numeric_values(m, _MOJIVE_HAZE_NUMERIC)
        return LightSet(
            lights=lights,
            headlight=self._headlight(),
            ambient=(
                np.asarray(ambient[:3], np.float32).copy()
                if ambient is not None and len(ambient) >= 3
                else self._global_ambient()
            ),
            fog_color=np.asarray(m.vis.rgba.fog[:3], np.float32).copy(),
            fog_start=float(m.vis.map.fogstart) * extent,
            fog_end=float(m.vis.map.fogend) * extent,
            haze_color=np.asarray(m.vis.rgba.haze[:3], np.float32).copy(),
            haze_density=float(m.vis.map.haze),
            horizon_haze=bool(haze[0]) if haze is not None and len(haze) >= 1 else True,
            horizon_haze_slices=(
                max(3, round(haze[1]))
                if haze is not None and len(haze) >= 2
                else max(3, int(m.vis.quality.numslices))
            ),
        )

    def _headlight(self) -> Light | None:
        hl = self._m.vis.headlight
        if not bool(hl.active):
            return None
        return Light(
            type=DEFAULT_HEADLIGHT.type,
            diffuse=np.asarray(hl.diffuse, np.float32).copy(),
            specular=np.asarray(hl.specular, np.float32).copy(),
            ambient=np.asarray(hl.ambient, np.float32).copy(),
            cast_shadow=DEFAULT_HEADLIGHT.cast_shadow,
        )

    def _global_ambient(self) -> np.ndarray:
        total = np.zeros(3, np.float32)
        hl = self._m.vis.headlight
        if bool(hl.active):
            total += np.asarray(hl.ambient, np.float32)
        if self._m.nlight:
            lights = np.asarray(self._m.light_ambient, np.float32)
            active = np.asarray(self._m.light_active, np.float32).reshape(-1, 1)
            total += (lights * active).sum(axis=0)
        return np.clip(total, 0.0, 1.0)

    def _build_nodes(self) -> list[SceneNode]:
        m = self._m
        nodes: list[SceneNode] = []
        body_node: dict[int, int] = {}
        self._node_body = {}
        self._node_element = {}
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}

        body_parent = np.asarray(m.body_parentid, np.int32)
        has_child = np.zeros(m.nbody, bool)
        if m.nbody > 1:
            has_child[body_parent[1:]] = True
        has_kinematic_dof = np.zeros(m.nbody, bool)
        for body in range(1, m.nbody):
            has_kinematic_dof[body] = (
                bool(m.body_jntnum[body]) or has_kinematic_dof[body_parent[body]]
            )

        def add(name: str, node_type: NodeType, parent: int, body: int, **kw) -> int:
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id=node_id,
                    name=name,
                    type=node_type,
                    parent=parent,
                    body_index=body,
                    **kw,
                )
            )
            if parent >= 0:
                nodes[parent].children.append(node_id)
            self._node_body[node_id] = body
            return node_id

        world_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, 0) or "world"
        body_node[0] = add(world_name, NodeType.WORLD, -1, 0, object_id=0)
        self._node_element[body_node[0]] = (0, NodeType.WORLD, world_name)
        model_parents = {
            item.model_id: add(
                item.name,
                NodeType.MODEL,
                body_node[0],
                -1,
                model_id=item.model_id,
                object_id=MODEL_OBJECT_BASE + item.model_id,
                posable=True,
            )
            for item in self._attached_models
        }
        self._node_model = {node_id: model_id for model_id, node_id in model_parents.items()}
        for model_id, node_id in model_parents.items():
            self._node_element[node_id] = (model_id, NodeType.MODEL, "")

        for b in range(1, m.nbody):
            compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            name = compiled_name or f"body{b}"
            parent = int(body_parent[b])
            node_type = NodeType.ROBOT if parent == 0 and has_child[b] else NodeType.LINK
            model_id, raw_name = self._model_element_name(
                compiled_name or "", mujoco.mjtObj.mjOBJ_BODY
            )
            source_editable = (
                bool(raw_name) and self._element(model_id, "link", raw_name) is not None
            )
            parent_node = body_node[parent]
            if parent == 0 and model_id in model_parents:
                parent_node = model_parents[model_id]
            body_node[b] = add(
                name,
                node_type,
                parent_node,
                b,
                object_id=b,
                posable=self._is_posable_body(b) or (source_editable and not has_kinematic_dof[b]),
                source_editable=source_editable,
            )
            nodes[body_node[b]].model_id = model_id
            self._node_element[body_node[b]] = (model_id, node_type, raw_name)

        for b in range(m.nbody):
            parent = body_node[b]
            adr, num = int(m.body_geomadr[b]), int(m.body_geomnum[b])
            for gi in range(adr, adr + num):
                if not self._visual_groups["geom"][int(m.geom_group[gi])]:
                    continue
                compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
                gname = compiled_name or f"geom{gi}"
                model_id, raw_name = self._model_element_name(
                    compiled_name or "", mujoco.mjtObj.mjOBJ_GEOM
                )
                source_editable = (
                    bool(raw_name) and self._element(model_id, "geom", raw_name) is not None
                )
                is_plane = int(m.geom_type[gi]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                is_infinite_plane = is_plane and (
                    float(m.geom_size[gi, 0]) == 0.0 or float(m.geom_size[gi, 1]) == 0.0
                )
                identity = (model_id, raw_name or gname)
                object_id = (
                    self._geometry_object_id(*identity)
                    if identity in self._geometry_object_ids
                    or (b == 0 and is_plane and not is_infinite_plane)
                    else 0
                )
                self._geom_nodes[gi] = add(
                    gname,
                    NodeType.GEOM,
                    parent,
                    b,
                    object_id=object_id,
                    geom_index=gi,
                    posable=source_editable,
                    source_editable=source_editable,
                )
                nodes[self._geom_nodes[gi]].model_id = model_id
                self._node_element[self._geom_nodes[gi]] = (
                    model_id,
                    NodeType.GEOM,
                    raw_name,
                )
            ja, jn = int(m.body_jntadr[b]), int(m.body_jntnum[b])
            for ji in range(ja, ja + jn):
                if not self._visual_groups["joint"][int(m.jnt_group[ji])]:
                    continue
                compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji)
                jname = compiled_name or f"joint{ji}"
                node_id = add(jname, NodeType.JOINT, parent, b, joint_index=ji)
                model_id, raw_name = self._model_element_name(
                    compiled_name or "", mujoco.mjtObj.mjOBJ_JOINT
                )
                nodes[node_id].model_id = model_id
                nodes[node_id].source_editable = bool(raw_name) and (
                    self._element(model_id, "joint", raw_name) is not None
                )
                self._node_element[node_id] = (model_id, NodeType.JOINT, raw_name)

        for li in range(m.nlight):
            b = int(m.light_bodyid[li])
            compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, li)
            name = compiled_name or f"light{li}"
            node_id = add(
                name,
                NodeType.LIGHT,
                body_node[b],
                b,
                object_id=LIGHT_OBJECT_BASE + li,
                visible=bool(m.light_active[li]),
                light_index=li,
            )
            model_id, raw_name = self._model_element_name(
                compiled_name or "", mujoco.mjtObj.mjOBJ_LIGHT
            )
            nodes[node_id].model_id = model_id
            nodes[node_id].source_editable = bool(raw_name) and (
                self._element(model_id, "light", raw_name) is not None
            )
            self._node_element[node_id] = (model_id, NodeType.LIGHT, raw_name)
        for ci in range(m.ncam):
            b = int(m.cam_bodyid[ci])
            compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, ci)
            name = compiled_name or f"camera{ci}"
            node_id = add(
                name,
                NodeType.CAMERA,
                body_node[b],
                b,
                object_id=CAMERA_OBJECT_BASE + ci,
                camera_index=ci,
            )
            model_id, raw_name = self._model_element_name(
                compiled_name or "", mujoco.mjtObj.mjOBJ_CAMERA
            )
            nodes[node_id].model_id = model_id
            nodes[node_id].source_editable = bool(raw_name) and (
                self._element(model_id, "camera", raw_name) is not None
            )
            self._node_element[node_id] = (model_id, NodeType.CAMERA, raw_name)
        for si in range(m.nsite):
            if not self._visual_groups["site"][int(m.site_group[si])]:
                continue
            b = int(m.site_bodyid[si])
            compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, si)
            name = compiled_name or f"site{si}"
            model_id, raw_name = self._model_element_name(
                compiled_name or "", mujoco.mjtObj.mjOBJ_SITE
            )
            source_editable = (
                bool(raw_name) and self._element(model_id, "site", raw_name) is not None
            )
            self._site_nodes[si] = add(
                name,
                NodeType.SITE,
                body_node[b],
                b,
                site_index=si,
                # A site pose is authored relative to its body. Joint-driven
                # parent motion does not make that local transform read-only.
                posable=source_editable,
                source_editable=source_editable,
            )
            nodes[self._site_nodes[si]].model_id = model_id
            self._node_element[self._site_nodes[si]] = (model_id, NodeType.SITE, raw_name)
        for fi in range(m.nflex):
            if not self._visual_groups["flex"][int(m.flex_group[fi])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fi) or f"flex{fi}"
            self._flex_nodes[fi] = add(name, NodeType.FLEX, body_node[0], 0, object_id=m.nbody + fi)
        for si in range(m.nskin):
            if not self._visual_groups["skin"][int(m.skin_group[si])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SKIN, si) or f"skin{si}"
            self._skin_nodes[si] = add(
                name,
                NodeType.SKIN,
                body_node[0],
                0,
                object_id=m.nbody + m.nflex + si,
            )
        return nodes

    def _rebuild_model_element_names(self) -> None:
        """Index compiled model names by owning editable MjSpec in O(E)."""
        collections = (
            (mujoco.mjtObj.mjOBJ_BODY, "bodies"),
            (mujoco.mjtObj.mjOBJ_GEOM, "geoms"),
            (mujoco.mjtObj.mjOBJ_JOINT, "joints"),
            (mujoco.mjtObj.mjOBJ_SITE, "sites"),
            (mujoco.mjtObj.mjOBJ_CAMERA, "cameras"),
            (mujoco.mjtObj.mjOBJ_LIGHT, "lights"),
            (mujoco.mjtObj.mjOBJ_MATERIAL, "materials"),
            (mujoco.mjtObj.mjOBJ_TEXTURE, "textures"),
            (mujoco.mjtObj.mjOBJ_HFIELD, "hfields"),
            (mujoco.mjtObj.mjOBJ_KEY, "keys"),
        )
        index: dict[tuple[int, str], tuple[int, str]] = {}
        specs = [(item.model_id, item.prefix, item.spec) for item in self._attached_models]
        if self._root_spec is not None:
            # Root names win if they merely resemble an attached-model prefix.
            specs.append((0, "", self._root_spec))
        for model_id, prefix, spec in specs:
            for object_type, collection_name in collections:
                for element in getattr(spec, collection_name, ()):
                    raw_name = str(element.name or "")
                    if raw_name:
                        index[(int(object_type), f"{prefix}{raw_name}")] = (
                            int(model_id),
                            raw_name,
                        )
        self._model_element_names = index

    def _model_element_name(self, compiled_name: str, object_type) -> tuple[int, str]:
        name = str(compiled_name)
        return self._model_element_names.get((int(object_type), name), (0, name))

    def _is_free_body(self, body: int) -> bool:
        m = self._m
        adr, num = int(m.body_jntadr[body]), int(m.body_jntnum[body])
        return num == 1 and int(m.jnt_type[adr]) == mujoco.mjtJoint.mjJNT_FREE

    def _is_posable_body(self, body: int) -> bool:
        return int(self._m.body_mocapid[body]) >= 0 or self._is_free_body(body)
