from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from .. import math3d
from ..types import (
    DEFAULT_HEADLIGHT,
    CameraView,
    InstancePoseSource,
    Light,
    LightKind,
    LightSet,
    Material,
    MeshData,
    MeshKey,
    MeshShape,
    TextureData,
    TextureKind,
)
from .base import (
    ActuatorInfo,
    ActuatorVisualKind,
    AdapterCaps,
    CameraInfo,
    DiagnosticFrame,
    DiagnosticSource,
    FrameNeeds,
    JointInfo,
    JointVisualKind,
    KeyframeInfo,
    NodeKind,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
    VisualGroupInfo,
)
from .mujoco_deformables import build_deformables, update_deformables

if TYPE_CHECKING:
    from pathlib import Path

try:
    import mujoco
except ImportError as exc:
    mujoco = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None


DEFAULT_GEOM_GROUPS: tuple[int, ...] = (0, 1, 2)
VISUAL_GROUP_CATEGORIES = ("geom", "site", "joint", "tendon", "actuator", "flex", "skin")


_GEOM_RGBA_DEFAULT = np.array([0.5, 0.5, 0.5, 1.0], np.float32)


_TEXROLE_RGB = 1
_TEXROLE_RGBA = 8


_ACTUATOR_POSE_JOINT_AXIS = 0
_ACTUATOR_POSE_JOINT_BODY = 1
_ACTUATOR_POSE_SITE = 2
_ACTUATOR_POSE_GEOM = 3


class MuJoCoAdapter:
    def __init__(self, path: Path | None = None) -> None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(
                f"MuJoCo is not installed: {_IMPORT_ERROR}. Install the [mujoco] optional dependency."
            )
        self.caps = AdapterCaps(
            name="mujoco",
            simulation=True,
            write_pose=True,
            write_qpos=True,
            perturb=True,
            raycast=True,
            inverse_kinematics=False,
            contacts=True,
            model_cameras=True,
            keyframes=True,
            sensors=True,
            visual_groups=True,
            reload=True,
        )
        self._m = None
        self._d = None
        self._path: Path | None = None
        self._structure_revision = 0
        self._notes: list[str] = []

        self._geom_xpos_buf = np.zeros((0, 3), np.float32)
        self._geom_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((0, 3), np.float32)
        self._site_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((0, 3), np.float32)
        self._body_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._diagnostic_frame = DiagnosticFrame()
        self._qpos_buf = np.zeros(0, np.float32)
        self._qvel_buf = np.zeros(0, np.float32)
        self._ctrl_buf = np.zeros(0, np.float32)
        self._sensor_buf = np.zeros(0, np.float32)
        self._contact_buf = np.zeros((0, 7), np.float32)
        self._contact_force = np.zeros(6, np.float64)
        self._contact_view = self._contact_buf
        self._tendon_segments = np.zeros((0, 2, 3), np.float32)
        self._tendon_ids = np.zeros(0, np.int32)
        self._tendon_widths = np.zeros(0, np.float32)
        self._actuator_visual_pose_kinds = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)

        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_body_xpos = None
        self._mj_body_xmat3 = None
        self._fast_pose = False

        self._ray_pnt = np.zeros(3, np.float64)
        self._ray_vec = np.zeros(3, np.float64)
        self._ray_geomid = np.zeros(1, np.int32)

        defaults = np.array([g in DEFAULT_GEOM_GROUPS for g in range(6)], dtype=bool)
        self._visual_groups = {name: defaults.copy() for name in VISUAL_GROUP_CATEGORIES}
        self._ray_geomgroup = self._visual_groups["geom"].astype(np.uint8)

        self._perturb = mujoco.MjvPerturb()
        self._perturb_body = -1
        self._perturb_jac = np.zeros((3, 0), np.float64)
        self._perturb_jac_m2 = np.zeros((3, 0), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(0, np.float64)
        self._perturb_quat = np.zeros(4, np.float64)

        self._frame = SceneFrame()
        self._source: SceneSource | None = None
        self._nodes: list[SceneNode] = []
        self._node_body: dict[int, int] = {}
        self._geom_nodes: dict[int, int] = {}
        self._site_nodes: dict[int, int] = {}
        self._flex_nodes: dict[int, int] = {}
        self._skin_nodes: dict[int, int] = {}
        self._deformables = []
        self._mesh_updates = {}
        self._lights_dynamic = False
        self._lights_edited = False

        if path is not None:
            self.load(path)

    def load(self, path: Path) -> None:

        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        self._path = path
        self._install(model)

    def reload(self) -> None:
        if self._path is None:
            raise RuntimeError("No asset has been loaded")
        self.load(self._path)

    def _install(self, model) -> None:
        self._m = model
        self._d = mujoco.MjData(model)
        self._notes = []
        mujoco.mj_forward(self._m, self._d)

        g, b = model.ngeom, model.nbody
        self._geom_xpos_buf = np.zeros((g, 3), np.float32)
        self._geom_xmat_buf = np.zeros((g, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((model.nsite, 3), np.float32)
        self._site_xmat_buf = np.zeros((model.nsite, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((b, 3), np.float32)
        self._body_xmat_buf = np.zeros((b, 3, 3), np.float32)
        self._diagnostic_frame = DiagnosticFrame(
            joint_xpos=np.zeros((model.njnt, 3), np.float32),
            joint_xaxis=np.zeros((model.njnt, 3), np.float32),
            subtree_com=np.zeros((b, 3), np.float32),
            body_xipos=np.zeros((b, 3), np.float32),
            body_ximat=np.zeros((b, 3, 3), np.float32),
        )
        self._qpos_buf = np.zeros(model.nq, np.float32)
        self._qvel_buf = np.zeros(model.nv, np.float32)
        self._ctrl_buf = np.zeros(model.nu, np.float32)
        self._activation_buf = np.zeros(model.nactuator, np.float32)
        self._actuator_ctrl_address = np.asarray(model.actuator_ctrladr, np.int32).copy()
        self._ctrl_actuator = np.full(model.nu, -1, np.int32)
        for actuator, (address, count) in enumerate(
            zip(model.actuator_ctrladr, model.actuator_ctrlnum, strict=True)
        ):
            self._ctrl_actuator[int(address) : int(address) + int(count)] = actuator
        self._actuator_act_index = np.where(
            np.asarray(model.actuator_dyntype) != 0,
            np.asarray(model.actuator_actadr) + np.asarray(model.actuator_actnum) - 1,
            -1,
        ).astype(np.int32)
        self._sensor_buf = np.zeros(model.nsensordata, np.float32)
        self._contact_buf = np.zeros((max(model.ngeom, 64), 7), np.float32)
        self._contact_view = self._contact_buf[:0]
        d = self._d
        wrap_capacity = d.wrap_xpos.size // 3
        self._tendon_segments = np.zeros((wrap_capacity, 2, 3), np.float32)
        self._tendon_ids = np.zeros(wrap_capacity, np.int32)
        self._tendon_widths = np.zeros(wrap_capacity, np.float32)
        self._actuator_visual_pose_kinds = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)
        self._mj_geom_xpos = d.geom_xpos
        self._mj_geom_xmat3 = d.geom_xmat.reshape(g, 3, 3)
        self._mj_site_xmat3 = d.site_xmat.reshape(model.nsite, 3, 3)
        # MuJoCo 3.11 exposes wrap_xpos as (nwrap, 6): ten_wrapadr/num address
        # the flattened xyz triples, not the first column of each row.
        self._mj_wrap_points = d.wrap_xpos.reshape(-1, 3)
        self._mj_wrap_objects = d.wrap_obj.reshape(-1)
        self._mj_body_xpos = d.xpos
        self._mj_body_xmat3 = d.xmat.reshape(b, 3, 3)
        self._fast_pose = self._verify_pose_layout()

        self._frame = SceneFrame()
        self._source = None
        self._nodes = []
        self._node_body = {}
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}
        self._deformables = []
        self._mesh_updates = {}
        for groups in self._visual_groups.values():
            groups[:] = [g in DEFAULT_GEOM_GROUPS for g in range(6)]
        self._ray_geomgroup[:] = self._visual_groups["geom"]
        self._lights_dynamic = bool(model.nlight) and bool(np.any(model.light_bodyid != 0))
        self._lights_edited = False
        self._perturb = mujoco.MjvPerturb()
        self._perturb_body = -1
        self._perturb_jac = np.zeros((3, model.nv), np.float64)
        self._perturb_jac_m2 = np.zeros((3, model.nv), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(model.nv, np.float64)
        self._structure_revision += 1

    @property
    def structure_revision(self) -> int:
        return self._structure_revision

    def _verify_pose_layout(self) -> bool:

        m, d = self._m, self._d
        g = m.ngeom
        if g == 0:
            return True
        try:
            xpos, xmat3 = self._mj_geom_xpos, self._mj_geom_xmat3
            if xpos.shape != (g, 3) or xmat3.shape != (g, 3, 3):
                return False
            if xpos.dtype != np.float64 or xmat3.dtype != np.float64:
                return False
            if not (xpos.flags["C_CONTIGUOUS"] and xmat3.flags["C_CONTIGUOUS"]):
                return False

            probe_pos = np.arange(g * 3, dtype=np.float64).reshape(g, 3) * 0.5 + 1.0
            probe_mat = np.arange(g * 9, dtype=np.float64).reshape(g, 3, 3) * 0.25 - 3.0
            xpos[:] = probe_pos
            xmat3[:] = probe_mat.reshape(g, 3, 3)
            for i in range(g):
                view = d.geom(i)
                if not np.array_equal(view.xpos, probe_pos[i]):
                    return False
                if not np.array_equal(view.xmat.reshape(3, 3), probe_mat[i]):
                    return False
            return True
        except Exception:
            return False
        finally:
            mujoco.mj_forward(m, d)

    def _fill_poses(self) -> None:

        if self._fast_pose:
            np.copyto(self._geom_xpos_buf, self._mj_geom_xpos, casting="unsafe")
            np.copyto(self._geom_xmat_buf, self._mj_geom_xmat3, casting="unsafe")
            np.copyto(self._body_xpos_buf, self._mj_body_xpos, casting="unsafe")
            np.copyto(self._body_xmat_buf, self._mj_body_xmat3, casting="unsafe")
            return

        d = self._d
        for i in range(len(self._geom_xpos_buf)):
            view = d.geom(i)
            self._geom_xpos_buf[i] = view.xpos
            self._geom_xmat_buf[i] = view.xmat.reshape(3, 3)
        for i in range(len(self._body_xpos_buf)):
            view = d.body(i)
            self._body_xpos_buf[i] = view.xpos
            self._body_xmat_buf[i] = view.xmat.reshape(3, 3)

    def reset(self) -> None:
        mujoco.mj_resetData(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)

    def set_paused(self, paused: bool) -> bool:
        """Pause ownership lives in Session; local MuJoCo needs no additional state."""
        return True

    def step(self, count: int = 1) -> None:
        for _ in range(max(1, int(count))):
            mujoco.mj_step(self._m, self._d)

    def timestep(self) -> float:
        return float(self._m.opt.timestep)

    def frame(self, needs: FrameNeeds) -> SceneFrame:

        d = self._d
        f = self._frame
        f.time = float(d.time)

        if needs.poses:
            self._fill_poses()
            f.geom_xpos = self._geom_xpos_buf
            f.geom_xmat = self._geom_xmat_buf
            np.copyto(self._site_xpos_buf, d.site_xpos, casting="unsafe")
            np.copyto(self._site_xmat_buf, self._mj_site_xmat3, casting="unsafe")
            f.site_xpos = self._site_xpos_buf
            f.site_xmat = self._site_xmat_buf
            f.body_xpos = self._body_xpos_buf
            f.body_xmat = self._body_xmat_buf
        else:
            f.geom_xpos = f.geom_xmat = f.site_xpos = f.site_xmat = None
            f.body_xpos = f.body_xmat = None

        if needs.qpos:
            np.copyto(self._qpos_buf, d.qpos, casting="unsafe")
            f.qpos = self._qpos_buf
        else:
            f.qpos = None

        if needs.qvel:
            np.copyto(self._qvel_buf, d.qvel, casting="unsafe")
            f.qvel = self._qvel_buf
        else:
            f.qvel = None

        if needs.actuator:
            np.copyto(self._ctrl_buf, d.ctrl, casting="unsafe")
            f.ctrl = self._ctrl_buf
            np.take(d.ctrl, self._actuator_ctrl_address, out=self._activation_buf)
            active = self._actuator_act_index >= 0
            self._activation_buf[active] = d.act[self._actuator_act_index[active]]
            f.actuator_activation = self._activation_buf
        else:
            f.ctrl = None
            f.actuator_activation = None

        if needs.sensors:
            np.copyto(self._sensor_buf, d.sensordata, casting="unsafe")
            f.sensors = self._sensor_buf
        else:
            f.sensors = None

        f.contacts = self._fill_contacts() if needs.contacts else None
        if needs.tendons:
            f.tendon_segments, f.tendon_ids, f.tendon_widths = self._fill_tendons()
        else:
            f.tendon_segments = f.tendon_ids = f.tendon_widths = None
        if needs.deformables:
            update_deformables(self._deformables, d)
            f.mesh_updates = self._mesh_updates
        else:
            f.mesh_updates = None

        if needs.diagnostics:
            diagnostics = self._diagnostic_frame
            np.copyto(diagnostics.joint_xpos, d.xanchor, casting="unsafe")
            np.copyto(diagnostics.joint_xaxis, d.xaxis, casting="unsafe")
            np.copyto(diagnostics.subtree_com, d.subtree_com, casting="unsafe")
            np.copyto(diagnostics.body_xipos, d.xipos, casting="unsafe")
            np.copyto(
                diagnostics.body_ximat,
                d.ximat.reshape(self._m.nbody, 3, 3),
                casting="unsafe",
            )
            self._fill_actuator_visual_poses(diagnostics)
            f.diagnostics = diagnostics
        else:
            f.diagnostics = None

        f.lights = self._dynamic_lights() if self._lights_dynamic or self._lights_edited else None
        return f

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

    def _fill_contacts(self) -> np.ndarray:

        d, m = self._d, self._m
        n = int(d.ncon)
        if n > len(self._contact_buf):
            self._contact_buf = np.zeros((max(n, 2 * len(self._contact_buf)), 7), np.float32)
            self._contact_view = self._contact_buf[:0]
        for i in range(n):
            c = d.contact[i]
            self._contact_buf[i, 0:3] = c.pos
            self._contact_buf[i, 3:6] = c.frame[0:3]
            mujoco.mj_contactForce(m, d, i, self._contact_force)
            self._contact_buf[i, 6] = self._contact_force[0]
        if len(self._contact_view) != n:
            self._contact_view = self._contact_buf[:n]
        return self._contact_view

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
        src.skybox = next((t.name for t in textures.values() if t.kind is TextureKind.SKYBOX), None)
        materials, mat_of_matid = self._build_materials(textures)
        src.materials = materials

        meshes: dict[MeshKey, MeshData] = {}
        mesh_keys: list[MeshKey] = []
        mats: list[int] = []
        sizes: list[np.ndarray] = []
        rgbas: list[np.ndarray] = []
        object_ids: list[int] = []
        bodies: list[int] = []
        sources: list[int] = []
        pose_sources: list[int] = []
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
        ) -> None:
            for key, scale, cap_offset in parts:
                if key.shape is not MeshShape.ASSET and key not in meshes:
                    meshes[key] = None
                mesh_keys.append(key)
                mats.append(mat_index)
                sizes.append(np.asarray(scale, np.float32))
                rgbas.append(rgba)
                object_ids.append(object_id)
                bodies.append(body)
                sources.append(source)
                pose_sources.append(int(pose_source))
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
            else:
                skipped.add(gtype)
                continue

            append_parts(
                parts,
                mat_index=mat_index,
                rgba=rgba,
                body=body,
                source=gi,
                pose_source=InstancePoseSource.GEOM,
                node_id=self._geom_nodes.get(gi, -1),
                object_id=body,
                is_infinite=is_infinite,
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
            )

        self._deformables = build_deformables(m, self._d, flex_groups, skin_groups)
        self._mesh_updates = {spec.key: spec.update_data for spec in self._deformables}
        for spec in self._deformables:
            meshes[spec.key] = spec.mesh
            mat_index = mat_of_matid[spec.matid] if spec.matid >= 0 else mat_of_matid[-1]
            rgba = spec.rgba
            if spec.matid >= 0 and np.array_equal(rgba, _GEOM_RGBA_DEFAULT):
                rgba = np.asarray(m.mat_rgba[spec.matid], np.float32)
            node_id = (
                self._flex_nodes.get(spec.key.index, -1)
                if spec.key.shape is MeshShape.FLEX
                else self._skin_nodes.get(spec.key.index, -1)
            )
            object_id = (
                m.nbody + spec.key.index
                if spec.key.shape is MeshShape.FLEX
                else m.nbody + m.nflex + spec.key.index
            )
            append_parts(
                [(spec.key, np.ones(3), None)],
                mat_index=mat_index,
                rgba=np.asarray(rgba, np.float32).copy(),
                body=0,
                source=0,
                pose_source=InstancePoseSource.WORLD,
                node_id=node_id,
                object_id=object_id,
            )

        src.meshes = {k: v for k, v in meshes.items() if v is not None}
        src.dynamic_meshes = frozenset(spec.key for spec in self._deformables)
        src.geom_mesh = mesh_keys
        src.geom_material = mats
        n = len(mesh_keys)
        src.geom_size = np.stack(sizes) if n else np.zeros((0, 3), np.float32)
        src.geom_rgba = np.stack(rgbas) if n else np.zeros((0, 4), np.float32)
        src.geom_object_id = np.array(object_ids, np.uint32)
        src.geom_body = np.array(bodies, np.int32)
        src.geom_source = np.array(sources, np.int32)
        src.geom_pose_source = np.array(pose_sources, np.uint8)
        src.geom_node = np.array(node_ids, np.int32)
        src.geom_local = np.stack(locals_) if n else np.zeros((0, 4, 4), np.float32)
        src.geom_infinite_plane = np.array(infinite, bool)
        src.lights = self._build_lights()
        src.scene_extent = float(m.stat.extent)

        src.shadow_clip = float(m.vis.map.shadowclip) or 1.0
        src.scene_center = np.asarray(m.stat.center, np.float32)

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

        if skipped:
            names = ", ".join(sorted(str(mujoco.mjtGeom(t)) for t in skipped))

            note = f"Skipped unsupported geom types: {names}"
            if note not in self._notes:
                self._notes.append(note)
            self.caps = replace(self.caps, notes=tuple(self._notes))
        return src

    def _build_diagnostic_source(self) -> DiagnosticSource:
        m = self._m
        joint_kinds = np.empty(m.njnt, np.uint8)
        joint_kind_map = {
            int(mujoco.mjtJoint.mjJNT_FREE): JointVisualKind.FREE,
            int(mujoco.mjtJoint.mjJNT_BALL): JointVisualKind.BALL,
            int(mujoco.mjtJoint.mjJNT_SLIDE): JointVisualKind.SLIDE,
            int(mujoco.mjtJoint.mjJNT_HINGE): JointVisualKind.HINGE,
        }
        for source_kind, visual_kind in joint_kind_map.items():
            joint_kinds[np.asarray(m.jnt_type) == source_kind] = int(visual_kind)

        meansize = float(m.stat.meansize)
        com_bodies = np.flatnonzero(np.asarray(m.body_parentid[1:]) == 0).astype(np.int32) + 1
        inertia_bodies = np.flatnonzero(np.asarray(m.body_dofnum) > 0).astype(np.int32)
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

        visual_kinds: list[int] = []
        visual_actuators: list[int] = []
        visual_sizes: list[np.ndarray] = []
        pose_kinds: list[int] = []
        pose_indices: list[int] = []
        primitive_kinds = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE): ActuatorVisualKind.SPHERE,
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ActuatorVisualKind.ELLIPSOID,
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): ActuatorVisualKind.CAPSULE,
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): ActuatorVisualKind.CYLINDER,
            int(mujoco.mjtGeom.mjGEOM_BOX): ActuatorVisualKind.BOX,
        }

        def append_primitive(actuator: int, geom_type: int, size, pose_kind: int, source: int):
            kind = primitive_kinds.get(int(geom_type))
            if kind is None:
                return
            raw = np.asarray(size, np.float32)
            if kind is ActuatorVisualKind.SPHERE:
                normalized = np.full(3, raw[0], np.float32)
            elif kind in (ActuatorVisualKind.CAPSULE, ActuatorVisualKind.CYLINDER):
                normalized = np.array((raw[0], raw[0], raw[1]), np.float32)
            else:
                normalized = raw[:3].copy()
            visual_kinds.append(int(kind))
            visual_actuators.append(actuator)
            visual_sizes.append(1.05 * normalized)
            pose_kinds.append(pose_kind)
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
                    kind = ActuatorVisualKind.SLIDE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    kind = ActuatorVisualKind.HINGE
                elif joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
                    kind = ActuatorVisualKind.BALL
                else:
                    kind = ActuatorVisualKind.FREE
                if kind in (ActuatorVisualKind.SLIDE, ActuatorVisualKind.HINGE):
                    size = (
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorwidth),
                        meansize * float(m.vis.scale.actuatorlength),
                    )
                    pose_kind = _ACTUATOR_POSE_JOINT_AXIS
                else:
                    radius = meansize * float(m.vis.scale.jointlength) * 0.33
                    size = (radius, radius, radius)
                    pose_kind = _ACTUATOR_POSE_JOINT_BODY
                visual_kinds.append(int(kind))
                visual_actuators.append(actuator)
                visual_sizes.append(np.asarray(size, np.float32))
                pose_kinds.append(pose_kind)
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

        count = len(visual_kinds)
        self._actuator_visual_pose_kinds = np.asarray(pose_kinds, np.uint8)
        self._actuator_visual_pose_indices = np.asarray(pose_indices, np.int32)
        self._diagnostic_frame.actuator_xpos = np.zeros((count, 3), np.float32)
        self._diagnostic_frame.actuator_xmat = np.zeros((count, 3, 3), np.float32)

        return DiagnosticSource(
            joint_kinds=joint_kinds,
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
            actuator_visual_kinds=np.asarray(visual_kinds, np.uint8),
            actuator_visual_actuators=np.asarray(visual_actuators, np.int32),
            actuator_visual_sizes=(
                np.stack(visual_sizes) if count else np.zeros((0, 3), np.float32)
            ),
        )

    def _fill_actuator_visual_poses(self, diagnostics: DiagnosticFrame) -> None:
        d, m = self._d, self._m
        for record, (pose_kind, source) in enumerate(
            zip(self._actuator_visual_pose_kinds, self._actuator_visual_pose_indices, strict=True)
        ):
            source = int(source)
            if pose_kind == _ACTUATOR_POSE_JOINT_AXIS:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = self._axis_rotation(d.xaxis[source])
            elif pose_kind == _ACTUATOR_POSE_JOINT_BODY:
                diagnostics.actuator_xpos[record] = d.xanchor[source]
                diagnostics.actuator_xmat[record] = d.xmat[int(m.jnt_bodyid[source])].reshape(3, 3)
            elif pose_kind == _ACTUATOR_POSE_SITE:
                diagnostics.actuator_xpos[record] = d.site_xpos[source]
                diagnostics.actuator_xmat[record] = d.site_xmat[source].reshape(3, 3)
            else:
                diagnostics.actuator_xpos[record] = d.geom_xpos[source]
                diagnostics.actuator_xmat[record] = d.geom_xmat[source].reshape(3, 3)

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
                kind = TextureKind.CUBE
            elif ttype == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                kind = TextureKind.SKYBOX
            else:
                kind = TextureKind.TWO_D
            if kind is TextureKind.TWO_D:
                pixels = raw.reshape(h, w, c)
            else:
                pixels = raw.reshape(6, h // 6, w, c)
            out[name] = TextureData(name=name, kind=kind, pixels=pixels.copy(), srgb=True)
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

        #   mjLIGHT_DIRECTIONAL=1、mjLIGHT_POINT=2、mjLIGHT_SPOT=0、mjLIGHT_IMAGE=3。

        if ltype == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
            kind = LightKind.DIRECTIONAL
        elif ltype == mujoco.mjtLightType.mjLIGHT_POINT:
            kind = LightKind.POINT
        elif ltype == mujoco.mjtLightType.mjLIGHT_SPOT:
            kind = LightKind.SPOT
        else:
            kind = LightKind.DIRECTIONAL
        return Light(
            kind=kind,
            position=np.asarray(pos, np.float32).copy(),
            direction=np.asarray(direction, np.float32).copy(),
            diffuse=np.asarray(m.light_diffuse[i], np.float32).copy(),
            specular=np.asarray(m.light_specular[i], np.float32).copy(),
            ambient=np.asarray(m.light_ambient[i], np.float32).copy(),
            attenuation=np.asarray(m.light_attenuation[i], np.float32).copy(),
            range=float(m.light_range[i]),
            cutoff=float(m.light_cutoff[i]),
            exponent=float(m.light_exponent[i]),
            cast_shadow=bool(m.light_castshadow[i]),
            active=bool(m.light_active[i]),
        )

    def _build_lights(self) -> LightSet:
        return self._light_set(
            tuple(
                self._light(i, self._m.light_pos[i], self._m.light_dir[i])
                for i in range(self._m.nlight)
            )
        )

    def _dynamic_lights(self) -> LightSet:
        d = self._d
        return self._light_set(
            tuple(self._light(i, d.light_xpos[i], d.light_xdir[i]) for i in range(self._m.nlight))
        )

    def _light_set(self, lights: tuple[Light, ...]) -> LightSet:
        m = self._m
        extent = float(m.stat.extent) or 1.0
        return LightSet(
            lights=lights,
            headlight=self._headlight(),
            ambient=self._global_ambient(),
            fog_color=np.asarray(m.vis.rgba.fog[:3], np.float32).copy(),
            fog_start=float(m.vis.map.fogstart) * extent,
            fog_end=float(m.vis.map.fogend) * extent,
            haze_color=np.asarray(m.vis.rgba.haze[:3], np.float32).copy(),
            haze_density=float(m.vis.map.haze) / extent,
        )

    def _headlight(self) -> Light | None:

        hl = self._m.vis.headlight
        if not bool(hl.active):
            return None
        return Light(
            kind=DEFAULT_HEADLIGHT.kind,
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
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}

        def add(name: str, kind: NodeKind, parent: int, body: int, **kw) -> int:
            node_id = len(nodes)
            nodes.append(
                SceneNode(
                    node_id=node_id, name=name, kind=kind, parent=parent, body_index=body, **kw
                )
            )
            if parent >= 0:
                nodes[parent].children.append(node_id)
            self._node_body[node_id] = body
            return node_id

        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            if b == 0:
                body_node[b] = add(name or "world", NodeKind.WORLD, -1, 0, object_id=0)
                continue
            parent = int(m.body_parentid[b])
            has_child = bool(np.any(m.body_parentid[b + 1 :] == b))

            kind = NodeKind.ROBOT if parent == 0 and has_child else NodeKind.LINK
            body_node[b] = add(
                name,
                kind,
                body_node[parent],
                b,
                object_id=b,
                posable=self._is_free_body(b),
            )

        for b in range(m.nbody):
            parent = body_node[b]
            adr, num = int(m.body_geomadr[b]), int(m.body_geomnum[b])
            for gi in range(adr, adr + num):
                if not self._visual_groups["geom"][int(m.geom_group[gi])]:
                    continue
                gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or f"geom{gi}"
                self._geom_nodes[gi] = add(gname, NodeKind.GEOM, parent, b)
            ja, jn = int(m.body_jntadr[b]), int(m.body_jntnum[b])
            for ji in range(ja, ja + jn):
                if not self._visual_groups["joint"][int(m.jnt_group[ji])]:
                    continue
                jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}"
                add(jname, NodeKind.JOINT, parent, b)

        for li in range(m.nlight):
            b = int(m.light_bodyid[li])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_LIGHT, li) or f"light{li}"
            add(
                name,
                NodeKind.LIGHT,
                body_node[b],
                b,
                object_id=m.nbody + m.nflex + m.nskin + li,
                visible=bool(m.light_active[li]),
                light_index=li,
            )
        for ci in range(m.ncam):
            b = int(m.cam_bodyid[ci])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, ci) or f"camera{ci}"
            add(name, NodeKind.CAMERA, body_node[b], b)
        for si in range(m.nsite):
            if not self._visual_groups["site"][int(m.site_group[si])]:
                continue
            b = int(m.site_bodyid[si])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, si) or f"site{si}"
            self._site_nodes[si] = add(name, NodeKind.SITE, body_node[b], b)
        for fi in range(m.nflex):
            if not self._visual_groups["flex"][int(m.flex_group[fi])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, fi) or f"flex{fi}"
            self._flex_nodes[fi] = add(name, NodeKind.FLEX, body_node[0], 0, object_id=m.nbody + fi)
        for si in range(m.nskin):
            if not self._visual_groups["skin"][int(m.skin_group[si])]:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SKIN, si) or f"skin{si}"
            self._skin_nodes[si] = add(
                name,
                NodeKind.SKIN,
                body_node[0],
                0,
                object_id=m.nbody + m.nflex + si,
            )
        return nodes

    def _is_free_body(self, body: int) -> bool:

        m = self._m
        adr, num = int(m.body_jntadr[body]), int(m.body_jntnum[body])
        return num == 1 and int(m.jnt_type[adr]) == mujoco.mjtJoint.mjJNT_FREE

    def joints(self) -> list[JointInfo]:
        m = self._m
        kinds = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        dofs = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}
        out = []
        for ji in range(m.njnt):
            kind = kinds.get(int(m.jnt_type[ji]), "hinge")
            out.append(
                JointInfo(
                    joint_id=ji,
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}",
                    kind=kind,
                    limited=bool(m.jnt_limited[ji]),
                    range=(float(m.jnt_range[ji][0]), float(m.jnt_range[ji][1])),
                    qpos_adr=int(m.jnt_qposadr[ji]),
                    qvel_adr=int(m.jnt_dofadr[ji]),
                    dof=dofs[kind],
                    body=int(m.jnt_bodyid[ji]),
                )
            )
        return out

    def actuators(self) -> list[ActuatorInfo]:
        m = self._m
        out = []
        for ai in range(m.nactuator):
            joint = -1
            if int(m.actuator_trntype[ai]) in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            ):
                joint = int(m.actuator_trnid[ai][0])
            out.append(
                ActuatorInfo(
                    actuator_id=ai,
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ai) or f"act{ai}",
                    ctrl_range=(
                        float(m.actuator_ctrlrange[ai][0]),
                        float(m.actuator_ctrlrange[ai][1]),
                    ),
                    ctrl_limited=bool(m.actuator_ctrllimited[ai]),
                    ctrl_address=int(m.actuator_ctrladr[ai]),
                    ctrl_count=int(m.actuator_ctrlnum[ai]),
                    gain=float(m.actuator_gainprm[ai][0]),
                    joint=joint,
                )
            )
        return out

    def cameras(self) -> list[CameraInfo]:
        m = self._m
        return [
            CameraInfo(
                camera_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera{i}",
            )
            for i in range(m.ncam)
        ]

    def keyframes(self) -> list[KeyframeInfo]:
        m = self._m
        return [
            KeyframeInfo(
                keyframe_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) or f"Key {i:03d}",
                time=float(m.key_time[i]),
            )
            for i in range(m.nkey)
        ]

    def sensors(self) -> list[SensorInfo]:
        m = self._m
        return [
            SensorInfo(
                sensor_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor{i}",
                kind=str(mujoco.mjtSensor(int(m.sensor_type[i]))).split(".")[-1],
                data_adr=int(m.sensor_adr[i]),
                dim=int(m.sensor_dim[i]),
            )
            for i in range(m.nsensor)
        ]

    def load_keyframe(self, keyframe_id: int) -> bool:
        i = int(keyframe_id)
        if not 0 <= i < self._m.nkey:
            return False
        mujoco.mj_resetDataKeyframe(self._m, self._d, i)
        mujoco.mj_forward(self._m, self._d)
        self._perturb_body = -1
        return True

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Resolve a model camera from MuJoCo's current forward-kinematics result."""
        i = int(camera_id)
        m, d = self._m, self._d
        if not 0 <= i < m.ncam:
            return None
        rot = np.asarray(d.cam_xmat[i], np.float32).reshape(3, 3)
        eye = np.asarray(d.cam_xpos[i], np.float32).copy()
        distance = max(float(m.stat.extent), 1e-3)
        projection = getattr(m, "cam_projection", None)
        orthographic = bool(
            projection is not None
            and int(projection[i]) == int(mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        )
        fovy = float(m.cam_fovy[i])
        intrinsics = np.asarray(m.cam_intrinsic[i], np.float32)
        sensor_size = np.asarray(m.cam_sensorsize[i], np.float32)
        return CameraView(
            eye=eye,
            target=(eye - rot[:, 2] * distance).astype(np.float32),
            up=rot[:, 1].copy(),
            fov_y=float(np.deg2rad(fovy if not orthographic else 45.0)),
            near=max(float(m.vis.map.znear) * distance, 1e-4),
            far=max(float(m.vis.map.zfar) * distance, distance),
            orthographic=orthographic,
            ortho_height=fovy if orthographic else 2.0 * distance * np.tan(np.deg2rad(fovy) * 0.5),
            focal_length=intrinsics[:2].copy(),
            sensor_size=sensor_size.copy(),
            principal_offset=intrinsics[2:4].copy(),
        )

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        return tuple(
            VisualGroupInfo(name, tuple(bool(x) for x in self._visual_groups[name]))
            for name in VISUAL_GROUP_CATEGORIES
        )

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        groups = self._visual_groups.get(str(category))
        i = int(group)
        if groups is None or not 0 <= i < len(groups):
            return False
        value = bool(visible)
        if bool(groups[i]) == value:
            return True
        groups[i] = value
        if category == "geom":
            self._ray_geomgroup[i] = int(value)
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def _group_visibility(self, model_groups, category: str) -> np.ndarray:
        groups = self._visual_groups[category]
        return groups[np.asarray(model_groups, np.intp)].copy()

    def set_qpos(self, index: int, value: float) -> bool:

        if not 0 <= int(index) < self._m.nq:
            return False
        self._d.qpos[int(index)] = float(value)
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_ctrl(self, index: int, value: float) -> bool:
        m, i = self._m, int(index)
        if not 0 <= i < m.nu:
            return False
        v = float(value)
        actuator = int(self._ctrl_actuator[i])
        if actuator >= 0 and bool(m.actuator_ctrllimited[actuator]):
            lo, hi = m.actuator_ctrlrange[actuator]
            v = float(np.clip(v, lo, hi))
        self._d.ctrl[i] = v
        return True

    def set_light(self, light_id: int, light: Light) -> bool:
        i = int(light_id)
        if not 0 <= i < self._m.nlight:
            return False
        kinds = {
            LightKind.DIRECTIONAL: mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            LightKind.POINT: mujoco.mjtLightType.mjLIGHT_POINT,
            LightKind.SPOT: mujoco.mjtLightType.mjLIGHT_SPOT,
            # AREA is a Forge render extension.  MuJoCo keeps its local pose as
            # a point light; Session restores the authored AREA kind afterwards.
            LightKind.AREA: mujoco.mjtLightType.mjLIGHT_POINT,
        }
        if light.kind not in kinds:
            return False
        m = self._m
        m.light_type[i] = int(kinds[light.kind])
        m.light_pos[i] = light.position
        direction = np.asarray(light.direction, np.float64)
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            m.light_dir[i] = direction / length
        m.light_diffuse[i] = light.diffuse
        m.light_specular[i] = light.specular
        m.light_ambient[i] = light.ambient
        m.light_attenuation[i] = light.attenuation
        m.light_range[i] = light.range
        m.light_cutoff[i] = light.cutoff
        m.light_exponent[i] = light.exponent
        m.light_castshadow[i] = light.cast_shadow
        m.light_active[i] = light.active
        mujoco.mj_forward(m, self._d)
        self._lights_edited = True
        if self._source is not None:
            self._source.lights = self._build_lights()
        return True

    def set_pose(self, node_id: int, position, rotation) -> bool:
        body = self._node_body.get(int(node_id), -1)
        if body < 0 or not self._is_free_body(body):
            return False
        adr = int(self._m.jnt_qposadr[int(self._m.body_jntadr[body])])
        self._d.qpos[adr : adr + 3] = np.asarray(position, np.float64).reshape(3)
        self._d.qpos[adr + 3 : adr + 7] = math3d.mat3_to_quat(rotation)

        dof = int(self._m.jnt_dofadr[int(self._m.body_jntadr[body])])
        self._d.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self._m, self._d)
        return True

    def apply_perturb(
        self, node_id: int, target_position: np.ndarray, target_rotation: np.ndarray, mode: str
    ) -> bool:

        body = self._node_body.get(int(node_id), -1)
        if body <= 0:
            return False
        if int(self._m.body_weldid[body]) == 0:
            return False

        pert = self._perturb
        if self._perturb_body != body:
            point = np.asarray(self._d.xpos[body], np.float64)
            np.sqrt(self._d.qLDiagInv, out=self._perturb_sqrt_inv_d)
            mujoco.mj_jac(self._m, self._d, self._perturb_jac, None, point, body)
            mujoco.mj_solveM2(
                self._m,
                self._d,
                self._perturb_jac_m2,
                self._perturb_jac,
                self._perturb_sqrt_inv_d,
            )
            invmass = float(np.sum(self._perturb_jac_m2 * self._perturb_jac_m2))
            pert.localmass = 3.0 / max(invmass, 1e-15)
            pert.select = body
            pert.localpos[:] = 0.0
            self._perturb_body = body

        pert.active2 = 0
        if mode == "translate":
            pert.active = int(mujoco.mjtPertBit.mjPERT_TRANSLATE)
            pert.refselpos[:] = np.asarray(target_position, np.float64).reshape(3)
        elif mode == "rotate":
            pert.active = int(mujoco.mjtPertBit.mjPERT_ROTATE)
            body_quat = np.asarray(math3d.mat3_to_quat(target_rotation), np.float64)
            mujoco.mju_mulQuat(self._perturb_quat, body_quat, self._m.body_iquat[body])
            pert.refquat[:] = self._perturb_quat
        else:
            return False
        mujoco.mjv_applyPerturbForce(self._m, self._d, pert)
        return True

    def clear_perturb(self) -> None:
        self._d.xfrc_applied[:] = 0.0
        self._perturb.active = 0
        self._perturb.active2 = 0
        self._perturb_body = -1

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:

        self._ray_pnt[:] = np.asarray(origin, np.float64).reshape(3)
        v = np.asarray(direction, np.float64).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return 0, float("inf")
        self._ray_vec[:] = v / n
        self._ray_geomid[0] = -1
        dist = mujoco.mj_ray(
            self._m,
            self._d,
            self._ray_pnt,
            self._ray_vec,
            self._ray_geomgroup,
            True,
            -1,
            self._ray_geomid,
        )
        gid = int(self._ray_geomid[0])
        if dist < 0.0 or gid < 0:
            return 0, float("inf")
        body = int(self._m.geom_bodyid[gid])
        if body == 0:
            return 0, float("inf")
        return body, float(dist)

    def camera_hint(self) -> CameraView | None:

        m = self._m
        extent = float(m.stat.extent) or 1.0
        center = np.asarray(m.stat.center, np.float32)
        az = np.deg2rad(float(m.vis.global_.azimuth))
        el = np.deg2rad(float(m.vis.global_.elevation))
        forward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32
        )
        distance = 1.5 * extent
        return CameraView(
            eye=(center - forward * distance).astype(np.float32),
            target=center.copy(),
            up=np.array([0.0, 0.0, 1.0], np.float32),
            fov_y=float(np.deg2rad(float(m.vis.global_.fovy))),
            near=float(m.vis.map.znear) * extent,
            far=float(m.vis.map.zfar) * extent,
        )

    def release(self) -> None:
        self._m = None
        self._d = None
        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_body_xpos = None
        self._mj_body_xmat3 = None

    @property
    def model(self):

        return self._m

    @property
    def data(self):
        return self._d

    @property
    def fast_pose(self) -> bool:

        return self._fast_pose
