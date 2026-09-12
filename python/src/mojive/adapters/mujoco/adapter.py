"""MuJoCo lifecycle, live data ownership and scene-frame publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ... import math3d
from ...types import (
    CameraView,
)
from ..base import (
    GEOMETRY_OBJECT_BASE,
    AdapterCaps,
    ContactObservation,
    DiagnosticFrame,
    FrameNeeds,
    NodeType,
    PhysicsObservation,
    PhysicsState,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSource,
)
from .composition import _ModelComposition
from .constants import _MOJIVE_AREA_LIGHTS_TEXT, DEFAULT_GEOM_GROUPS, VISUAL_GROUP_CATEGORIES
from .deformables import update_deformables
from .diagnostics import _Diagnostics, _RangefinderSpec
from .engine import _IMPORT_ERROR, mujoco
from .export import _MjcfExport
from .keyframes import _Keyframes
from .model_editing import _ModelEditing
from .properties import _ModelProperties
from .source import _SceneConversion
from .spec import _compiled_text_names, _load_editable_spec
from .state import _AttachedModel, _ModelComponentEntry, _ModelTransformPreview


class MuJoCoAdapter(
    _ModelComposition,
    _ModelEditing,
    _MjcfExport,
    _SceneConversion,
    _Diagnostics,
    _ModelProperties,
    _Keyframes,
    SceneAdapterBase,
):
    """Own one model/data pair and expose backend-neutral scene contracts."""

    def __init__(self, path: Path | None = None, *, external_clock: bool = False) -> None:
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(
                f"MuJoCo is not installed: {_IMPORT_ERROR}. Install the [mujoco] optional dependency."
            )
        self._model_edit_batch_depth = 0
        self._model_edit_batch_rebuild = False
        self.caps = AdapterCaps(
            name="mujoco",
            backend_version=mujoco.__version__,
            model_formats=(".xml", ".mjcf", ".urdf"),
            features=(("mujoco.mjcf", 1), ("model.components", 1), ("model.keyframe_edit", 1)),
            simulation=True,
            external_clock=external_clock,
            clock_control=not external_clock,
            asset_loading=True,
            write_pose=True,
            write_qpos=True,
            write_ctrl=True,
            perturb=True,
            raycast=True,
            state_snapshots=True,
            contacts=True,
            model_cameras=True,
            keyframes=True,
            sensors=True,
            equality_constraints=True,
            visual_groups=True,
            reload=True,
            model_composition=True,
            topology_editing=True,
            model_properties=True,
            model_assets=True,
        )
        self._m = None
        self._d = None
        self._path: Path | None = None
        self._root_path: Path | None = None
        self._root_spec = None
        self._root_edited = False
        self._attached_models: list[_AttachedModel] = []
        self._model_transform_preview: _ModelTransformPreview | None = None
        self._next_model_id = 1
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
        self._equality_enabled_buf = np.zeros(0, bool)
        self._sensor_buf = np.zeros(0, np.float32)
        self._contact_buf = np.zeros((0, 7), np.float32)
        self._contact_force = np.zeros(6, np.float64)
        self._contact_view = self._contact_buf
        self._contact_force_buf = np.zeros((0, 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf
        self._contact_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._contact_island_rgba_view = self._contact_island_rgba_buf
        self._island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._body_island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_segments = np.zeros((0, 2, 3), np.float32)
        self._tendon_ids = np.zeros(0, np.int32)
        self._tendon_widths = np.zeros(0, np.float32)
        self._actuator_visual_pose_types = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)
        self._slider_crank_actuators = np.zeros(0, np.int32)
        self._bvh_pose_type = np.zeros(0, np.uint8)
        self._bvh_pose_source = np.zeros(0, np.int32)
        self._bvh_global_index = np.zeros(0, np.int32)
        self._bvh_local_center = np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.zeros((0, 3), np.float32)
        self._bvh_control_body = np.zeros((0, 2), np.int32)
        self._bvh_control_local = np.zeros((0, 2, 3), np.float32)
        self._bvh_source_ready = False
        self._rangefinder_specs: tuple[_RangefinderSpec, ...] = ()

        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_wrap_objects = None
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
        self._node_model: dict[int, int] = {}
        self._node_element: dict[int, tuple[int, NodeType, str]] = {}
        self._model_element_names: dict[tuple[int, str], tuple[int, str]] = {}
        self._geometry_object_ids: dict[tuple[int, str], int] = {}
        self._next_geometry_object_id = GEOMETRY_OBJECT_BASE
        self._component_entries: dict[tuple[int, str], tuple[_ModelComponentEntry, ...]] = {}
        self._next_component_id: dict[tuple[int, str], int] = {}
        self._geom_nodes: dict[int, int] = {}
        self._site_nodes: dict[int, int] = {}
        self._flex_nodes: dict[int, int] = {}
        self._skin_nodes: dict[int, int] = {}
        self._deformables = []
        self._mesh_updates = {}
        self._lights_dynamic = False
        self._lights_edited = False
        self._area_lights = np.zeros(0, bool)

        if path is not None:
            self.load(path)

    def load(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        try:
            spec = _load_editable_spec(path)
            model = spec.compile()
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path}: {exc}") from exc
        self._path = path
        self._root_path = path
        self._root_spec = spec
        self._root_edited = False
        self._attached_models.clear()
        self._reset_next_model_id()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=True)
        self._install(model)

    def new_scene(self) -> None:
        self._path = None
        self._root_path = None
        self._root_spec = mujoco.MjSpec()
        self._root_edited = False
        self._attached_models.clear()
        self._reset_next_model_id()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=True)
        self._install(self._root_spec.compile())

    def load_model(self, model, data=None) -> None:
        """Install an existing MuJoCo model and optional data object."""
        if not isinstance(model, mujoco.MjModel):
            raise TypeError("model must be a mujoco.MjModel")
        if data is not None and not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data is not None and data.model is not model:
            raise ValueError("data was created for a different MuJoCo model")
        self._path = None
        self._root_path = None
        self._root_spec = None
        self._root_edited = False
        self._attached_models.clear()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self.caps = replace(self.caps, model_composition=False)
        self._install(model, data)

    def use_data(self, data) -> None:
        """Bind dynamic state supplied by a programmatic rendering workflow."""
        if not isinstance(data, mujoco.MjData):
            raise TypeError("data must be a mujoco.MjData")
        if data.model is not self._m:
            raise ValueError("data was created for a different MuJoCo model")
        if data is self._d:
            return
        self._d = data
        self._bind_data_views()

    def apply_scene_option(self, option) -> bool:
        """Apply MuJoCo visual-group visibility to the stable scene source."""
        changed = False
        fields = {
            "geom": "geomgroup",
            "site": "sitegroup",
            "joint": "jointgroup",
            "tendon": "tendongroup",
            "flex": "flexgroup",
            "skin": "skingroup",
        }
        for category, field in fields.items():
            visible = np.asarray(getattr(option, field), bool)
            groups = self._visual_groups[category]
            if not np.array_equal(groups, visible):
                groups[:] = visible
                changed = True
        if not changed:
            return False
        self._ray_geomgroup[:] = self._visual_groups["geom"]
        self._source = None
        self._nodes = []
        self._structure_revision += 1
        return True

    def refresh_model_visuals(self) -> bool:
        """Refresh cached scene data after direct MjModel visual edits."""
        source_changed = self._refresh_snapshots(self._visual_state)
        lights_changed = self._refresh_snapshots(self._light_state)
        if source_changed:
            self._source = None
            self._structure_revision += 1
        if lights_changed:
            self._lights_edited = True
        return source_changed or lights_changed

    def _refresh_snapshots(self, snapshots: dict[str, np.ndarray]) -> bool:
        changed = False
        for name, previous in snapshots.items():
            current = np.asarray(getattr(self._m, name))
            if not np.array_equal(current, previous):
                np.copyto(previous, current)
                changed = True
        return changed

    def reload(self) -> None:
        if self._root_path is None and self._root_spec is None:
            raise RuntimeError("No asset has been loaded")
        if self._root_path is not None:
            self._root_spec = _load_editable_spec(self._root_path)
        self._install(self._compile_composed_model())

    def current_pose_modified(self) -> bool:
        if self._m is None or self._d is None:
            return False
        return not np.allclose(self._d.qpos, self._m.qpos0, rtol=1e-6, atol=1e-7)

    def _install(self, model, data=None) -> None:
        if self._model_edit_batch_depth and model is self._m:
            return
        # A compiled model is authoritative. Transient placement indices refer to
        # the previous model layout and must never survive an install.
        self._model_transform_preview = None
        self._m = model
        self._d = data if data is not None else mujoco.MjData(model)
        self._notes = []
        self._rebuild_model_element_names()
        mujoco.mj_forward(self._m, self._d)

        g, b = model.ngeom, model.nbody
        self._rangefinder_specs = self._build_rangefinder_specs(model)
        rangefinder_count = sum(spec.ray_count for spec in self._rangefinder_specs)
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
            rangefinder_starts=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_ends=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_normals=np.zeros((rangefinder_count, 3), np.float32),
            rangefinder_lines=np.zeros(rangefinder_count, bool),
            rangefinder_points=np.zeros(rangefinder_count, bool),
            rangefinder_normal_arrows=np.zeros(rangefinder_count, bool),
            constraint_starts=np.zeros((model.neq, 3), np.float32),
            constraint_ends=np.zeros((model.neq, 3), np.float32),
            constraint_visible=np.zeros(model.neq, bool),
        )
        self._qpos_buf = np.zeros(model.nq, np.float32)
        self._qvel_buf = np.zeros(model.nv, np.float32)
        self._ctrl_buf = np.zeros(model.nu, np.float32)
        self._equality_enabled_buf = np.zeros(model.neq, bool)
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
        self._flex_vertices_buf = np.zeros((model.nflexvert, 3), np.float32)
        self._contact_buf = np.zeros((max(model.ngeom, 64), 7), np.float32)
        self._contact_view = self._contact_buf[:0]
        self._contact_force_buf = np.zeros((len(self._contact_buf), 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf[:0]
        self._contact_island_rgba_buf = np.zeros((len(self._contact_buf), 4), np.float32)
        self._contact_island_rgba_view = self._contact_island_rgba_buf[:0]
        self._island_rgba_buf = np.zeros((0, 4), np.float32)
        self._tendon_island_rgba_buf = np.zeros((model.ntendon, 4), np.float32)
        self._flex_island_rgba_buf = np.zeros((model.nflex, 4), np.float32)
        self._body_island_rgba_buf = np.zeros((model.nbody, 4), np.float32)
        d = self._d
        wrap_capacity = d.wrap_xpos.size // 3
        self._tendon_segments = np.zeros((wrap_capacity, 2, 3), np.float32)
        self._tendon_ids = np.zeros(wrap_capacity, np.int32)
        self._tendon_widths = np.zeros(wrap_capacity, np.float32)
        self._actuator_visual_pose_types = np.zeros(0, np.uint8)
        self._actuator_visual_pose_indices = np.zeros(0, np.int32)
        self._slider_crank_actuators = np.zeros(0, np.int32)
        self._bvh_pose_type = np.zeros(0, np.uint8)
        self._bvh_pose_source = np.zeros(0, np.int32)
        self._bvh_global_index = np.zeros(0, np.int32)
        self._bvh_local_center = np.zeros((0, 3), np.float32)
        self._bvh_local_size = np.zeros((0, 3), np.float32)
        self._bvh_control_body = np.zeros((0, 2), np.int32)
        self._bvh_control_local = np.zeros((0, 2, 3), np.float32)
        self._bvh_source_ready = False
        self._bind_data_views()
        self._fast_pose = self._verify_pose_layout()

        self._frame = SceneFrame()
        self._source = None
        self._nodes = []
        self._node_body = {}
        self._node_model = {}
        self._node_element = {}
        self._geom_nodes = {}
        self._site_nodes = {}
        self._flex_nodes = {}
        self._skin_nodes = {}
        self._deformables = []
        self._mesh_updates = {}
        for groups in self._visual_groups.values():
            groups[:] = [g in DEFAULT_GEOM_GROUPS for g in range(6)]
        self._ray_geomgroup[:] = self._visual_groups["geom"]
        self._lights_dynamic = bool(model.nlight) and bool(
            np.any(model.light_bodyid != 0)
            or np.any(model.light_mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        )
        area_names: set[str] = set()
        for prefix, names in _compiled_text_names(model, _MOJIVE_AREA_LIGHTS_TEXT):
            area_names.update(f"{prefix}{name}" for name in names)
        self._area_lights = np.asarray(
            [
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or "") in area_names
                for i in range(model.nlight)
            ],
            bool,
        )
        visual_fields = (
            "geom_rgba",
            "site_rgba",
            "flex_rgba",
            "skin_rgba",
            "tendon_rgba",
            "tendon_width",
            "mat_rgba",
            "mat_emission",
            "mat_specular",
            "mat_shininess",
            "mat_reflectance",
            "mat_texrepeat",
            "mat_texuniform",
            "mat_texid",
            "tex_data",
        )
        light_fields = (
            "light_type",
            "light_pos",
            "light_dir",
            "light_diffuse",
            "light_specular",
            "light_ambient",
            "light_attenuation",
            "light_range",
            "light_cutoff",
            "light_exponent",
            "light_intensity",
            "light_castshadow",
            "light_active",
        )
        # Compiled model array sizes are fixed until the next installation.
        # Empty fields cannot change and need no per-frame comparison.
        self._visual_state = {
            name: value.copy()
            for name in visual_fields
            if (value := np.asarray(getattr(model, name, ()))).size
        }
        self._light_state = {
            name: value.copy()
            for name in light_fields
            if (value := np.asarray(getattr(model, name, ()))).size
        }
        self._lights_edited = False
        self._perturb = mujoco.MjvPerturb()
        self._perturb_body = -1
        self._perturb_jac = np.zeros((3, model.nv), np.float64)
        self._perturb_jac_m2 = np.zeros((3, model.nv), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(model.nv, np.float64)
        self._structure_revision += 1

    def _bind_data_views(self) -> None:
        m, d = self._m, self._d
        self._mj_geom_xpos = d.geom_xpos
        self._mj_geom_xmat3 = d.geom_xmat.reshape(m.ngeom, 3, 3)
        self._mj_site_xmat3 = d.site_xmat.reshape(m.nsite, 3, 3)
        # MuJoCo exposes wrap_xpos as packed xyz triples addressed by ten_wrapadr/num.
        self._mj_wrap_points = d.wrap_xpos.reshape(-1, 3)
        self._mj_wrap_objects = d.wrap_obj.reshape(-1)
        self._mj_body_xpos = d.xpos
        self._mj_body_xmat3 = d.xmat.reshape(m.nbody, 3, 3)

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
        if self.caps.external_clock:
            raise RuntimeError("Reset belongs to the external physics caller")
        mujoco.mj_resetData(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)

    def set_paused(self, paused: bool) -> bool:
        """Pause ownership lives in Session; local MuJoCo needs no additional state."""
        return not self.caps.external_clock

    def step(self, count: int = 1) -> None:
        if self.caps.external_clock:
            raise RuntimeError("Physics stepping belongs to the external caller")
        mujoco.mj_step(self._m, self._d, nstep=max(1, int(count)))

    def timestep(self) -> float:
        return float(self._m.opt.timestep)

    def create_simulation_driver(self):
        """Create a worker only when this adapter owns the physics clock."""
        if self.caps.external_clock:
            return None
        from .simulation import MuJoCoSimulation

        return MuJoCoSimulation(self)

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        if self.prepare_frame(needs):
            self.scene_source()
        d = self._d
        f = self._frame
        f.time = float(d.time)
        if self.caps.external_clock:
            f.paused = False

        if needs.poses:
            self._fill_poses()
            np.copyto(self._site_xpos_buf, d.site_xpos, casting="unsafe")
            np.copyto(self._site_xmat_buf, self._mj_site_xmat3, casting="unsafe")
            self._apply_model_transform_preview_poses()
            f.geom_xpos = self._geom_xpos_buf
            f.geom_xmat = self._geom_xmat_buf
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

        np.copyto(self._equality_enabled_buf, d.eq_active, casting="unsafe")
        f.equality_enabled = self._equality_enabled_buf

        if needs.contacts:
            f.contacts = self._fill_contacts(needs.islands)
            f.contact_forces = self._contact_force_view
            f.contact_island_rgba = self._contact_island_rgba_view if needs.islands else None
        else:
            f.contacts = None
            f.contact_forces = None
            f.contact_island_rgba = None
        if needs.tendons:
            f.tendon_segments, f.tendon_ids, f.tendon_widths = self._fill_tendons()
        else:
            f.tendon_segments = f.tendon_ids = f.tendon_widths = None
        if needs.deformables:
            update_deformables(self._deformables, d)
            f.mesh_updates = self._mesh_updates
            np.copyto(self._flex_vertices_buf, d.flexvert_xpos, casting="unsafe")
            f.flex_vertices = self._flex_vertices_buf
        else:
            f.mesh_updates = None
            f.flex_vertices = None

        if needs.islands:
            f.island_rgba, f.tendon_island_rgba, f.flex_island_rgba = self._fill_island_colors()
        else:
            f.island_rgba = None
            f.tendon_island_rgba = None
            f.flex_island_rgba = None

        if needs.diagnostics or needs.joint_frames:
            diagnostics = self._diagnostic_frame
            np.copyto(diagnostics.joint_xpos, d.xanchor, casting="unsafe")
            np.copyto(diagnostics.joint_xaxis, d.xaxis, casting="unsafe")
            if needs.diagnostics:
                np.copyto(diagnostics.subtree_com, d.subtree_com, casting="unsafe")
                np.copyto(diagnostics.body_xipos, d.xipos, casting="unsafe")
                np.copyto(
                    diagnostics.body_ximat,
                    d.ximat.reshape(self._m.nbody, 3, 3),
                    casting="unsafe",
                )
                self._fill_actuator_visual_poses(diagnostics)
                self._fill_slider_crank_visuals(diagnostics)
                self._fill_autoconnect_visuals(diagnostics)
                self._fill_rangefinder_visuals(diagnostics)
                self._fill_constraint_visuals(diagnostics)
                if needs.bvh:
                    self._fill_bvh_visuals(diagnostics)
            self._apply_model_transform_preview_diagnostics(
                diagnostics,
                include_full_frame=needs.diagnostics,
            )
            f.diagnostics = diagnostics
            f.cameras = (
                tuple(self.camera_view(i) for i in range(self._m.ncam))
                if needs.diagnostics
                else None
            )
        else:
            f.diagnostics = None
            f.cameras = None

        f.lights = (
            self._dynamic_lights()
            if self._lights_dynamic
            or self._lights_edited
            or self._model_transform_preview is not None
            else None
        )
        return f

    def set_qpos(self, index: int, value: float) -> bool:
        if not 0 <= int(index) < self._m.nq:
            return False
        self._d.qpos[int(index)] = float(value)
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_qpos_batch(self, indices: np.ndarray, values: np.ndarray) -> bool:
        raw_slots = np.asarray(indices).reshape(-1)
        if not np.issubdtype(raw_slots.dtype, np.integer):
            return False
        slots = raw_slots.astype(np.intp, copy=False)
        coordinates = np.asarray(values, np.float64).reshape(-1)
        if (
            not len(slots)
            or len(slots) != len(coordinates)
            or np.any(slots < 0)
            or np.any(slots >= self._m.nq)
            or len(np.unique(slots)) != len(slots)
            or not np.all(np.isfinite(coordinates))
        ):
            return False
        self._d.qpos[slots] = coordinates
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool:
        i = int(constraint_id)
        if not 0 <= i < self._m.neq:
            return False
        self._d.eq_active[i] = bool(enabled)
        mujoco.mj_forward(self._m, self._d)
        return True

    def set_ctrl(self, index: int, value: float) -> bool:
        m, i = self._m, int(index)
        if not 0 <= i < m.nu:
            return False
        v = float(value)
        if not np.isfinite(v):
            return False
        actuator = int(self._ctrl_actuator[i])
        if actuator >= 0 and bool(m.actuator_ctrllimited[actuator]):
            lo, hi = m.actuator_ctrlrange[actuator]
            v = float(np.clip(v, lo, hi))
        self._d.ctrl[i] = v
        return True

    def set_ctrl_vector(self, values: np.ndarray) -> bool:
        """Clamp all flat control coordinates and commit them together."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != self._d.ctrl.shape or not np.all(np.isfinite(values)):
            return False
        if not values.size:
            return True
        indices = self._ctrl_actuator
        limited = self._m.actuator_ctrllimited[indices].astype(bool)
        ranges = self._m.actuator_ctrlrange[indices]
        np.clip(
            values,
            np.where(limited, ranges[:, 0], -np.inf),
            np.where(limited, ranges[:, 1], np.inf),
            out=self._d.ctrl,
        )
        return True

    def capture_observation(self) -> PhysicsObservation:
        """Copy the current solver and sensor outputs without changing simulation state."""
        m, d = self._m, self._d
        contacts = []
        for index in range(d.ncon):
            contact = d.contact[index]
            frame = contact.frame.reshape(3, 3).copy()
            wrench = np.zeros(6, np.float64)
            mujoco.mj_contactForce(m, d, index, wrench)
            geom_ids = tuple(int(value) for value in contact.geom)
            contacts.append(
                ContactObservation(
                    index=index,
                    geom_ids=geom_ids,
                    body_indices=tuple(int(m.geom_bodyid[g]) if g >= 0 else -1 for g in geom_ids),
                    flex_ids=tuple(int(value) for value in contact.flex),
                    position=contact.pos.copy(),
                    frame=frame,
                    distance=float(contact.dist),
                    dimension=int(contact.dim),
                    wrench=wrench,
                    world_wrench=(wrench.reshape(2, 3) @ frame).reshape(6),
                )
            )
        return PhysicsObservation(
            time=float(d.time),
            sensordata=d.sensordata.copy(),
            sensors=tuple(self.sensors()),
            actuator_force=d.actuator_force.copy(),
            contacts=tuple(contacts),
        )

    def capture_state(self) -> PhysicsState:
        data = self._d
        return PhysicsState(
            qpos=np.asarray(data.qpos, np.float64).copy(),
            qvel=np.asarray(data.qvel, np.float64).copy(),
            act=np.asarray(data.act, np.float64).copy(),
            ctrl=np.asarray(data.ctrl, np.float64).copy(),
            time=float(data.time),
            mocap_pos=np.asarray(data.mocap_pos, np.float64).copy(),
            mocap_quat=np.asarray(data.mocap_quat, np.float64).copy(),
        )

    def restore_state(self, state: PhysicsState) -> bool:
        data = self._d
        arrays = (
            (data.qpos, state.qpos),
            (data.qvel, state.qvel),
            (data.act, state.act),
            (data.ctrl, state.ctrl),
            (data.mocap_pos, state.mocap_pos),
            (data.mocap_quat, state.mocap_quat),
        )
        if any(np.shape(dst) != np.shape(src) for dst, src in arrays):
            return False
        for dst, src in arrays:
            np.copyto(dst, src)
        data.time = float(state.time)
        mujoco.mj_forward(self._m, data)
        self._perturb_body = -1
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
        node_id = self._geom_nodes.get(gid, -1)
        node = self._node_for_id(node_id)
        if node is not None and node.object_id:
            return int(node.object_id), float(dist)
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
            # The free editor camera is not constrained to MuJoCo's classic
            # viewport range. Keep distant authored and composed entities visible.
            far=max(float(m.vis.map.zfar), 200.0) * extent,
        )

    def release(self) -> None:
        self._model_transform_preview = None
        self._m = None
        self._d = None
        self._root_spec = None
        self._attached_models.clear()
        self._source = None
        self._nodes.clear()
        self._node_body.clear()
        self._reset_geometry_object_ids()
        self._component_entries.clear()
        self._next_component_id.clear()
        self._node_model.clear()
        self._node_element.clear()
        self._geom_nodes.clear()
        self._site_nodes.clear()
        self._flex_nodes.clear()
        self._skin_nodes.clear()
        self._deformables.clear()
        self._mesh_updates.clear()
        self._mj_geom_xpos = None
        self._mj_geom_xmat3 = None
        self._mj_site_xmat3 = None
        self._mj_wrap_points = None
        self._mj_wrap_objects = None
        self._mj_body_xpos = None
        self._mj_body_xmat3 = None
        self._geom_xpos_buf = np.zeros((0, 3), np.float32)
        self._geom_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._site_xpos_buf = np.zeros((0, 3), np.float32)
        self._site_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._body_xpos_buf = np.zeros((0, 3), np.float32)
        self._body_xmat_buf = np.zeros((0, 3, 3), np.float32)
        self._qpos_buf = np.zeros(0, np.float32)
        self._qvel_buf = np.zeros(0, np.float32)
        self._ctrl_buf = np.zeros(0, np.float32)
        self._sensor_buf = np.zeros(0, np.float32)
        self._contact_buf = np.zeros((0, 7), np.float32)
        self._contact_view = self._contact_buf
        self._contact_force_buf = np.zeros((0, 2, 3), np.float32)
        self._contact_force_view = self._contact_force_buf
        self._tendon_segments = np.zeros((0, 2, 3), np.float32)
        self._tendon_ids = np.zeros(0, np.int32)
        self._tendon_widths = np.zeros(0, np.float32)
        self._flex_vertices_buf = np.zeros((0, 3), np.float32)
        self._perturb_jac = np.zeros((3, 0), np.float64)
        self._perturb_jac_m2 = np.zeros((3, 0), np.float64)
        self._perturb_sqrt_inv_d = np.zeros(0, np.float64)
        self._visual_state = {}
        self._light_state = {}

    @property
    def model(self):
        return self._m

    @property
    def data(self):
        return self._d

    @property
    def fast_pose(self) -> bool:
        return self._fast_pose
