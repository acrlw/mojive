"""Capability-specific model property access and write-back."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from ... import math3d
from ...types import (
    CameraView,
    InstancePoseSource,
    Light,
    LightType,
    Material,
    MeshShape,
    TextureType,
)
from ..base import (
    CAMERA_OBJECT_BASE,
    ActuatorInfo,
    BodyProperties,
    CameraInfo,
    EqualityConstraintInfo,
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    NodeType,
    SensorInfo,
    SiteProperties,
    VisualGroupInfo,
)
from .constants import (
    _MOJIVE_AREA_LIGHTS_TEXT,
    _TEXROLE_RGB,
    _TEXROLE_RGBA,
    VISUAL_GROUP_CATEGORIES,
)
from .engine import mujoco
from .schema import _model_asset_element
from .spec import (
    _component_xml,
    _editable_spec_xml,
    _format_mjcf_values,
    _relative_pose,
    _serialize_component_xml,
    _set_text_names,
    _spec_text_names,
)


class _ModelProperties:
    """Capability-specific model property access and write-back.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def joints(self) -> list[JointInfo]:
        m = self._m
        joint_types = {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }
        dofs = {"free": 6, "ball": 3, "slide": 1, "hinge": 1}
        out = []
        for ji in range(m.njnt):
            joint_type = joint_types.get(int(m.jnt_type[ji]), "hinge")
            out.append(
                JointInfo(
                    joint_id=ji,
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, ji) or f"joint{ji}",
                    type=joint_type,
                    limited=bool(m.jnt_limited[ji]),
                    range=(float(m.jnt_range[ji][0]), float(m.jnt_range[ji][1])),
                    qpos_adr=int(m.jnt_qposadr[ji]),
                    qvel_adr=int(m.jnt_dofadr[ji]),
                    dof=dofs[joint_type],
                    body=int(m.jnt_bodyid[ji]),
                    axis=tuple(float(value) for value in m.jnt_axis[ji]),
                    damping=float(m.dof_damping[int(m.jnt_dofadr[ji])]),
                    stiffness=float(m.jnt_stiffness[ji]),
                )
            )
        return out

    def set_joint_properties(
        self,
        joint_id: int,
        axis: np.ndarray,
        limited: bool,
        value_range: tuple[float, float],
        damping: float,
        stiffness: float,
    ) -> bool:
        joint = int(joint_id)
        if not 0 <= joint < self._m.njnt:
            return False
        joint_type = int(self._m.jnt_type[joint])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            return False
        values = np.asarray(value_range, np.float64).reshape(2)
        axis = np.asarray(axis, np.float64).reshape(3)
        damping = float(damping)
        stiffness = float(stiffness)
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(axis))
            or not np.isfinite((damping, stiffness)).all()
            or damping < 0.0
            or stiffness < 0.0
        ):
            return False
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            if bool(limited) and values[1] <= 0.0:
                return False
            values[0] = 0.0
        elif bool(limited) and values[1] <= values[0]:
            return False
        if joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            norm = float(np.linalg.norm(axis))
            if norm <= 1e-12:
                return False
            axis /= norm

        node = next((item for item in self.nodes() if item.joint_index == joint), None)
        identity = self._node_element.get(node.node_id) if node is not None else None
        if identity is None:
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        spec = self._spec_for_model(model_id)
        if element is None or spec is None:
            return False

        if joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            element.axis = axis
            self._m.jnt_axis[joint] = axis
        element.limited = bool(limited)
        authored_range = values.copy()
        if bool(spec.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            authored_range = np.degrees(authored_range)
        element.range = authored_range
        authored_damping = np.asarray(element.damping, np.float64).copy()
        authored_stiffness = np.asarray(element.stiffness, np.float64).copy()
        authored_damping[0] = damping
        authored_stiffness[0] = stiffness
        element.damping = authored_damping
        element.stiffness = authored_stiffness

        self._m.jnt_limited[joint] = bool(limited)
        self._m.jnt_range[joint] = values
        dof_address = int(self._m.jnt_dofadr[joint])
        dof_count = 3 if joint_type == int(mujoco.mjtJoint.mjJNT_BALL) else 1
        self._m.dof_damping[dof_address : dof_address + dof_count] = damping
        self._m.jnt_stiffness[joint] = stiffness
        mujoco.mj_setConst(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)
        self._mark_model_edited(model_id)
        self._structure_revision += 1
        return True

    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None:
        joint = int(joint_id)
        if not 0 <= joint < self._m.njnt:
            return None
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_JOINT)
        if not name:
            return None
        spec = self._spec_for_model(model_id)
        element = spec.joint(name) if spec is not None else None
        if spec is None or element is None:
            return None
        reference = float(element.ref)
        spring_reference = float(element.springref)
        joint_type = int(self._m.jnt_type[joint])
        if bool(spec.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            reference = float(np.radians(reference))
            spring_reference = float(np.radians(spring_reference))
        force_limit_modes = {
            int(mujoco.mjtLimited.mjLIMITED_AUTO): "auto",
            int(mujoco.mjtLimited.mjLIMITED_FALSE): "unlimited",
            int(mujoco.mjtLimited.mjLIMITED_TRUE): "limited",
        }
        return JointAdvancedProperties(
            joint_id=joint,
            group=int(element.group),
            armature=float(element.armature),
            friction_loss=float(element.frictionloss),
            reference=reference,
            spring_reference=spring_reference,
            margin=float(element.margin),
            limit_solver_reference=tuple(float(value) for value in element.solref_limit),
            limit_solver_impedance=tuple(float(value) for value in element.solimp_limit),
            friction_solver_reference=tuple(float(value) for value in element.solref_friction),
            friction_solver_impedance=tuple(float(value) for value in element.solimp_friction),
            actuator_force_limit_mode=force_limit_modes[int(element.actfrclimited)],
            actuator_force_range=tuple(float(value) for value in self._m.jnt_actfrcrange[joint]),
            actuator_gravity_compensation=bool(element.actgravcomp),
        )

    def set_joint_advanced_properties(self, properties: JointAdvancedProperties) -> bool:
        joint = int(properties.joint_id)
        if not 0 <= joint < self._m.njnt:
            return False
        compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_JOINT)
        if not name:
            return False
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.joint(name)
        if element is None:
            return False
        reference = float(properties.reference)
        spring_reference = float(properties.spring_reference)
        joint_type = int(self._m.jnt_type[joint])
        if bool(working.compiler.degree) and joint_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        }:
            reference = float(np.degrees(reference))
            spring_reference = float(np.degrees(spring_reference))
        element.group = int(properties.group)
        element.armature = float(properties.armature)
        element.frictionloss = float(properties.friction_loss)
        element.ref = reference
        element.springref = spring_reference
        element.margin = float(properties.margin)
        element.solref_limit = properties.limit_solver_reference
        element.solimp_limit = properties.limit_solver_impedance
        element.solref_friction = properties.friction_solver_reference
        element.solimp_friction = properties.friction_solver_impedance
        element.actfrclimited = {
            "auto": mujoco.mjtLimited.mjLIMITED_AUTO,
            "unlimited": mujoco.mjtLimited.mjLIMITED_FALSE,
            "limited": mujoco.mjtLimited.mjLIMITED_TRUE,
        }[properties.actuator_force_limit_mode]
        element.actfrcrange = properties.actuator_force_range
        element.actgravcomp = bool(properties.actuator_gravity_compensation)
        return self._replace_model_spec(model_id, working)

    def site_properties(self, node_id: int) -> SiteProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.SITE
            or not 0 <= node.site_index < self._m.nsite
        ):
            return None
        model_id, _node_type, name = identity
        spec = self._spec_for_model(model_id)
        element = spec.site(name) if spec is not None else None
        if element is None:
            return None
        type_names = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        }
        site_type = type_names.get(int(self._m.site_type[node.site_index]))
        if site_type is None:
            return None
        authored_from_to = np.asarray(element.fromto, np.float64).reshape(6)
        use_from_to = bool(np.all(np.isfinite(authored_from_to)))
        if use_from_to:
            from_to = authored_from_to
        else:
            center = np.asarray(element.pos, np.float64).reshape(3)
            axis = math3d.quat_to_mat3(element.quat)[:, 2]
            half_length = (
                float(self._m.site_size[node.site_index, 1])
                if site_type in ("capsule", "cylinder")
                else 0.5
            )
            from_to = np.concatenate((center - axis * half_length, center + axis * half_length))
        return SiteProperties(
            node_id=int(node_id),
            type=site_type,
            group=int(element.group),
            use_from_to=use_from_to,
            from_to=tuple(float(value) for value in from_to),
        )

    def set_site_properties(self, properties: SiteProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.SITE
            or not 0 <= node.site_index < self._m.nsite
        ):
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        working = source_spec.copy()
        element = working.site(name)
        if element is None:
            return False
        site_types = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
        }
        site_type = site_types.get(str(properties.type))
        if site_type is None:
            return False
        element.type = site_type
        element.group = int(properties.group)
        if properties.use_from_to:
            if str(properties.type) not in ("capsule", "cylinder"):
                return False
            element.fromto = np.asarray(properties.from_to, np.float64).reshape(6)
        else:
            index = int(node.site_index)
            element.fromto = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            element.pos = np.asarray(self._m.site_pos[index], np.float64)
            element.quat = np.asarray(self._m.site_quat[index], np.float64)
            element.size = np.asarray(self._m.site_size[index], np.float64)
        return self._replace_model_spec(model_id, working)

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        node = self._node_for_id(node_id)
        if node is None or node.type is not NodeType.GEOM or node.geom_index < 0:
            return None
        geom = int(node.geom_index)
        return GeometryProperties(
            node_id=int(node_id),
            friction=tuple(float(value) for value in self._m.geom_friction[geom]),
            collision_type_mask=int(self._m.geom_contype[geom]),
            collision_affinity_mask=int(self._m.geom_conaffinity[geom]),
            contact_dimension=int(self._m.geom_condim[geom]),
            contact_priority=int(self._m.geom_priority[geom]),
            margin=float(self._m.geom_margin[geom]),
            gap=float(self._m.geom_gap[geom]),
            solver_mix=float(self._m.geom_solmix[geom]),
            solver_reference=tuple(float(value) for value in self._m.geom_solref[geom]),
            solver_impedance=tuple(float(value) for value in self._m.geom_solimp[geom]),
            adhesion=float(self._m.geom_adhesion[geom]),
            surface_velocity=tuple(float(value) for value in self._m.geom_surfacevel[geom]),
        )

    def set_geometry_properties(self, properties: GeometryProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return False
        try:
            friction = np.asarray(properties.friction, np.float64).reshape(3)
            masks = (
                int(properties.collision_type_mask),
                int(properties.collision_affinity_mask),
            )
            contact_dimension = int(properties.contact_dimension)
            contact_priority = int(properties.contact_priority)
            values = np.asarray(
                (properties.margin, properties.gap, properties.solver_mix), np.float64
            )
            solver_reference = np.asarray(properties.solver_reference, np.float64).reshape(2)
            solver_impedance = np.asarray(properties.solver_impedance, np.float64).reshape(5)
            adhesion = float(properties.adhesion)
            surface_velocity = np.asarray(properties.surface_velocity, np.float64).reshape(6)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not np.all(np.isfinite(friction))
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(solver_reference))
            or not np.all(np.isfinite(solver_impedance))
            or not np.all(np.isfinite(surface_velocity))
            or not np.isfinite(adhesion)
            or np.any(friction < 0.0)
            or any(value < 0 or value > np.iinfo(np.int32).max for value in masks)
            or contact_dimension not in (1, 3, 4, 6)
            or not 0 <= contact_priority <= np.iinfo(np.int32).max
            or values[0] < 0.0
            or values[1] < 0.0
            or not 0.0 <= values[2] <= 1.0
            or adhesion < 0.0
            or not (np.all(solver_reference > 0.0) or np.all(solver_reference <= 0.0))
            or not 0.0 <= solver_impedance[0] <= solver_impedance[1] <= 1.0
            or solver_impedance[2] <= 0.0
            or not 0.0 <= solver_impedance[3] <= 1.0
            or solver_impedance[4] < 1.0
        ):
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False
        element.friction = friction
        element.contype = masks[0]
        element.conaffinity = masks[1]
        element.condim = contact_dimension
        element.priority = contact_priority
        element.margin = float(values[0])
        element.gap = float(values[1])
        element.solmix = float(values[2])
        element.solref = solver_reference
        element.solimp = solver_impedance
        element.adhesion = adhesion
        element.surfacevel = surface_velocity

        geom = int(node.geom_index)
        self._m.geom_friction[geom] = friction
        self._m.geom_contype[geom] = masks[0]
        self._m.geom_conaffinity[geom] = masks[1]
        self._m.geom_condim[geom] = contact_dimension
        self._m.geom_priority[geom] = contact_priority
        self._m.geom_margin[geom] = values[0]
        self._m.geom_gap[geom] = values[1]
        self._m.geom_solmix[geom] = values[2]
        self._m.geom_solref[geom] = solver_reference
        self._m.geom_solimp[geom] = solver_impedance
        self._m.geom_adhesion[geom] = adhesion
        self._m.geom_surfacevel[geom] = surface_velocity
        self._mark_model_edited(model_id)
        mujoco.mj_forward(self._m, self._d)
        self._structure_revision += 1
        return True

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return None
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return None
        authored_mass = float(element.mass)
        mass_mode = "mass" if np.isfinite(authored_mass) else "density"
        return GeometryAdvancedProperties(
            node_id=int(node_id),
            visual_group=int(element.group),
            mass_mode=mass_mode,
            mass=authored_mass if mass_mode == "mass" else 1.0,
            density=float(element.density),
            inertia_mode=(
                "shell"
                if int(element.typeinertia) == int(mujoco.mjtGeomInertia.mjINERTIA_SHELL)
                else "volume"
            ),
            fluid_ellipsoid=bool(float(element.fluid_ellipsoid) > 0.0),
            fluid_coefficients=tuple(float(value) for value in element.fluid_coefs),
        )

    def set_geometry_advanced_properties(self, properties: GeometryAdvancedProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if node is None or identity is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        edited = source_spec.copy()
        element = edited.geom(name)
        if element is None:
            return False
        mass_mode = str(properties.mass_mode).strip().lower()
        inertia_mode = str(properties.inertia_mode).strip().lower()
        if mass_mode not in ("density", "mass") or inertia_mode not in ("volume", "shell"):
            return False
        try:
            fluid_coefficients = np.asarray(properties.fluid_coefficients, np.float64).reshape(5)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not 0 <= int(properties.visual_group) < 6
            or not np.isfinite((properties.mass, properties.density, *fluid_coefficients)).all()
            or (mass_mode == "mass" and properties.mass <= 0.0)
            or (mass_mode == "density" and properties.density <= 0.0)
            or np.any(fluid_coefficients < 0.0)
        ):
            return False
        element.group = int(properties.visual_group)
        element.density = float(properties.density)
        element.mass = float(properties.mass) if mass_mode == "mass" else np.nan
        element.typeinertia = (
            mujoco.mjtGeomInertia.mjINERTIA_SHELL
            if inertia_mode == "shell"
            else mujoco.mjtGeomInertia.mjINERTIA_VOLUME
        )
        element.fluid_ellipsoid = 1.0 if properties.fluid_ellipsoid else 0.0
        element.fluid_coefs = fluid_coefficients
        return self._replace_model_spec(model_id, edited)

    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type is not NodeType.GEOM
            or node.geom_index < 0
        ):
            return None
        model_id, node_type, name = identity
        spec = self._spec_for_model(model_id)
        element = self._element(model_id, node_type.value, name)
        if spec is None or element is None:
            return None
        geom_types = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }
        geom_type = geom_types.get(int(element.type))
        if geom_type is None:
            return None
        resource_name = (
            str(element.meshname)
            if geom_type == "mesh"
            else str(element.hfieldname)
            if geom_type == "hfield"
            else ""
        )
        return GeometryShapeProperties(
            node_id=int(node_id),
            type=geom_type,
            resource_name=resource_name,
            mesh_names=tuple(str(item.name) for item in spec.meshes if item.name),
            height_field_names=tuple(str(item.name) for item in spec.hfields if item.name),
        )

    def set_geometry_shape(self, node_id: int, geom_type: str, resource_name: str) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        types = {
            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
            "hfield": mujoco.mjtGeom.mjGEOM_HFIELD,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "mesh": mujoco.mjtGeom.mjGEOM_MESH,
        }
        kind = str(geom_type).strip().lower()
        resource = str(resource_name).strip()
        if kind not in types:
            return False
        element = self._element(model_id, "geom", name)
        if element is None:
            return False
        if kind == "mesh" and source_spec.mesh(resource) is None:
            return False
        if kind == "hfield" and source_spec.hfield(resource) is None:
            return False
        size = np.asarray(element.size, np.float64).reshape(3).copy()
        defaults = {
            "plane": np.array((1.0, 1.0, 0.1)),
            "sphere": np.array((0.1, 0.1, 0.1)),
            "capsule": np.array((0.1, 0.2, 0.0)),
            "ellipsoid": np.array((0.1, 0.1, 0.1)),
            "cylinder": np.array((0.1, 0.2, 0.0)),
            "box": np.array((0.1, 0.1, 0.1)),
        }
        required = {
            "plane": (0, 1),
            "sphere": (0,),
            "capsule": (0, 1),
            "ellipsoid": (0, 1, 2),
            "cylinder": (0, 1),
            "box": (0, 1, 2),
        }.get(kind, ())
        if required and any(
            not np.isfinite(size[index]) or size[index] <= 0.0 for index in required
        ):
            size = defaults[kind]
        root, _xml = _component_xml(source_spec)
        target = next(
            (item for item in root.iter("geom") if str(item.attrib.get("name", "")) == name),
            None,
        )
        if target is None:
            return False
        target.set("type", kind)
        if kind == "mesh":
            target.set("mesh", resource)
        else:
            target.attrib.pop("mesh", None)
        if kind == "hfield":
            target.set("hfield", resource)
        else:
            target.attrib.pop("hfield", None)
        if required:
            target.set("size", _format_mjcf_values(size))
        else:
            target.attrib.pop("size", None)
        if kind not in ("capsule", "cylinder"):
            target.attrib.pop("fromto", None)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def import_model_geometry_resource(
        self, node_id: int, resource_type: str, path: Path, name: str
    ) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type is not NodeType.GEOM:
            return False
        model_id, _node_type, element_name = identity
        source_spec = self._spec_for_model(model_id)
        source = Path(path).expanduser().resolve()
        kind = str(resource_type).strip().lower()
        value = str(name).strip()
        if (
            source_spec is None
            or not source.is_file()
            or kind not in ("mesh", "hfield")
            or not value
        ):
            return False
        if kind == "mesh" and source_spec.mesh(value) is not None:
            return False
        if kind == "hfield" and source_spec.hfield(value) is not None:
            return False
        working = source_spec.copy()
        if kind == "mesh":
            working.add_mesh(name=value, file=str(source))
        else:
            working.add_hfield(name=value, file=str(source), size=(1.0, 1.0, 1.0, 0.1))
        element = working.geom(element_name)
        if element is None:
            return False
        element.type = (
            mujoco.mjtGeom.mjGEOM_MESH if kind == "mesh" else mujoco.mjtGeom.mjGEOM_HFIELD
        )
        element.meshname = value if kind == "mesh" else ""
        element.hfieldname = value if kind == "hfield" else ""
        element.fromto = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self._replace_model_spec(model_id, working)

    def body_properties(self, node_id: int) -> BodyProperties | None:
        node = self._node_for_id(node_id)
        identity = self._node_element.get(int(node_id))
        if (
            node is None
            or identity is None
            or node.type not in (NodeType.LINK, NodeType.ROBOT)
            or not 0 < node.body_index < self._m.nbody
        ):
            return None
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return None
        body = int(node.body_index)
        authored_full = np.asarray(element.fullinertia, np.float64).reshape(6)
        if not bool(element.explicitinertial):
            inertia_mode = "auto"
        elif np.isfinite(authored_full[0]):
            inertia_mode = "full"
        else:
            inertia_mode = "diagonal"
        diagonal = np.asarray(self._m.body_inertia[body], np.float64)
        full = (
            authored_full
            if inertia_mode == "full"
            else np.array((*diagonal, 0.0, 0.0, 0.0), np.float64)
        )
        sleep_policies = {
            int(mujoco.mjtSleepPolicy.mjSLEEP_NEVER): "never",
            int(mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED): "allowed",
            int(mujoco.mjtSleepPolicy.mjSLEEP_INIT): "init",
        }
        return BodyProperties(
            node_id=int(node_id),
            inertia_mode=inertia_mode,
            mass=float(self._m.body_mass[body]),
            inertial_position=tuple(float(value) for value in self._m.body_ipos[body]),
            inertial_quaternion=tuple(float(value) for value in self._m.body_iquat[body]),
            diagonal_inertia=tuple(float(value) for value in diagonal),
            full_inertia=tuple(float(value) for value in full),
            gravity_compensation=float(element.gravcomp),
            mocap=bool(element.mocap),
            sleep_policy=sleep_policies.get(int(element.sleep), "auto"),
        )

    def set_body_properties(self, properties: BodyProperties) -> bool:
        node = self._node_for_id(properties.node_id)
        identity = self._node_element.get(int(properties.node_id))
        if (
            node is None
            or identity is None
            or node.type not in (NodeType.LINK, NodeType.ROBOT)
            or node.body_index <= 0
        ):
            return False
        model_id, node_type, name = identity
        source_spec = self._spec_for_model(model_id)
        if source_spec is None:
            return False
        edited = source_spec.copy()
        lookup = "body" if node_type in (NodeType.LINK, NodeType.ROBOT) else node_type.value
        element = getattr(edited, lookup)(name)
        if element is None:
            return False
        mode = str(properties.inertia_mode).strip().lower()
        sleep_policy = str(properties.sleep_policy).strip().lower()
        if mode not in ("auto", "diagonal", "full") or sleep_policy not in (
            "auto",
            "never",
            "allowed",
            "init",
        ):
            return False
        element.gravcomp = float(properties.gravity_compensation)
        element.mocap = bool(properties.mocap)
        element.sleep = {
            "auto": mujoco.mjtSleepPolicy.mjSLEEP_AUTO,
            "never": mujoco.mjtSleepPolicy.mjSLEEP_NEVER,
            "allowed": mujoco.mjtSleepPolicy.mjSLEEP_ALLOWED,
            "init": mujoco.mjtSleepPolicy.mjSLEEP_INIT,
        }[sleep_policy]
        if mode == "auto":
            # MjSpec retains explicit inertial values after toggling the flag. Clear every
            # sentinel-backed field so the next compile really derives inertia from geoms.
            element.explicitinertial = False
            element.mass = 0.0
            element.ipos = [np.nan, 0.0, 0.0]
            element.iquat = [1.0, 0.0, 0.0, 0.0]
            element.inertia = [0.0, 0.0, 0.0]
            element.fullinertia = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            quaternion = np.asarray(properties.inertial_quaternion, np.float64).reshape(4)
            quaternion /= np.linalg.norm(quaternion)
            element.explicitinertial = True
            element.mass = float(properties.mass)
            element.ipos = np.asarray(properties.inertial_position, np.float64).reshape(3)
            element.iquat = quaternion
            if mode == "diagonal":
                element.inertia = np.asarray(properties.diagonal_inertia, np.float64).reshape(3)
                element.fullinertia = [np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                element.inertia = [0.0, 0.0, 0.0]
                element.fullinertia = np.asarray(properties.full_inertia, np.float64).reshape(6)
        return self._replace_model_spec(model_id, edited)

    def actuators(self) -> list[ActuatorInfo]:
        m = self._m
        nodes = self.nodes()
        joint_nodes = {n.joint_index: n.node_id for n in nodes if n.type is NodeType.JOINT}
        body_nodes = {
            n.body_index: n.node_id
            for n in nodes
            if n.type in (NodeType.WORLD, NodeType.LINK, NodeType.ROBOT)
        }
        out = []
        for ai in range(m.nactuator):
            joint = -1
            target = -1
            transmission = int(m.actuator_trntype[ai])
            source = int(m.actuator_trnid[ai][0])
            if transmission in (
                int(mujoco.mjtTrn.mjTRN_JOINT),
                int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            ):
                joint = source
                target = joint_nodes.get(joint, body_nodes.get(int(m.jnt_bodyid[joint]), -1))
            elif transmission == int(mujoco.mjtTrn.mjTRN_BODY):
                target = body_nodes.get(source, -1)
            elif transmission in (
                int(mujoco.mjtTrn.mjTRN_SITE),
                int(mujoco.mjtTrn.mjTRN_SLIDERCRANK),
            ):
                target = self._site_nodes.get(
                    source, body_nodes.get(int(m.site_bodyid[source]), -1)
                )
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
                    act_address=int(m.actuator_actadr[ai]),
                    act_count=int(m.actuator_actnum[ai]),
                    gain=float(m.actuator_gainprm[ai][0]),
                    joint=joint,
                    target_node_id=target,
                )
            )
        return out

    def cameras(self) -> list[CameraInfo]:
        m = self._m
        return [
            CameraInfo(
                camera_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera{i}",
                object_id=CAMERA_OBJECT_BASE + i,
            )
            for i in range(m.ncam)
        ]

    def sensors(self) -> list[SensorInfo]:
        m = self._m
        return [
            SensorInfo(
                sensor_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor{i}",
                type=str(mujoco.mjtSensor(int(m.sensor_type[i]))).split(".")[-1],
                data_adr=int(m.sensor_adr[i]),
                dim=int(m.sensor_dim[i]),
            )
            for i in range(m.nsensor)
        ]

    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        m = self._m
        return [
            EqualityConstraintInfo(
                constraint_id=i,
                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"equality{i}",
                type=str(mujoco.mjtEq(int(m.eq_type[i]))).split(".")[-1],
                enabled=bool(self._d.eq_active[i]),
            )
            for i in range(m.neq)
        ]

    def camera_view(self, camera_id: int) -> CameraView | None:
        """Resolve a model camera from MuJoCo's current forward-kinematics result."""
        i = int(camera_id)
        m, d = self._m, self._d
        if not 0 <= i < m.ncam:
            return None
        rot = np.asarray(d.cam_xmat[i], np.float32).reshape(3, 3)
        eye = np.asarray(d.cam_xpos[i], np.float32).copy()
        preview = self._model_transform_preview
        if preview is not None and preview.camera_mask[i]:
            eye = (
                preview.position
                + preview.delta_rotation @ (np.asarray(eye, np.float64) - preview.previous_position)
            ).astype(np.float32)
            rot = (preview.delta_rotation @ np.asarray(rot, np.float64)).astype(np.float32)
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

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        i = int(camera_id)
        if not 0 <= i < self._m.ncam:
            return False
        identity = self._element_identity(NodeType.CAMERA, i)
        if identity is None:
            return False
        model_id, name = identity
        element = self._element(model_id, "camera", name)
        if element is None:
            return False

        body = int(self._m.cam_bodyid[i])
        body_position = np.asarray(self._d.xpos[body], np.float64)
        body_rotation = np.asarray(self._d.xmat[body], np.float64).reshape(3, 3)
        eye = np.asarray(camera.eye, np.float64).reshape(3)
        forward = math3d.normalize(np.asarray(camera.target, np.float64) - eye)
        up = math3d.normalize(np.asarray(camera.up, np.float64))
        right = math3d.normalize(np.cross(forward, up))
        if not np.any(right):
            right = math3d.normalize(np.cross(forward, np.array((0.0, 0.0, 1.0))))
        up = math3d.normalize(np.cross(right, forward))
        world_rotation = np.column_stack((right, up, -forward))
        local_position = body_rotation.T @ (eye - body_position)
        local_rotation = body_rotation.T @ world_rotation
        quaternion = math3d.mat3_to_quat(local_rotation)

        element.pos = local_position
        element.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        element.quat = quaternion
        element.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
        element.proj = (
            mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
            if camera.orthographic
            else mujoco.mjtProjection.mjPROJ_PERSPECTIVE
        )
        element.fovy = (
            float(camera.ortho_height) if camera.orthographic else float(np.degrees(camera.fov_y))
        )
        if camera.uses_intrinsics():
            element.focal_length = np.asarray(camera.focal_length, np.float64)
            element.sensor_size = np.asarray(camera.sensor_size, np.float64)
            element.principal_length = np.asarray(camera.principal_offset, np.float64)
        else:
            zeros = np.zeros(2, np.float64)
            element.focal_length = zeros
            element.focal_pixel = zeros
            element.sensor_size = zeros
            element.principal_length = zeros
            element.principal_pixel = zeros
        self._mark_model_edited(model_id)

        self._m.cam_pos[i] = local_position
        self._m.cam_quat[i] = quaternion
        self._m.cam_fovy[i] = element.fovy
        projection = getattr(self._m, "cam_projection", None)
        if projection is not None:
            projection[i] = int(element.proj)
        self._m.cam_intrinsic[i, :2] = camera.focal_length
        self._m.cam_intrinsic[i, 2:4] = camera.principal_offset
        self._m.cam_sensorsize[i] = camera.sensor_size
        extent = max(float(self._m.stat.extent), 1e-6)
        self._m.vis.map.znear = max(float(camera.near) / extent, 1e-7)
        self._m.vis.map.zfar = max(float(camera.far) / extent, self._m.vis.map.znear)
        mujoco.mj_forward(self._m, self._d)
        return True

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

    def set_light(self, light_index: int, light: Light) -> bool:
        i = int(light_index)
        if not 0 <= i < self._m.nlight:
            return False
        identity = self._element_identity(NodeType.LIGHT, i)
        light_types = {
            LightType.DIRECTIONAL: mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            LightType.POINT: mujoco.mjtLightType.mjLIGHT_POINT,
            LightType.SPOT: mujoco.mjtLightType.mjLIGHT_SPOT,
            LightType.IMAGE: mujoco.mjtLightType.mjLIGHT_IMAGE,
            # AREA is a OpenGL render extension represented by a MuJoCo point
            # light plus custom metadata.
            LightType.AREA: mujoco.mjtLightType.mjLIGHT_POINT,
        }
        if light.type not in light_types:
            return False
        m = self._m
        texture_id = -1
        texture_name = ""
        if light.type is LightType.IMAGE:
            texture_id = (
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, light.texture)
                if light.texture
                else -1
            )
            if texture_id < 0:
                return False
            texture_model, texture_name = self._model_element_name(
                light.texture or "", mujoco.mjtObj.mjOBJ_TEXTURE
            )
            # A stored child MjSpec can only reference assets from that child.
            # The composed model exposes prefixed names, so strip the prefix for
            # write-back and reject references that cannot survive standalone export.
            if identity is not None and texture_model != identity[0]:
                return False
        m.light_type[i] = int(light_types[light.type])
        m.light_pos[i] = light.position
        direction = np.asarray(light.direction, np.float64)
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            m.light_dir[i] = direction / length

        # MuJoCo compiles the reference poses used by track and trackcom lights
        # into light_pos0/light_poscom0/light_dir0. mj_forward() consumes those
        # arrays but does not rebuild them after an interactive model edit, so a
        # paused gizmo write would otherwise snap back to the compiled pose.
        body = int(m.light_bodyid[i])
        body_rotation = np.asarray(self._d.xmat[body], np.float64).reshape(3, 3)
        world_position = np.asarray(self._d.xpos[body], np.float64) + body_rotation @ np.asarray(
            m.light_pos[i], np.float64
        )
        world_direction = body_rotation @ np.asarray(m.light_dir[i], np.float64)
        target = int(m.light_targetbodyid[i])
        reference_body = target if target >= 0 else body
        m.light_pos0[i] = world_position - np.asarray(self._d.xpos[body], np.float64)
        m.light_poscom0[i] = world_position - np.asarray(
            self._d.subtree_com[reference_body], np.float64
        )
        world_direction_length = float(np.linalg.norm(world_direction))
        if world_direction_length > 0.0:
            m.light_dir0[i] = world_direction / world_direction_length
        m.light_diffuse[i] = light.diffuse
        m.light_specular[i] = light.specular
        m.light_ambient[i] = light.ambient
        m.light_attenuation[i] = light.attenuation
        m.light_range[i] = light.range
        m.light_cutoff[i] = light.cutoff
        m.light_exponent[i] = light.exponent
        m.light_bulbradius[i] = light.area_radius
        m.light_texid[i] = texture_id
        m.light_intensity[i] = light.intensity
        m.light_castshadow[i] = light.cast_shadow
        m.light_active[i] = light.active
        if identity is not None:
            model_id, name = identity
            element = self._element(model_id, "light", name)
            if element is not None:
                element.type = light_types[light.type]
                element.pos = light.position
                element.dir = light.direction
                element.diffuse = light.diffuse
                element.specular = light.specular
                element.ambient = light.ambient
                element.attenuation = light.attenuation
                element.range = light.range
                element.cutoff = light.cutoff
                element.exponent = light.exponent
                element.texture = texture_name
                element.intensity = light.intensity
                element.castshadow = light.cast_shadow
                element.active = light.active
                element.bulbradius = light.area_radius
                names = set(
                    _spec_text_names(self._spec_for_model(model_id), _MOJIVE_AREA_LIGHTS_TEXT)
                )
                if light.type is LightType.AREA:
                    names.add(name)
                else:
                    names.discard(name)
                _set_text_names(self._spec_for_model(model_id), _MOJIVE_AREA_LIGHTS_TEXT, names)
                self._mark_model_edited(model_id)
        if i < len(self._area_lights):
            self._area_lights[i] = light.type is LightType.AREA
        mujoco.mj_forward(m, self._d)
        self._lights_edited = True
        if self._source is not None:
            self._source.lights = self._build_lights()
        return True

    def set_skybox(self, texture: str | None) -> bool:
        selected = -1
        if texture is not None:
            selected = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, texture)
            if selected < 0 or int(self._m.tex_type[selected]) not in (
                int(mujoco.mjtTexture.mjTEXTURE_CUBE),
                int(mujoco.mjtTexture.mjTEXTURE_SKYBOX),
            ):
                return False

        updates = []
        for index in range(self._m.ntex):
            current = int(self._m.tex_type[index])
            next_type = (
                mujoco.mjtTexture.mjTEXTURE_SKYBOX
                if index == selected
                else (
                    mujoco.mjtTexture.mjTEXTURE_CUBE
                    if current == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
                    else current
                )
            )
            if int(next_type) == current:
                continue
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, index) or ""
            model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_TEXTURE)
            spec = self._spec_for_model(model_id)
            element = spec.texture(name) if spec is not None and name else None
            if element is None:
                return False
            updates.append((index, next_type, model_id, compiled_name, element))

        changed_models: set[int] = set()
        for index, next_type, model_id, _compiled_name, element in updates:
            self._m.tex_type[index] = int(next_type)
            element.type = next_type
            changed_models.add(model_id)
        for model_id in changed_models:
            self._mark_model_edited(model_id)
        if self._source is not None:
            for _index, next_type, _model_id, compiled_name, _element in updates:
                item = self._source.textures.get(compiled_name)
                if item is not None:
                    item_type = (
                        TextureType.SKYBOX
                        if int(next_type) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
                        else TextureType.CUBE
                    )
                    self._source.textures[compiled_name] = replace(item, type=item_type)
            self._source.skybox = texture
        return True

    def _element_identity(self, node_type: NodeType, slot: int) -> tuple[int, str] | None:
        self.nodes()
        field = "camera_index" if node_type is NodeType.CAMERA else "light_index"
        node = next(
            (
                node
                for node in self._nodes
                if node.type is node_type and getattr(node, field) == slot
            ),
            None,
        )
        identity = self._node_element.get(node.node_id) if node is not None else None
        return (identity[0], identity[2]) if identity is not None else None

    def set_material(self, material_index: int, material: Material) -> bool:
        i = int(material_index)
        if not 0 <= i < self._m.nmat:
            return False
        m = self._m
        texture_id = (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, material.texture)
            if material.texture
            else -1
        )
        if material.texture and texture_id < 0:
            return False
        compiled_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        model_id, name = self._model_element_name(compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL)
        spec = self._spec_for_model(model_id)
        element = spec.material(name) if spec is not None and name else None
        texture_name = ""
        if element is not None and material.texture:
            texture_model, texture_name = self._model_element_name(
                material.texture, mujoco.mjtObj.mjOBJ_TEXTURE
            )
            if texture_model != model_id:
                return False

        current_texture_id = next(
            (
                int(m.mat_texid[i, role])
                for role in (_TEXROLE_RGB, _TEXROLE_RGBA)
                if int(m.mat_texid[i, role]) >= 0
            ),
            -1,
        )
        current_texture = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, current_texture_id) or ""
            if current_texture_id >= 0
            else ""
        )
        texture_changed = current_texture != (material.texture or "")

        m.mat_rgba[i] = material.rgba
        m.mat_emission[i] = material.emission
        m.mat_specular[i] = material.specular
        m.mat_shininess[i] = material.shininess
        m.mat_reflectance[i] = material.reflectance
        m.mat_metallic[i] = material.metallic
        m.mat_roughness[i] = material.roughness
        m.mat_texrepeat[i] = material.tex_repeat
        m.mat_texuniform[i] = material.tex_uniform
        if texture_changed:
            m.mat_texid[i, _TEXROLE_RGB] = texture_id
            m.mat_texid[i, _TEXROLE_RGBA] = -1
        if element is not None:
            element.rgba = material.rgba
            element.emission = material.emission
            element.specular = material.specular
            element.shininess = material.shininess
            element.reflectance = material.reflectance
            element.metallic = material.metallic
            element.roughness = material.roughness
            element.texrepeat = material.tex_repeat
            element.texuniform = material.tex_uniform
            if texture_changed:
                textures = list(element.textures)
                textures.extend([""] * (int(mujoco.mjtTextureRole.mjNTEXROLE) - len(textures)))
                textures[_TEXROLE_RGB] = texture_name
                textures[_TEXROLE_RGBA] = ""
                element.textures = textures
            # MuJoCo 3.11 serializes MjsMaterial.textures correctly but retains
            # the old compiled texture reference in that live MjSpec. Reparse the
            # serialized spec so a later topology rebuild or export sees the edit.
            self._store_model_spec(
                model_id,
                self._spec_from_component_xml(model_id, _editable_spec_xml(spec)),
            )
        return True

    def model_material_indices(self, model_id: int) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(self._m.nmat)
            if (name := mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, index))
            and self._model_element_name(name, mujoco.mjtObj.mjOBJ_MATERIAL)[0] == int(model_id)
        )

    def model_texture_names(self, model_id: int) -> tuple[str, ...]:
        return tuple(
            name
            for index in range(self._m.ntex)
            if (name := mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_TEXTURE, index))
            and self._model_element_name(name, mujoco.mjtObj.mjOBJ_TEXTURE)[0] == int(model_id)
        )

    def create_model_material(self, model_id: int, name: str) -> int:
        model_id = int(model_id)
        value = str(name).strip()
        spec = self._spec_for_model(model_id)
        if spec is None or not value or spec.material(value) is not None:
            return -1
        working = spec.copy()
        working.add_material(name=value)
        if not self._replace_model_spec(model_id, working):
            return -1
        return mujoco.mj_name2id(
            self._m,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            f"{self._model_prefix(model_id)}{value}",
        )

    def add_model_material(self, node_id: int, name: str, copy_from: int = -1) -> int:
        identity = self._node_element.get(int(node_id))
        value = str(name).strip()
        if identity is None or identity[1] not in (NodeType.GEOM, NodeType.SITE) or not value:
            return -1
        model_id, node_type, element_name = identity
        spec = self._spec_for_model(model_id)
        if spec is None or spec.material(value) is not None:
            return -1
        source_name = ""
        source_index = int(copy_from)
        if source_index >= 0:
            if not 0 <= source_index < self._m.nmat:
                return -1
            compiled_name = (
                mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, source_index) or ""
            )
            source_model, source_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if source_model != model_id:
                return -1

        if source_name:
            root, _xml = _component_xml(spec)
            asset, source_element = _model_asset_element(root, "material", source_name)
            if asset is None or source_element is None:
                return -1
            duplicate = deepcopy(source_element)
            duplicate.set("name", value)
            children = tuple(asset)
            asset.insert(children.index(source_element) + 1, duplicate)
            working = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        else:
            working = spec.copy()
            working.add_material(name=value)
        target = getattr(working, node_type.value)(element_name)
        if target is None:
            return -1
        target.material = value
        if not self._replace_model_spec(model_id, working):
            return -1
        compiled_name = value
        if model_id > 0:
            attached = next(
                (item for item in self._attached_models if item.model_id == model_id), None
            )
            if attached is None:
                return -1
            compiled_name = f"{attached.prefix}{value}"
        return mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, compiled_name)

    def import_model_texture(
        self,
        model_id: int,
        path: Path,
        name: str,
        material_index: int = -1,
        texture_type: str = "2d",
    ) -> bool:
        model_id = int(model_id)
        source = Path(path).expanduser().resolve()
        value = str(name).strip()
        spec = self._spec_for_model(model_id)
        if spec is None or not source.is_file() or not value or spec.texture(value) is not None:
            return False
        texture_types = {
            "2d": mujoco.mjtTexture.mjTEXTURE_2D,
            "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
            "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        }
        kind = str(texture_type).strip().lower()
        if kind not in texture_types or (kind != "2d" and int(material_index) >= 0):
            return False
        working = spec.copy()
        working.add_texture(
            name=value,
            type=texture_types[kind],
            file=str(source),
        )
        material = int(material_index)
        if material >= 0:
            if not 0 <= material < self._m.nmat:
                return False
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material) or ""
            material_model, material_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            target = working.material(material_name) if material_model == model_id else None
            if target is None:
                return False
            textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
            textures[_TEXROLE_RGB] = value
            target.textures = textures
        return self._replace_model_spec(model_id, working)

    def set_geometry_material(self, node_id: int, material_index: int) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or identity[1] not in (NodeType.GEOM, NodeType.SITE):
            return False
        model_id, node_type, name = identity
        material = int(material_index)
        material_name = ""
        if material >= 0:
            if not 0 <= material < self._m.nmat:
                return False
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material) or ""
            material_model, material_name = self._model_element_name(
                compiled_name, mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if material_model != model_id:
                return False
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False
        element.material = material_name
        if node_type is NodeType.GEOM:
            self._m.geom_matid[node.geom_index] = material
        else:
            self._m.site_matid[node.site_index] = material
        self._mark_model_edited(model_id)
        self._source = None
        self._structure_revision += 1
        return True

    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool:
        source = self.scene_source()
        instances = np.flatnonzero(source.geom_node == int(node_id))
        if not len(instances):
            return False
        color = np.asarray(rgba, np.float32).reshape(4)
        for instance in instances:
            pose_source = InstancePoseSource(int(source.geom_pose_source[instance]))
            source_id = int(source.geom_source[instance])
            if pose_source is InstancePoseSource.GEOM:
                self._m.geom_rgba[source_id] = color
            elif pose_source is InstancePoseSource.SITE:
                self._m.site_rgba[source_id] = color
            else:
                mesh = source.geom_mesh[instance]
                if mesh.shape in (MeshShape.FLEX, MeshShape.FLEX_FACE):
                    self._m.flex_rgba[mesh.index] = color
                elif mesh.shape is MeshShape.SKIN:
                    self._m.skin_rgba[mesh.index] = color
        source.geom_rgba[instances] = color
        identity = self._node_element.get(int(node_id))
        if identity is not None:
            model_id, node_type, name = identity
            if node_type in (NodeType.GEOM, NodeType.SITE):
                element = self._element(model_id, node_type.value, name)
                if element is not None:
                    element.rgba = color
                    self._mark_model_edited(model_id)
        return True

    def set_geometry_size(self, node_id: int, size: np.ndarray) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None or node.type not in (NodeType.GEOM, NodeType.SITE):
            return False
        values = np.asarray(size, np.float64).reshape(3)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            return False

        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        if element is None:
            return False

        if node_type is NodeType.GEOM:
            index = int(node.geom_index)
            if not 0 <= index < self._m.ngeom:
                return False
            primitive_type = int(self._m.geom_type[index])
            compiled_size = np.asarray(self._m.geom_size[index], np.float64).copy()
        else:
            index = int(node.site_index)
            if not 0 <= index < self._m.nsite:
                return False
            primitive_type = int(self._m.site_type[index])
            compiled_size = np.asarray(self._m.site_size[index], np.float64).copy()

        supported = {
            int(mujoco.mjtGeom.mjGEOM_PLANE),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }
        if primitive_type not in supported or (
            node_type is NodeType.SITE and primitive_type == int(mujoco.mjtGeom.mjGEOM_PLANE)
        ):
            return False

        authored_size = np.asarray(element.size, np.float64).copy()
        if primitive_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            authored_size[:2] = values[:2]
            compiled_size[:2] = values[:2]
        elif primitive_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            authored_size[0] = values[0]
            compiled_size[0] = values[0]
        elif primitive_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }:
            authored_size[0] = values[0]
            compiled_size[:2] = (values[0], values[2])
            fromto = np.asarray(element.fromto, np.float64).copy()
            if np.all(np.isfinite(fromto)):
                center = 0.5 * (fromto[:3] + fromto[3:])
                axis = fromto[3:] - fromto[:3]
                length = float(np.linalg.norm(axis))
                if length <= 1e-12:
                    return False
                axis /= length
                fromto[:3] = center - axis * values[2]
                fromto[3:] = center + axis * values[2]
                element.fromto = fromto
            else:
                authored_size[1] = values[2]
        else:
            authored_size[:3] = values[:3]
            compiled_size[:3] = values[:3]
        element.size = authored_size
        if self._model_edit_batch_depth:
            self._model_edit_batch_rebuild = True
            self._mark_model_edited(model_id)
            return True
        if node_type is NodeType.GEOM:
            self._m.geom_size[index] = compiled_size
        else:
            self._m.site_size[index] = compiled_size
        mujoco.mj_setConst(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)

        self._mark_model_edited(model_id)
        self._source = None
        self._structure_revision += 1
        return True

    def set_pose(self, node_id: int, position, rotation) -> bool:
        model_id = self._node_model.get(int(node_id), -1)
        if model_id >= 0:
            return self.set_scene_model_transform(model_id, position, rotation)
        identity = self._node_element.get(int(node_id))
        if identity is not None:
            _model_id, node_type, _name = identity
            body = self._node_body.get(int(node_id), -1)
            if node_type in (NodeType.LINK, NodeType.ROBOT) and self._is_posable_body(body):
                return self._set_dynamic_body_pose(body, position, rotation)
            node = self._node_for_id(node_id)
            if (
                node is not None
                and node.posable
                and node_type
                in (
                    NodeType.LINK,
                    NodeType.ROBOT,
                    NodeType.GEOM,
                    NodeType.SITE,
                )
            ):
                return self._set_model_element_pose(int(node_id), position, rotation)
        body = self._node_body.get(int(node_id), -1)
        if body < 0 or not self._is_posable_body(body):
            return False
        return self._set_dynamic_body_pose(body, position, rotation)

    def _set_dynamic_body_pose(self, body: int, position, rotation) -> bool:
        mocap = int(self._m.body_mocapid[body])
        if mocap >= 0:
            self._d.mocap_pos[mocap] = np.asarray(position, np.float64).reshape(3)
            self._d.mocap_quat[mocap] = math3d.mat3_to_quat(rotation)
            mujoco.mj_forward(self._m, self._d)
            return True
        adr = int(self._m.jnt_qposadr[int(self._m.body_jntadr[body])])
        self._d.qpos[adr : adr + 3] = np.asarray(position, np.float64).reshape(3)
        self._d.qpos[adr + 3 : adr + 7] = math3d.mat3_to_quat(rotation)

        dof = int(self._m.jnt_dofadr[int(self._m.body_jntadr[body])])
        self._d.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self._m, self._d)
        return True

    def _set_model_element_pose(self, node_id: int, position, rotation) -> bool:
        identity = self._node_element.get(int(node_id))
        node = self._node_for_id(node_id)
        if identity is None or node is None:
            return False
        model_id, node_type, name = identity
        element = self._element(model_id, node_type.value, name)
        spec = self._spec_for_model(model_id)
        if element is None or spec is None:
            return False

        world_position = np.asarray(position, np.float64).reshape(3)
        world_rotation = np.asarray(rotation, np.float64).reshape(3, 3)
        parent_body = (
            int(self._m.body_parentid[node.body_index])
            if node_type in (NodeType.LINK, NodeType.ROBOT)
            else int(node.body_index)
        )
        compiled_parent_position = np.asarray(self._d.xpos[parent_body], np.float64)
        compiled_parent_rotation = np.asarray(self._d.xmat[parent_body], np.float64).reshape(3, 3)
        compiled_position, compiled_rotation = _relative_pose(
            world_position,
            world_rotation,
            compiled_parent_position,
            compiled_parent_rotation,
        )

        spec_parent_position = compiled_parent_position
        spec_parent_rotation = compiled_parent_rotation
        if element.parent is spec.worldbody:
            attached = next(
                (item for item in self._attached_models if item.model_id == model_id), None
            )
            if attached is not None:
                spec_parent_position = attached.position
                spec_parent_rotation = attached.rotation
        local_position, local_rotation = _relative_pose(
            world_position,
            world_rotation,
            spec_parent_position,
            spec_parent_rotation,
        )
        fromto = (
            np.asarray(element.fromto, np.float64).copy()
            if node_type in (NodeType.GEOM, NodeType.SITE)
            else np.empty(0, np.float64)
        )
        if fromto.shape == (6,) and np.all(np.isfinite(fromto)):
            half_length = 0.5 * float(np.linalg.norm(fromto[3:] - fromto[:3]))
            axis = local_rotation[:, 2]
            fromto[:3] = local_position - axis * half_length
            fromto[3:] = local_position + axis * half_length
            element.fromto = fromto
        else:
            element.pos = local_position
            element.quat = math3d.mat3_to_quat(local_rotation)

        if node_type in (NodeType.LINK, NodeType.ROBOT):
            self._m.body_pos[node.body_index] = compiled_position
            self._m.body_quat[node.body_index] = math3d.mat3_to_quat(compiled_rotation)
        elif node_type is NodeType.GEOM:
            self._m.geom_pos[node.geom_index] = compiled_position
            self._m.geom_quat[node.geom_index] = math3d.mat3_to_quat(compiled_rotation)
        elif node_type is NodeType.SITE:
            self._m.site_pos[node.site_index] = compiled_position
            self._m.site_quat[node.site_index] = math3d.mat3_to_quat(compiled_rotation)
        else:
            return False
        self._mark_model_edited(model_id)
        if node_type in (NodeType.GEOM, NodeType.SITE):
            # MuJoCo caches local geom/site poses as model constants;
            # mj_forward alone leaves their world pose unchanged.
            mujoco.mj_setConst(self._m, self._d)
        mujoco.mj_forward(self._m, self._d)
        return True
