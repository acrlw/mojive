"""Session commands: properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mojive import commands as cmd
from mojive.adapters.base import (
    BodyProperties,
    GeometryAdvancedProperties,
    GeometryProperties,
    JointAdvancedProperties,
    SiteProperties,
)
from mojive.commands import CommandResult

if TYPE_CHECKING:
    from .. import Session


def set_joint_properties(self: Session, c: cmd.SetJointProperties) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing joint properties")
    joint = next((item for item in self._joints if item.joint_id == c.joint_id), None)
    if joint is None:
        return CommandResult.bad(f"Joint {c.joint_id} is unavailable")
    joint_node = next(
        (node for node in self._nodes if node.joint_index == c.joint_id),
        None,
    )
    if joint_node is None or not joint_node.source_editable:
        return CommandResult.bad(f"Joint {joint.name} has no editable source element")
    if joint.type == "free":
        return CommandResult.bad("Free-joint properties are not editable here")
    axis = np.asarray(c.axis, np.float64).reshape(3)
    value_range = np.asarray(c.range, np.float64).reshape(2)
    damping = float(c.damping)
    stiffness = float(c.stiffness)
    if (
        not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(value_range))
        or not np.isfinite((damping, stiffness)).all()
    ):
        return CommandResult.bad("Joint properties must contain finite values")
    if joint.type in ("hinge", "slide") and np.linalg.norm(axis) <= 1e-12:
        return CommandResult.bad("Joint axis must be non-zero")
    if joint.type == "ball":
        if bool(c.limited) and value_range[1] <= 0.0:
            return CommandResult.bad("Ball-joint limit must be positive")
        value_range[0] = 0.0
    elif bool(c.limited) and value_range[1] <= value_range[0]:
        return CommandResult.bad("Joint range upper bound must exceed its lower bound")
    if damping < 0.0 or stiffness < 0.0:
        return CommandResult.bad("Joint damping and stiffness cannot be negative")
    changed = self._adapter.set_joint_properties(
        c.joint_id,
        axis,
        bool(c.limited),
        (float(value_range[0]), float(value_range[1])),
        damping,
        stiffness,
    )
    if not changed:
        return CommandResult.bad(f"Joint {joint.name} properties cannot be edited")
    self._refresh_joint_metadata()
    self._adapter_revision = self._adapter.structure_revision
    self._structure_generation += 1
    return CommandResult.good("")


def set_joint_advanced_properties(
    self: Session, c: cmd.SetJointAdvancedProperties
) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing joint properties")
    joint_node = next(
        (node for node in self._nodes if node.joint_index == c.joint_id),
        None,
    )
    if joint_node is None or not joint_node.source_editable:
        return CommandResult.bad(f"Joint {c.joint_id} has no editable source element")
    if self._adapter.joint_advanced_properties(c.joint_id) is None:
        return CommandResult.bad(f"Joint {c.joint_id} is unavailable")
    try:
        group = int(c.group)
        scalars = np.asarray(
            (c.armature, c.friction_loss, c.reference, c.spring_reference, c.margin),
            np.float64,
        )
        limit_reference = np.asarray(c.limit_solver_reference, np.float64).reshape(2)
        limit_impedance = np.asarray(c.limit_solver_impedance, np.float64).reshape(5)
        friction_reference = np.asarray(c.friction_solver_reference, np.float64).reshape(2)
        friction_impedance = np.asarray(c.friction_solver_impedance, np.float64).reshape(5)
        force_range = np.asarray(c.actuator_force_range, np.float64).reshape(2)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Advanced joint properties have invalid value types")
    values = np.concatenate(
        (
            scalars,
            limit_reference,
            limit_impedance,
            friction_reference,
            friction_impedance,
            force_range,
        )
    )
    if not np.all(np.isfinite(values)):
        return CommandResult.bad("Advanced joint properties must be finite")
    if not 0 <= group < 6:
        return CommandResult.bad("Joint group must be between 0 and 5")
    if scalars[0] < 0.0 or scalars[1] < 0.0 or scalars[4] < 0.0:
        return CommandResult.bad("Armature, friction loss, and margin cannot be negative")
    force_limit_mode = str(c.actuator_force_limit_mode)
    if force_limit_mode not in {"auto", "unlimited", "limited"}:
        return CommandResult.bad("Actuator force limit mode is invalid")
    if force_limit_mode == "limited" and force_range[1] <= force_range[0]:
        return CommandResult.bad("Actuator force upper bound must exceed its lower bound")
    properties = JointAdvancedProperties(
        joint_id=int(c.joint_id),
        group=group,
        armature=float(scalars[0]),
        friction_loss=float(scalars[1]),
        reference=float(scalars[2]),
        spring_reference=float(scalars[3]),
        margin=float(scalars[4]),
        limit_solver_reference=tuple(float(value) for value in limit_reference),
        limit_solver_impedance=tuple(float(value) for value in limit_impedance),
        friction_solver_reference=tuple(float(value) for value in friction_reference),
        friction_solver_impedance=tuple(float(value) for value in friction_impedance),
        actuator_force_limit_mode=force_limit_mode,
        actuator_force_range=tuple(float(value) for value in force_range),
        actuator_gravity_compensation=bool(c.actuator_gravity_compensation),
    )
    try:
        changed = self._adapter.set_joint_advanced_properties(properties)
    except Exception as exc:
        return CommandResult.bad(f"Joint properties could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Advanced joint properties could not be edited")
    self._refresh_structure()
    return CommandResult.good("")


def set_site_properties(self: Session, c: cmd.SetSiteProperties) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing site properties")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Site node {c.node_id} has no editable source element")
    if self._adapter.site_properties(c.node_id) is None:
        return CommandResult.bad(f"Site node {c.node_id} is unavailable")
    site_type = str(c.type).strip().lower()
    if site_type not in {"sphere", "ellipsoid", "capsule", "cylinder", "box"}:
        return CommandResult.bad("Site type is invalid")
    try:
        group = int(c.group)
        from_to = np.asarray(c.from_to, np.float64).reshape(6)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Site properties have invalid value types")
    if not 0 <= group < 6:
        return CommandResult.bad("Site group must be between 0 and 5")
    if not np.all(np.isfinite(from_to)):
        return CommandResult.bad("Site endpoints must be finite")
    if bool(c.use_from_to):
        if site_type not in {"capsule", "cylinder"}:
            return CommandResult.bad("Only capsule and cylinder sites use endpoints")
        if np.linalg.norm(from_to[3:] - from_to[:3]) <= 1e-9:
            return CommandResult.bad("Site endpoints must be distinct")
    properties = SiteProperties(
        node_id=int(c.node_id),
        type=site_type,
        group=group,
        use_from_to=bool(c.use_from_to),
        from_to=tuple(float(value) for value in from_to),
    )
    try:
        changed = self._adapter.set_site_properties(properties)
    except Exception as exc:
        return CommandResult.bad(f"Site properties could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Site properties could not be edited")
    self._refresh_structure()
    return CommandResult.good("")


def set_geometry_properties(self: Session, c: cmd.SetGeometryProperties) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing geometry properties")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    current = self._adapter.geometry_properties(c.node_id)
    if current is None:
        return CommandResult.bad(f"Geometry node {c.node_id} is unavailable")
    try:
        friction = np.asarray(c.friction, np.float64).reshape(3)
        collision_type_mask = int(c.collision_type_mask)
        collision_affinity_mask = int(c.collision_affinity_mask)
        contact_dimension = int(c.contact_dimension)
        contact_priority = int(c.contact_priority)
        margin = float(c.margin)
        gap = float(c.gap)
        solver_mix = float(c.solver_mix)
        solver_reference = np.asarray(
            current.solver_reference if c.solver_reference is None else c.solver_reference,
            np.float64,
        ).reshape(2)
        solver_impedance = np.asarray(
            current.solver_impedance if c.solver_impedance is None else c.solver_impedance,
            np.float64,
        ).reshape(5)
        adhesion = float(current.adhesion if c.adhesion is None else c.adhesion)
        surface_velocity = np.asarray(
            current.surface_velocity if c.surface_velocity is None else c.surface_velocity,
            np.float64,
        ).reshape(6)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Geometry contact properties have invalid value types")
    finite_values = np.concatenate(
        (
            friction,
            np.array((margin, gap, solver_mix, adhesion)),
            solver_reference,
            solver_impedance,
            surface_velocity,
        )
    )
    if not np.all(np.isfinite(finite_values)):
        return CommandResult.bad("Geometry properties must contain finite values")
    if np.any(friction < 0.0):
        return CommandResult.bad("Geometry friction cannot be negative")
    if collision_type_mask < 0 or collision_affinity_mask < 0:
        return CommandResult.bad("Collision masks cannot be negative")
    if max(collision_type_mask, collision_affinity_mask) > np.iinfo(np.int32).max:
        return CommandResult.bad("Collision masks exceed MuJoCo's 31-bit positive range")
    if contact_dimension not in (1, 3, 4, 6):
        return CommandResult.bad("Contact dimension must be 1, 3, 4, or 6")
    if not 0 <= contact_priority <= np.iinfo(np.int32).max:
        return CommandResult.bad("Contact priority exceeds MuJoCo's positive integer range")
    if margin < 0.0 or gap < 0.0:
        return CommandResult.bad("Contact margin and gap cannot be negative")
    if not 0.0 <= solver_mix <= 1.0:
        return CommandResult.bad("Solver mix must be between 0 and 1")
    standard_reference = np.all(solver_reference > 0.0)
    direct_reference = np.all(solver_reference <= 0.0)
    if not standard_reference and not direct_reference:
        return CommandResult.bad("Solver reference values must both use standard or direct format")
    if (
        not 0.0 <= solver_impedance[0] <= solver_impedance[1] <= 1.0
        or solver_impedance[2] <= 0.0
        or not 0.0 <= solver_impedance[3] <= 1.0
        or solver_impedance[4] < 1.0
    ):
        return CommandResult.bad("Solver impedance values are outside MuJoCo limits")
    if adhesion < 0.0:
        return CommandResult.bad("Geometry adhesion cannot be negative")
    properties = GeometryProperties(
        node_id=int(c.node_id),
        friction=tuple(float(value) for value in friction),
        collision_type_mask=collision_type_mask,
        collision_affinity_mask=collision_affinity_mask,
        contact_dimension=contact_dimension,
        contact_priority=contact_priority,
        margin=margin,
        gap=gap,
        solver_mix=solver_mix,
        solver_reference=tuple(float(value) for value in solver_reference),
        solver_impedance=tuple(float(value) for value in solver_impedance),
        adhesion=adhesion,
        surface_velocity=tuple(float(value) for value in surface_velocity),
    )
    if not self._adapter.set_geometry_properties(properties):
        return CommandResult.bad("Geometry contact properties could not be edited")
    self._adapter_revision = self._adapter.structure_revision
    self._structure_generation += 1
    return CommandResult.good("")


def set_geometry_advanced_properties(
    self: Session, c: cmd.SetGeometryAdvancedProperties
) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing geometry properties")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    if self._adapter.geometry_advanced_properties(c.node_id) is None:
        return CommandResult.bad(f"Geometry node {c.node_id} is unavailable")
    mass_mode = str(c.mass_mode).strip().lower()
    inertia_mode = str(c.inertia_mode).strip().lower()
    if mass_mode not in ("density", "mass"):
        return CommandResult.bad("Geometry mass mode must be density or mass")
    if inertia_mode not in ("volume", "shell"):
        return CommandResult.bad("Geometry inertia mode must be volume or shell")
    try:
        visual_group = int(c.visual_group)
        mass = float(c.mass)
        density = float(c.density)
        fluid_coefficients = np.asarray(c.fluid_coefficients, np.float64).reshape(5)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Advanced geometry properties have invalid value types")
    if not np.isfinite((mass, density, *fluid_coefficients)).all():
        return CommandResult.bad("Advanced geometry properties must be finite")
    if not 0 <= visual_group < 6:
        return CommandResult.bad("Geometry visual group must be between 0 and 5")
    if mass_mode == "mass" and mass <= 0.0:
        return CommandResult.bad("Geometry mass must be positive")
    if mass_mode == "density" and density <= 0.0:
        return CommandResult.bad("Geometry density must be positive")
    if np.any(fluid_coefficients < 0.0):
        return CommandResult.bad("Geometry fluid coefficients cannot be negative")
    properties = GeometryAdvancedProperties(
        node_id=int(c.node_id),
        visual_group=visual_group,
        mass_mode=mass_mode,
        mass=mass,
        density=density,
        inertia_mode=inertia_mode,
        fluid_ellipsoid=bool(c.fluid_ellipsoid),
        fluid_coefficients=tuple(float(value) for value in fluid_coefficients),
    )
    try:
        changed = self._adapter.set_geometry_advanced_properties(properties)
    except Exception as exc:
        return CommandResult.bad(f"Geometry properties could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Advanced geometry properties could not be edited")
    self._refresh_structure()
    return CommandResult.good("")


def set_geometry_shape(self: Session, c: cmd.SetGeometryShape) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing geometry shape")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Geometry node {c.node_id} has no editable source element")
    current = self._adapter.geometry_shape_properties(c.node_id)
    if current is None:
        return CommandResult.bad(f"Geometry node {c.node_id} is unavailable")
    geom_type = str(c.type).strip().lower()
    resource_name = str(c.resource_name).strip()
    supported = (
        "plane",
        "hfield",
        "sphere",
        "capsule",
        "ellipsoid",
        "cylinder",
        "box",
        "mesh",
    )
    if geom_type not in supported:
        return CommandResult.bad(f"Unsupported geometry type {geom_type!r}")
    choices = (
        current.mesh_names
        if geom_type == "mesh"
        else current.height_field_names
        if geom_type == "hfield"
        else ()
    )
    if geom_type in ("mesh", "hfield") and resource_name not in choices:
        return CommandResult.bad(
            f"{geom_type} resource {resource_name!r} is unavailable in this model"
        )
    try:
        changed = self._adapter.set_geometry_shape(c.node_id, geom_type, resource_name)
    except Exception as exc:
        return CommandResult.bad(f"Geometry shape could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Geometry shape could not be edited")
    self._refresh_structure()
    return CommandResult.good("")


def set_body_properties(self: Session, c: cmd.SetBodyProperties) -> CommandResult:
    caps = self._adapter.caps
    if not caps.model_properties:
        return CommandResult.bad(f"{caps.name} does not support model property editing")
    if caps.simulation and not self._paused:
        return CommandResult.bad("Pause the simulation before editing body properties")
    node = self.node(c.node_id)
    if node is None or not node.source_editable:
        return CommandResult.bad(f"Body node {c.node_id} has no editable source element")
    if self._adapter.body_properties(c.node_id) is None:
        return CommandResult.bad(f"Body node {c.node_id} is unavailable")
    inertia_mode = str(c.inertia_mode).strip().lower()
    sleep_policy = str(c.sleep_policy).strip().lower()
    if inertia_mode not in ("auto", "diagonal", "full"):
        return CommandResult.bad("Body inertia mode must be auto, diagonal, or full")
    if sleep_policy not in ("auto", "never", "allowed", "init"):
        return CommandResult.bad(
            "Body sleep policy must be auto, never, allowed, or initially asleep"
        )
    try:
        mass = float(c.mass)
        inertial_position = np.asarray(c.inertial_position, np.float64).reshape(3)
        inertial_quaternion = np.asarray(c.inertial_quaternion, np.float64).reshape(4)
        diagonal_inertia = np.asarray(c.diagonal_inertia, np.float64).reshape(3)
        full_inertia = np.asarray(c.full_inertia, np.float64).reshape(6)
        gravity_compensation = float(c.gravity_compensation)
    except (TypeError, ValueError, OverflowError):
        return CommandResult.bad("Body properties have invalid value types")
    values = np.concatenate(
        (
            np.array((mass, gravity_compensation), np.float64),
            inertial_position,
            inertial_quaternion,
            diagonal_inertia,
            full_inertia,
        )
    )
    if not np.all(np.isfinite(values)):
        return CommandResult.bad("Body properties must contain finite values")
    quaternion_norm = float(np.linalg.norm(inertial_quaternion))
    if quaternion_norm <= 1e-12:
        return CommandResult.bad("Body inertial rotation must be non-zero")
    inertial_quaternion /= quaternion_norm
    if inertia_mode != "auto" and mass <= 0.0:
        return CommandResult.bad("Explicit body mass must be positive")
    if inertia_mode == "diagonal":
        if np.any(diagonal_inertia <= 0.0):
            return CommandResult.bad("Diagonal inertia values must be positive")
        if 2.0 * float(np.max(diagonal_inertia)) > float(np.sum(diagonal_inertia)):
            return CommandResult.bad("Diagonal inertia violates the triangle inequality")
    if inertia_mode == "full":
        ixx, iyy, izz, ixy, ixz, iyz = full_inertia
        tensor = np.array(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)), np.float64)
        principal = np.linalg.eigvalsh(tensor)
        if np.any(principal <= 0.0):
            return CommandResult.bad("Full inertia tensor must be positive definite")
        if 2.0 * float(np.max(principal)) > float(np.sum(principal)):
            return CommandResult.bad("Full inertia tensor violates the triangle inequality")
    properties = BodyProperties(
        node_id=int(c.node_id),
        inertia_mode=inertia_mode,
        mass=mass,
        inertial_position=tuple(float(value) for value in inertial_position),
        inertial_quaternion=tuple(float(value) for value in inertial_quaternion),
        diagonal_inertia=tuple(float(value) for value in diagonal_inertia),
        full_inertia=tuple(float(value) for value in full_inertia),
        gravity_compensation=gravity_compensation,
        mocap=bool(c.mocap),
        sleep_policy=sleep_policy,
    )
    try:
        changed = self._adapter.set_body_properties(properties)
    except Exception as exc:
        return CommandResult.bad(f"Body properties could not be applied: {exc}")
    if not changed:
        return CommandResult.bad("Body properties could not be edited")
    self._refresh_structure()
    return CommandResult.good("")
