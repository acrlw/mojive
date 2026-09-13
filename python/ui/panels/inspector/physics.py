"""Inspector: physics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive import math3d
from mojive.adapters.base import (
    JointInfo,
    NodeType,
    SceneNode,
)
from mojive.ui.panels import (
    PanelContext,
)

from .fields import (
    _begin_property_table,
    _property_control_row,
    _property_section,
    _property_vector_row,
    _vector_fields,
)
from .support import (
    _nearest_euler_degrees,
)


class _Physics:
    """Private physics methods of InspectorPanel; state belongs to its owner."""

    def _body_properties(self, ctx: PanelContext, node: SceneNode) -> None:
        current = ctx.session.body_properties(node.node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._body_property_node != node.node_id
            or self._body_property_generation != generation
            or self._body_property_edit is None
        ):
            self._body_property_node = node.node_id
            self._body_property_generation = generation
            self._body_property_edit = current
            self._body_inertial_euler = _nearest_euler_degrees(
                math3d.quat_to_mat3(current.inertial_quaternion), None
            )
            self._body_property_error = ""
        properties = self._body_property_edit
        if properties is None or not imgui.collapsing_header(ctx.tr("body inertial and dynamics")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        inertia_modes = ("auto from geoms", "diagonal", "full tensor")
        mode_values = ("auto", "diagonal", "full")
        mode = mode_values.index(properties.inertia_mode)
        inertia_mode = properties.inertia_mode
        edited = properties
        mode_changed = mass_changed = position_changed = rotation_changed = False
        diagonal_changed = full_diagonal_changed = full_cross_changed = False
        gravity_changed = mocap_changed = sleep_changed = False
        mass = float(edited.mass)
        inertial_position = np.asarray(edited.inertial_position, np.float64)
        inertial_euler = self._body_inertial_euler.copy()
        diagonal_inertia = np.asarray(edited.diagonal_inertia, np.float64)
        full_diagonal = np.asarray(edited.full_inertia[:3], np.float64)
        full_cross = np.asarray(edited.full_inertia[3:], np.float64)
        gravity_compensation = float(edited.gravity_compensation)
        mocap = bool(edited.mocap)
        parent = ctx.session.node(node.parent)
        root_body = parent is not None and parent.type in (NodeType.WORLD, NodeType.MODEL)
        movable_root = root_body and bool(ctx.session.joints_for_body(node.body_index))
        sleep_values = ("auto", "never", "allowed", "init")
        sleep_labels = ("automatic", "never", "allowed", "initially asleep")
        sleep_policy = (
            sleep_values.index(edited.sleep_policy) if edited.sleep_policy in sleep_values else 0
        )
        if _begin_property_table("body_inertial_properties"):
            _property_control_row(ctx, "inertia mode")
            mode_changed, mode = imgui.combo(
                "##body-inertia-mode", mode, tuple(ctx.tr(label) for label in inertia_modes)
            )
            inertia_mode = mode_values[mode]
            edited = replace(properties, inertia_mode=inertia_mode) if mode_changed else properties

            derived = inertia_mode == "auto"
            if derived:
                imgui.begin_disabled()
            _property_control_row(ctx, "mass")
            mass_changed, mass = imgui.drag_float(
                "##body-mass", float(edited.mass), 0.01, 0.000001, 1000000000.0, "%.6g kg"
            )
            position_changed, inertial_position = _property_vector_row(
                ctx,
                node,
                "inertial position",
                "body_inertial_position",
                edited.inertial_position,
                editable=not derived,
                speed=0.001,
                lo=-1000000.0,
                hi=1000000.0,
                fmt="%.5f m",
                reset_values=current.inertial_position,
            )
            rotation_disabled = derived or inertia_mode == "full"
            rotation_changed, inertial_euler = _property_vector_row(
                ctx,
                node,
                "inertial rotation",
                "body_inertial_rotation",
                self._body_inertial_euler,
                editable=not rotation_disabled,
                speed=0.25,
                lo=-360000.0,
                hi=360000.0,
                fmt="%.2f°",
                reset_values=_nearest_euler_degrees(
                    math3d.quat_to_mat3(current.inertial_quaternion), None
                ),
            )
            imgui.set_item_tooltip(
                ctx.tr(
                    "Body-frame rotation of diagonal principal axes; full tensors derive this rotation"
                )
            )
            if inertia_mode in ("auto", "diagonal"):
                diagonal_changed, diagonal_inertia = _property_vector_row(
                    ctx,
                    node,
                    "diagonal inertia",
                    "body_diagonal_inertia",
                    edited.diagonal_inertia,
                    editable=not derived,
                    speed=0.001,
                    lo=0.000000001,
                    hi=1000000000000.0,
                    fmt="%.6g",
                    reset_values=current.diagonal_inertia,
                )
            else:
                _property_control_row(ctx, "tensor xx / yy / zz")
                full_diagonal_changed, full_diagonal = imgui.drag_float3(
                    "##body-full-diagonal",
                    np.asarray(edited.full_inertia[:3], np.float32),
                    0.001,
                    -1000000000000.0,
                    1000000000000.0,
                    "%.6g",
                )
                _property_control_row(ctx, "tensor xy / xz / yz")
                full_cross_changed, full_cross = imgui.drag_float3(
                    "##body-full-cross",
                    np.asarray(edited.full_inertia[3:], np.float32),
                    0.001,
                    -1000000000000.0,
                    1000000000000.0,
                    "%.6g",
                )
            if derived:
                imgui.end_disabled()

            _property_control_row(ctx, "gravity compensation")
            gravity_changed, gravity_compensation = imgui.drag_float(
                "##body-gravity-compensation",
                float(edited.gravity_compensation),
                0.01,
                -1000000.0,
                1000000.0,
                "%.4f",
            )
            _property_control_row(ctx, "mocap body")
            mocap_changed, mocap = imgui.checkbox("##body-mocap", bool(edited.mocap))
            _property_control_row(ctx, "sleep policy")
            if not movable_root:
                imgui.begin_disabled()
            sleep_changed, sleep_policy = imgui.combo(
                "##body-sleep-policy",
                sleep_policy,
                tuple(ctx.tr(label) for label in sleep_labels),
            )
            if not movable_root:
                imgui.end_disabled()
                imgui.set_item_tooltip(
                    ctx.tr("MuJoCo sleep policies apply only to movable root bodies")
                )
            imgui.end_table()
        derived = inertia_mode == "auto"
        rotation_disabled = derived or inertia_mode == "full"
        sleep_policy_value = sleep_values[sleep_policy]

        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit model body properties"))

        if mass_changed:
            edited = replace(edited, mass=float(mass))
        if position_changed:
            edited = replace(
                edited,
                inertial_position=tuple(float(value) for value in inertial_position),
            )
        if rotation_changed and not rotation_disabled:
            self._body_inertial_euler = np.asarray(inertial_euler, np.float64)
            quaternion = math3d.mat3_to_quat(
                math3d.euler_xyz_to_mat3(np.radians(self._body_inertial_euler))
            )
            edited = replace(
                edited, inertial_quaternion=tuple(float(value) for value in quaternion)
            )
        if diagonal_changed:
            edited = replace(
                edited,
                diagonal_inertia=tuple(float(value) for value in diagonal_inertia),
            )
        if full_diagonal_changed or full_cross_changed:
            edited = replace(
                edited,
                full_inertia=tuple(float(value) for value in (*full_diagonal, *full_cross)),
            )
        if gravity_changed:
            edited = replace(edited, gravity_compensation=float(gravity_compensation))
        if mocap_changed:
            edited = replace(edited, mocap=bool(mocap))
        if sleep_changed and movable_root:
            edited = replace(edited, sleep_policy=sleep_policy_value)
        self._body_property_edit = edited

        dirty = edited != current
        if not editable or not dirty:
            imgui.begin_disabled()
        if imgui.button(f"{ctx.tr('Apply')}##body-properties"):
            result = ctx.submit(
                cmd.SetBodyProperties(
                    node_id=edited.node_id,
                    inertia_mode=edited.inertia_mode,
                    mass=edited.mass,
                    inertial_position=edited.inertial_position,
                    inertial_quaternion=edited.inertial_quaternion,
                    diagonal_inertia=edited.diagonal_inertia,
                    full_inertia=edited.full_inertia,
                    gravity_compensation=edited.gravity_compensation,
                    mocap=edited.mocap,
                    sleep_policy=edited.sleep_policy,
                )
            )
            if result.ok:
                self._body_property_generation = -1
                self._body_property_error = ""
            else:
                self._body_property_error = result.message
        imgui.set_item_tooltip(ctx.tr("Apply rebuilds the model once"))
        if not editable or not dirty:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button(f"{ctx.tr('Revert')}##body-properties"):
            self._body_property_edit = current
            self._body_inertial_euler = _nearest_euler_degrees(
                math3d.quat_to_mat3(current.inertial_quaternion), None
            )
            self._body_property_error = ""
        if self._body_property_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._body_property_error)
            if imgui.button(f"{ctx.tr('Copy error')}##body-properties"):
                imgui.set_clipboard_text(self._body_property_error)

    def _joint(self, ctx: PanelContext, node: SceneNode) -> None:
        joint = next(
            (item for item in ctx.session.joints if item.joint_id == node.joint_index), None
        )
        if joint is None:
            imgui.text_disabled(ctx.tr("joint metadata is unavailable"))
            return
        if _begin_property_table("joint_identity"):
            for label, value in (
                ("type", joint.type),
                ("qpos address", str(joint.qpos_adr)),
                ("dof", str(joint.dof)),
            ):
                _property_control_row(ctx, label)
                imgui.align_text_to_frame_padding()
                imgui.text(value)
            imgui.end_table()
        if _property_section(ctx, "joint properties"):
            self._joint_properties(ctx, node, joint)
        self._joint_advanced_properties(ctx, joint)

    def _joint_properties(self, ctx: PanelContext, node: SceneNode, joint: JointInfo) -> None:
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and ctx.session.paused
            and joint.type != "free"
            and node.source_editable
        )
        if not editable:
            imgui.begin_disabled()
        axis = np.asarray(joint.axis, np.float32)
        axis_changed = False
        limited = bool(joint.limited)
        limited_changed = False
        range_changed = False
        damping_changed = False
        damping = float(joint.damping)
        stiffness_changed = False
        stiffness = float(joint.stiffness)
        value_range = np.asarray(joint.range, np.float64).copy()
        range_valid = (
            value_range[1] > 0.0 if joint.type == "ball" else value_range[1] > value_range[0]
        )
        default_range = np.array(
            (0.0, np.pi)
            if joint.type == "ball"
            else ((-np.pi, np.pi) if joint.type == "hinge" else (-1.0, 1.0)),
            np.float64,
        )
        displayed_range = value_range.copy() if range_valid else default_range
        if _begin_property_table("joint_properties_table"):
            if joint.type in ("hinge", "slide"):
                axis_changed, axis = _property_vector_row(
                    ctx,
                    node,
                    "axis",
                    "joint_axis",
                    axis,
                    editable=editable,
                    speed=0.01,
                    lo=-1000000.0,
                    hi=1000000.0,
                    fmt="%.4f",
                    reset_values=joint.axis,
                    label_tooltip="Axis is expressed in the body frame",
                )

            if joint.type in ("hinge", "slide", "ball"):
                _property_control_row(ctx, "limited")
                limited_changed, limited = imgui.checkbox("##joint_limited", limited)
                if joint.type == "ball":
                    upper_deg = float(np.degrees(displayed_range[1]))
                    _property_control_row(ctx, "limit angle")
                    range_changed, upper_deg = imgui.drag_float(
                        "##joint_limit_angle", upper_deg, 0.5, 0.001, 360.0, "%.2f deg"
                    )
                    displayed_range[:] = (0.0, np.radians(upper_deg))
                elif joint.type == "hinge":
                    degrees = np.degrees(displayed_range)
                    _property_control_row(ctx, "range")
                    range_changed, degrees = imgui.drag_float2(
                        "##joint_range", degrees, 0.5, -36000.0, 36000.0, "%.2f deg"
                    )
                    displayed_range[:] = np.radians(degrees)
                else:
                    _property_control_row(ctx, "range")
                    range_changed, displayed_range = imgui.drag_float2(
                        "##joint_range",
                        displayed_range,
                        0.01,
                        -1000000.0,
                        1000000.0,
                        "%.4f m",
                    )
                    displayed_range = np.asarray(displayed_range, np.float64)
                if range_changed or (limited_changed and limited and not range_valid):
                    value_range = displayed_range

            _property_control_row(ctx, "damping")
            damping_changed, damping = imgui.drag_float(
                "##joint_damping", damping, 0.01, 0.0, 1000000.0, "%.4f"
            )
            _property_control_row(ctx, "stiffness")
            stiffness_changed, stiffness = imgui.drag_float(
                "##joint_stiffness", stiffness, 0.01, 0.0, 1000000.0, "%.4f"
            )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            reason = (
                "Free-joint properties stay defined by the free body"
                if joint.type == "free"
                else (
                    "Pause the simulation to edit model properties"
                    if not ctx.session.paused
                    else "This adapter cannot write model properties"
                    if not ctx.session.adapter.caps.model_properties
                    else "This compiled joint has no editable source element"
                )
            )
            imgui.text_disabled(reason)

        changed = any(
            (
                axis_changed,
                limited_changed,
                range_changed,
                damping_changed,
                stiffness_changed,
            )
        )
        invalid_axis = joint.type in ("hinge", "slide") and np.linalg.norm(axis) <= 1e-6
        invalid_range = bool(limited) and value_range[1] <= value_range[0]
        if invalid_axis:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "Axis must be non-zero")
        if invalid_range:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning), "Range upper bound must exceed lower bound"
            )
        if changed and editable and not invalid_axis and not invalid_range:
            self._submit_edit(
                ctx,
                cmd.SetJointProperties(
                    joint.joint_id,
                    np.asarray(axis, np.float64),
                    bool(limited),
                    (float(value_range[0]), float(value_range[1])),
                    float(damping),
                    float(stiffness),
                ),
            )

    def _joint_advanced_properties(self, ctx: PanelContext, joint: JointInfo) -> None:
        current = ctx.session.joint_advanced_properties(joint.joint_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._joint_advanced_id != joint.joint_id
            or self._joint_advanced_generation != generation
            or self._joint_advanced_edit is None
        ):
            self._joint_advanced_id = joint.joint_id
            self._joint_advanced_generation = generation
            self._joint_advanced_edit = current
            self._joint_advanced_error = ""
        properties = self._joint_advanced_edit
        if properties is None or not _property_section(ctx, "advanced joint properties"):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        rotational = joint.type in ("hinge", "ball")
        group_changed, group = False, int(properties.group)
        armature_changed, armature = False, float(properties.armature)
        friction_changed, friction_loss = False, float(properties.friction_loss)
        reference = (
            float(np.degrees(properties.reference)) if rotational else float(properties.reference)
        )
        spring_reference = (
            float(np.degrees(properties.spring_reference))
            if rotational
            else float(properties.spring_reference)
        )
        reference_changed = False
        spring_reference_changed = False
        margin_changed, margin = False, float(properties.margin)
        limit_reference_changed = False
        limit_reference = np.asarray(properties.limit_solver_reference, np.float32)
        limit_impedance_first_changed = False
        limit_impedance_first = np.asarray(properties.limit_solver_impedance[:3], np.float32)
        limit_impedance_shape_changed = False
        limit_impedance_shape = np.asarray(properties.limit_solver_impedance[3:], np.float32)
        friction_reference_changed = False
        friction_reference = np.asarray(properties.friction_solver_reference, np.float32)
        friction_impedance_first_changed = False
        friction_impedance_first = np.asarray(properties.friction_solver_impedance[:3], np.float32)
        friction_impedance_shape_changed = False
        friction_impedance_shape = np.asarray(properties.friction_solver_impedance[3:], np.float32)
        force_modes = ("auto", "unlimited", "limited")
        force_mode = force_modes.index(properties.actuator_force_limit_mode)
        force_mode_changed = False
        force_range_changed = False
        force_range = np.asarray(properties.actuator_force_range, np.float32)
        gravity_changed = False
        gravity_compensation = properties.actuator_gravity_compensation

        if _begin_property_table("joint_advanced_properties_table"):
            _property_control_row(ctx, "group")
            group_changed, group = imgui.combo(
                "##joint_advanced_group", group, tuple(str(value) for value in range(6))
            )
            _property_control_row(ctx, "armature")
            armature_changed, armature = imgui.drag_float(
                "##joint_armature", armature, 0.001, 0.0, 1000000000.0, "%.6g"
            )
            _property_control_row(ctx, "friction loss")
            friction_changed, friction_loss = imgui.drag_float(
                "##joint_friction_loss",
                friction_loss,
                0.001,
                0.0,
                1000000000.0,
                "%.6g",
            )
            _property_control_row(ctx, "reference")
            reference_changed, reference = imgui.drag_float(
                "##joint_reference",
                reference,
                0.25 if rotational else 0.001,
                -360000.0 if rotational else -1000000.0,
                360000.0 if rotational else 1000000.0,
                "%.3f deg" if rotational else "%.6g m",
            )
            _property_control_row(ctx, "spring reference")
            spring_reference_changed, spring_reference = imgui.drag_float(
                "##joint_spring_reference",
                spring_reference,
                0.25 if rotational else 0.001,
                -360000.0 if rotational else -1000000.0,
                360000.0 if rotational else 1000000.0,
                "%.3f deg" if rotational else "%.6g m",
            )
            _property_control_row(ctx, "limit margin")
            margin_changed, margin = imgui.drag_float(
                "##joint_limit_margin", margin, 0.001, 0.0, 1000000.0, "%.6g"
            )
            _property_control_row(ctx, "limit solver reference")
            limit_reference_changed, limit_reference = imgui.drag_float2(
                "##joint_limit_solver_reference",
                limit_reference,
                0.001,
                -1000000.0,
                1000000.0,
                "%.5g",
            )
            _property_control_row(ctx, "limit impedance min / max / width")
            limit_impedance_first_changed, limit_impedance_first = imgui.drag_float3(
                "##joint_limit_impedance_first",
                limit_impedance_first,
                0.001,
                0.0,
                1.0,
                "%.5g",
            )
            _property_control_row(ctx, "limit impedance midpoint / power")
            limit_impedance_shape_changed, limit_impedance_shape = imgui.drag_float2(
                "##joint_limit_impedance_shape",
                limit_impedance_shape,
                0.01,
                0.0,
                1000.0,
                "%.4g",
            )
            _property_control_row(ctx, "friction solver reference")
            friction_reference_changed, friction_reference = imgui.drag_float2(
                "##joint_friction_solver_reference",
                friction_reference,
                0.001,
                -1000000.0,
                1000000.0,
                "%.5g",
            )
            _property_control_row(ctx, "friction impedance min / max / width")
            friction_impedance_first_changed, friction_impedance_first = imgui.drag_float3(
                "##joint_friction_impedance_first",
                friction_impedance_first,
                0.001,
                0.0,
                1.0,
                "%.5g",
            )
            _property_control_row(ctx, "friction impedance midpoint / power")
            friction_impedance_shape_changed, friction_impedance_shape = imgui.drag_float2(
                "##joint_friction_impedance_shape",
                friction_impedance_shape,
                0.01,
                0.0,
                1000.0,
                "%.4g",
            )
            _property_control_row(ctx, "actuator force limit")
            force_mode_changed, force_mode = imgui.combo(
                "##joint_actuator_force_limit",
                force_mode,
                tuple(ctx.tr(value) for value in force_modes),
            )
            _property_control_row(ctx, "actuator force range")
            force_range_changed, force_range = imgui.drag_float2(
                "##joint_actuator_force_range",
                force_range,
                0.01,
                -1000000000.0,
                1000000000.0,
                "%.6g N",
            )
            imgui.set_item_tooltip(
                ctx.tr(
                    "Auto enables the limit when a valid range is authored; "
                    "unlimited ignores the range"
                )
            )
            _property_control_row(ctx, "actuator gravity compensation")
            gravity_changed, gravity_compensation = imgui.checkbox(
                "##joint_actuator_gravity_compensation", gravity_compensation
            )
            imgui.end_table()
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit advanced joint properties"))

        edited = properties
        if group_changed:
            edited = replace(edited, group=int(group))
        if armature_changed:
            edited = replace(edited, armature=float(armature))
        if friction_changed:
            edited = replace(edited, friction_loss=float(friction_loss))
        if reference_changed:
            edited = replace(
                edited,
                reference=float(np.radians(reference) if rotational else reference),
            )
        if spring_reference_changed:
            edited = replace(
                edited,
                spring_reference=float(
                    np.radians(spring_reference) if rotational else spring_reference
                ),
            )
        if margin_changed:
            edited = replace(edited, margin=float(margin))
        if limit_reference_changed:
            edited = replace(
                edited,
                limit_solver_reference=tuple(float(value) for value in limit_reference),
            )
        if limit_impedance_first_changed or limit_impedance_shape_changed:
            edited = replace(
                edited,
                limit_solver_impedance=tuple(
                    float(value) for value in (*limit_impedance_first, *limit_impedance_shape)
                ),
            )
        if friction_reference_changed:
            edited = replace(
                edited,
                friction_solver_reference=tuple(float(value) for value in friction_reference),
            )
        if friction_impedance_first_changed or friction_impedance_shape_changed:
            edited = replace(
                edited,
                friction_solver_impedance=tuple(
                    float(value) for value in (*friction_impedance_first, *friction_impedance_shape)
                ),
            )
        if force_mode_changed:
            edited = replace(edited, actuator_force_limit_mode=force_modes[force_mode])
        if force_range_changed:
            edited = replace(
                edited,
                actuator_force_range=tuple(float(value) for value in force_range),
            )
        if gravity_changed:
            edited = replace(edited, actuator_gravity_compensation=bool(gravity_compensation))
        self._joint_advanced_edit = edited

        dirty = edited != current
        invalid_force_range = (
            edited.actuator_force_limit_mode == "limited"
            and edited.actuator_force_range[1] <= edited.actuator_force_range[0]
        )
        if invalid_force_range:
            imgui.text_colored(
                imgui.ImVec4(*ctx.theme.warning),
                "Actuator force upper bound must exceed its lower bound",
            )
        apply_clicked = False
        revert_clicked = False
        if _begin_property_table("joint_advanced_actions"):
            _property_control_row(ctx, "changes")
            if not editable or not dirty or invalid_force_range:
                imgui.begin_disabled()
            apply_clicked = imgui.button(f"{ctx.tr('Apply')}##joint-advanced")
            imgui.set_item_tooltip(ctx.tr("Apply rebuilds the model once"))
            if not editable or not dirty or invalid_force_range:
                imgui.end_disabled()
            imgui.same_line()
            revert_clicked = imgui.button(f"{ctx.tr('Revert')}##joint-advanced")
            imgui.end_table()
        if apply_clicked:
            result = ctx.submit(
                cmd.SetJointAdvancedProperties(
                    joint_id=edited.joint_id,
                    group=edited.group,
                    armature=edited.armature,
                    friction_loss=edited.friction_loss,
                    reference=edited.reference,
                    spring_reference=edited.spring_reference,
                    margin=edited.margin,
                    limit_solver_reference=edited.limit_solver_reference,
                    limit_solver_impedance=edited.limit_solver_impedance,
                    friction_solver_reference=edited.friction_solver_reference,
                    friction_solver_impedance=edited.friction_solver_impedance,
                    actuator_force_limit_mode=edited.actuator_force_limit_mode,
                    actuator_force_range=edited.actuator_force_range,
                    actuator_gravity_compensation=edited.actuator_gravity_compensation,
                )
            )
            if result.ok:
                self._joint_advanced_generation = -1
                self._joint_advanced_error = ""
            else:
                self._joint_advanced_error = result.message
        if revert_clicked:
            self._joint_advanced_edit = current
            self._joint_advanced_error = ""
        if self._joint_advanced_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._joint_advanced_error)
            if imgui.button(f"{ctx.tr('Copy error')}##joint-advanced"):
                imgui.set_clipboard_text(self._joint_advanced_error)

    def _site_properties(self, ctx: PanelContext, node: SceneNode) -> None:
        current = ctx.session.site_properties(node.node_id)
        if current is None:
            return
        generation = ctx.session.structure_generation
        if (
            self._site_property_node != node.node_id
            or self._site_property_generation != generation
            or self._site_property_edit is None
        ):
            self._site_property_node = node.node_id
            self._site_property_generation = generation
            self._site_property_edit = current
            self._site_property_error = ""
        properties = self._site_property_edit
        if properties is None or not imgui.collapsing_header(ctx.tr("site shape and endpoints")):
            return
        editable = bool(
            ctx.session.adapter.caps.model_properties
            and (not ctx.session.adapter.caps.simulation or ctx.session.paused)
        )
        if not editable:
            imgui.begin_disabled()

        site_types = ("sphere", "ellipsoid", "capsule", "cylinder", "box")
        site_type = site_types.index(properties.type)
        type_changed = group_changed = endpoints_changed = False
        type_value = site_types[site_type]
        supports_endpoints = type_value in ("capsule", "cylinder")
        use_from_to = properties.use_from_to if supports_endpoints else False
        if _begin_property_table("site_shape_properties"):
            _property_control_row(ctx, "type")
            type_changed, site_type = imgui.combo(
                "##site-type",
                site_type,
                tuple(value.title() for value in site_types),
            )
            type_value = site_types[site_type]
            supports_endpoints = type_value in ("capsule", "cylinder")
            if not supports_endpoints:
                use_from_to = False
            _property_control_row(ctx, "visual group")
            group_changed, group = imgui.combo(
                "##site-visual-group",
                int(properties.group),
                tuple(str(value) for value in range(6)),
            )
            _property_control_row(ctx, "define with endpoints")
            if not supports_endpoints:
                imgui.begin_disabled()
            endpoints_changed, use_from_to = imgui.checkbox("##site-define-endpoints", use_from_to)
            if not supports_endpoints:
                imgui.end_disabled()
                imgui.set_item_tooltip(ctx.tr("Endpoints apply only to capsule and cylinder sites"))
            imgui.end_table()
        from_to = np.asarray(properties.from_to, np.float32)
        first_changed = second_changed = False
        if use_from_to:
            (first_changed, first), (second_changed, second) = _vector_fields(
                ctx,
                node,
                "site_endpoints",
                (
                    ("endpoint A (body frame)", from_to[:3], 0.001, "%.5f m", from_to[:3]),
                    ("endpoint B (body frame)", from_to[3:], 0.001, "%.5f m", from_to[3:]),
                ),
                editable=editable,
            )
            from_to = np.asarray((*first, *second), np.float32)
        if not editable:
            imgui.end_disabled()
            imgui.text_disabled(ctx.tr("Pause the simulation to edit site properties"))

        edited = properties
        if type_changed:
            edited = replace(
                edited,
                type=type_value,
                use_from_to=bool(use_from_to),
            )
        if group_changed:
            edited = replace(edited, group=int(group))
        if endpoints_changed:
            edited = replace(edited, use_from_to=bool(use_from_to))
        if first_changed or second_changed:
            edited = replace(edited, from_to=tuple(float(value) for value in from_to))
        self._site_property_edit = edited

        invalid_endpoints = (
            edited.use_from_to
            and np.linalg.norm(np.asarray(edited.from_to[3:]) - np.asarray(edited.from_to[:3]))
            <= 1e-9
        )
        if invalid_endpoints:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), "Site endpoints must be distinct")
        dirty = edited != current
        if not editable or not dirty or invalid_endpoints:
            imgui.begin_disabled()
        if imgui.button(f"{ctx.tr('Apply')}##site-properties"):
            result = ctx.submit(
                cmd.SetSiteProperties(
                    node_id=edited.node_id,
                    type=edited.type,
                    group=edited.group,
                    use_from_to=edited.use_from_to,
                    from_to=edited.from_to,
                )
            )
            if result.ok:
                self._site_property_generation = -1
                self._site_property_error = ""
            else:
                self._site_property_error = result.message
        imgui.set_item_tooltip(ctx.tr("Apply rebuilds the model once"))
        if not editable or not dirty or invalid_endpoints:
            imgui.end_disabled()
        imgui.same_line()
        if imgui.button(f"{ctx.tr('Revert')}##site-properties"):
            self._site_property_edit = current
            self._site_property_error = ""
        if self._site_property_error:
            imgui.text_colored(imgui.ImVec4(*ctx.theme.warning), self._site_property_error)
            if imgui.button(f"{ctx.tr('Copy error')}##site-properties"):
                imgui.set_clipboard_text(self._site_property_error)
