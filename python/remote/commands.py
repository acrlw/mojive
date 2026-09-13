"""Remote: commands."""

from __future__ import annotations

from mojive import commands as cmd
from mojive.commands import CommandResult


def handle_session_command(session, message: dict):
    """Translate one remote command payload into a session command or query."""
    if not isinstance(message, dict):
        return CommandResult.bad("Remote command must be a mapping")
    op = message.get("op")
    if not isinstance(op, str):
        return CommandResult.bad("Remote command op must be a string")
    revision = message.get("operation_version", 1)
    if type(revision) is not int or revision != 1:
        return CommandResult.bad(f"Unsupported revision {revision!r} of remote command {op}")
    if op == "raycast":
        try:
            return session.query(cmd.Pick(message["origin"], message["direction"]))
        except (KeyError, TypeError, ValueError) as exc:
            return CommandResult.bad(f"Invalid raycast arguments: {exc}")
    # Native publisher messages retain their established wire names. Their scene
    # edits share the application catalog's validation and command construction.
    from mojive.control.errors import ControlError
    from mojive.control.operations import apply_session_operation

    shared = {
        "pose": "set_pose",
        "geometry_color": "set_geometry_color",
        "geometry_size": "set_geometry_size",
        "scene_camera": "set_scene_camera",
        **{
            name: name
            for name in (
                "add_scene_object",
                "remove_scene_object",
                "add_scene_light",
                "remove_scene_light",
                "add_scene_camera",
                "remove_scene_camera",
                "duplicate_scene_entity",
                "remove_scene_entity",
                "rename_scene_entity",
            )
        },
    }
    if op in shared:
        try:
            return apply_session_operation(
                session,
                shared[op],
                {
                    key: value
                    for key, value in message.items()
                    if key not in {"op", "operation_version"}
                },
            )
        except ControlError as exc:
            return CommandResult.bad(str(exc))
    commands = {
        "pause": lambda: cmd.Pause(),
        "play": lambda: cmd.Play(),
        "step": lambda: cmd.Step(message.get("count", 1)),
        "reset": lambda: cmd.Reset(),
        "reload": lambda: cmd.Reload(),
        "keyframe": lambda: cmd.LoadKeyframe(message["keyframe_id"]),
        "visual_group": lambda: cmd.SetVisualGroup(
            message["category"], message["group"], message["visible"]
        ),
        "qpos": lambda: cmd.SetQpos(message["index"], message["value"]),
        "qpos_batch": lambda: cmd.SetQposBatch(message["indices"], message["values"]),
        "joint_properties": lambda: cmd.SetJointProperties(
            message["joint_id"],
            message["axis"],
            message["limited"],
            message["range"],
            message["damping"],
            message["stiffness"],
        ),
        "joint_advanced_properties": lambda: cmd.SetJointAdvancedProperties(
            message["joint_id"],
            message["group"],
            message["armature"],
            message["friction_loss"],
            message["reference"],
            message["spring_reference"],
            message["margin"],
            message["limit_solver_reference"],
            message["limit_solver_impedance"],
            message["friction_solver_reference"],
            message["friction_solver_impedance"],
            message["actuator_force_limit_mode"],
            message["actuator_force_range"],
            message["actuator_gravity_compensation"],
        ),
        "site_properties": lambda: cmd.SetSiteProperties(
            message["node_id"],
            message["type"],
            message["group"],
            message["use_from_to"],
            message["from_to"],
        ),
        "geometry_properties": lambda: cmd.SetGeometryProperties(
            message["node_id"],
            message["friction"],
            message["collision_type_mask"],
            message["collision_affinity_mask"],
            message["contact_dimension"],
            message["contact_priority"],
            message["margin"],
            message["gap"],
            message["solver_mix"],
            message.get("solver_reference"),
            message.get("solver_impedance"),
            message.get("adhesion"),
            message.get("surface_velocity"),
        ),
        "geometry_advanced_properties": lambda: cmd.SetGeometryAdvancedProperties(
            message["node_id"],
            message["visual_group"],
            message["mass_mode"],
            message["mass"],
            message["density"],
            message["inertia_mode"],
            message["fluid_ellipsoid"],
            message["fluid_coefficients"],
        ),
        "geometry_shape": lambda: cmd.SetGeometryShape(
            message["node_id"], message["type"], message.get("resource_name", "")
        ),
        "body_properties": lambda: cmd.SetBodyProperties(
            message["node_id"],
            message["inertia_mode"],
            message["mass"],
            message["inertial_position"],
            message["inertial_quaternion"],
            message["diagonal_inertia"],
            message["full_inertia"],
            message["gravity_compensation"],
            message["mocap"],
            message["sleep_policy"],
        ),
        "equality": lambda: cmd.SetEqualityEnabled(message["constraint_id"], message["enabled"]),
        "ctrl": lambda: cmd.SetCtrl(message["index"], message["value"]),
        "ctrl_vector": lambda: cmd.SetCtrlVector(message["values"]),
        "light": lambda: cmd.SetLight(
            message.get("light_index", message.get("light_id")), message["light"]
        ),
        "environment": lambda: cmd.SetEnvironment(message["environment"]),
        "skybox": lambda: cmd.SetSkybox(message.get("texture")),
        "material": lambda: cmd.SetMaterial(
            message.get("material_index", message.get("material_id")), message["material"]
        ),
        "perturb": lambda: cmd.Perturb(
            message["node_id"],
            message["target_position"],
            message["target_rotation"],
            message["mode"],
        ),
        "clear_perturb": lambda: cmd.ClearPerturb(),
    }
    factory = commands.get(op)
    if factory is None:
        return CommandResult.bad(f"unknown op {op!r}")
    try:
        command = factory()
    except (KeyError, TypeError, ValueError) as error:
        return CommandResult.bad(f"Invalid {op} arguments: {error}")
    return session.submit(command)
