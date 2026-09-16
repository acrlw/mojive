"""Dependencies of cached scene data on writable MuJoCo model fields.

Physics values are consumed by MuJoCo in place. Only inputs baked into the
scene source invalidate it; changing mjModel does not imply recompilation.
"""

# Resource and visual-structure inputs used by source.py and deformables.py.
_SOURCE_PREFIXES = (
    "mesh_",
    "hfield_",
    "tex_",
    "mat_",
    "flex_",
    "skin_",
    "bvh_",
    "oct_",
    "name_",
    "numeric_",
    "text_",
    "tuple_",
    "key_",
    "cam_",
    "vis.",
)
_SOURCE_FIELDS = frozenset(
    {
        "names",
        "qpos0",
        "body_parentid",
        "body_weldid",
        "body_geomadr",
        "body_geomnum",
        "body_jntadr",
        "body_jntnum",
        "body_dofnum",
        "body_bvhadr",
        "body_bvhnum",
        "geom_type",
        "geom_bodyid",
        "geom_dataid",
        "geom_size",
        "geom_aabb",
        "geom_group",
        "geom_matid",
        "geom_rgba",
        "geom_contype",
        "geom_conaffinity",
        "pair_geom1",
        "pair_geom2",
        "site_type",
        "site_bodyid",
        "site_size",
        "site_group",
        "site_matid",
        "site_rgba",
        "jnt_type",
        "jnt_group",
        "tendon_group",
        "tendon_matid",
        "tendon_rgba",
        "tendon_width",
        "actuator_trnid",
        "actuator_trntype",
        "actuator_group",
        "actuator_dyntype",
        "sensor_type",
        "sensor_objid",
        "sensor_objtype",
        "sensor_intprm",
        "sensor_dim",
        "cam_bodyid",
        "stat.extent",
        "stat.center",
    }
)
_DIAGNOSTIC_FIELDS = frozenset({"body_mass", "body_inertia", "stat.meanmass", "stat.meansize"})
CONTROL_FIELDS = frozenset(
    {
        "jnt_axis",
        "jnt_limited",
        "jnt_range",
        "dof_damping",
        "jnt_stiffness",
        "actuator_gainprm",
        "actuator_ctrllimited",
        "actuator_ctrlrange",
    }
)
ACTUATOR_DISPLAY_FIELDS = (
    ("actuator_ctrllimited", "actuator_ctrl_limited"),
    ("actuator_ctrlrange", "actuator_ctrl_range"),
    ("actuator_actlimited", "actuator_act_limited"),
    ("actuator_actrange", "actuator_act_range"),
)


def scene_fields_changed(fields: set[str]) -> bool:
    return any(name in _SOURCE_FIELDS or name.startswith(_SOURCE_PREFIXES) for name in fields)


def diagnostic_fields_changed(fields: set[str]) -> bool:
    return not fields.isdisjoint(_DIAGNOSTIC_FIELDS)


def camera_fields_changed(fields: set[str]) -> bool:
    return any(name.startswith(("cam_", "vis.global", "vis.map.", "stat.")) for name in fields)
