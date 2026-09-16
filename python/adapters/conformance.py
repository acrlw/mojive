"""Backend-neutral structural checks for third-party SceneAdapter implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshShape
from .base import (
    AdapterCaps,
    FrameNeeds,
    KeyframeCatalog,
    SceneAdapter,
    SceneAdapterBase,
    SceneAppearance,
    SceneFrame,
    SceneProvider,
    SceneSource,
    SimulationAccess,
)


@dataclass(frozen=True)
class ConformanceCheck:
    """One named adapter invariant and its diagnostic result."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ConformanceReport:
    """Complete backend-neutral adapter validation report."""

    backend: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def ok(self) -> bool:
        """Return whether every conformance invariant passed."""

        return all(check.ok for check in self.checks)


def check_scene_provider(provider: SceneProvider) -> ConformanceReport:
    """Validate a read-only scene stream without requiring simulation or editor methods."""
    _source, _frame, checks = _check_scene_stream(provider)
    return ConformanceReport(type(provider).__name__, tuple(checks))


def _check_scene_stream(provider: SceneProvider):
    checks: list[ConformanceCheck] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(ConformanceCheck(name, bool(ok), detail))

    missing = [
        name for name in ("scene_source", "frame") if not callable(getattr(provider, name, None))
    ]
    add("required methods", not missing, "all present" if not missing else f"missing {missing}")
    if missing:
        return None, None, checks
    try:
        revision = provider.structure_revision
        source = provider.scene_source()
        frame = provider.frame(
            FrameNeeds(
                poses=True,
                qpos=True,
                qvel=True,
                contacts=True,
                tendons=True,
                actuator=True,
                sensors=True,
                deformables=True,
                diagnostics=True,
                islands=True,
            )
        )
    except Exception as exc:
        add("scene stream", False, f"{type(exc).__name__}: {exc}")
        return None, None, checks
    if not isinstance(source, SceneSource) or not isinstance(frame, SceneFrame):
        add("scene stream", False, "scene_source/frame must return SceneSource/SceneFrame")
        return None, None, checks

    n = source.instance_count
    instance_fields = (
        "geom_material",
        "geom_size",
        "geom_rgba",
        "geom_object_id",
        "geom_body",
        "geom_source",
        "geom_pose_source",
        "geom_visual",
        "geom_static",
        "geom_node",
        "geom_local",
        "geom_infinite_plane",
    )
    lengths = {name: len(getattr(source, name)) for name in instance_fields}
    add("instance columns", all(length == n for length in lengths.values()), str(lengths))
    add(
        "instance dtypes",
        source.geom_object_id.dtype == np.uint32 and source.geom_source.dtype == np.int32,
        f"object_id={source.geom_object_id.dtype}, source={source.geom_source.dtype}",
    )

    ids = [node.node_id for node in source.nodes]
    known = set(ids)
    parents_ok = all(node.parent == -1 or node.parent in known for node in source.nodes)
    add("node graph", len(ids) == len(known) and parents_ok, f"{len(ids)} nodes")

    light_nodes = {node.light_index for node in source.nodes if node.light_index >= 0}
    lights_ok = light_nodes == set(range(len(source.lights.lights)))
    if frame.lights is not None:
        lights_ok &= len(frame.lights.lights) == len(source.lights.lights)
    add(
        "light entities", lights_ok, f"{len(source.lights.lights)} lights, {len(light_nodes)} nodes"
    )

    camera_nodes = {node.camera_index for node in source.nodes if node.camera_index >= 0}
    cameras_ok = camera_nodes == set(range(len(source.cameras)))
    if frame.cameras is not None:
        cameras_ok &= len(frame.cameras) == len(source.cameras)
    add("camera entities", cameras_ok, f"{len(source.cameras)} cameras, {len(camera_nodes)} nodes")

    mesh_ok = True
    mesh_detail = []
    for key in (*source.geom_mesh, *source.geom_convex_mesh):
        if key.shape in (
            MeshShape.ASSET,
            MeshShape.CONVEX_HULL,
            MeshShape.HEIGHTFIELD,
            MeshShape.FLEX,
            MeshShape.SKIN,
        ):
            present = key in source.meshes
            mesh_ok &= present
            if not present:
                mesh_detail.append(str(key))
    add("mesh references", mesh_ok, "all present" if mesh_ok else f"missing {mesh_detail}")

    poses_ok = frame.geom_xpos is not None and frame.geom_xmat is not None
    if poses_ok:
        positions = np.asarray(frame.geom_xpos)
        rotations = np.asarray(frame.geom_xmat)
        poses_ok = positions.ndim == 2 and positions.shape[1:] == (3,)
        poses_ok &= rotations.shape == (len(positions), 3, 3)
        poses_ok &= np.isfinite(positions).all() and np.isfinite(rotations).all()
        pose_detail = f"{len(positions)} source poses"
    else:
        pose_detail = "geom_xpos/geom_xmat missing"
    add("pose frame", poses_ok, pose_detail)

    updates = frame.mesh_updates or {}
    dynamic_ok = set(updates) <= set(source.dynamic_meshes)
    dynamic_ok &= all(
        key in source.meshes
        and update.positions.shape == source.meshes[key].positions.shape
        and update.normals.shape == source.meshes[key].normals.shape
        for key, update in updates.items()
    )
    add("dynamic meshes", dynamic_ok, f"{len(updates)} updates")

    tendon_count = len(source.tendon_rgba)
    tendon_meta_ok = (
        len(source.tendon_material) == tendon_count
        and len(source.tendon_visible) == tendon_count
        and (
            not tendon_count
            or (
                int(source.tendon_material.min()) >= 0
                and int(source.tendon_material.max()) < len(source.materials)
            )
        )
        and len(source.actuator_visible) == len(source.actuator_tendon)
        and len(source.actuator_ctrl_address) == len(source.actuator_visible)
        and len(source.actuator_ctrl_limited) == len(source.actuator_visible)
        and len(source.actuator_ctrl_range) == len(source.actuator_visible)
    )
    add("tendon metadata", tendon_meta_ok, f"{tendon_count} tendons")

    tendons_ok = (
        frame.tendon_segments is None and frame.tendon_ids is None and frame.tendon_widths is None
    )
    tendon_detail = "not produced"
    if (
        frame.tendon_segments is not None
        and frame.tendon_ids is not None
        and frame.tendon_widths is not None
    ):
        segments = np.asarray(frame.tendon_segments)
        tendon_ids = np.asarray(frame.tendon_ids)
        widths = np.asarray(frame.tendon_widths)
        tendons_ok = (
            segments.shape == (len(tendon_ids), 2, 3)
            and widths.shape == (len(tendon_ids),)
            and np.isfinite(segments).all()
            and np.isfinite(widths).all()
            and (widths > 0.0).all()
        )
        tendons_ok &= not len(tendon_ids) or (
            int(tendon_ids.min()) >= 0 and int(tendon_ids.max()) < len(source.tendon_rgba)
        )
        tendon_detail = f"{len(segments)} segments"
    add("tendon frame", tendons_ok, tendon_detail)

    add(
        "stable structure",
        provider.structure_revision == revision,
        f"revision={provider.structure_revision}",
    )
    return source, frame, checks


def check_adapter(adapter: SceneAdapter) -> ConformanceReport:
    """Validate one frame and declared optional capabilities without invoking writes.

    Unsupported base-class defaults are diagnosed only for advertised write contracts.
    This detects declaration mistakes; it does not prove transactional write behavior.
    """
    source, frame, checks = _check_scene_stream(adapter)
    caps = getattr(adapter, "caps", None)
    if not isinstance(caps, AdapterCaps):
        checks.append(ConformanceCheck("adapter capabilities", False, "caps must be AdapterCaps"))
        return ConformanceReport(type(adapter).__name__, tuple(checks))
    required = ("prepare_frame", "nodes", "camera_hint", "release", "step", "reset", "set_paused")
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    checks.append(
        ConformanceCheck(
            "editor runtime methods",
            not missing,
            "all present" if not missing else f"missing {missing}",
        )
    )
    checks.extend(_check_capability_implementations(adapter, caps))
    if source is not None and frame is not None:
        groups = (
            ("camera metadata", ("cameras",), _check_cameras, (adapter, caps, source)),
            (
                "simulation metadata",
                ("sensors", "equality_constraints", "actuators", "timestep"),
                _check_simulation,
                (adapter, caps, frame),
            ),
            ("keyframe metadata", ("keyframes",), _check_keyframes, (adapter, caps)),
        )
        for name, methods, check, args in groups:
            missing = [method for method in methods if not callable(getattr(adapter, method, None))]
            if missing:
                checks.append(ConformanceCheck(name, False, f"missing {missing}"))
                continue
            # Keep broken third-party metadata visible as a named diagnostic.
            try:
                checks.extend(check(*args))
            except Exception as exc:
                checks.append(ConformanceCheck(name, False, f"{type(exc).__name__}: {exc}"))
    return ConformanceReport(caps.name, tuple(checks))


def _check_cameras(adapter: SceneAppearance, caps: AdapterCaps, source: SceneSource):
    cameras = adapter.cameras() if caps.model_cameras else []
    ok = len(cameras) == len(source.cameras) and len(
        {camera.camera_id for camera in cameras}
    ) == len(cameras)
    return (ConformanceCheck("camera metadata", ok, f"{len(cameras)} cameras"),)


def _check_keyframes(adapter: KeyframeCatalog, caps: AdapterCaps):
    checks = []

    def add(name, ok, detail):
        checks.append(ConformanceCheck(name, bool(ok), detail))

    keyframes = adapter.keyframes()
    key_ids = [key.keyframe_id for key in keyframes]
    keyframes_ok = not caps.keyframes or len(key_ids) == len(set(key_ids))
    add("keyframe metadata", keyframes_ok, f"{len(keyframes)} keyframes")

    return checks


def _check_simulation(adapter: SimulationAccess, caps: AdapterCaps, frame: SceneFrame):
    checks = []

    def add(name, ok, detail):
        checks.append(ConformanceCheck(name, bool(ok), detail))

    sensors = adapter.sensors()
    sensor_values = frame.sensors
    sensors_ok = not caps.sensors or sensor_values is not None
    if sensor_values is not None:
        sensors_ok &= all(info.data_adr + info.dim <= len(sensor_values) for info in sensors)
    add("sensor frame", sensors_ok, f"{len(sensors)} sensors")

    equalities = adapter.equality_constraints()
    equality_ids = [constraint.constraint_id for constraint in equalities]
    equalities_ok = not caps.equality_constraints or (
        frame.equality_enabled is not None
        and len(frame.equality_enabled) == len(equalities)
        and len(equality_ids) == len(set(equality_ids))
    )
    add("equality constraints", equalities_ok, f"{len(equalities)} constraints")

    actuators = adapter.actuators()
    controls_ok = not actuators or (
        frame.ctrl is not None
        and frame.actuator_activation is not None
        and all(a.ctrl_address + a.ctrl_count <= len(frame.ctrl) for a in actuators)
        and len(frame.actuator_activation) == len(actuators)
    )
    add("actuator frame", controls_ok, f"{len(actuators)} actuators")

    dt = float(adapter.timestep())
    timing_ok = np.isfinite(dt) and (dt > 0.0 if caps.simulation else dt >= 0.0)
    add("simulation timing", timing_ok, f"simulation={caps.simulation}, dt={dt:g}")
    return checks


# Only methods whose Base implementation explicitly cannot perform the advertised
# write belong here. Empty inventories and functional inherited defaults are valid.
_CAPABILITY_WRITES = {
    "asset_loading": ("load",),
    "reload": ("reload",),
    "write_pose": ("set_pose",),
    "write_scale": ("set_scale",),
    "write_qpos": ("set_qpos", "set_qpos_batch"),
    "write_ctrl": ("set_ctrl", "set_ctrl_vector"),
    "perturb": ("apply_perturb", "clear_perturb"),
    "raycast": ("raycast",),
    "state_snapshots": ("capture_state", "restore_state"),
    "equality_constraints": ("set_equality_enabled",),
    "visual_groups": ("set_visual_group",),
    "keyframes": ("load_keyframe",),
    "edit_history": ("capture_edit_state", "restore_edit_state"),
    "scene_new": ("new_scene",),
    "scene_open": ("open_scene",),
    "scene_save": ("save_scene",),
    "document_checkpoints": ("capture_document_state", "restore_document_state"),
    "scene_authoring": (
        "add_scene_object",
        "remove_scene_object",
        "add_scene_light",
        "remove_scene_light",
        "add_scene_camera",
        "remove_scene_camera",
        "duplicate_scene_entity",
        "remove_scene_entity",
        "rename_scene_entity",
    ),
    "model_composition": (
        "add_scene_model",
        "remove_scene_model",
        "set_scene_model_transform",
        "preview_scene_model_transform",
        "clear_scene_model_transform_preview",
    ),
    "topology_editing": (
        "add_model_element",
        "duplicate_model_element",
        "remove_model_element",
        "rename_model_element",
        "apply_model_edit_batch",
    ),
    "model_properties": (
        "set_joint_properties",
        "set_joint_advanced_properties",
        "set_site_properties",
        "set_geometry_properties",
        "set_geometry_advanced_properties",
        "set_geometry_shape",
        "set_body_properties",
    ),
    "model_assets": (
        "import_model_geometry_resource",
        "import_model_asset",
        "set_height_field_size",
        "rename_model_asset",
        "duplicate_model_asset",
        "replace_model_asset_file",
        "remove_model_asset",
        "create_model_material",
        "add_model_material",
        "import_model_texture",
        "set_geometry_material",
    ),
    "mujoco.mjcf": ("set_scene_model_xml",),
    "model.components": ("add_model_component", "update_model_component", "remove_model_component"),
    "model.keyframe_edit": (
        "add_model_keyframe",
        "set_keyframe_properties",
        "remove_model_keyframe",
    ),
}


def _check_capability_implementations(adapter: SceneAdapter, caps: AdapterCaps):
    contracts = [
        (name, methods) for name, methods in _CAPABILITY_WRITES.items() if caps.supports(name)
    ]
    if caps.simulation and caps.clock_control:
        contracts.append(("simulation", ("step", "reset")))
    for feature, methods in contracts:
        missing = []
        for name in methods:
            method = getattr(adapter, name, None)
            if not callable(method) or getattr(method, "__func__", method) is getattr(
                SceneAdapterBase, name
            ):
                missing.append(name)
        yield ConformanceCheck(
            f"capability {feature}",
            not missing,
            f"unsupported defaults or missing methods: {', '.join(missing)}"
            if missing
            else "declared methods implemented",
        )
