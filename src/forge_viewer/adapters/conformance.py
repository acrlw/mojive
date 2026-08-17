"""Backend-neutral structural checks for third-party SceneAdapter implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import MeshShape
from .base import FrameNeeds, SceneAdapter


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class ConformanceReport:
    backend: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)


def check_adapter(adapter: SceneAdapter) -> ConformanceReport:
    """Exercise structure and one full-data frame without assuming a physics library."""
    revision = adapter.structure_revision
    source = adapter.scene_source()
    frame = adapter.frame(
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
        )
    )
    checks: list[ConformanceCheck] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(ConformanceCheck(name, bool(ok), detail))

    required = ("scene_source", "frame", "step", "reset", "set_paused", "release")
    missing = [name for name in required if not callable(getattr(adapter, name, None))]
    add("required methods", not missing, "all present" if not missing else f"missing {missing}")

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

    cameras = adapter.cameras() if adapter.caps.model_cameras else []
    camera_nodes = {node.camera_index for node in source.nodes if node.camera_index >= 0}
    cameras_ok = camera_nodes == set(range(len(source.cameras)))
    cameras_ok &= len(cameras) == len(source.cameras)
    cameras_ok &= len({camera.camera_id for camera in cameras}) == len(cameras)
    if frame.cameras is not None:
        cameras_ok &= len(frame.cameras) == len(source.cameras)
    add("camera entities", cameras_ok, f"{len(source.cameras)} cameras, {len(camera_nodes)} nodes")

    mesh_ok = True
    mesh_detail = []
    for key in source.geom_mesh:
        if key.shape in (MeshShape.ASSET, MeshShape.HEIGHTFIELD, MeshShape.FLEX, MeshShape.SKIN):
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
        update.positions.shape == source.meshes[key].positions.shape
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

    sensors = adapter.sensors()
    sensor_values = frame.sensors
    sensors_ok = not adapter.caps.sensors or sensor_values is not None
    if sensor_values is not None:
        sensors_ok &= all(info.data_adr + info.dim <= len(sensor_values) for info in sensors)
    add("sensor frame", sensors_ok, f"{len(sensors)} sensors")

    keyframes = adapter.keyframes()
    key_ids = [key.keyframe_id for key in keyframes]
    keyframes_ok = not adapter.caps.keyframes or len(key_ids) == len(set(key_ids))
    add("keyframe metadata", keyframes_ok, f"{len(keyframes)} keyframes")

    equalities = adapter.equality_constraints()
    equality_ids = [constraint.constraint_id for constraint in equalities]
    equalities_ok = not adapter.caps.equality_constraints or (
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
    timing_ok = np.isfinite(dt) and (dt > 0.0 if adapter.caps.simulation else dt >= 0.0)
    add("simulation timing", timing_ok, f"simulation={adapter.caps.simulation}, dt={dt:g}")
    add(
        "stable structure",
        adapter.structure_revision == revision,
        f"revision={adapter.structure_revision}",
    )
    return ConformanceReport(adapter.caps.name, tuple(checks))
