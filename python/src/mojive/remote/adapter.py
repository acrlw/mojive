"""Remote: adapter."""

from __future__ import annotations

import contextlib
import math
import pickle
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge

import numpy as np

from mojive.adapters.base import (
    ActuatorInfo,
    AdapterCaps,
    BodyProperties,
    CameraInfo,
    EqualityConstraintInfo,
    FrameNeeds,
    GeometryAdvancedProperties,
    GeometryProperties,
    GeometryShapeProperties,
    JointAdvancedProperties,
    JointInfo,
    KeyframeInfo,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
    SiteProperties,
    VisualGroupInfo,
)
from mojive.commands import CommandResult
from mojive.types import CameraView, Environment, Light, Material

from .protocol import (
    _REMOTE_REQUIREMENTS,
    AUTHKEY,
    DEFAULT_PORT,
    STREAM_PROTOCOL_VERSION,
    RemoteFrame,
    RemoteStructure,
    _close_connection,
    remote_command_versions,
)


class RemoteSceneAdapter(SceneAdapterBase):
    """Consume a publisher through the normal session, renderer, and UI path."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 8.0
    ) -> None:
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Remote timeout must be finite and positive")
        self.host, self.port = host, int(port)
        self._lock = threading.Condition()
        self._structure: RemoteStructure | None = None
        self._camera_slot_by_id: dict[int, int] = {}
        self._geometry_properties_by_node: dict[int, GeometryProperties] = {}
        self._body_properties_by_node: dict[int, BodyProperties] = {}
        self._geometry_advanced_properties_by_node: dict[int, GeometryAdvancedProperties] = {}
        self._geometry_shape_properties_by_node: dict[int, GeometryShapeProperties] = {}
        self._joint_advanced_properties_by_id: dict[int, JointAdvancedProperties] = {}
        self._site_properties_by_node: dict[int, SiteProperties] = {}
        self._latest: RemoteFrame | None = None
        self._delivered_sequence = -1
        self._error = ""
        self._closed = False
        self._timeout = float(timeout)
        self._command_lock = threading.Lock()
        self._command_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mojive-remote-control"
        )
        self._state: Connection | None = None
        self._command: Connection | None = None
        try:
            self._state = self._connect((host, self.port), timeout)
            self._command = self._connect((host, self.port + 1), timeout)
            threading.Thread(
                target=self._receive, name="mojive-remote-receive", daemon=True
            ).start()
            self._wait(lambda: self._structure is not None, timeout, "scene structure")
        except Exception:
            self.release()
            raise

    def _update_capabilities(self, caps: AdapterCaps) -> None:
        advertised = dict(self._structure.command_versions)
        supported = dict(remote_command_versions(caps))
        self._command_versions = {
            name: version
            for name, version in advertised.items()
            if type(version) is int and supported.get(name) == version
        }
        writable = {feature for feature in _REMOTE_REQUIREMENTS.values() if feature}
        gated = {
            feature: caps.supports(feature)
            and all(
                name in self._command_versions
                for name, requirement in _REMOTE_REQUIREMENTS.items()
                if requirement == feature
            )
            for feature in writable
            if feature != "simulation"
        }
        if caps.simulation:
            gated["clock_control"] = caps.clock_control and all(
                name in self._command_versions for name in ("pause", "play", "step", "reset")
            )
        caps = replace(caps, **gated)
        self.caps = replace(
            caps,
            name=f"remote:{caps.name}",
            asset_loading=False,
            model_formats=(),
            features=(),
            external_clock=True,
            state_snapshots=False,
            edit_history=False,
            model_composition=False,
            model_cameras=caps.model_cameras,
            scene_files=False,
            topology_editing=False,
            model_assets=False,
            notes=(*caps.notes, f"attached to {self.host}:{self.port}"),
        )

    def _connect(self, address, timeout: float) -> Connection:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            connection = None
            try:
                with socket.create_connection(address, timeout=deadline - time.monotonic()) as peer:
                    peer.setblocking(True)
                    connection = Connection(peer.detach())
                handshake = self._command_executor.submit(self._authenticate, connection)
                handshake.result(timeout=max(0.0, deadline - time.monotonic()))
                return connection
            except TimeoutError as exc:
                last_error = exc
                if connection is not None:
                    _close_connection(connection)
                break
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                last_error = exc
                if connection is not None:
                    _close_connection(connection)
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
            except Exception:
                if connection is not None:
                    _close_connection(connection)
                raise
        raise ConnectionError(f"remote connection to {address} timed out: {last_error}")

    @staticmethod
    def _authenticate(connection: Connection) -> None:
        # Match multiprocessing Client's handshake while retaining ownership on timeout.
        answer_challenge(connection, AUTHKEY)
        deliver_challenge(connection, AUTHKEY)

    def _receive(self) -> None:
        try:
            while not self._closed:
                if self._state is None:
                    return
                packet = pickle.loads(self._state.recv_bytes())
                with self._lock:
                    if isinstance(packet, RemoteStructure):
                        if (
                            type(packet.protocol_version) is not int
                            or packet.protocol_version != STREAM_PROTOCOL_VERSION
                        ):
                            raise ValueError(
                                f"Unsupported remote stream version: {packet.protocol_version}"
                            )
                        self._structure = packet
                        self._update_capabilities(packet.caps)
                        self._camera_slot_by_id = {
                            camera.camera_id: slot for slot, camera in enumerate(packet.cameras)
                        }
                        self._geometry_properties_by_node = {
                            item.node_id: item for item in packet.geometry_properties
                        }
                        self._body_properties_by_node = {
                            item.node_id: item for item in packet.body_properties
                        }
                        self._geometry_advanced_properties_by_node = {
                            item.node_id: item for item in packet.geometry_advanced_properties
                        }
                        self._geometry_shape_properties_by_node = {
                            item.node_id: item for item in packet.geometry_shape_properties
                        }
                        self._joint_advanced_properties_by_id = {
                            item.joint_id: item for item in packet.joint_advanced_properties
                        }
                        self._site_properties_by_node = {
                            item.node_id: item for item in packet.site_properties
                        }
                        self._latest = None
                        self._delivered_sequence = -1
                    elif (
                        isinstance(packet, RemoteFrame)
                        and self._structure is not None
                        and packet.structure_revision == self._structure.structure_revision
                    ):
                        self._latest = packet
                    self._lock.notify_all()
        except (EOFError, OSError, pickle.PickleError, ValueError) as exc:
            with self._lock:
                self._error = str(exc) or type(exc).__name__
                self._lock.notify_all()

    def _wait(self, predicate, timeout: float, what: str) -> None:
        with self._lock:
            if not self._lock.wait_for(lambda: predicate() or self._error, timeout):
                raise TimeoutError(f"timed out waiting for remote {what}")
            if self._error:
                raise ConnectionError(
                    f"remote stream closed while waiting for {what}: {self._error}"
                )

    @property
    def structure_revision(self) -> int:
        with self._lock:
            return self._structure.structure_revision if self._structure is not None else -1

    def scene_source(self) -> SceneSource:
        self._wait(lambda: self._structure is not None, self._timeout, "scene structure")
        return self._structure.source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        del needs
        self._wait(lambda: self._latest is not None, self._timeout, "first frame")
        with self._lock:
            packet = self._latest
            packet.frame.debug_commands = (
                packet.debug_commands if packet.frame_sequence != self._delivered_sequence else None
            )
            self._delivered_sequence = packet.frame_sequence
            return packet.frame

    def nodes(self) -> list[SceneNode]:
        return self._structure.nodes

    def joints(self) -> list[JointInfo]:
        return self._structure.joints

    def actuators(self) -> list[ActuatorInfo]:
        return self._structure.actuators

    def cameras(self) -> list[CameraInfo]:
        return self._structure.cameras

    def camera_view(self, camera_id: int) -> CameraView | None:
        with self._lock:
            slot = self._camera_slot_by_id.get(int(camera_id), -1)
            if slot < 0:
                return None
            frame = self._latest.frame if self._latest is not None else None
            cameras = frame.cameras if frame is not None else None
            if cameras is not None and slot < len(cameras):
                return cameras[slot]
            source_cameras = self._structure.source.cameras
            return source_cameras[slot] if slot < len(source_cameras) else None

    def keyframes(self) -> list[KeyframeInfo]:
        return self._structure.keyframes

    def sensors(self) -> list[SensorInfo]:
        return self._structure.sensors

    def equality_constraints(self) -> list[EqualityConstraintInfo]:
        return self._structure.equality_constraints

    def geometry_properties(self, node_id: int) -> GeometryProperties | None:
        with self._lock:
            return self._geometry_properties_by_node.get(int(node_id))

    def body_properties(self, node_id: int) -> BodyProperties | None:
        with self._lock:
            return self._body_properties_by_node.get(int(node_id))

    def geometry_advanced_properties(self, node_id: int) -> GeometryAdvancedProperties | None:
        with self._lock:
            return self._geometry_advanced_properties_by_node.get(int(node_id))

    def geometry_shape_properties(self, node_id: int) -> GeometryShapeProperties | None:
        with self._lock:
            return self._geometry_shape_properties_by_node.get(int(node_id))

    def joint_advanced_properties(self, joint_id: int) -> JointAdvancedProperties | None:
        with self._lock:
            return self._joint_advanced_properties_by_id.get(int(joint_id))

    def site_properties(self, node_id: int) -> SiteProperties | None:
        with self._lock:
            return self._site_properties_by_node.get(int(node_id))

    def load_keyframe(self, keyframe_id: int) -> bool:
        return self._ok(self._send("keyframe", keyframe_id=int(keyframe_id)))

    def camera_hint(self) -> CameraView | None:
        return self._structure.camera_hint

    def timestep(self) -> float:
        return self._structure.timestep

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        return self._structure.visual_groups

    def _send(self, op: str, **args):
        versions = self._command_versions
        if versions.get(op) != 1:
            return CommandResult.bad(f"Remote publisher does not support {op} (revision 1)")
        deadline = time.monotonic() + self._timeout
        if not self._command_lock.acquire(timeout=self._timeout):
            return CommandResult.bad("remote command channel is busy; request was not sent")
        try:
            if self._command is None:
                return CommandResult.bad("remote command channel is closed")
            pending = self._command_executor.submit(
                self._exchange_command, self._command, {"op": op, "operation_version": 1, **args}
            )
            return pending.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            self._close_command_channel()
            return CommandResult.bad(
                "remote command timed out; completion unknown; inspect before retrying"
            )
        except (EOFError, OSError, RuntimeError):
            self._close_command_channel()
            return CommandResult.bad("remote command channel is closed; completion unknown")
        finally:
            self._command_lock.release()

    @staticmethod
    def _exchange_command(connection: Connection, payload: dict):
        # poll() alone cannot bound a blocked send or a reply with only its header received.
        connection.send(payload)
        return connection.recv()

    def _close_command_channel(self) -> None:
        connection, self._command = self._command, None
        if connection is not None:
            with contextlib.suppress(OSError):
                _close_connection(connection)

    @staticmethod
    def _ok(result) -> bool:
        return bool(result.ok) if isinstance(result, CommandResult) else bool(result)

    def set_paused(self, paused: bool) -> bool:
        return self._ok(self._send("pause" if paused else "play"))

    def _run_control_command(self, operation: str, **args) -> None:
        result = self._send(operation, **args)
        if not self._ok(result):
            message = (
                result.message
                if isinstance(result, CommandResult)
                else f"remote {operation} failed"
            )
            raise RuntimeError(message)

    def step(self, count: int = 1) -> None:
        self._run_control_command("step", count=int(count))

    def reset(self) -> None:
        self._run_control_command("reset")

    def reload(self) -> None:
        self._run_control_command("reload")

    def set_visual_group(self, category: str, group: int, visible: bool) -> bool:
        return self._ok(
            self._send(
                "visual_group",
                category=str(category),
                group=int(group),
                visible=bool(visible),
            )
        )

    def set_qpos(self, index: int, value: float) -> bool:
        return self._ok(self._send("qpos", index=int(index), value=float(value)))

    def set_qpos_batch(self, indices: np.ndarray, values: np.ndarray) -> bool:
        return self._ok(
            self._send(
                "qpos_batch",
                indices=np.asarray(indices, np.intp),
                values=np.asarray(values, np.float64),
            )
        )

    def set_equality_enabled(self, constraint_id: int, enabled: bool) -> bool:
        return self._ok(
            self._send(
                "equality",
                constraint_id=int(constraint_id),
                enabled=bool(enabled),
            )
        )

    def set_ctrl(self, index: int, value: float) -> bool:
        return self._ok(self._send("ctrl", index=int(index), value=float(value)))

    def set_ctrl_vector(self, values: np.ndarray) -> bool:
        return self._ok(self._send("ctrl_vector", values=np.asarray(values, np.float64)))

    def set_pose(self, node_id: int, position, rotation) -> bool:
        return self._ok(
            self._send(
                "pose",
                node_id=int(node_id),
                position=np.asarray(position, np.float32),
                rotation=np.asarray(rotation, np.float32),
            )
        )

    def set_light(self, light_index: int, light: Light) -> bool:
        return self._ok(self._send("light", light_index=int(light_index), light=light))

    def set_environment(self, environment: Environment) -> bool:
        return self._ok(self._send("environment", environment=environment))

    def set_skybox(self, texture: str | None) -> bool:
        return self._ok(self._send("skybox", texture=texture))

    def set_material(self, material_index: int, material: Material) -> bool:
        return self._ok(
            self._send("material", material_index=int(material_index), material=material)
        )

    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool:
        return self._ok(
            self._send(
                "geometry_color",
                node_id=int(node_id),
                rgba=np.asarray(rgba, np.float32),
            )
        )

    def set_geometry_size(self, node_id: int, size: np.ndarray) -> bool:
        return self._ok(
            self._send_structure_edit(
                "geometry_size",
                node_id=int(node_id),
                size=np.asarray(size, np.float32),
            )
        )

    def set_joint_properties(
        self,
        joint_id: int,
        axis: np.ndarray,
        limited: bool,
        value_range: tuple[float, float],
        damping: float,
        stiffness: float,
    ) -> bool:
        return self._ok(
            self._send_structure_edit(
                "joint_properties",
                joint_id=int(joint_id),
                axis=np.asarray(axis, np.float64),
                limited=bool(limited),
                range=tuple(float(value) for value in value_range),
                damping=float(damping),
                stiffness=float(stiffness),
            )
        )

    def set_joint_advanced_properties(self, properties: JointAdvancedProperties) -> bool:
        return self._ok(
            self._send_structure_edit(
                "joint_advanced_properties",
                joint_id=int(properties.joint_id),
                group=int(properties.group),
                armature=float(properties.armature),
                friction_loss=float(properties.friction_loss),
                reference=float(properties.reference),
                spring_reference=float(properties.spring_reference),
                margin=float(properties.margin),
                limit_solver_reference=properties.limit_solver_reference,
                limit_solver_impedance=properties.limit_solver_impedance,
                friction_solver_reference=properties.friction_solver_reference,
                friction_solver_impedance=properties.friction_solver_impedance,
                actuator_force_limit_mode=properties.actuator_force_limit_mode,
                actuator_force_range=properties.actuator_force_range,
                actuator_gravity_compensation=properties.actuator_gravity_compensation,
            )
        )

    def set_site_properties(self, properties: SiteProperties) -> bool:
        return self._ok(
            self._send_structure_edit(
                "site_properties",
                node_id=int(properties.node_id),
                type=properties.type,
                group=int(properties.group),
                use_from_to=bool(properties.use_from_to),
                from_to=properties.from_to,
            )
        )

    def set_geometry_properties(self, properties: GeometryProperties) -> bool:
        return self._ok(
            self._send_structure_edit(
                "geometry_properties",
                node_id=int(properties.node_id),
                friction=properties.friction,
                collision_type_mask=int(properties.collision_type_mask),
                collision_affinity_mask=int(properties.collision_affinity_mask),
                contact_dimension=int(properties.contact_dimension),
                contact_priority=int(properties.contact_priority),
                margin=float(properties.margin),
                gap=float(properties.gap),
                solver_mix=float(properties.solver_mix),
                solver_reference=properties.solver_reference,
                solver_impedance=properties.solver_impedance,
                adhesion=float(properties.adhesion),
                surface_velocity=properties.surface_velocity,
            )
        )

    def set_geometry_advanced_properties(self, properties: GeometryAdvancedProperties) -> bool:
        return self._ok(
            self._send_structure_edit(
                "geometry_advanced_properties",
                node_id=int(properties.node_id),
                visual_group=int(properties.visual_group),
                mass_mode=properties.mass_mode,
                mass=float(properties.mass),
                density=float(properties.density),
                inertia_mode=properties.inertia_mode,
                fluid_ellipsoid=bool(properties.fluid_ellipsoid),
                fluid_coefficients=properties.fluid_coefficients,
            )
        )

    def set_geometry_shape(self, node_id: int, geom_type: str, resource_name: str) -> bool:
        return self._ok(
            self._send_structure_edit(
                "geometry_shape",
                node_id=int(node_id),
                type=str(geom_type),
                resource_name=str(resource_name),
            )
        )

    def set_body_properties(self, properties: BodyProperties) -> bool:
        return self._ok(
            self._send_structure_edit(
                "body_properties",
                node_id=int(properties.node_id),
                inertia_mode=properties.inertia_mode,
                mass=float(properties.mass),
                inertial_position=properties.inertial_position,
                inertial_quaternion=properties.inertial_quaternion,
                diagonal_inertia=properties.diagonal_inertia,
                full_inertia=properties.full_inertia,
                gravity_compensation=float(properties.gravity_compensation),
                mocap=bool(properties.mocap),
                sleep_policy=properties.sleep_policy,
            )
        )

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        return self._ok(self._send("scene_camera", camera_id=int(camera_id), camera=camera))

    def _send_structure_edit(self, op: str, **args) -> CommandResult:
        revision = self.structure_revision
        result = self._send(op, **args)
        if isinstance(result, CommandResult) and result.ok:
            self._wait(
                lambda: self.structure_revision != revision,
                self._timeout,
                "scene structure update",
            )
        return result

    def add_scene_object(self, shape, name, size, position, rotation, color, material) -> int:
        result = self._send_structure_edit(
            "add_scene_object",
            shape=shape,
            name=str(name),
            size=tuple(size),
            position=tuple(position),
            rotation=np.asarray(rotation, np.float32),
            color=tuple(color),
            material=material,
        )
        return result.entity_id if result.ok else -1

    def remove_scene_object(self, object_id: int) -> bool:
        return self._ok(self._send_structure_edit("remove_scene_object", object_id=int(object_id)))

    def add_scene_light(self, name: str, light: Light) -> int:
        result = self._send_structure_edit("add_scene_light", name=str(name), light=light)
        return result.entity_id if result.ok else -1

    def remove_scene_light(self, light_id: int) -> bool:
        return self._ok(self._send_structure_edit("remove_scene_light", light_id=int(light_id)))

    def add_scene_camera(self, name: str, camera: CameraView) -> int:
        result = self._send_structure_edit("add_scene_camera", name=str(name), camera=camera)
        return result.entity_id if result.ok else -1

    def remove_scene_camera(self, camera_id: int) -> bool:
        return self._ok(self._send_structure_edit("remove_scene_camera", camera_id=int(camera_id)))

    def duplicate_scene_entity(self, object_id: int) -> int:
        result = self._send_structure_edit("duplicate_scene_entity", object_id=int(object_id))
        return result.entity_id if result.ok else 0

    def remove_scene_entity(self, object_id: int) -> bool:
        return self._ok(self._send_structure_edit("remove_scene_entity", object_id=int(object_id)))

    def rename_scene_entity(self, object_id: int, name: str) -> bool:
        return self._ok(
            self._send_structure_edit(
                "rename_scene_entity", object_id=int(object_id), name=str(name)
            )
        )

    def apply_perturb(self, node_id: int, target_position, target_rotation, mode: str) -> bool:
        return self._ok(
            self._send(
                "perturb",
                node_id=int(node_id),
                target_position=np.asarray(target_position, np.float32),
                target_rotation=np.asarray(target_rotation, np.float32),
                mode=str(mode),
            )
        )

    def clear_perturb(self) -> None:
        self._send("clear_perturb")

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> tuple[int, float]:
        result = self._send(
            "raycast",
            origin=np.asarray(origin, np.float64),
            direction=np.asarray(direction, np.float64),
        )
        return tuple(result) if isinstance(result, tuple) else (0, float("inf"))

    def release(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._closed = True
            self._error = self._error or "adapter released"
            self._lock.notify_all()
        if self._state is not None:
            with contextlib.suppress(OSError):
                _close_connection(self._state)
            self._state = None
        self._close_command_channel()
        self._command_executor.shutdown(wait=False, cancel_futures=True)
