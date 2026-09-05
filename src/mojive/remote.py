"""Remote scene snapshots with reliable structure, latest-only frames, and commands."""

from __future__ import annotations

import contextlib
import math
import pickle
import queue
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection, Listener, answer_challenge, deliver_challenge
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import (
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
from .commands import CommandResult
from .types import CameraView, Environment, Light, Material

DEFAULT_PORT = 47650
AUTHKEY = b"mojive-local"


def _close_connection(connection: Connection) -> None:
    """Interrupt pending TCP reads/writes before releasing the Connection's descriptor."""
    with contextlib.suppress(OSError):
        transport = socket.socket(fileno=connection.fileno())
        try:
            transport.shutdown(socket.SHUT_RDWR)
        finally:
            # Connection owns the descriptor; the temporary socket only shuts it down.
            transport.detach()
    connection.close()


@dataclass(frozen=True)
class RemoteStructure:
    """Reliable stable structure sent when a scene revision changes."""

    structure_revision: int
    source: SceneSource
    caps: AdapterCaps
    nodes: list[SceneNode]
    joints: list[JointInfo]
    actuators: list[ActuatorInfo]
    cameras: list[CameraInfo]
    keyframes: list[KeyframeInfo]
    sensors: list[SensorInfo]
    equality_constraints: list[EqualityConstraintInfo]
    camera_hint: CameraView | None
    timestep: float
    visual_groups: tuple[VisualGroupInfo, ...] = ()
    geometry_properties: tuple[GeometryProperties, ...] = ()
    body_properties: tuple[BodyProperties, ...] = ()
    geometry_advanced_properties: tuple[GeometryAdvancedProperties, ...] = ()
    geometry_shape_properties: tuple[GeometryShapeProperties, ...] = ()
    joint_advanced_properties: tuple[JointAdvancedProperties, ...] = ()
    site_properties: tuple[SiteProperties, ...] = ()


@dataclass(frozen=True)
class RemoteFrame:
    """Latest-only dynamic scene frame and its debug commands."""

    frame_sequence: int
    frame: SceneFrame
    debug_commands: tuple[dict, ...] = ()
    structure_revision: int = -1


def snapshot_structure(session) -> RemoteStructure:
    """Capture adapter structure and metadata from a session."""
    return RemoteStructure(
        session.structure_generation,
        session.source,
        session.adapter.caps,
        session.nodes,
        session.joints,
        session.actuators,
        session.cameras,
        session.keyframes,
        session.sensor_infos,
        session.equality_constraints,
        session.camera_hint(),
        session.adapter.timestep(),
        session.visual_groups(),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.body_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_advanced_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.geometry_shape_properties(node.node_id)) is not None
        ),
        tuple(
            properties
            for joint in session.joints
            if (properties := session.joint_advanced_properties(joint.joint_id)) is not None
        ),
        tuple(
            properties
            for node in session.nodes
            if (properties := session.site_properties(node.node_id)) is not None
        ),
    )


class _LatestSender:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._reliable: deque[bytes] = deque()
        self._latest: bytes | None = None
        self._condition = threading.Condition()
        self.closed = False
        threading.Thread(target=self._run, name="mojive-remote-send", daemon=True).start()

    def reliable(self, payload: bytes, *, clear_latest: bool = False) -> None:
        with self._condition:
            if clear_latest:
                self._latest = None
            self._reliable.append(payload)
            self._condition.notify()

    def latest(self, payload: bytes) -> None:
        with self._condition:
            self._latest = payload
            self._condition.notify()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self.closed or self._reliable or self._latest is not None
                    )
                    if self.closed:
                        return
                    if self._reliable:
                        payload = self._reliable.popleft()
                    else:
                        payload = self._latest
                        self._latest = None
                self.connection.send_bytes(payload)
        except (EOFError, OSError):
            pass
        finally:
            self.close()

    def close(self) -> None:
        with self._condition:
            if self.closed:
                return
            self.closed = True
            self._condition.notify_all()
        _close_connection(self.connection)


@dataclass
class _CommandRequest:
    payload: dict
    result: Future = field(default_factory=Future)


class SnapshotPublisher:
    """Transport reliable scene structure and latest-only frames."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        *,
        command_timeout: float = 10.0,
    ) -> None:
        self.host, self.port = host, int(port)
        self.command_port = self.port + 1
        self._state_listener = Listener((host, self.port), authkey=AUTHKEY)
        self._command_listener = Listener((host, self.command_port), authkey=AUTHKEY)
        self._clients: list[_LatestSender] = []
        self._clients_lock = threading.Lock()
        self._command_clients: set[Connection] = set()
        self._command_clients_lock = threading.Lock()
        self._commands: queue.SimpleQueue[_CommandRequest] = queue.SimpleQueue()
        self._structure: bytes | None = None
        self._structure_revision = -1
        self._frame: bytes | None = None
        self._frame_sequence = 0
        self._command_timeout = max(0.0, float(command_timeout))
        self._closed = False
        threading.Thread(target=self._accept_state, name="mojive-remote-state", daemon=True).start()
        threading.Thread(
            target=self._accept_commands, name="mojive-remote-command", daemon=True
        ).start()

    def publish_structure(self, structure: RemoteStructure) -> None:
        """Reliably publish stable structure to current and future clients."""
        payload = pickle.dumps(structure, protocol=pickle.HIGHEST_PROTOCOL)
        self._structure = payload
        self._structure_revision = int(structure.structure_revision)
        self._frame = None
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        for client in clients:
            client.reliable(payload, clear_latest=True)

    def publish_frame(self, frame: SceneFrame, debug_commands=None) -> int:
        """Publish a latest-only frame and return its sequence number."""
        self._frame_sequence += 1
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        # Retain one bootstrap frame for a future viewer, but do not serialize
        # every training step while nobody is connected.
        if not clients and self._frame is not None:
            return self._frame_sequence
        commands = frame.debug_commands if debug_commands is None else debug_commands
        packet = RemoteFrame(
            frame_sequence=self._frame_sequence,
            frame=frame,
            debug_commands=tuple(commands or ()),
            structure_revision=self._structure_revision,
        )
        payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        self._frame = payload
        for client in clients:
            client.latest(payload)
        return self._frame_sequence

    def pump_commands(self, handler: Callable[[dict], Any], budget: int = 256) -> int:
        """Handle queued viewer commands on the publisher thread."""
        count = 0
        for _ in range(max(0, int(budget))):
            try:
                request = self._commands.get_nowait()
            except queue.Empty:
                break
            if not request.result.set_running_or_notify_cancel():
                count += 1
                continue
            try:
                result = handler(request.payload)
            except Exception as exc:
                result = CommandResult.bad(f"remote command failed: {exc}")
            request.result.set_result(result)
            count += 1
        return count

    def _accept_state(self) -> None:
        while not self._closed:
            try:
                connection = self._state_listener.accept()
            except (EOFError, OSError):
                return
            client = _LatestSender(connection)
            with self._clients_lock:
                self._clients.append(client)
            if self._structure is not None:
                client.reliable(self._structure)
            if self._frame is not None:
                client.latest(self._frame)

    def _accept_commands(self) -> None:
        while not self._closed:
            try:
                connection = self._command_listener.accept()
            except (EOFError, OSError):
                return
            with self._command_clients_lock:
                if self._closed:
                    connection.close()
                    return
                self._command_clients.add(connection)
            threading.Thread(
                target=self._command_client,
                args=(connection,),
                name="mojive-remote-command-client",
                daemon=True,
            ).start()

    def _command_client(self, connection: Connection) -> None:
        try:
            while not self._closed:
                request = _CommandRequest(connection.recv())
                self._commands.put(request)
                try:
                    result = request.result.result(timeout=self._command_timeout)
                except TimeoutError:
                    cancelled = request.result.cancel()
                    result = CommandResult.bad(
                        "remote command timed out before execution"
                        if cancelled
                        else "remote command timed out; completion unknown; inspect before retrying"
                    )
                connection.send(result)
        except (EOFError, OSError):
            pass
        finally:
            with self._command_clients_lock:
                self._command_clients.discard(connection)
            with contextlib.suppress(OSError):
                connection.close()

    def close(self) -> None:
        """Close listeners and all connected clients."""
        if self._closed:
            return
        self._closed = True
        self._state_listener.close()
        self._command_listener.close()
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()
        with self._command_clients_lock:
            command_clients, self._command_clients = self._command_clients, set()
        for connection in command_clients:
            with contextlib.suppress(OSError):
                _close_connection(connection)


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
        self.caps = replace(
            caps,
            name=f"remote:{caps.name}",
            asset_loading=False,
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
        except (EOFError, OSError, pickle.PickleError) as exc:
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
        deadline = time.monotonic() + self._timeout
        if not self._command_lock.acquire(timeout=self._timeout):
            return CommandResult.bad("remote command channel is busy; request was not sent")
        try:
            if self._command is None:
                return CommandResult.bad("remote command channel is closed")
            pending = self._command_executor.submit(
                self._exchange_command, self._command, {"op": op, **args}
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


def handle_session_command(session, message: dict):
    """Translate one remote command payload into a session command or query."""
    op = message.get("op")
    if op == "raycast":
        return session.query(cmd.Pick(message["origin"], message["direction"]))
    # Native publisher messages retain their established wire names. Their scene
    # edits share the application catalog's validation and command construction.
    from .control_errors import ControlError
    from .operations import apply_session_operation

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
                session, shared[op], {key: value for key, value in message.items() if key != "op"}
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
    factory = commands.get(str(op))
    return (
        session.submit(factory())
        if factory is not None
        else CommandResult.bad(f"unknown op {op!r}")
    )
