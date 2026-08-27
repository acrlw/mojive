"""Remote scene snapshots with reliable structure, latest-only frames, and commands."""

from __future__ import annotations

import contextlib
import pickle
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Client, Connection, Listener
from typing import Any

import numpy as np

from . import commands as cmd
from .adapters.base import (
    ActuatorInfo,
    AdapterCaps,
    CameraInfo,
    EqualityConstraintInfo,
    FrameNeeds,
    JointInfo,
    KeyframeInfo,
    SceneAdapterBase,
    SceneFrame,
    SceneNode,
    SceneSource,
    SensorInfo,
    VisualGroupInfo,
)
from .commands import CommandResult
from .types import CameraView, Environment, Light, Material

DEFAULT_PORT = 47650
AUTHKEY = b"forge-viewer-local"


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
    )


class _LatestSender:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        self._reliable: deque[bytes] = deque()
        self._latest: bytes | None = None
        self._condition = threading.Condition()
        self.closed = False
        threading.Thread(target=self._run, name="forge-remote-send", daemon=True).start()

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
        self.connection.close()


@dataclass
class _CommandRequest:
    payload: dict
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: Any = None


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
        threading.Thread(target=self._accept_state, name="forge-remote-state", daemon=True).start()
        threading.Thread(
            target=self._accept_commands, name="forge-remote-command", daemon=True
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
            if request.cancelled.is_set():
                request.done.set()
                count += 1
                continue
            try:
                request.result = handler(request.payload)
            except Exception as exc:
                request.result = CommandResult.bad(f"remote command failed: {exc}")
            request.done.set()
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
                name="forge-remote-command-client",
                daemon=True,
            ).start()

    def _command_client(self, connection: Connection) -> None:
        try:
            while not self._closed:
                request = _CommandRequest(connection.recv())
                self._commands.put(request)
                if not request.done.wait(self._command_timeout):
                    request.cancelled.set()
                    connection.send(CommandResult.bad("remote command timed out"))
                else:
                    connection.send(request.result)
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
                connection.close()


class RemoteSceneAdapter(SceneAdapterBase):
    """Consumes a publisher while continuing through the normal Session/forge/UI path."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 8.0
    ) -> None:
        self.host, self.port = host, int(port)
        self._lock = threading.Condition()
        self._structure: RemoteStructure | None = None
        self._camera_slot_by_id: dict[int, int] = {}
        self._latest: RemoteFrame | None = None
        self._delivered_sequence = -1
        self._error = ""
        self._closed = False
        self._timeout = float(timeout)
        self._command_lock = threading.Lock()
        self._state: Connection | None = None
        self._command: Connection | None = None
        try:
            self._state = self._connect((host, self.port), timeout)
            self._command = self._connect((host, self.port + 1), timeout)
            threading.Thread(target=self._receive, name="forge-remote-receive", daemon=True).start()
            self._wait(lambda: self._structure is not None, timeout, "scene structure")
        except Exception:
            self.release()
            raise
        caps = self._structure.caps
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
            notes=(*caps.notes, f"attached to {host}:{port}"),
        )

    @staticmethod
    def _connect(address, timeout: float) -> Connection:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return Client(address, authkey=AUTHKEY)
            except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
                last_error = exc
                time.sleep(0.05)
        raise ConnectionError(
            f"cannot connect to remote viewer publisher at {address}: {last_error}"
        )

    def _receive(self) -> None:
        try:
            while not self._closed:
                if self._state is None:
                    return
                packet = pickle.loads(self._state.recv_bytes())
                with self._lock:
                    if isinstance(packet, RemoteStructure):
                        self._structure = packet
                        self._camera_slot_by_id = {
                            camera.camera_id: slot for slot, camera in enumerate(packet.cameras)
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
                self._error = str(exc)
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

    def load_keyframe(self, keyframe_id: int) -> bool:
        return self._ok(self._send("keyframe", keyframe_id=int(keyframe_id)))

    def camera_hint(self) -> CameraView | None:
        return self._structure.camera_hint

    def timestep(self) -> float:
        return self._structure.timestep

    def visual_groups(self) -> tuple[VisualGroupInfo, ...]:
        return self._structure.visual_groups

    def _send(self, op: str, **args):
        with self._command_lock:
            try:
                if self._command is None:
                    return CommandResult.bad("remote command channel is closed")
                self._command.send({"op": op, **args})
                return self._command.recv()
            except (EOFError, OSError):
                return CommandResult.bad("remote command channel is closed")

    @staticmethod
    def _ok(result) -> bool:
        return bool(result.ok) if isinstance(result, CommandResult) else bool(result)

    def set_paused(self, paused: bool) -> bool:
        return self._ok(self._send("pause" if paused else "play"))

    def step(self, count: int = 1) -> None:
        self._send("step", count=int(count))

    def reset(self) -> None:
        self._send("reset")

    def reload(self) -> None:
        self._send("reload")

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
        self._closed = True
        if self._state is not None:
            with contextlib.suppress(OSError):
                self._state.close()
            self._state = None
        if self._command is not None:
            with contextlib.suppress(OSError):
                self._command.close()
            self._command = None


def handle_session_command(session, message: dict):
    """Translate one remote command payload into a session command or query."""
    op = message.get("op")
    if op == "raycast":
        return session.query(cmd.Pick(message["origin"], message["direction"]))
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
        "equality": lambda: cmd.SetEqualityEnabled(message["constraint_id"], message["enabled"]),
        "ctrl": lambda: cmd.SetCtrl(message["index"], message["value"]),
        "pose": lambda: cmd.SetPose(message["node_id"], message["position"], message["rotation"]),
        "light": lambda: cmd.SetLight(
            message.get("light_index", message.get("light_id")), message["light"]
        ),
        "environment": lambda: cmd.SetEnvironment(message["environment"]),
        "skybox": lambda: cmd.SetSkybox(message.get("texture")),
        "material": lambda: cmd.SetMaterial(
            message.get("material_index", message.get("material_id")), message["material"]
        ),
        "geometry_color": lambda: cmd.SetGeometryColor(message["node_id"], message["rgba"]),
        "geometry_size": lambda: cmd.SetGeometrySize(message["node_id"], message["size"]),
        "scene_camera": lambda: cmd.SetSceneCamera(message["camera_id"], message["camera"]),
        "add_scene_object": lambda: cmd.AddSceneObject(
            message["shape"],
            message["name"],
            message["size"],
            message["position"],
            message["rotation"],
            message["color"],
            message["material"],
        ),
        "remove_scene_object": lambda: cmd.RemoveSceneObject(message["object_id"]),
        "add_scene_light": lambda: cmd.AddSceneLight(message["name"], message["light"]),
        "remove_scene_light": lambda: cmd.RemoveSceneLight(message["light_id"]),
        "add_scene_camera": lambda: cmd.AddSceneCamera(message["name"], message["camera"]),
        "remove_scene_camera": lambda: cmd.RemoveSceneCamera(message["camera_id"]),
        "duplicate_scene_entity": lambda: cmd.DuplicateSceneEntity(message["object_id"]),
        "remove_scene_entity": lambda: cmd.RemoveSceneEntity(message["object_id"]),
        "rename_scene_entity": lambda: cmd.RenameSceneEntity(message["object_id"], message["name"]),
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
