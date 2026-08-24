"""Remote scene snapshots with reliable structure, latest-only frames, and commands."""

from __future__ import annotations

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
    frame_sequence: int
    frame: SceneFrame
    debug_commands: tuple[dict, ...] = ()


def snapshot_structure(session) -> RemoteStructure:
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

    def reliable(self, payload: bytes) -> None:
        with self._condition:
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
    result: Any = None


class SnapshotPublisher:
    """Transport reliable scene structure and latest-only frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        self.host, self.port = host, int(port)
        self.command_port = self.port + 1
        self._state_listener = Listener((host, self.port), authkey=AUTHKEY)
        self._command_listener = Listener((host, self.command_port), authkey=AUTHKEY)
        self._clients: list[_LatestSender] = []
        self._clients_lock = threading.Lock()
        self._commands: queue.SimpleQueue[_CommandRequest] = queue.SimpleQueue()
        self._structure: bytes | None = None
        self._frame: bytes | None = None
        self._frame_sequence = 0
        self._closed = False
        threading.Thread(target=self._accept_state, name="forge-remote-state", daemon=True).start()
        threading.Thread(
            target=self._accept_commands, name="forge-remote-command", daemon=True
        ).start()

    def publish_structure(self, structure: RemoteStructure) -> None:
        payload = pickle.dumps(structure, protocol=pickle.HIGHEST_PROTOCOL)
        self._structure = payload
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        for client in clients:
            client.reliable(payload)

    def publish_frame(self, frame: SceneFrame, debug_commands=None) -> int:
        self._frame_sequence += 1
        commands = frame.debug_commands if debug_commands is None else debug_commands
        packet = RemoteFrame(self._frame_sequence, frame, tuple(commands or ()))
        payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        self._frame = payload
        with self._clients_lock:
            self._clients = [client for client in self._clients if not client.closed]
            clients = tuple(self._clients)
        for client in clients:
            client.latest(payload)
        return self._frame_sequence

    def pump_commands(self, handler: Callable[[dict], Any], budget: int = 256) -> int:
        count = 0
        for _ in range(max(0, int(budget))):
            try:
                request = self._commands.get_nowait()
            except queue.Empty:
                break
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
                if not request.done.wait(10.0):
                    connection.send(CommandResult.bad("remote command timed out"))
                else:
                    connection.send(request.result)
        except (EOFError, OSError):
            pass
        finally:
            connection.close()

    def close(self) -> None:
        self._closed = True
        self._state_listener.close()
        self._command_listener.close()
        with self._clients_lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.close()


class RemoteSceneAdapter(SceneAdapterBase):
    """Consumes a publisher while continuing through the normal Session/forge/UI path."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 8.0
    ) -> None:
        self.host, self.port = host, int(port)
        self._lock = threading.Condition()
        self._structure: RemoteStructure | None = None
        self._latest: RemoteFrame | None = None
        self._delivered_sequence = -1
        self._error = ""
        self._closed = False
        self._command_lock = threading.Lock()
        self._state = self._connect((host, self.port), timeout)
        self._command = self._connect((host, self.port + 1), timeout)
        threading.Thread(target=self._receive, name="forge-remote-receive", daemon=True).start()
        self._wait(lambda: self._structure is not None, timeout, "scene structure")
        caps = self._structure.caps
        self.caps = replace(
            caps,
            name=f"remote:{caps.name}",
            external_clock=True,
            edit_history=False,
            model_composition=False,
            model_cameras=bool(self._structure.cameras),
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
                packet = pickle.loads(self._state.recv_bytes())
                with self._lock:
                    if isinstance(packet, RemoteStructure):
                        self._structure = packet
                    elif isinstance(packet, RemoteFrame):
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
        self._wait(lambda: self._structure is not None, 8.0, "scene structure")
        return self._structure.source

    def frame(self, needs: FrameNeeds) -> SceneFrame:
        self._wait(lambda: self._latest is not None, 8.0, "first frame")
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
        slot = next(
            (
                slot
                for slot, camera in enumerate(self._structure.cameras)
                if camera.camera_id == int(camera_id)
            ),
            -1,
        )
        if slot < 0:
            return None
        with self._lock:
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

    def set_light(self, light_id: int, light: Light) -> bool:
        return self._ok(self._send("light", light_id=int(light_id), light=light))

    def set_environment(self, environment: Environment) -> bool:
        return self._ok(self._send("environment", environment=environment))

    def set_material(self, material_id: int, material: Material) -> bool:
        return self._ok(self._send("material", material_id=int(material_id), material=material))

    def set_geometry_color(self, node_id: int, rgba: np.ndarray) -> bool:
        return self._ok(
            self._send(
                "geometry_color",
                node_id=int(node_id),
                rgba=np.asarray(rgba, np.float32),
            )
        )

    def set_camera_view(self, camera_id: int, camera: CameraView) -> bool:
        return self._ok(self._send("scene_camera", camera_id=int(camera_id), camera=camera))

    def _send_structure_edit(self, op: str, **args) -> CommandResult:
        revision = self.structure_revision
        result = self._send(op, **args)
        if isinstance(result, CommandResult) and result.ok:
            self._wait(lambda: self.structure_revision != revision, 8.0, "scene structure update")
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
        self._closed = True
        self._state.close()
        self._command.close()


def handle_session_command(session, message: dict):
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
        "equality": lambda: cmd.SetEqualityEnabled(message["constraint_id"], message["enabled"]),
        "ctrl": lambda: cmd.SetCtrl(message["index"], message["value"]),
        "pose": lambda: cmd.SetPose(message["node_id"], message["position"], message["rotation"]),
        "light": lambda: cmd.SetLight(message["light_id"], message["light"]),
        "environment": lambda: cmd.SetEnvironment(message["environment"]),
        "material": lambda: cmd.SetMaterial(message["material_id"], message["material"]),
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
