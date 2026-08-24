"""Remote snapshots keep physics independent from viewer frame rate."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from forge_viewer import commands as cmd
from forge_viewer.adapters.base import (
    CAMERA_OBJECT_BASE,
    LIGHT_OBJECT_BASE,
    AdapterCaps,
    FrameNeeds,
    SceneAdapterBase,
)
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.commands import CommandResult
from forge_viewer.remote import (
    RemoteSceneAdapter,
    SnapshotPublisher,
    handle_session_command,
    snapshot_structure,
)
from forge_viewer.scene import Scene
from forge_viewer.session import Session
from forge_viewer.types import CameraView, Environment, Light, Material, MeshShape


def _port_pair() -> int:
    for port in range(47000, 49000, 2):
        sockets = []
        try:
            for candidate in (port, port + 1):
                sock = socket.socket()
                sock.bind(("127.0.0.1", candidate))
                sockets.append(sock)
            return port
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no free consecutive TCP ports")


def _eventually(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_publisher_delivers_structure_then_latest_frame_and_debug_once():
    scene = Scene()
    scene.box(name="remote box")
    source_session = Session(StaticSceneAdapter(scene))
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source_session))
    publisher.publish_frame(
        replace(scene.frame, step=7),
        ({"op": "text", "id": 9, "text": "remote"},),
    )

    remote = RemoteSceneAdapter(port=port)
    try:
        assert remote.scene_source().instance_count == 1
        first = remote.frame(FrameNeeds())
        assert first.step == 7
        assert first.debug_commands[0]["text"] == "remote"
        assert remote.frame(FrameNeeds()).debug_commands is None

        for step in range(8, 30):
            publisher.publish_frame(replace(scene.frame, step=step))
        latest = _eventually(
            lambda: frame if (frame := remote.frame(FrameNeeds())).step == 29 else None
        )
        assert latest.step == 29

        newer = replace(snapshot_structure(source_session), structure_revision=12)
        publisher.publish_structure(newer)
        assert _eventually(lambda: remote.structure_revision == 12)
    finally:
        remote.release()
        publisher.close()
        source_session.release()


def test_commands_use_a_separate_round_trip_channel():
    scene = Scene()
    scene.sphere()
    source_session = Session(StaticSceneAdapter(scene))
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source_session))
    publisher.publish_frame(scene.frame)
    remote = RemoteSceneAdapter(port=port)
    received = []
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            publisher.pump_commands(
                lambda message: received.append(message) or CommandResult.good("accepted")
            )
            stop.wait(0.002)

    worker = threading.Thread(target=pump)
    worker.start()
    try:
        assert remote.set_paused(True)
        assert received == [{"op": "pause"}]
        assert remote.set_environment(Environment(ambient=np.array([0.2, 0.3, 0.4], np.float32)))
        assert received[-1]["op"] == "environment"
        assert received[-1]["environment"].ambient == pytest.approx([0.2, 0.3, 0.4])
    finally:
        stop.set()
        worker.join()
        remote.release()
        publisher.close()
        source_session.release()


def test_keyframe_command_keeps_its_typed_remote_boundary():
    class Sink:
        def submit(self, command):
            return command

    command = handle_session_command(Sink(), {"op": "keyframe", "keyframe_id": 17})
    assert command == cmd.LoadKeyframe(17)


def test_scene_camera_command_keeps_its_typed_remote_boundary():
    class Sink:
        def submit(self, command):
            return command

    camera = CameraView()
    command = handle_session_command(
        Sink(), {"op": "scene_camera", "camera_id": 7, "camera": camera}
    )
    assert command == cmd.SetSceneCamera(7, camera)


def test_scene_entity_commands_keep_their_typed_remote_boundary():
    class Sink:
        def submit(self, command):
            return command

    light = Light(diffuse=np.array([0.2, 0.4, 0.8], np.float32))
    environment = Environment(ambient=np.array([0.1, 0.2, 0.3], np.float32))
    material = Material(name="remote", emission=0.4)
    color = np.array([0.3, 0.5, 0.7, 0.8], np.float32)

    command = handle_session_command(Sink(), {"op": "light", "light_id": 3, "light": light})
    assert isinstance(command, cmd.SetLight)
    assert command.light_id == 3
    assert command.light is light

    command = handle_session_command(Sink(), {"op": "environment", "environment": environment})
    assert isinstance(command, cmd.SetEnvironment)
    assert command.environment is environment

    command = handle_session_command(
        Sink(), {"op": "material", "material_id": 2, "material": material}
    )
    assert isinstance(command, cmd.SetMaterial)
    assert command.material_id == 2
    assert command.material is material

    command = handle_session_command(Sink(), {"op": "geometry_color", "node_id": 9, "rgba": color})
    assert isinstance(command, cmd.SetGeometryColor)
    assert command.node_id == 9
    assert np.array_equal(command.rgba, color)

    size = np.array((2.0, 3.0, 0.02), np.float32)
    command = handle_session_command(Sink(), {"op": "geometry_size", "node_id": 9, "size": size})
    assert isinstance(command, cmd.SetGeometrySize)
    assert command.node_id == 9
    assert np.array_equal(command.size, size)

    command = handle_session_command(
        Sink(),
        {
            "op": "add_scene_object",
            "shape": MeshShape.BOX,
            "name": "box",
            "size": (0.5, 0.5, 0.5),
            "position": (1.0, 2.0, 3.0),
            "rotation": np.eye(3, dtype=np.float32),
            "color": (0.2, 0.4, 0.8, 1.0),
            "material": material,
        },
    )
    assert isinstance(command, cmd.AddSceneObject)
    assert command.shape is MeshShape.BOX
    assert command.name == "box"


def test_remote_scene_authoring_publishes_structure_updates():
    scene = Scene()
    source = Session(StaticSceneAdapter(scene))
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source))
    publisher.publish_frame(source.frame)
    remote = Session(RemoteSceneAdapter(port=port))
    stop = threading.Event()

    def pump():
        generation = source.structure_generation
        while not stop.is_set():
            publisher.pump_commands(lambda message: handle_session_command(source, message))
            if source.structure_generation != generation:
                generation = source.structure_generation
                publisher.publish_structure(snapshot_structure(source))
                publisher.publish_frame(source.frame)
            stop.wait(0.002)

    worker = threading.Thread(target=pump)
    worker.start()
    try:
        added = remote.submit(
            cmd.AddSceneObject(MeshShape.SPHERE, "live sphere", position=(0.0, 0.0, 1.0))
        )
        assert added.ok and added.entity_id > 0
        assert remote.node_by_object_id(added.entity_id).name == "live sphere"
        assert source.node_by_object_id(added.entity_id).name == "live sphere"

        light = remote.submit(cmd.AddSceneLight("live light", Light()))
        camera = remote.submit(cmd.AddSceneCamera("live camera", CameraView()))
        assert remote.node_by_object_id(LIGHT_OBJECT_BASE + light.entity_id).name == "live light"
        assert remote.node_by_object_id(CAMERA_OBJECT_BASE + camera.entity_id).name == "live camera"

        assert remote.submit(cmd.RemoveSceneObject(added.entity_id))
        assert remote.submit(cmd.RemoveSceneLight(light.entity_id))
        assert remote.submit(cmd.RemoveSceneCamera(camera.entity_id))
        assert remote.node_by_object_id(added.entity_id) is None
        assert remote.node_by_object_id(LIGHT_OBJECT_BASE + light.entity_id) is None
        assert remote.node_by_object_id(CAMERA_OBJECT_BASE + camera.entity_id) is None
        assert source.node_by_object_id(added.entity_id) is None
    finally:
        stop.set()
        worker.join()
        remote.release()
        publisher.close()
        source.release()


def test_equality_command_keeps_its_typed_remote_boundary():
    class Sink:
        def submit(self, command):
            return command

    command = handle_session_command(
        Sink(), {"op": "equality", "constraint_id": 3, "enabled": False}
    )
    assert command == cmd.SetEqualityEnabled(3, False)


def test_remote_camera_metadata_and_edits_use_the_shared_scene_contract():
    scene = Scene(camera=CameraView())
    source_session = Session(StaticSceneAdapter(scene))
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source_session))
    publisher.publish_frame(source_session.frame)
    remote = Session(RemoteSceneAdapter(port=port))
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            publisher.pump_commands(lambda message: handle_session_command(source_session, message))
            stop.wait(0.002)

    worker = threading.Thread(target=pump)
    worker.start()
    try:
        assert remote.adapter.caps.model_cameras
        info = remote.cameras[0]
        assert remote.camera_view(info.camera_id) is not None
        edited = CameraView(eye=np.array([4.0, -3.0, 2.0], np.float32))
        assert remote.submit(cmd.SetSceneCamera(info.camera_id, edited))
        assert source_session.camera_view(info.camera_id).eye == pytest.approx(edited.eye)
        assert remote.frame.cameras == (edited,)
    finally:
        stop.set()
        worker.join()
        remote.release()
        publisher.close()
        source_session.release()


def test_two_viewers_receive_the_same_latest_frame_independently():
    scene = Scene()
    scene.box()
    source_session = Session(StaticSceneAdapter(scene))
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source_session))
    publisher.publish_frame(replace(scene.frame, step=1))
    effect = RemoteSceneAdapter(port=port)
    debug = RemoteSceneAdapter(port=port)
    try:
        publisher.publish_frame(replace(scene.frame, step=8))
        assert _eventually(lambda: effect.frame(FrameNeeds()).step == 8)
        assert _eventually(lambda: debug.frame(FrameNeeds()).step == 8)
    finally:
        effect.release()
        debug.release()
        publisher.close()
        source_session.release()


def test_pause_round_trip_changes_the_source_session():
    from forge_viewer.adapters.toy import ToyPhysicsAdapter

    source = Session(ToyPhysicsAdapter())
    port = _port_pair()
    publisher = SnapshotPublisher(port=port)
    publisher.publish_structure(snapshot_structure(source))
    publisher.publish_frame(source.frame)
    remote = Session(RemoteSceneAdapter(port=port))
    stop = threading.Event()

    def pump():
        while not stop.is_set():
            publisher.pump_commands(lambda message: handle_session_command(source, message))
            stop.wait(0.002)

    worker = threading.Thread(target=pump)
    worker.start()
    try:
        assert remote.submit(cmd.Pause())
        assert source.paused
        assert remote.paused
    finally:
        stop.set()
        worker.join()
        remote.release()
        publisher.close()
        source.release()


class _ExternalClock(SceneAdapterBase):
    caps = AdapterCaps(name="remote-test", simulation=True, external_clock=True)

    def __init__(self):
        self.steps = 0
        self.paused = False
        self.scene = Scene()
        self.scene.box()

    def scene_source(self):
        return self.scene.source

    def frame(self, needs):
        return replace(self.scene.frame, step=42, paused=self.paused)

    def step(self, count=1):
        self.steps += count

    def set_paused(self, paused):
        self.paused = bool(paused)
        return True


def test_external_clock_is_not_stepped_or_overwritten_by_render_ticks():
    adapter = _ExternalClock()
    session = Session(adapter)

    frame = session.tick(FrameNeeds(), wall_dt=1.0)

    assert adapter.steps == 0
    assert frame.step == 42
    assert not session.paused
    assert session.submit(cmd.Pause())
    assert adapter.paused
