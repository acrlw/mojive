from __future__ import annotations

import argparse
import socket
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .. import commands as cmd
from ..adapters.base import FrameNeeds
from ..adapters.static import StaticSceneAdapter
from ..composition import build_from_adapter
from ..remote import (
    RemoteSceneAdapter,
    SnapshotPublisher,
    handle_session_command,
    snapshot_structure,
)
from ..render.backend import RenderFlag
from ..scene import Scene
from ..session import Session
from ..types import CameraView, Light, LightKind, MeshShape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Live View scene authoring")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/remote-authoring.png"))
    args = parser.parse_args(argv)

    camera = CameraView(
        eye=np.array([5.0, -6.0, 4.0], np.float32),
        target=np.array([0.0, 0.0, 0.6], np.float32),
    )
    source = Session(StaticSceneAdapter(Scene(camera=camera)))
    publisher = SnapshotPublisher(port=_port_pair())
    publisher.publish_structure(snapshot_structure(source))
    publisher.publish_frame(source.frame)
    stop = threading.Event()
    worker = threading.Thread(target=_publish, args=(publisher, source, stop), daemon=True)
    worker.start()
    viewer = build_from_adapter(
        RemoteSceneAdapter(port=publisher.port),
        vsync=False,
        width=1400,
        height=900,
        title="forge Live View authoring",
    )
    try:
        _author_scene(viewer.session, camera)
        viewer.backend.set_flag(RenderFlag.LIGHT, True)
        viewer.backend.set_flag(RenderFlag.CAMERA, True)
        for _ in range(5):
            viewer.sync()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pixels = viewer.window.read_frame()[::-1, :, :3]
        Image.fromarray(pixels, "RGB").save(args.output)
    finally:
        viewer.release()
        stop.set()
        worker.join()
        publisher.close()
        source.release()

    print(args.output.resolve())
    return 0


def _publish(publisher: SnapshotPublisher, source: Session, stop: threading.Event) -> None:
    generation = source.structure_generation
    while not stop.is_set():
        publisher.pump_commands(lambda message: handle_session_command(source, message))
        if source.structure_generation != generation:
            generation = source.structure_generation
            publisher.publish_structure(snapshot_structure(source))
        publisher.publish_frame(source.tick(FrameNeeds()))
        stop.wait(1.0 / 120.0)


def _author_scene(session: Session, camera: CameraView) -> None:
    objects = (
        cmd.AddSceneObject(
            MeshShape.PLANE,
            "ground",
            size=(4.0, 4.0, 0.03),
            position=(0.0, 0.0, -0.03),
            color=(0.18, 0.22, 0.28, 1.0),
        ),
        cmd.AddSceneObject(
            MeshShape.BOX,
            "remote box",
            size=(0.55, 0.55, 0.55),
            position=(-0.9, 0.0, 0.55),
            color=(0.95, 0.35, 0.22, 1.0),
        ),
        cmd.AddSceneObject(
            MeshShape.SPHERE,
            "remote sphere",
            size=(0.65, 0.65, 0.65),
            position=(0.9, 0.0, 0.65),
            color=(0.22, 0.55, 0.95, 1.0),
        ),
    )
    for command in objects:
        _require(session.submit(command))
    _require(
        session.submit(
            cmd.AddSceneLight(
                "remote key",
                Light(
                    kind=LightKind.POINT,
                    position=np.array([0.0, -1.5, 3.5], np.float32),
                    diffuse=np.array([1.0, 0.85, 0.7], np.float32),
                    range=12.0,
                ),
            )
        )
    )
    _require(session.submit(cmd.AddSceneCamera("remote camera", camera)))


def _require(result) -> None:
    if not result.ok:
        raise RuntimeError(result.message)


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


if __name__ == "__main__":
    raise SystemExit(main())
