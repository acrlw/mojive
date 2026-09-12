"""Cli: viewer."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from mojive.render.backend import DebugView, RenderFlag

from .common import _resolve, log


def cmd_view(args: argparse.Namespace) -> int:
    from mojive.application.composition import build, build_workspace

    compose = build_workspace if args.backend == "mujoco" else build
    viewer = compose(
        _resolve(args.asset),
        args.backend,
        paused=args.paused,
        vsync=not args.no_vsync,
    )
    try:
        if getattr(args, "rpc_socket", None):
            viewer.start_rpc(Path(args.rpc_socket))
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Publish simulation snapshots from a headless physics process."""
    import time

    from mojive import commands as cmd
    from mojive.adapters.base import FrameNeeds
    from mojive.application.backends import make_adapter
    from mojive.capture.recording import SnapshotWriter
    from mojive.remote import (
        RemoteFrame,
        SnapshotPublisher,
        handle_session_command,
        snapshot_structure,
    )
    from mojive.session import Session

    path = _resolve(args.asset)
    with ExitStack() as resources:
        session = Session(make_adapter(args.backend, path), path)
        resources.callback(session.release)
        publisher = SnapshotPublisher(args.host, args.port)
        resources.callback(publisher.close)
        writer = (
            resources.enter_context(SnapshotWriter(Path(args.record_snapshot)))
            if args.record_snapshot
            else None
        )
        if args.paused and not session.paused:
            session.submit(cmd.Pause())
        needs = FrameNeeds(
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
        period = 1.0 / args.hz
        previous = time.perf_counter()
        deadline = previous
        published_generation = -1
        log.info("Publishing {} at {}:{}", path.name, args.host, args.port)

        def handle_command(message):
            return handle_session_command(session, message)

        while True:
            now = time.perf_counter()
            publisher.pump_commands(handle_command)
            frame = session.tick(needs, wall_dt=max(0.0, now - previous))
            previous = now
            if session.structure_generation != published_generation:
                structure = snapshot_structure(session)
                publisher.publish_structure(structure)
                if writer is not None:
                    writer.write(structure)
                published_generation = session.structure_generation
            sequence = publisher.publish_frame(frame)
            if writer is not None:
                writer.write(
                    RemoteFrame(
                        sequence,
                        frame,
                        tuple(frame.debug_commands or ()),
                        structure_revision=published_generation,
                    )
                )
            deadline += period
            delay = deadline - time.perf_counter()
            if delay > 0.0:
                # Publication cadence must not delay remote command responses.
                while delay > 0.0:
                    time.sleep(min(delay, 0.01))
                    publisher.pump_commands(handle_command)
                    delay = deadline - time.perf_counter()
            else:
                deadline = time.perf_counter()


def cmd_attach(args: argparse.Namespace) -> int:
    """Attach an independent Mojive window to a snapshot publisher."""
    from mojive.application.composition import build_from_adapter
    from mojive.remote import RemoteSceneAdapter

    viewer = build_from_adapter(
        RemoteSceneAdapter(args.host, args.port),
        vsync=not args.no_vsync,
        title=args.title,
    )
    try:
        viewer.backend.set_debug_view(DebugView(args.debug_view))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Republish a recorded snapshot stream through the normal attach path."""
    import time
    from dataclasses import replace

    from mojive.adapters.base import AdapterCaps
    from mojive.capture.recording import read_snapshots
    from mojive.commands import CommandResult
    from mojive.remote import RemoteFrame, RemoteStructure, SnapshotPublisher

    publisher = SnapshotPublisher(args.host, args.port)
    path = Path(args.snapshot)
    log.info("Replaying {} at {}:{}", path.name, args.host, args.port)

    def wait(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0.0:
            publisher.pump_commands(lambda _message: CommandResult.bad("replay is read-only"))
            time.sleep(min(remaining, 0.01))

    try:
        while True:
            previous_time = None
            packets = 0
            for packet in read_snapshots(path):
                packets += 1
                if isinstance(packet, RemoteStructure):
                    caps = AdapterCaps(
                        name=f"replay:{packet.caps.name}",
                        external_clock=True,
                        contacts=packet.caps.contacts,
                        model_cameras=bool(packet.cameras),
                        sensors=packet.caps.sensors,
                        notes=(*packet.caps.notes, "Recorded playback is read-only"),
                    )
                    publisher.publish_structure(replace(packet, caps=caps))
                elif isinstance(packet, RemoteFrame):
                    frame_time = float(packet.frame.time)
                    if previous_time is not None:
                        wait(max(0.0, frame_time - previous_time) / args.speed)
                    publisher.publish_frame(
                        replace(packet.frame, paused=True), packet.debug_commands
                    )
                    previous_time = frame_time
            if packets == 0:
                raise ValueError("snapshot recording is empty")
            if not args.loop:
                return 0
    finally:
        publisher.close()


def cmd_canvas(args: argparse.Namespace) -> int:
    from mojive.application.composition import build_scene
    from mojive.application.demos import canvas_scene, lighting_scene
    from mojive.scene import Scene

    if args.demo == "empty":
        scene = Scene()
    else:
        scene = lighting_scene() if args.demo == "lighting" else canvas_scene()
    viewer = build_scene(scene, vsync=not args.no_vsync, title=f"Mojive · {args.demo}")
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        if args.demo == "lighting":
            viewer.backend.set_flag(RenderFlag.FOG, True)
            viewer.backend.set_flag(RenderFlag.HAZE, True)
            if viewer.backend.debug is not None:
                from mojive.render.debugdraw import Occlusion

                viewer.backend.debug.layer("demo.lighting.help", Occlusion.ALWAYS).text(
                    "atmosphere",
                    (0.0, -1.0, 3.2),
                    "F9: toggle fog / haze   |   near → far",
                    align=(0.5, 1.0),
                )
        if args.demo == "text" and viewer.backend.debug is not None:
            from mojive.render.debugdraw import Occlusion

            depth = viewer.backend.debug.layer("demo.text.depth", Occlusion.DEPTH)
            depth.text("crate", (-0.7, 0.0, 1.0), "crate", offset_px=(0, -8), align=(0.5, 1))
            depth.text("ball", (0.45, -0.3, 0.92), "ball 0.42 m", offset_px=(0, -8), align=(0.5, 1))
            always = viewer.backend.debug.layer("demo.text.always", Occlusion.ALWAYS)
            always.text("origin", (0, 0, 0), "world origin", offset_px=(8, 8), align=(0, 0))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_editor(args: argparse.Namespace) -> int:
    from mojive.application.composition import build_editor, build_workspace

    viewer = (
        build_workspace(
            _resolve(args.asset),
            "mujoco",
            vsync=not args.no_vsync,
            title="Mojive",
        )
        if args.asset
        else build_editor(vsync=not args.no_vsync, title="Mojive")
    )
    try:
        if getattr(args, "rpc_socket", None):
            viewer.start_rpc(Path(args.rpc_socket))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_toy(args: argparse.Namespace) -> int:
    """Open the dependency-free reference physics adapter through the production viewer."""
    from mojive.application.backends import make_adapter
    from mojive.application.composition import build_from_adapter

    viewer = build_from_adapter(
        make_adapter("toy"), vsync=not args.no_vsync, title="Mojive toy physics"
    )
    try:
        viewer.run()
    finally:
        viewer.release()
    return 0
