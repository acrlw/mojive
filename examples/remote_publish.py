"""Publish a moving Mojive scene to independent remote viewers."""

from __future__ import annotations

import argparse
import math
import time

from mojive import FrameNeeds, Scene, SnapshotPublisher
from mojive.adapters.static import StaticSceneAdapter
from mojive.remote import handle_session_command, snapshot_structure
from mojive.session import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47650)
    parser.add_argument("--hz", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Scene()
    scene.plane(name="floor", size=(5.0, 5.0, 0.04))
    marker = scene.sphere(name="marker", size=(0.25, 0.25, 0.25))
    session = Session(StaticSceneAdapter(scene))
    publisher = SnapshotPublisher(args.host, args.port)
    publisher.publish_structure(snapshot_structure(session))
    period = 1.0 / max(args.hz, 1.0)
    started = time.perf_counter()
    print(f"Publishing on {args.host}:{args.port}; attach with uv run mojive attach")
    try:
        while True:
            frame_start = time.perf_counter()
            elapsed = frame_start - started
            marker.set_pose((math.cos(elapsed), math.sin(elapsed), 0.5))
            publisher.pump_commands(lambda message: handle_session_command(session, message))
            publisher.publish_frame(session.tick(FrameNeeds(poses=True)))
            time.sleep(max(0.0, period - (time.perf_counter() - frame_start)))
    finally:
        publisher.close()
        session.release()


if __name__ == "__main__":
    main()
