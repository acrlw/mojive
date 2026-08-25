"""Publish a moving Forge scene to independent remote viewers."""

from __future__ import annotations

import argparse
import math
import time

from forge_viewer import FrameNeeds, Scene, SnapshotPublisher
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.remote import handle_session_command, snapshot_structure
from forge_viewer.session import Session


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
    print(f"Publishing on {args.host}:{args.port}; attach with forge-viewer attach")
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
