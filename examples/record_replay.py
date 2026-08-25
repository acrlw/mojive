"""Record programmatic scene frames for later replay."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from forge_viewer import FrameNeeds, Scene, SnapshotWriter
from forge_viewer.adapters.static import StaticSceneAdapter
from forge_viewer.remote import RemoteFrame, snapshot_structure
from forge_viewer.session import Session


def parse_args() -> argparse.Namespace:
    """Parse recording duration and destination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/examples/orbit.fvs"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    """Write structure once and a sequence of dynamic frame packets."""
    args = parse_args()
    scene = Scene()
    scene.plane(name="floor", size=(4.0, 4.0, 0.04))
    marker = scene.sphere(name="marker", size=(0.25, 0.25, 0.25))
    session = Session(StaticSceneAdapter(scene))
    try:
        with SnapshotWriter(args.output) as writer:
            writer.write(snapshot_structure(session))
            for sequence in range(1, max(1, args.frames) + 1):
                time = (sequence - 1) / max(args.fps, 1.0)
                marker.set_pose((math.cos(time), math.sin(time), 0.5))
                frame = session.tick(FrameNeeds(poses=True), wall_dt=0.0)
                frame.time = time
                frame.step = sequence - 1
                writer.write(RemoteFrame(sequence, frame, tuple(frame.debug_commands or ())))
    finally:
        session.release()
    path = args.output.expanduser().resolve()
    print(path)
    print(f"Replay with: forge-viewer replay {path}")


if __name__ == "__main__":
    main()
