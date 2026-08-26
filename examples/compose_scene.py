"""Compose MJCF or URDF models and save a Forge workspace or MJCF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from forge_viewer import CameraView, Light, LightType, MuJoCoAdapter
from forge_viewer.adapters.workspace import WorkspaceAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary = MuJoCoAdapter()
    primary.new_scene()
    workspace = WorkspaceAdapter(primary)
    for index, model in enumerate(args.models):
        workspace.add_scene_model(
            model.expanduser().resolve(),
            np.array((index * args.spacing, 0.0, 0.0), np.float32),
            np.eye(3, dtype=np.float32),
        )
    workspace.add_scene_light(
        "key",
        Light(
            type=LightType.SPOT,
            position=np.array((3.0, -4.0, 6.0), np.float32),
            direction=np.array((-0.4, 0.5, -1.0), np.float32),
        ),
    )
    workspace.add_scene_camera(
        "overview",
        CameraView(
            eye=np.array((6.0, -8.0, 5.0), np.float32),
            target=np.array((args.spacing * (len(args.models) - 1) * 0.5, 0.0, 0.5), np.float32),
        ),
    )
    try:
        workspace.save_scene(args.output)
    finally:
        workspace.release()
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
