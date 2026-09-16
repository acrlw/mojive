"""MuJoCo-style interactive entry points; rendering uses the full Mojive desktop UI.

Use ``import mojive.viewer as viewer`` when migrating ``mujoco.viewer`` calls.
The separately scheduled, process-based extension remains ``mojive.launch_passive``.
"""

from mojive.app.mujoco_viewer.handle import Handle as Handle
from mojive.app.mujoco_viewer.launch import launch as launch
from mojive.app.mujoco_viewer.launch import launch_from_path as launch_from_path
from mojive.app.mujoco_viewer.launch import launch_passive as launch_passive

__all__ = ["Handle", "launch", "launch_from_path", "launch_passive"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch the interactive Mojive viewer")
    parser.add_argument("--mjcf", help="MuJoCo model path")
    args = parser.parse_args()
    launch_from_path(args.mjcf) if args.mjcf else launch()
