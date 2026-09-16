"""Managed viewer: Mojive owns the simulation loop; close the window to return."""

import argparse
from pathlib import Path

import mujoco

import mojive.viewer as viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).resolve().parents[1] / "assets/joint_types.xml"
    )
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    print(
        "Managed mode: Mojive steps physics. Space toggles playback; close the window to exit.",
        flush=True,
    )
    viewer.launch(model, data, renderer=args.renderer)
    print(f"Viewer closed; simulation time: {data.time:.3f} s", flush=True)


if __name__ == "__main__":
    main()
