"""Run the MuJoCo passive-viewer loop with the Mojive compatibility import."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

import mojive.viewer as viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    paused = False

    def key_callback(keycode: int):
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused

    with viewer.launch_passive(
        model, data, key_callback=key_callback, renderer=args.renderer
    ) as handle:
        with handle.lock():
            handle.cam.distance = 5
            handle.cam.elevation = -25
            handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            mujoco.mjv_initGeom(
                handle.user_scn.geoms[0],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.1, 0, 0],
                [0, 0, 2],
                np.eye(3).reshape(-1),
                [1, 0.2, 0.1, 1],
            )
            handle.user_scn.ngeom = 1
        deadline = time.monotonic() + args.seconds
        while handle.is_running() and time.monotonic() < deadline:
            started = time.monotonic()
            if not paused:
                mujoco.mj_step(model, data)
            handle.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
