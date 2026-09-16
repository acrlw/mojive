"""Passive viewer: this program owns physics; the UI runs on its own thread."""

import argparse
import math
import threading
import time
from pathlib import Path

import mujoco

import mojive.viewer as viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).resolve().parents[1] / "assets/joint_types.xml"
    )
    parser.add_argument("--renderer", choices=("opengl", "wgpu", "bgfx"), default="opengl")
    parser.add_argument(
        "--seconds", type=float, default=0, help="Close after this duration; 0 waits for the window"
    )
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("seconds must be finite and nonnegative")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    paused = threading.Event()

    def key_callback(keycode):
        if keycode == ord(" "):
            if paused.is_set():
                paused.clear()
            else:
                paused.set()

    print(
        "Passive mode: this script steps physics. Space pauses this script; close the window to exit.",
        flush=True,
    )
    with viewer.launch_passive(
        model, data, key_callback=key_callback, renderer=args.renderer
    ) as handle:
        deadline = time.monotonic() + args.seconds if args.seconds else math.inf
        while handle.is_running() and time.monotonic() < deadline:
            started = time.monotonic()
            if not paused.is_set():
                mujoco.mj_step(model, data)
            handle.sync()
            time.sleep(max(0, model.opt.timestep - (time.monotonic() - started)))
    print(f"Viewer closed; simulation time: {data.time:.3f} s", flush=True)


if __name__ == "__main__":
    main()
