"""Open an interactive Mojive window inside a private, nested Weston desktop."""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run_wayland_acceptance import ROOT, prepare, stop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--runtime", type=Path, default=ROOT / "output/wayland-runtime")
    parser.add_argument("--output", type=Path, default=ROOT / "output/native-wayland-viewer")
    parser.add_argument("--renderer", choices=("bgfx", "opengl", "wgpu"), default="bgfx")
    parser.add_argument("--scene", default="test_scene")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("viewer_args", nargs=argparse.REMAINDER, help="Viewer flags after --")
    args = parser.parse_args()
    if args.width < 640 or args.height < 480:
        parser.error("The desktop must be at least 640 x 480")
    env = dict(os.environ)
    # Resolve the parent socket before giving the nested compositor its own runtime.
    if env.get("WAYLAND_DISPLAY"):
        backend = "wayland"
        parent = Path(env["WAYLAND_DISPLAY"])
        if not parent.is_absolute():
            parent = Path(env["XDG_RUNTIME_DIR"]) / parent
        env["WAYLAND_DISPLAY"] = str(parent)
    elif env.get("DISPLAY"):
        backend = "x11"
    else:
        parser.error("Run from a graphical desktop; use native-wayland-test for headless tests")

    prefix = args.runtime.resolve() / "root/usr"
    if args.prepare:
        prefix, _ = prepare(args.runtime.resolve())
    weston = prefix / "bin/weston"
    if not weston.is_file():
        parser.error("Run with --prepare to extract the private Weston runtime")
    lib = next((prefix / "lib").glob("**/libweston-9.so.0")).parent
    bundle = Path(importlib.util.find_spec("imgui_bundle").origin).parent
    glfw = bundle / "libmojive_glfw.so.3"
    if not glfw.is_file():
        parser.error("Run make setup-imgui to build the dual-platform GLFW dependency")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(lib / "weston"), str(lib), env.get("LD_LIBRARY_PATH", ""))
    )
    env["WESTON_MODULE_MAP"] = ";".join(f"{path.name}={path}" for path in lib.glob("*/*.so"))
    env.pop("WAYLAND_SOCKET", None)
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{args.renderer}-", dir=args.output.resolve()))

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with tempfile.TemporaryDirectory(prefix="mojive-desktop-") as runtime:
        config = Path(runtime) / "weston.ini"
        config.write_text(
            f"[shell]\nclient={lib}/weston-desktop-shell\n"
            "locking=false\npanel-position=none\nbackground-color=0xff242830\n"
            "background-image=\nstartup-animation=none\nclose-animation=none\n"
            "[input-method]\npath=\n"
        )
        env["XDG_RUNTIME_DIR"] = runtime
        compositor = child = None
        with (output / "weston.log").open("w") as log:
            try:
                compositor = subprocess.Popen(
                    [
                        str(weston),
                        f"--config={config}",
                        "--idle-time=0",
                        f"--backend={lib}/libweston-9/{backend}-backend.so",
                        f"--shell={lib}/weston/desktop-shell.so",
                        "--socket=mojive-viewer",
                        f"--width={args.width}",
                        f"--height={args.height}",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                for _ in range(100):
                    if compositor.poll() is not None:
                        raise RuntimeError(f"Weston exited; see {log.name}")
                    if Path(runtime, "mojive-viewer").exists():
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError(f"Weston did not create its socket; see {log.name}")

                # Only Weston connects to the parent desktop. Mojive uses native Wayland.
                for name in ("DISPLAY", "PYOPENGL_PLATFORM", "MOJIVE_GL"):
                    env.pop(name, None)
                env.update(
                    WAYLAND_DISPLAY="mojive-viewer",
                    XDG_SESSION_TYPE="wayland",
                    PYGLFW_LIBRARY=str(glfw),
                    MOJIVE_RENDERER=args.renderer,
                    MOJIVE_CONFIG_DIR=runtime,
                    MOJIVE_IMGUI_INI=str(Path(runtime) / "imgui.ini"),
                    MOJIVE_SETTINGS=str(Path(runtime) / "settings.json"),
                )
                env.setdefault("MOJIVE_NATIVE_BUILD", str(ROOT / "output/cpp-build"))
                viewer_args = args.viewer_args
                if viewer_args[:1] == ["--"]:
                    viewer_args = viewer_args[1:]
                print(f"Interactive Wayland {args.renderer} viewer; logs: {output}", flush=True)
                print("Close Mojive, close Weston, or press Ctrl+C here to stop.", flush=True)
                with (output / "viewer.log").open("w") as viewer_log:
                    child = subprocess.Popen(
                        [sys.executable, "-m", "mojive.cli", "view", args.scene, *viewer_args],
                        cwd=ROOT,
                        env=env,
                        stdout=viewer_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    while child.poll() is None and compositor.poll() is None:
                        time.sleep(0.1)
                    if child.poll() not in (None, 0):
                        print(viewer_log.name + ":\n" + Path(viewer_log.name).read_text())
                        return 1
                    return 0
            except KeyboardInterrupt:
                return 0
            finally:
                if child is not None:
                    stop(child, group=True)
                if compositor is not None:
                    stop(compositor, group=True)


if __name__ == "__main__":
    raise SystemExit(main())
